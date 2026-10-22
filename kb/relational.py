"""PostgreSQL store: the source of truth for files, code structure and rules.

Tables (every row hangs off files.id, the shared "<repo>/<path>" id):

    files              one row per file in the repo
    classes            classes found by the parser
    class_attributes   attributes of each class
    functions          functions and methods (class_id set for methods)
    business_rules     one-sentence rules extracted by the LLM, one per feature

Deleting a repo's rows from `files` removes everything else for that repo,
because every other table references files with ON DELETE CASCADE.

Connection string: pass one in, or set DATABASE_URL, e.g.
    postgresql://postgres@localhost:5432/repo_analysis
"""

from __future__ import annotations

import os
from typing import Iterable

import psycopg

from kb.parser import FileNode, FileStructure

DEFAULT_DSN = "postgresql://postgres@localhost:5432/repo_analysis"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,
    repo        TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    name        TEXT NOT NULL,
    extension   TEXT NOT NULL,
    language    TEXT NOT NULL,
    size_bytes  BIGINT NOT NULL,
    line_count  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_repo_idx ON files (repo);

CREATE TABLE IF NOT EXISTS classes (
    id          SERIAL PRIMARY KEY,
    file_id     TEXT NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    bases       TEXT[] NOT NULL,
    decorators  TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS class_attributes (
    id          SERIAL PRIMARY KEY,
    class_id    INTEGER NOT NULL REFERENCES classes (id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    line        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS functions (
    id          SERIAL PRIMARY KEY,
    file_id     TEXT NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    class_id    INTEGER REFERENCES classes (id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    parameters  TEXT[] NOT NULL,
    decorators  TEXT[] NOT NULL,
    is_async    BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS business_rules (
    id          SERIAL PRIMARY KEY,
    file_id     TEXT NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    feature     TEXT NOT NULL,
    rule        TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class RelationalStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN)
        # autocommit: each method is one unit of work and is visible at once.
        self.conn = psycopg.connect(self.dsn, autocommit=True)

    def close(self) -> None:
        self.conn.close()

    # ---------- schema and reset ----------

    def create_schema(self) -> None:
        self.conn.execute(SCHEMA)

    def delete_repo(self, repo: str) -> None:
        """Remove a repo's files and, by cascade, everything attached to them."""
        self.conn.execute("DELETE FROM files WHERE repo = %s", (repo,))

    def delete_business_rules(self, repo: str) -> None:
        """Remove only a repo's rules, e.g. before re-running extraction."""
        self.conn.execute(
            """DELETE FROM business_rules
               WHERE file_id IN (SELECT id FROM files WHERE repo = %s)""", (repo,))

    # ---------- writing ----------

    def add_files(self, files: Iterable[FileNode]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO files (id, repo, rel_path, name, extension,
                                      language, size_bytes, line_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [(f.id, f.repo, f.rel_path, f.name, f.extension, f.language,
                  f.size_bytes, f.line_count) for f in files])

    def add_structure(self, file_id: str, structure: FileStructure) -> None:
        """Store one file's classes, their attributes, and its functions."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            class_ids: dict[tuple[str, int], int] = {}
            for c in structure.classes:
                cur.execute(
                    """INSERT INTO classes (file_id, name, start_line, end_line,
                                            bases, decorators)
                       VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                    (file_id, c.name, c.start_line, c.end_line, c.bases, c.decorators))
                class_id = cur.fetchone()[0]
                class_ids[(c.name, c.start_line)] = class_id
                cur.executemany(
                    """INSERT INTO class_attributes (class_id, name, line)
                       VALUES (%s, %s, %s)""",
                    [(class_id, name, line) for name, line in c.attributes])

            for f in structure.functions:
                # (name, start line) identifies a class even when two classes
                # in one file share a name.
                class_id = class_ids.get((f.class_name, f.class_start_line))
                cur.execute(
                    """INSERT INTO functions (file_id, class_id, name, start_line,
                                              end_line, parameters, decorators,
                                              is_async)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (file_id, class_id, f.name, f.start_line, f.end_line,
                     f.parameters, f.decorators, f.is_async))

    def add_business_rules(self, file_id: str, rules: Iterable[tuple[str, str]],
                           model: str) -> list[int]:
        """Store (feature, rule) pairs for one file; return the new row ids."""
        ids = []
        with self.conn.transaction(), self.conn.cursor() as cur:
            for feature, rule in rules:
                cur.execute(
                    """INSERT INTO business_rules (file_id, feature, rule, model)
                       VALUES (%s, %s, %s, %s) RETURNING id""",
                    (file_id, feature, rule, model))
                ids.append(cur.fetchone()[0])
        return ids

    # ---------- reading ----------

    def fetch_business_rules(self, repo: str) -> list[dict]:
        """All rules for a repo, with the file they came from."""
        with self.conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """SELECT r.id, r.file_id, f.rel_path, f.language, r.feature,
                          r.rule, r.model
                   FROM business_rules r JOIN files f ON f.id = r.file_id
                   WHERE f.repo = %s ORDER BY f.rel_path, r.id""", (repo,))
            return cur.fetchall()

    def fetch_file_ids_with_rules(self, repo: str) -> set[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT r.file_id FROM business_rules r
                   JOIN files f ON f.id = r.file_id WHERE f.repo = %s""", (repo,))
            return {row[0] for row in cur.fetchall()}

    def counts(self, repo: str) -> dict[str, int]:
        """Row counts per table for one repo, for checking a run."""
        queries = {
            "files": "SELECT count(*) FROM files WHERE repo = %s",
            "classes": """SELECT count(*) FROM classes c JOIN files f
                          ON f.id = c.file_id WHERE f.repo = %s""",
            "class_attributes": """SELECT count(*) FROM class_attributes a
                                   JOIN classes c ON c.id = a.class_id
                                   JOIN files f ON f.id = c.file_id
                                   WHERE f.repo = %s""",
            "functions": """SELECT count(*) FROM functions fn JOIN files f
                            ON f.id = fn.file_id WHERE f.repo = %s""",
            "business_rules": """SELECT count(*) FROM business_rules r JOIN files f
                                 ON f.id = r.file_id WHERE f.repo = %s""",
        }
        result = {}
        with self.conn.cursor() as cur:
            for table, sql in queries.items():
                cur.execute(sql, (repo,))
                result[table] = cur.fetchone()[0]
        return result