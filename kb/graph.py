"""Kùzu graph store: the structure graph and the dependency graph.

Schema (one database; both graphs share the same File nodes):

    Structure graph
        (:Folder)-[:CONTAINS_FOLDER]->(:Folder)
        (:Folder)-[:CONTAINS_FILE]->(:File)

    Dependency graph
        (:File)-[:IMPORTS]->(:File)              import resolved to a repo file
        (:File)-[:IMPORTS_EXTERNAL]->(:Module)   stdlib or outside the repo

Because File nodes are shared, one query can mix structure and dependencies,
e.g. "which folders does src/flask/app.py depend on?".

Note: Kùzu's repository was archived in October 2025 and 0.11.3 is the final
release. All Kùzu calls live in this file, so swapping the store later only
changes this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import kuzu

from kb.parser import FileNode, FolderNode

SCHEMA = [
    """CREATE NODE TABLE IF NOT EXISTS Folder(
        id STRING, name STRING, repo STRING, rel_path STRING, depth INT64,
        PRIMARY KEY (id))""",
    """CREATE NODE TABLE IF NOT EXISTS File(
        id STRING, name STRING, repo STRING, rel_path STRING,
        extension STRING, language STRING, size_bytes INT64, line_count INT64,
        PRIMARY KEY (id))""",
    # Modules are shared across repos ("os" is the same module everywhere).
    """CREATE NODE TABLE IF NOT EXISTS Module(
        name STRING, kind STRING,
        PRIMARY KEY (name))""",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS_FOLDER(FROM Folder TO Folder)",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS_FILE(FROM Folder TO File)",
    # One edge per import statement, so each edge traces back to a line.
    """CREATE REL TABLE IF NOT EXISTS IMPORTS(
        FROM File TO File, line INT64, statement STRING, names STRING[],
        type_only BOOLEAN)""",
    """CREATE REL TABLE IF NOT EXISTS IMPORTS_EXTERNAL(
        FROM File TO Module, line INT64, statement STRING, names STRING[],
        type_only BOOLEAN)""",
]


class GraphStore:
    def __init__(self, db_path: str = "./kuzu_db", read_only: bool = False):
        # Kùzu creates db_path itself as a file and fails if it is a folder, so
        # only its parent folder is created here.
        Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        # Kùzu allows one read-write connection or many read-only ones.
        self.db = kuzu.Database(db_path, read_only=read_only)
        self.conn = kuzu.Connection(self.db)

    def close(self) -> None:
        self.conn.close()
        self.db.close()

    # ---------- schema and reset ----------

    def create_schema(self) -> None:
        for statement in SCHEMA:
            self.conn.execute(statement)

    def delete_repo(self, repo: str) -> None:
        """Remove a repo's folders and files (and their edges). Module nodes
        are shared between repos and are left in place."""
        self.conn.execute("MATCH (f:File) WHERE f.repo = $repo DETACH DELETE f",
                          {"repo": repo})
        self.conn.execute("MATCH (d:Folder) WHERE d.repo = $repo DETACH DELETE d",
                          {"repo": repo})

    # ---------- structure graph ----------

    def add_folders(self, folders: Iterable[FolderNode]) -> None:
        folders = list(folders)
        for f in folders:
            self.conn.execute(
                """CREATE (:Folder {id: $id, name: $name, repo: $repo,
                                    rel_path: $rel_path, depth: $depth})""",
                {"id": f.id, "name": f.name, "repo": f.repo,
                 "rel_path": f.rel_path, "depth": f.depth})
        # Edges after all folders exist, so walk order does not matter.
        for f in folders:
            if f.parent_id is not None:
                self.conn.execute(
                    """MATCH (p:Folder {id: $parent}), (c:Folder {id: $child})
                       CREATE (p)-[:CONTAINS_FOLDER]->(c)""",
                    {"parent": f.parent_id, "child": f.id})

    def add_files(self, files: Iterable[FileNode]) -> None:
        for f in files:
            self.conn.execute(
                """CREATE (:File {id: $id, name: $name, repo: $repo,
                                  rel_path: $rel_path, extension: $extension,
                                  language: $language, size_bytes: $size,
                                  line_count: $lines})""",
                {"id": f.id, "name": f.name, "repo": f.repo,
                 "rel_path": f.rel_path, "extension": f.extension,
                 "language": f.language, "size": f.size_bytes,
                 "lines": f.line_count})
            self.conn.execute(
                """MATCH (d:Folder {id: $folder}), (f:File {id: $file})
                   CREATE (d)-[:CONTAINS_FILE]->(f)""",
                {"folder": f.parent_id, "file": f.id})

    # ---------- dependency graph ----------

    def add_internal_import(self, source_id: str, target_id: str, line: int,
                            statement: str, names: list[str], type_only: bool) -> None:
        self.conn.execute(
            """MATCH (a:File {id: $src}), (b:File {id: $dst})
               CREATE (a)-[:IMPORTS {line: $line, statement: $stmt,
                                     names: $names, type_only: $type_only}]->(b)""",
            {"src": source_id, "dst": target_id, "line": line,
             "stmt": statement, "names": names, "type_only": type_only})

    def add_external_import(self, source_id: str, module: str, kind: str,
                            line: int, statement: str, names: list[str],
                            type_only: bool) -> None:
        self.conn.execute(
            "MERGE (m:Module {name: $name}) ON CREATE SET m.kind = $kind",
            {"name": module, "kind": kind})
        self.conn.execute(
            """MATCH (a:File {id: $src}), (m:Module {name: $name})
               CREATE (a)-[:IMPORTS_EXTERNAL {line: $line, statement: $stmt,
                                              names: $names,
                                              type_only: $type_only}]->(m)""",
            {"src": source_id, "name": module, "line": line,
             "stmt": statement, "names": names, "type_only": type_only})

    # ---------- reading ----------

    def query(self, cypher: str, params: dict | None = None) -> list[list]:
        result = self.conn.execute(cypher, params or {})
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def dependencies_of(self, file_id: str, include_type_only: bool = False
                        ) -> list[str]:
        """Repo files that file_id imports directly."""
        return [row[0] for row in self.query(
            """MATCH (a:File {id: $id})-[r:IMPORTS]->(b:File)
               WHERE $include_type_only OR NOT r.type_only
               RETURN DISTINCT b.id ORDER BY b.id""",
            {"id": file_id, "include_type_only": include_type_only})]

    def dependents_of(self, file_id: str, include_type_only: bool = False
                      ) -> list[str]:
        """Repo files that import file_id directly."""
        return [row[0] for row in self.query(
            """MATCH (a:File)-[r:IMPORTS]->(b:File {id: $id})
               WHERE $include_type_only OR NOT r.type_only
               RETURN DISTINCT a.id ORDER BY a.id""",
            {"id": file_id, "include_type_only": include_type_only})]