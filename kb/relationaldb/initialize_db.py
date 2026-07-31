import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "repo_analysis"

DB_CONFIG = {
    "host": "localhost",
    "dbname": DB_NAME,
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}

ADMIN_CONFIG = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def get_connection():
    """Returns a new connection to the analysis DB."""
    return psycopg.connect(**DB_CONFIG)


def create_db():
    """Creates the repo_analysis database if it does not already exist."""
    conn = psycopg.connect(**ADMIN_CONFIG, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            if cur.fetchone() is None:
                cur.execute(f"CREATE DATABASE {DB_NAME}")
                print(f"Created database '{DB_NAME}'")
            else:
                print(f"Database '{DB_NAME}' already exists")
    finally:
        conn.close()


# ── Schema migrations ─────────────────────────────────────────────────────────
# Add idempotent ALTER statements here whenever the schema evolves. Because
# CREATE TABLE IF NOT EXISTS does not modify existing tables, these keep older
# databases in sync. All use IF NOT EXISTS so they are safe to run every time.

MIGRATIONS = [
    "ALTER TABLE files ADD COLUMN IF NOT EXISTS rules_extracted BOOLEAN DEFAULT FALSE",
]


def init_db():
    """Creates all tables. Safe to run repeatedly."""
    conn = psycopg.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id BIGSERIAL PRIMARY KEY,
                repository_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT,
                classes INTEGER DEFAULT 0,
                functions INTEGER DEFAULT 0,
                methods INTEGER DEFAULT 0,
                async_functions INTEGER DEFAULT 0,
                imports INTEGER DEFAULT 0,
                lines INTEGER DEFAULT 0,
                rules_extracted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(repository_name, file_path)
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS business_rules (
                id BIGSERIAL PRIMARY KEY,
                file_id BIGINT NOT NULL
                    REFERENCES files(id) ON DELETE CASCADE,
                rule_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS key_points (
                id BIGSERIAL PRIMARY KEY,
                repository_name TEXT NOT NULL,
                point_index INTEGER NOT NULL,
                point_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(repository_name, point_index)
            )
            """)

            # Structural detail — names, not just the counts in files.classes/
            # functions/methods. Python is exact (ast); C#/JS/TS are regex +
            # brace-counting, same approach and same tradeoff as kb/diagram.py's
            # UML class scanner, so the two stay consistent with each other.
            cur.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id BIGSERIAL PRIMARY KEY,
                file_id BIGINT NOT NULL
                    REFERENCES files(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                bases TEXT[] DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS class_attributes (
                id BIGSERIAL PRIMARY KEY,
                class_id BIGINT NOT NULL
                    REFERENCES classes(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                type TEXT,
                visibility TEXT
            )
            """)

            # class_id NULL = a free (module-level) function; set = a method.
            # ON DELETE CASCADE on class_id means deleting a class also cleans
            # up its methods here — free functions are cleaned up by file_id.
            cur.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                id BIGSERIAL PRIMARY KEY,
                file_id BIGINT NOT NULL
                    REFERENCES files(id) ON DELETE CASCADE,
                class_id BIGINT
                    REFERENCES classes(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                params TEXT[] DEFAULT '{}',
                return_type TEXT,
                is_async BOOLEAN DEFAULT FALSE,
                visibility TEXT
            )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_classes_file ON classes(file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_class_attrs_class ON class_attributes(class_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_functions_class ON functions(class_id)")

            # ── Migrations ────────────────────────────────────────────────────
            # Idempotent ALTERs so existing tables pick up columns added after
            # their initial creation. Each is safe to run on every startup.
            for migration in MIGRATIONS:
                cur.execute(migration)

        conn.commit()
        print("Tables ensured: files, business_rules, key_points, classes, class_attributes, functions")
        print(f"Migrations applied: {len(MIGRATIONS)}")
    finally:
        conn.close()


# ── File metrics ──────────────────────────────────────────────────────────────

def save_file(conn, repository_name, file_path, language, metrics):
    """Inserts or updates a file record (metrics only). Returns file_id.
    Note: does NOT touch rules_extracted, so re-scanning won't wipe rule state."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO files (
                repository_name, file_path, language,
                classes, functions, methods,
                async_functions, imports, lines
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (repository_name, file_path)
            DO UPDATE SET
                language = EXCLUDED.language,
                classes = EXCLUDED.classes,
                functions = EXCLUDED.functions,
                methods = EXCLUDED.methods,
                async_functions = EXCLUDED.async_functions,
                imports = EXCLUDED.imports,
                lines = EXCLUDED.lines
            RETURNING id
            """,
            (
                repository_name, file_path, language,
                metrics["classes"], metrics["functions"], metrics["methods"],
                metrics["async_functions"], metrics["imports"], metrics["lines"],
            ),
        )
        return cur.fetchone()[0]


# ── Structural detail (names, not just counts) ────────────────────────────────

def save_structure(conn, file_id, classes, functions):
    """
    Replaces all classes/attributes/methods/free-functions for a file with
    fresh ones. Shapes match what parser.py's analyze_file() produces:

      classes:   [{"name", "bases": [...], "attrs": [{"name","type","vis"}],
                   "methods": [{"name","params","ret","vis","is_async"}]}]
      functions: [{"name","params","ret","vis","is_async"}]   (free functions only)

    Deleting a file's classes cascades to class_attributes and to any
    functions row using it as class_id (methods); free functions
    (class_id IS NULL) aren't covered by that cascade, so they're deleted
    by file_id separately.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM classes WHERE file_id = %s", (file_id,))
        cur.execute("DELETE FROM functions WHERE file_id = %s AND class_id IS NULL", (file_id,))

        for c in classes:
            cur.execute(
                "INSERT INTO classes (file_id, name, bases) VALUES (%s, %s, %s) RETURNING id",
                (file_id, c["name"], c.get("bases") or []),
            )
            class_id = cur.fetchone()[0]

            for a in c.get("attrs", []):
                cur.execute(
                    "INSERT INTO class_attributes (class_id, name, type, visibility) "
                    "VALUES (%s, %s, %s, %s)",
                    (class_id, a["name"], a.get("type") or None, a.get("vis")),
                )
            for m in c.get("methods", []):
                cur.execute(
                    "INSERT INTO functions (file_id, class_id, name, params, return_type, "
                    "is_async, visibility) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (file_id, class_id, m["name"], m.get("params") or [],
                     m.get("ret") or None, m.get("is_async", False), m.get("vis")),
                )

        for f in functions:
            cur.execute(
                "INSERT INTO functions (file_id, class_id, name, params, return_type, "
                "is_async, visibility) VALUES (%s, NULL, %s, %s, %s, %s, %s)",
                (file_id, f["name"], f.get("params") or [], f.get("ret") or None,
                 f.get("is_async", False), f.get("vis")),
            )


# ── Business rules ────────────────────────────────────────────────────────────

def save_business_rules(conn, file_id, business_rules):
    """Replaces all business rules for a file and marks it extracted."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM business_rules WHERE file_id = %s", (file_id,))
        for rule in business_rules:
            cur.execute(
                "INSERT INTO business_rules (file_id, rule_text) VALUES (%s, %s)",
                (file_id, rule),
            )
        cur.execute(
            "UPDATE files SET rules_extracted = TRUE WHERE id = %s",
            (file_id,),
        )


# ── Key points ────────────────────────────────────────────────────────────────

def save_key_points(conn, repository_name, key_points):
    """Replaces all key points for a repository. List order = point_index."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM key_points WHERE repository_name = %s", (repository_name,))
        for idx, point in enumerate(key_points):
            cur.execute(
                "INSERT INTO key_points (repository_name, point_index, point_text) VALUES (%s, %s, %s)",
                (repository_name, idx, point),
            )


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_files(conn, repository_name, only_unprocessed=False):
    """
    Returns a list of file dicts for a repo:
      { file_id, file_path, language, rules_extracted }
    If only_unprocessed, returns only files whose rules haven't been extracted.
    """
    query = """
        SELECT id, file_path, language, rules_extracted
        FROM files
        WHERE repository_name = %s
    """
    if only_unprocessed:
        query += " AND rules_extracted = FALSE"
    query += " ORDER BY file_path"

    with conn.cursor() as cur:
        cur.execute(query, (repository_name,))
        return [
            {"file_id": r[0], "file_path": r[1], "language": r[2], "rules_extracted": r[3]}
            for r in cur.fetchall()
        ]


def load_business_rules_from_db(conn, repository_name):
    """Returns { file_path: [rule_text, ...] } for a repository."""
    result = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.file_path, b.rule_text
            FROM files f
            JOIN business_rules b ON b.file_id = f.id
            WHERE f.repository_name = %s
            ORDER BY f.file_path, b.id
            """,
            (repository_name,),
        )
        for file_path, rule_text in cur.fetchall():
            result.setdefault(file_path, []).append(rule_text)
    return result


def load_key_points_from_db(conn, repository_name):
    """Returns an ordered list of key point strings for a repository."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT point_text FROM key_points WHERE repository_name = %s ORDER BY point_index",
            (repository_name,),
        )
        return [row[0] for row in cur.fetchall()]


def load_structure_from_db(conn, repository_name):
    """
    Returns { file_path: {"classes": [...], "functions": [...]} } for a repo,
    reassembled from classes/class_attributes/functions. Free functions
    (class_id IS NULL) land in each file's "functions" list; everything else
    nests under its class's "methods"/"attrs" — same shape parser.py produces,
    so this is the inverse of save_structure().
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT f.file_path, c.id, c.name, c.bases
            FROM classes c JOIN files f ON f.id = c.file_id
            WHERE f.repository_name = %s
            ORDER BY f.file_path, c.id
        """, (repository_name,))
        class_rows = cur.fetchall()

        cur.execute("""
            SELECT c.id, a.name, a.type, a.visibility
            FROM class_attributes a
            JOIN classes c ON c.id = a.class_id
            JOIN files f ON f.id = c.file_id
            WHERE f.repository_name = %s
            ORDER BY a.id
        """, (repository_name,))
        attr_rows = cur.fetchall()

        cur.execute("""
            SELECT f.file_path, fn.class_id, fn.name, fn.params, fn.return_type,
                   fn.is_async, fn.visibility
            FROM functions fn
            JOIN files f ON f.id = fn.file_id
            WHERE f.repository_name = %s
            ORDER BY f.file_path, fn.id
        """, (repository_name,))
        func_rows = cur.fetchall()

    result: dict = {}
    classes_by_id = {}
    for file_path, class_id, name, bases in class_rows:
        entry = {"name": name, "bases": list(bases or []), "attrs": [], "methods": []}
        classes_by_id[class_id] = entry
        result.setdefault(file_path, {"classes": [], "functions": []})["classes"].append(entry)

    for class_id, name, type_, vis in attr_rows:
        if class_id in classes_by_id:
            classes_by_id[class_id]["attrs"].append({"name": name, "type": type_, "vis": vis})

    for file_path, class_id, name, params, ret, is_async, vis in func_rows:
        entry = {"name": name, "params": list(params or []), "ret": ret,
                "vis": vis, "is_async": is_async}
        if class_id is None:
            result.setdefault(file_path, {"classes": [], "functions": []})["functions"].append(entry)
        elif class_id in classes_by_id:
            classes_by_id[class_id]["methods"].append(entry)

    return result


def list_repositories(conn):
    """Returns all distinct repository names in the DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT repository_name FROM files ORDER BY repository_name")
        return [row[0] for row in cur.fetchall()]


def get_repo_status(conn, repository_name):
    """Returns a dict summarizing how far the pipeline has progressed."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM files WHERE repository_name = %s", (repository_name,))
        total_files = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM files WHERE repository_name = %s AND rules_extracted = TRUE",
            (repository_name,),
        )
        files_with_rules = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM business_rules b
            JOIN files f ON f.id = b.file_id
            WHERE f.repository_name = %s
        """, (repository_name,))
        total_rules = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM key_points WHERE repository_name = %s", (repository_name,))
        total_key_points = cur.fetchone()[0]

    return {
        "total_files": total_files,
        "files_with_rules": files_with_rules,
        "total_rules": total_rules,
        "total_key_points": total_key_points,
    }


# ── Clearing / teardown ───────────────────────────────────────────────────────

def clear_repo(conn, repository_name):
    """
    Removes a single repository from PostgreSQL: its files (business_rules,
    classes, class_attributes, and functions all cascade via ON DELETE
    CASCADE) and its key points.
    Returns a dict of how many rows were removed.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM files WHERE repository_name = %s",
            (repository_name,),
        )
        file_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM business_rules b
            JOIN files f ON f.id = b.file_id
            WHERE f.repository_name = %s
        """, (repository_name,))
        rule_count = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM key_points WHERE repository_name = %s",
            (repository_name,),
        )
        kp_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM classes c
            JOIN files f ON f.id = c.file_id
            WHERE f.repository_name = %s
        """, (repository_name,))
        class_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM functions fn
            JOIN files f ON f.id = fn.file_id
            WHERE f.repository_name = %s
        """, (repository_name,))
        function_count = cur.fetchone()[0]

        # business_rules, classes (and their class_attributes), and functions
        # all go away automatically via ON DELETE CASCADE on files.
        cur.execute("DELETE FROM files WHERE repository_name = %s", (repository_name,))
        cur.execute("DELETE FROM key_points WHERE repository_name = %s", (repository_name,))

    conn.commit()
    return {"files": file_count, "business_rules": rule_count, "key_points": kp_count,
            "classes": class_count, "functions": function_count}


def clear_all(conn):
    """
    Wipes ALL data from PostgreSQL (every repository). Truncates every table
    and resets their id sequences. Schema is preserved.

    All tables are named explicitly rather than relying on CASCADE to reach
    them — TRUNCATE ... RESTART IDENTITY only resets sequences for tables
    named in the statement itself; CASCADE truncates dependents but leaves
    their sequences alone.
    """
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE TABLE business_rules, key_points, class_attributes, "
            "functions, classes, files RESTART IDENTITY CASCADE"
        )
    conn.commit()


def drop_all_tables(conn):
    """
    Hard reset: drops every table entirely. Use init_db() afterward to
    recreate them. Rarely needed — clear_all() is usually enough.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS business_rules CASCADE")
        cur.execute("DROP TABLE IF EXISTS key_points CASCADE")
        cur.execute("DROP TABLE IF EXISTS class_attributes CASCADE")
        cur.execute("DROP TABLE IF EXISTS functions CASCADE")
        cur.execute("DROP TABLE IF EXISTS classes CASCADE")
        cur.execute("DROP TABLE IF EXISTS files CASCADE")
    conn.commit()


if __name__ == "__main__":
    create_db()
    init_db()