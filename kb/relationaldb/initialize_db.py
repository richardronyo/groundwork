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

            # ── Migrations ────────────────────────────────────────────────────
            # Idempotent ALTERs so existing tables pick up columns added after
            # their initial creation. Each is safe to run on every startup.
            for migration in MIGRATIONS:
                cur.execute(migration)

        conn.commit()
        print("Tables ensured: files, business_rules, key_points")
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
    Removes a single repository from PostgreSQL: its files (business_rules
    cascade via ON DELETE CASCADE) and its key points.
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

        # business_rules go away automatically via ON DELETE CASCADE
        cur.execute("DELETE FROM files WHERE repository_name = %s", (repository_name,))
        cur.execute("DELETE FROM key_points WHERE repository_name = %s", (repository_name,))

    conn.commit()
    return {"files": file_count, "business_rules": rule_count, "key_points": kp_count}


def clear_all(conn):
    """
    Wipes ALL data from PostgreSQL (every repository). Truncates the three
    tables and resets their id sequences. Schema is preserved.
    """
    with conn.cursor() as cur:
        # RESTART IDENTITY resets BIGSERIAL counters; CASCADE handles FKs
        cur.execute("TRUNCATE TABLE business_rules, key_points, files RESTART IDENTITY CASCADE")
    conn.commit()


def drop_all_tables(conn):
    """
    Hard reset: drops the three tables entirely. Use init_db() afterward to
    recreate them. Rarely needed — clear_all() is usually enough.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS business_rules CASCADE")
        cur.execute("DROP TABLE IF EXISTS key_points CASCADE")
        cur.execute("DROP TABLE IF EXISTS files CASCADE")
    conn.commit()


if __name__ == "__main__":
    create_db()
    init_db()