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

def create_db():
    """
    Creates the repo_analysis database if it does not already exist.
    """

    conn = psycopg.connect(
        **ADMIN_CONFIG,
        autocommit=True,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (DB_NAME,),
            )

            if cur.fetchone() is None:
                cur.execute(f"CREATE DATABASE {DB_NAME}")
                print(f"Created database '{DB_NAME}'")
            else:
                print(f"Database '{DB_NAME}' already exists")

    finally:
        conn.close()

def init_db():
    """
    Creates all tables.
    """

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

                created_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(repository_name, file_path)
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS business_rules (
                id BIGSERIAL PRIMARY KEY,

                file_id BIGINT NOT NULL
                    REFERENCES files(id)
                    ON DELETE CASCADE,

                rule_text TEXT NOT NULL,

                created_at TIMESTAMP DEFAULT NOW()
            )
            """)

        conn.commit()

    finally:
        conn.close()

def save_file(conn, repository_name, file_path, language, metrics):
    """
    Inserts or updates a file record.
    Returns file_id.
    """

    with conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO files (
                repository_name,
                file_path,
                language,
                classes,
                functions,
                methods,
                async_functions,
                imports,
                lines
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)

            ON CONFLICT (
                repository_name,
                file_path
            )
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
                repository_name,
                file_path,
                language,
                metrics["classes"],
                metrics["functions"],
                metrics["methods"],
                metrics["async_functions"],
                metrics["imports"],
                metrics["lines"],
            ),
        )

        return cur.fetchone()[0]

def save_business_rules(conn, file_id, business_rules):
    """
    Replaces all business rules for a file.
    """

    with conn.cursor() as cur:

        cur.execute(
            "DELETE FROM business_rules WHERE file_id = %s",
            (file_id,),
        )

        for rule in business_rules:
            cur.execute(
                """
                INSERT INTO business_rules (
                    file_id,
                    rule_text
                )
                VALUES (%s, %s)
                """,
                (
                    file_id,
                    rule,
                ),
            )

def save_file_and_rules(repository_name, file_path, language, metrics, business_rules):
    """
    Convenience wrapper.
    """

    conn = psycopg.connect(**DB_CONFIG)

    try:
        file_id = save_file(
            conn,
            repository_name,
            file_path,
            language,
            metrics,
        )

        save_business_rules(
            conn,
            file_id,
            business_rules,
        )

        conn.commit()

    finally:
        conn.close()

if __name__ == "__main__":
    create_db()
    init_db()