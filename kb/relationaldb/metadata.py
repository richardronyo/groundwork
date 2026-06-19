#!/usr/bin/env python3

import ast
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv()
DB_CONFIG = {
    "host": "localhost",
    "dbname": "repo_analysis",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}


class MetricsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = 0
        self.functions = 0
        self.methods = 0
        self.async_functions = 0
        self.imports = 0
        self.class_depth = 0

    def visit_ClassDef(self, node):
        self.classes += 1
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_FunctionDef(self, node):
        if self.class_depth:
            self.methods += 1
        else:
            self.functions += 1

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.async_functions += 1

        if self.class_depth:
            self.methods += 1
        else:
            self.functions += 1

        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports += len(node.names)

    def visit_ImportFrom(self, node):
        self.imports += len(node.names)


def analyze_file(path):
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)

        visitor = MetricsVisitor()
        visitor.visit(tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "methods": visitor.methods,
            "async_functions": visitor.async_functions,
            "imports": visitor.imports,
            "lines": len(source.splitlines()),
        }

    except Exception as e:
        print(f"Failed to analyze {path}: {e}")
        return None


def analyze_repo(repo_root):
    repo_root = Path(repo_root)
    results = {}

    for py_file in sorted(repo_root.rglob("*.py")):
        metrics = analyze_file(py_file)

        if metrics:
            rel_path = str(py_file.relative_to(repo_root))
            results[rel_path] = metrics

    return results


def load_business_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_file(
    conn,
    repository_name,
    file_path,
    language,
    metrics,
):
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
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s
            )
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


def replace_business_rules(conn, file_id, rules):
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM business_rules
            WHERE file_id = %s
            """,
            (file_id,),
        )

        for rule in rules:
            cur.execute(
                """
                INSERT INTO business_rules (
                    file_id,
                    rule_text
                )
                VALUES (%s,%s)
                """,
                (file_id, rule),
            )


def migrate_to_db(repo_root, metrics_by_file, rules_by_file):
    repo_root = Path(repo_root)
    repo_name = repo_root.name

    conn = psycopg.connect(**DB_CONFIG)

    try:
        migrated_files = 0
        migrated_rules = 0

        for relative_path, metrics in metrics_by_file.items():
            absolute_path = str((repo_root / relative_path).resolve())

            rules = (
                rules_by_file.get(absolute_path)
                or rules_by_file.get(relative_path)
                or []
            )

            file_id = upsert_file(
                conn=conn,
                repository_name=repo_name,
                file_path=relative_path,
                language="Python",
                metrics=metrics,
            )

            replace_business_rules(conn, file_id, rules)

            migrated_files += 1
            migrated_rules += len(rules)

        conn.commit()

        print()
        print(f"Migrated files: {migrated_files}")
        print(f"Migrated business rules: {migrated_rules}")

    finally:
        conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze repository metadata and migrate to PostgreSQL"
    )

    parser.add_argument(
        "repo",
        help="Path to repository root",
    )

    parser.add_argument(
        "--rules",
        default="business_rules.json",
        help="Path to business_rules.json",
    )

    args = parser.parse_args()

    print("Analyzing repository...")
    metrics = analyze_repo(args.repo)

    print(f"Found {len(metrics)} Python files")

    print("Loading business rules...")
    business_rules = load_business_rules(args.rules)

    print(f"Loaded business rules for {len(business_rules)} files")

    print("Migrating to PostgreSQL...")

    migrate_to_db(
        repo_root=args.repo,
        metrics_by_file=metrics,
        rules_by_file=business_rules,
    )

    print("Done.")


if __name__ == "__main__":
    main()