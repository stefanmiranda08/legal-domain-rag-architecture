#!/usr/bin/env python3
"""Run database migrations."""

import argparse
from pathlib import Path

from sqlalchemy import text

from src.config import get_settings
from src.database import get_postgres_engine, get_postgres_session


def run_migration(migration_file: Path) -> None:
    """Run a SQL migration file."""
    settings = get_settings()
    engine = get_postgres_engine(settings)

    print(f"Running migration: {migration_file.name}")

    with open(migration_file) as f:
        sql = f.read()

    # Split on semicolons to execute each statement
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    with get_postgres_session(engine) as session:
        for i, statement in enumerate(statements):
            # Skip empty statements and comments-only statements
            if not statement or statement.startswith("--"):
                continue

            try:
                result = session.execute(text(statement))
                # If it's a SELECT, print results
                if statement.upper().startswith("SELECT"):
                    rows = result.fetchall()
                    for row in rows:
                        print(f"  {row}")
                print(f"  Statement {i+1}: OK")
            except Exception as e:
                print(f"  Statement {i+1}: {e}")

        session.commit()

    print("Migration complete.")


def main():
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "migration",
        nargs="?",
        default="001_add_chunk_links.sql",
        help="Migration file to run (default: 001_add_chunk_links.sql)",
    )
    args = parser.parse_args()

    migrations_dir = Path(__file__).parent / "migrations"
    migration_file = migrations_dir / args.migration

    if not migration_file.exists():
        print(f"Migration file not found: {migration_file}")
        return

    run_migration(migration_file)


if __name__ == "__main__":
    main()
