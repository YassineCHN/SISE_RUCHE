import os
import duckdb


def get_connection(read_only=False):
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    motherduck_db = os.getenv("MOTHERDUCK_DB")

    if motherduck_token and motherduck_db:
        return duckdb.connect(
            f"md:{motherduck_db}",
            read_only=read_only,
            config={"motherduck_token": motherduck_token},
        )

    # Fallback DuckDB local
    db_path = os.getenv("DUCKDB_PATH", "/data/local.duckdb")
    return duckdb.connect(db_path, read_only=read_only)
