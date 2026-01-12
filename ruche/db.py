import os
import duckdb
from pathlib import Path
import platform


def get_connection(read_only=False):
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    motherduck_db = os.getenv("MOTHERDUCK_DB")
    conn_mode = os.getenv("CONNEXION_MODE")  # 'offline' | 'online' | None

    if conn_mode == "online" or (
        conn_mode is None and motherduck_token and motherduck_db
    ):
        return duckdb.connect(
            f"md:{motherduck_db}",
            read_only=read_only,
            config={"motherduck_token": motherduck_token},
        )
    # --- DuckDB local ---
    db_path = None
    if conn_mode == "offline":
        db_path_env = os.getenv("DUCKDB_PATH")
    else:
        db_path_env = os.getenv("DUCKDB_PATH")

    if db_path_env:
        # Cas path Unix fourni sur Windows (ex: /data/local.duckdb)
        if platform.system() == "Windows" and db_path_env.startswith("/"):
            db_path = None
        else:
            candidate = Path(db_path_env)
            if candidate.is_absolute() and not candidate.exists():
                db_path = None
            else:
                db_path = candidate

    if db_path is None:
        # fallback local propre
        project_root = Path(__file__).resolve().parents[1]
        db_path = project_root / "data" / "local.duckdb"

    return duckdb.connect(str(db_path), read_only=read_only)
