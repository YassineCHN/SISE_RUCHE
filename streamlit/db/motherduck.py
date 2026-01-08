import os
import duckdb
from pathlib import Path
from dotenv import load_dotenv

# Racine du projet = 2 niveaux au-dessus de ce fichier
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def get_connection():
    token = os.getenv("MOTHERDUCK_TOKEN")
    db_name = os.getenv("MOTHERDUCK_DB")

    if not token or not db_name:
        raise RuntimeError("MotherDuck credentials missing")

    return duckdb.connect(database=f"md:{db_name}", config={"motherduck_token": token})
