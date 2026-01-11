import duckdb
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")


# 1. Configuration
local_db_path = "data/local.duckdb"
target_db_name = "job_market_RUCHE"  # Nom de la base sur MotherDuck

try:
    # 2. Connexion à MotherDuck
    # Si le token n'est pas configuré dans vos variables d'environnement,
    # on peut le passer directement dans la chaîne de connexion.
    con = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")

    print(f"Connexion établie. Migration de {local_db_path} vers MotherDuck...")

    # 3. Attacher la base locale à la session MotherDuck
    con.execute(f"ATTACH '{local_db_path}' AS local_db (READ_ONLY);")

    # 4. Créer la base sur MotherDuck à partir de la base locale
    # Cette commande copie schémas et données d'un seul coup
    con.execute(f"CREATE DATABASE {target_db_name} FROM local_db;")

    print(
        f"Succès ! Votre base est maintenant disponible sous le nom '{target_db_name}'"
    )

except Exception as e:
    print(f"Erreur lors du transfert : {e}")

finally:
    # 5. Nettoyage
    con.close()
