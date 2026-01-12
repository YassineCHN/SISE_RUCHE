"""
MotherDuck Cold Storage Backup
==============================

Script d'exportation de base de données MotherDuck vers un fichier DuckDB local.
Ce script est conçu pour être exécuté de manière autonome ou planifiée via un scheduler.

Author: Data Engineering Team
"""

import os
import sys
import duckdb
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruche.db import get_connection

# Chargement des variables d'environnement
load_dotenv()


class MotherDuckBackup:
    """
    Gère le cycle de vie de la sauvegarde : connexion, extraction, copie et validation.
    """

    def __init__(
        self,
        motherduck_db: str,
        local_backup_path: str,
        token: Optional[str] = None,
        exclude_tables: Optional[List[str]] = None,
    ):
        """
        Initialise les paramètres de sauvegarde.

        Args:
            motherduck_db: Nom de la base source sur MotherDuck.
            local_backup_path: Chemin du fichier de destination (.duckdb).
            token: Jeton d'authentification (si None, utilise la variable d'env MOTHERDUCK_TOKEN).
            exclude_tables: Liste des tables à ignorer lors du backup.
        """
        self.motherduck_db = motherduck_db
        self.local_backup_path = Path(local_backup_path)
        self.token = token or os.getenv("MOTHERDUCK_TOKEN")
        # Tables à exclure (par défaut ou liste fournie)
        self.exclude_tables = (
            exclude_tables if exclude_tables is not None else ["job_offers_cleaned"]
        )
        self.con = None

        if not self.token:
            self._log(
                "ERREUR: MOTHERDUCK_TOKEN manquant dans l'environnement.", level="ERROR"
            )
            sys.exit(1)

    def _log(self, message: str, level: str = "INFO") -> None:
        """
        Affiche un message formaté avec timestamp.
        Les erreurs sont dirigées vers stderr.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level:<5}] {message}"

        if level in ("ERROR", "CRITICAL"):
            print(formatted_message, file=sys.stderr)
        else:
            print(formatted_message)

    def _connect(self) -> None:
        """Établit la connexion à l'instance distante."""
        self._log(f"Connexion a MotherDuck (DB: {self.motherduck_db})...")
        try:
            self.con = get_connection()
            self._log("Connexion etablie.")
        except Exception as e:
            self._log(f"Echec de connexion: {e}", level="ERROR")
            raise

    def _get_tables(self) -> List[str]:
        """Récupère la liste des tables éligibles au backup."""
        self._log("Recuperation de la liste des tables...")
        query = """
            SELECT DISTINCT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        try:
            tables = [t[0] for t in self.con.execute(query).fetchall()]

            # Filtrage des tables
            tables_to_process = [t for t in tables if t not in self.exclude_tables]
            excluded_count = len(tables) - len(tables_to_process)

            self._log(
                f"Tables detectees: {len(tables)}. A traiter: {len(tables_to_process)}. Exclues: {excluded_count}."
            )
            return tables_to_process
        except Exception as e:
            self._log(f"Erreur lors du listing des tables: {e}", level="ERROR")
            raise

    def _get_row_count(self, table_name: str, db_alias: str) -> int:
        """Retourne le nombre de lignes d'une table spécifique."""
        try:
            return self.con.execute(
                f"SELECT COUNT(*) FROM {db_alias}.{table_name}"
            ).fetchone()[0]
        except Exception:
            return -1

    def run_backup(self) -> bool:
        """
        Exécute la procédure complète de sauvegarde.
        Returns: True si succès complet, False si erreurs rencontrées.
        """
        try:
            # 1. Préparation du système de fichiers
            if not self.local_backup_path.parent.exists():
                self.local_backup_path.parent.mkdir(parents=True, exist_ok=True)

            if self.local_backup_path.exists():
                self._log(f"Suppression du fichier existant: {self.local_backup_path}")
                self.local_backup_path.unlink()

            # 2. Connexion et Attachement
            self._connect()

            self._log(f"Initialisation de la base locale: {self.local_backup_path}")
            self.con.execute(
                f"ATTACH '{self.local_backup_path}' AS local_backup (TYPE DUCKDB)"
            )

            # 3. Traitement des tables
            tables = self._get_tables()
            if not tables:
                self._log("Aucune table a sauvegarder.", level="WARNING")
                return True

            errors = 0
            total_rows = 0

            for i, table in enumerate(tables, 1):
                self._log(f"Traitement [{i}/{len(tables)}] : {table}")

                try:
                    # Copie des données (Create Table As Select)
                    self.con.execute(
                        f"CREATE OR REPLACE TABLE local_backup.{table} AS SELECT * FROM main.{table}"
                    )

                    # Vérification simple (Row Count)
                    count_src = self._get_row_count(table, "main")
                    count_dst = self._get_row_count(table, "local_backup")

                    if count_src == count_dst:
                        self._log(f"Succes copie {table} ({count_dst} lignes).")
                        total_rows += count_dst
                    else:
                        self._log(
                            f"Disparite de donnees pour {table} (Source: {count_src} vs Local: {count_dst})",
                            level="WARNING",
                        )

                except Exception as e:
                    self._log(f"Erreur copie table {table}: {e}", level="ERROR")
                    errors += 1

            # 4. Finalisation
            self.con.execute("DETACH local_backup")

            # Calcul de la taille finale
            size_mb = self.local_backup_path.stat().st_size / (1024 * 1024)

            self._log("-" * 60)
            self._log(
                f"BACKUP TERMINE. Succes: {len(tables) - errors}/{len(tables)}. Lignes: {total_rows}. Taille: {size_mb:.2f} MB"
            )

            return errors == 0

        except Exception as e:
            self._log(f"Erreur critique du script: {e}", level="CRITICAL")
            return False
        finally:
            if self.con:
                try:
                    self.con.close()
                    self._log("Connexion MotherDuck fermee.")
                except Exception:
                    pass


def main():
    # Configuration
    BACKUP_FILE = "data/backup_job_market.duckdb"
    EXCLUDE_LIST = ["job_offers_cleaned"]

    backup_job = MotherDuckBackup(
        motherduck_db=MOTHERDUCK_DATABASE,
        local_backup_path=BACKUP_FILE,
        exclude_tables=EXCLUDE_LIST,
    )

    success = backup_job.run_backup()

    # Code de sortie explicite pour l'orchestrateur (0=OK, 1=KO)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
