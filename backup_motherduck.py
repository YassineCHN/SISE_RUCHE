"""
MotherDuck Backup Script - Cold Storage
========================================
Script de sauvegarde pour télécharger une base MotherDuck vers un fichier local DuckDB.

Usage:
    python backup_motherduck.py

Requirements:
    - duckdb
    - python-dotenv (optionnel)
    
Environment Variables:
    MOTHERDUCK_TOKEN: Token d'authentification MotherDuck
"""

import os
import duckdb
# MotherDuck
    # import db name
from config import MOTHERDUCK_DATABASE
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class MotherDuckBackup:
    """Classe pour gérer le backup MotherDuck vers fichier local"""
    
    def __init__(self, motherduck_db: str, local_backup_path: str, token: str = None):
        """
        Initialise le backup manager
        
        Args:
            motherduck_db: Nom de la base MotherDuck (ex: "job_market_analytics")
            local_backup_path: Chemin du fichier de backup local (ex: "data/backup_job_market.duckdb")
            token: Token MotherDuck (si None, utilise MOTHERDUCK_TOKEN env var)
        """
        self.motherduck_db = motherduck_db
        self.local_backup_path = Path(local_backup_path)
        self.token = os.getenv('MOTHERDUCK_TOKEN')
        
        if not self.token:
            raise ValueError("MOTHERDUCK_TOKEN not found in environment variables")
        
        # Créer le répertoire parent si nécessaire
        self.local_backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.con = None
    
    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Établit la connexion à MotherDuck"""
        print("=" * 80)
        print("MOTHERDUCK BACKUP - COLD STORAGE")
        print("=" * 80)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connexion à MotherDuck...")
        
        try:
            # Connexion à MotherDuck
            connection_string = f"md:{self.motherduck_db}?motherduck_token={self.token}"
            self.con = duckdb.connect(connection_string)
            print(f"✅ Connecté à MotherDuck: {self.motherduck_db}")
            return self.con
        except Exception as e:
            print(f"❌ Erreur de connexion à MotherDuck: {e}")
            raise
    
    def _get_tables(self) -> List[str]:
        """Récupère la liste des tables dans la base MotherDuck"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Détection des tables...")
        
        try:
            # Requête pour lister les tables
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            
            tables = self.con.execute(query).fetchall()
            table_names = [table[0] for table in tables]
            
            print(f"✅ {len(table_names)} table(s) détectée(s):")
            for table in table_names:
                print(f"   • {table}")
            
            return table_names
        except Exception as e:
            print(f"❌ Erreur lors de la détection des tables: {e}")
            raise
    
    def _get_table_count(self, table_name: str, db_alias: str = "main") -> int:
        """Récupère le nombre de lignes d'une table"""
        try:
            query = f"SELECT COUNT(*) FROM {db_alias}.{table_name}"
            count = self.con.execute(query).fetchone()[0]
            return count
        except Exception as e:
            print(f"⚠️  Impossible de compter les lignes de {table_name}: {e}")
            return -1
    
    def backup(self) -> Tuple[int, int, List[str]]:
        """
        Exécute le backup complet
        
        Returns:
            Tuple (nb_tables, total_rows, table_names)
        """
        try:
            # 1. Connexion à MotherDuck
            self._connect()
            
            # 2. Détection des tables
            tables = self._get_tables()
            
            if not tables:
                print("\n⚠️  Aucune table trouvée dans la base MotherDuck")
                return 0, 0, []
            
            # 3. Attacher la base locale
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Préparation de la base locale...")
            print(f"   Fichier: {self.local_backup_path.absolute()}")
            
            # Supprimer l'ancien fichier de backup s'il existe
            if self.local_backup_path.exists():
                self.local_backup_path.unlink()
                print(f"   ♻️  Ancien backup supprimé")
            
            # Attacher la base locale
            attach_query = f"ATTACH '{self.local_backup_path}' AS local_backup (TYPE DUCKDB)"
            self.con.execute(attach_query)
            print(f"✅ Base locale attachée: local_backup")
            
            # 4. Copier chaque table
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Copie des tables...")
            print("-" * 80)
            
            total_rows = 0
            backup_stats = []
            
            for i, table in enumerate(tables, 1):
                print(f"\n[{i}/{len(tables)}] Copie de la table: {table}")
                
                # Compter les lignes source
                source_count = self._get_table_count(table, "main")
                print(f"   📊 Lignes dans MotherDuck: {source_count:,}")
                
                # Copier la table
                try:
                    copy_query = f"""
                        CREATE OR REPLACE TABLE local_backup.{table} AS 
                        SELECT * FROM main.{table}
                    """
                    self.con.execute(copy_query)
                    
                    # Vérifier la copie
                    local_count = self._get_table_count(table, "local_backup")
                    print(f"   💾 Lignes copiées localement: {local_count:,}")
                    
                    if source_count == local_count:
                        print(f"   ✅ Copie réussie ({local_count:,} lignes)")
                        backup_stats.append((table, local_count, "✅"))
                        total_rows += local_count
                    else:
                        print(f"   ⚠️  Attention: Incohérence de comptage (Source: {source_count:,}, Local: {local_count:,})")
                        backup_stats.append((table, local_count, "⚠️"))
                        total_rows += local_count
                        
                except Exception as e:
                    print(f"   ❌ Erreur lors de la copie: {e}")
                    backup_stats.append((table, 0, "❌"))
            
            # 5. Détacher la base locale
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finalisation...")
            self.con.execute("DETACH local_backup")
            print(f"✅ Base locale détachée")
            
            # 6. Résumé
            print("\n" + "=" * 80)
            print("RÉSUMÉ DU BACKUP")
            print("=" * 80)
            print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"☁️  Source: MotherDuck ({self.motherduck_db})")
            print(f"💾 Destination: {self.local_backup_path.absolute()}")
            print(f"\n📊 Statistiques:")
            print(f"   • Tables copiées: {len([s for s in backup_stats if s[2] == '✅'])}/{len(tables)}")
            print(f"   • Total lignes: {total_rows:,}")
            print(f"   • Taille fichier: {self._get_file_size()}")
            
            print(f"\n📋 Détail par table:")
            print("-" * 80)
            print(f"{'Table':<30} {'Lignes':>15} {'Statut':>10}")
            print("-" * 80)
            for table_name, row_count, status in backup_stats:
                print(f"{table_name:<30} {row_count:>15,} {status:>10}")
            print("-" * 80)
            
            print("\n✅ Backup terminé avec succès!")
            print("=" * 80)
            
            return len(tables), total_rows, [s[0] for s in backup_stats]
            
        except Exception as e:
            print(f"\n❌ ERREUR CRITIQUE: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Fermeture propre de la connexion
            if self.con:
                try:
                    self.con.close()
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connexion MotherDuck fermée")
                except:
                    pass
    
    def _get_file_size(self) -> str:
        """Retourne la taille du fichier de backup formatée"""
        if not self.local_backup_path.exists():
            return "N/A"
        
        size_bytes = self.local_backup_path.stat().st_size
        
        # Conversion en unité lisible
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


def backup_motherduck_to_local(
    motherduck_db: str = MOTHERDUCK_DATABASE,
    local_backup_path: str = "data/backup_job_market.duckdb",
    token: str = None
) -> bool:
    """
    Fonction principale de backup MotherDuck → Local
    
    Args:
        motherduck_db: Nom de la base MotherDuck
        local_backup_path: Chemin du fichier de backup local
        token: Token MotherDuck (optionnel, sinon via env var)
    
    Returns:
        True si succès, False sinon
    
    Example:
        >>> backup_motherduck_to_local()
        >>> backup_motherduck_to_local("my_db", "backups/my_backup.duckdb")
    """
    try:
        backup_manager = MotherDuckBackup(motherduck_db, local_backup_path, token)
        nb_tables, total_rows, table_names = backup_manager.backup()
        return True
    except Exception as e:
        print(f"\n❌ Le backup a échoué: {e}")
        return False


# ============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║           MOTHERDUCK BACKUP - COLD STORAGE UTILITY                    ║
    ║                                                                       ║
    ║  Ce script télécharge une copie complète de votre base MotherDuck    ║
    ║  vers un fichier DuckDB local pour sécurisation (Cold Storage).      ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration (peut être modifiée selon vos besoins)
    LOCAL_BACKUP_PATH = "data/backup_job_market.duckdb"
    
    # Exécution du backup
    success = backup_motherduck_to_local(
        motherduck_db=MOTHERDUCK_DATABASE,
        local_backup_path=LOCAL_BACKUP_PATH
    )
    
    # Code de sortie
    exit(0 if success else 1)