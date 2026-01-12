"""ETL Vectorization - Semantic Search Embeddings"""

import duckdb
import pandas as pd
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

from config_etl import EMBEDDING_MODEL, EMBEDDING_DIM
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ruche.db import get_connection


load_dotenv()


class EmbeddingETL:
    def __init__(self):
        self.con = None
        self.model = None
        self.embedding_dim = EMBEDDING_DIM

    def connect(self):
        print("Connecting to DuckDB / MotherDuck...")
        self.con = get_connection()

    def load_model(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[OK] Model loaded (dim: {self.embedding_dim})")

    def prepare_embedding_column(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing embedding column...")

        # Pas de préfixe, juste la table
        table_name = "f_offre"

        try:
            check_query = f"DESCRIBE {table_name}"
            result = self.con.execute(check_query).fetchdf()

            if "embedding" not in result["column_name"].values:
                print("[INFO] Column 'embedding' not found, creating...")
                alter_query = f"ALTER TABLE {table_name} ADD COLUMN embedding FLOAT[{self.embedding_dim}]"
                self.con.execute(alter_query)
                print(f"[OK] Column 'embedding' added")
            else:
                print(f"[INFO] Column 'embedding' already exists")

        except Exception as e:
            print(f"[WARNING] Cannot describe table: {e}")
            print("[INFO] Attempting to add column directly...")
            try:
                alter_query = f"ALTER TABLE {table_name} ADD COLUMN embedding FLOAT[{self.embedding_dim}]"
                self.con.execute(alter_query)
                print(f"[OK] Column 'embedding' added")
            except Exception as e2:
                if (
                    "already exists" in str(e2).lower()
                    or "duplicate" in str(e2).lower()
                ):
                    print(f"[INFO] Column 'embedding' already exists")
                else:
                    raise e2

    def extract_enriched_data(self) -> pd.DataFrame:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Extracting data with enrichment..."
        )

        query = """
        SELECT 
            f.job_id,
            CONCAT_WS(' | ',
                f.title,
                f.company_name,
                COALESCE(c.type_contrat, ''),
                COALESCE(l.ville, ''),
                COALESCE(l.code_postal, ''),
                f.description,
                COALESCE(f.hard_skills, ''),
                COALESCE(f.soft_skills, '')
            ) AS enriched_text
        FROM f_offre f
        LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
        LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
        WHERE f.description IS NOT NULL
        ORDER BY f.job_id
        """

        df = self.con.execute(query).fetchdf()
        print(f"[OK] Extracted {len(df):,} offers")
        return df

    def generate_embeddings(
        self, df: pd.DataFrame, batch_size: int = 32
    ) -> pd.DataFrame:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating embeddings...")

        texts = df["enriched_text"].tolist()
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, convert_to_numpy=True, show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        result_df = pd.DataFrame({"job_id": df["job_id"], "embedding": embeddings})

        print(f"[OK] Generated {len(embeddings):,} embeddings")
        return result_df

    def update_embeddings(self, df: pd.DataFrame):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Updating MotherDuck...")

        self.con.execute("DROP TABLE IF EXISTS temp_embeddings")
        self.con.execute(
            f"""
            CREATE TEMP TABLE temp_embeddings (
                job_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}]
            )
        """
        )

        df_copy = df.copy()
        df_copy["embedding"] = df_copy["embedding"].apply(lambda x: x.tolist())

        batch_size = 1000
        for i in tqdm(range(0, len(df_copy), batch_size), desc="Inserting"):
            batch = df_copy.iloc[i : i + batch_size]
            self.con.execute("INSERT INTO temp_embeddings SELECT * FROM batch")

        self.con.execute(
            """
            UPDATE f_offre f
            SET embedding = t.embedding
            FROM temp_embeddings t
            WHERE f.job_id = t.job_id
        """
        )

        count = self.con.execute(
            """
            SELECT COUNT(*) FROM f_offre WHERE embedding IS NOT NULL
        """
        ).fetchone()[0]

        print(f"[OK] Updated {count:,} embeddings")
        self.con.execute("DROP TABLE IF EXISTS temp_embeddings")

    def run(self):
        print("=" * 80)
        print("ETL VECTORIZATION PIPELINE")
        print("=" * 80)

        start_time = datetime.now()

        try:
            self.connect()
            self.load_model()
            self.prepare_embedding_column()
            df = self.extract_enriched_data()
            embeddings_df = self.generate_embeddings(df)
            self.update_embeddings(embeddings_df)

            elapsed = (datetime.now() - start_time).total_seconds()
            print("\n" + "=" * 80)
            print(f"COMPLETED in {elapsed:.2f}s ({len(df):,} offers)")
            print("=" * 80)

        finally:
            if self.con:
                self.con.close()


if __name__ == "__main__":
    MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
    etl = EmbeddingETL()
    etl.run()
