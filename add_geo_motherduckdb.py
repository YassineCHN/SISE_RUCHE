"""
Script de géocodage post-pipeline
==================================

Géocode les villes dans la table d_localisation de MotherDuck
en appelant l'API GeoRef France.

Usage:
    python geocode_motherduck.py
"""

import duckdb
import os
from dotenv import load_dotenv
from geolocation_enrichment import GeoRefFranceV2
import time

def geocode_motherduck_locations():
    """Géocoder les villes dans MotherDuck"""
    
    print("=" * 80)
    print("🌍 GÉOCODAGE DES VILLES DANS MOTHERDUCK")
    print("=" * 80)
    
    # Connexion MotherDuck
    load_dotenv()
    token = os.getenv('MOTHERDUCK_TOKEN')
    database = "job_market_RUCHE"
    
    print("\n📡 Connexion à MotherDuck...")
    con = duckdb.connect(f"md:{database}?motherduck_token={token}")
    print(f"✅ Connecté à {database}")
    
    # Récupérer les villes
    print("\n📊 Récupération des villes...")
    df_locations = con.execute("SELECT * FROM d_localisation").fetchdf()
    print(f"   Total : {len(df_locations)} villes")
    
    # Filtrer celles sans coordonnées
    missing_coords = df_locations['latitude'].isna()
    df_to_geocode = df_locations[missing_coords].copy()
    print(f"   À géocoder : {len(df_to_geocode)} villes")
    
    if len(df_to_geocode) == 0:
        print("\n✅ Toutes les villes ont déjà des coordonnées!")
        con.close()
        return
    
    # Géocodage
    print("\n🔍 Géocodage en cours...")
    geo_api = GeoRefFranceV2()
    
    success_count = 0
    fail_count = 0
    
    for idx, row in df_to_geocode.iterrows():
        ville = row['ville']
        departement = row['departement']
        id_ville = row['id_ville']
        
        # Skip UNKNOWN
        if ville == 'UNKNOWN':
            continue
        
        coords = None
        
        # Stratégie 1: Nom exact
        if ville and isinstance(ville, str) and ville.strip():
            coords = geo_api.get_coords_by_city(ville.strip())
        
        # Stratégie 2: Fuzzy search
        if coords is None and ville and isinstance(ville, str) and len(ville.strip()) > 3:
            coords = geo_api.search_city(ville.strip())
        
        # Stratégie 3: Département
        if coords is None and departement and isinstance(departement, str):
            coords = geo_api.get_coords_by_department(departement.strip())
        
        # Update si trouvé
        if coords and coords[0] is not None and coords[1] is not None:
            lat, lon = coords
            
            # Update dans MotherDuck
            con.execute(f"""
                UPDATE d_localisation 
                SET latitude = {lat}, longitude = {lon}
                WHERE id_ville = {id_ville}
            """)
            
            success_count += 1
            print(f"   ✅ {ville} ({departement}) → ({lat:.4f}, {lon:.4f})")
        else:
            fail_count += 1
            print(f"   ❌ {ville} ({departement}) → Not found")
        
        # Rate limiting (500ms entre requêtes)
        time.sleep(0.5)
        
        # Progression
        if (success_count + fail_count) % 50 == 0:
            print(f"\n   📊 Progression: {success_count + fail_count}/{len(df_to_geocode)}")
    
    # Résumé
    print("\n" + "=" * 80)
    print("✅ GÉOCODAGE TERMINÉ")
    print("=" * 80)
    print(f"   Succès : {success_count}/{len(df_to_geocode)} ({success_count/len(df_to_geocode)*100:.1f}%)")
    print(f"   Échecs : {fail_count}/{len(df_to_geocode)} ({fail_count/len(df_to_geocode)*100:.1f}%)")
    
    # Vérification finale
    print("\n🔍 Vérification dans MotherDuck...")
    result = con.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(latitude) as with_coords,
            COUNT(*) - COUNT(latitude) as without_coords
        FROM d_localisation
        WHERE ville != 'UNKNOWN'
    """).fetchdf()
    
    print(result.to_string(index=False))
    
    con.close()
    print("\n✅ Connexion fermée")


if __name__ == "__main__":
    geocode_motherduck_locations()