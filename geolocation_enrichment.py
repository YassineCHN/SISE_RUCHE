"""
Geolocation Enrichment - VERSION FINALE
========================================
Avec retry, timeout augmenté, et mapping des régions françaises
"""

import pandas as pd 
import requests
from typing import Optional, Tuple
import time

# ============================================================================
# MAPPING COMPLET DES RÉGIONS FRANÇAISES
# ============================================================================

COMPLETE_REGION_MAPPING = {
    # Île-de-France
    '75': ('Île-de-France', '11'), '77': ('Île-de-France', '11'),
    '78': ('Île-de-France', '11'), '91': ('Île-de-France', '11'),
    '92': ('Île-de-France', '11'), '93': ('Île-de-France', '11'),
    '94': ('Île-de-France', '11'), '95': ('Île-de-France', '11'),
    
    
    # Corse
    '2A': ('Corse', '94'), '2B': ('Corse', '94'),
}

# ============================================================================
# Coordonnées des arrondissements de Paris (fallback)
# ============================================================================

PARIS_ARRONDISSEMENTS = {
    '75001': (48.8632, 2.3367),  # 1er
    '75002': (48.8679, 2.3418),  # 2e
    '75003': (48.8644, 2.3630),  # 3e
    '75004': (48.8564, 2.3522),  # 4e
    '75005': (48.8462, 2.3481),  # 5e
    '75006': (48.8508, 2.3325),  # 6e
    '75007': (48.8564, 2.3107),  # 7e
    '75008': (48.8725, 2.3118),  # 8e
    '75009': (48.8768, 2.3394),  # 9e
    '75010': (48.8760, 2.3631),  # 10e
    '75011': (48.8594, 2.3790),  # 11e
    '75012': (48.8412, 2.3891),  # 12e
    '75013': (48.8322, 2.3561),  # 13e
    '75014': (48.8337, 2.3274),  # 14e ← CELUI-CI !
    '75015': (48.8401, 2.2986),  # 15e
    '75016': (48.8557, 2.2671),  # 16e
    '75017': (48.8873, 2.3090),  # 17e
    '75018': (48.8927, 2.3444),  # 18e
    '75019': (48.8838, 2.3789),  # 19e
    '75020': (48.8632, 2.3979),  # 20e
}


class GeoRefFranceV2:
    """Client with increased timeout and retry logic"""
    BASE_URL = "https://data.enseignementsup-recherche.gouv.fr/api/explore/v2.1/catalog/datasets/fr-esr-referentiel-geographique/records"

    def _request(self, params: dict, max_retries: int = 3) -> Optional[dict]:
        """API request with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.BASE_URL, 
                    params=params, 
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
            except requests.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                else:
                    return None
                    
            except requests.RequestException:
                return None
                
            except Exception:
                return None
        
        return None

    def _clean_city_name(self, city_name: str) -> str:
        """Clean city name for better matching"""
        if not city_name:
            return ""
        
        # Remove arrondissement mentions
        clean = city_name.strip()
        
        # "Paris 14e Arrondissement" → "Paris"
        if "arrondissement" in clean.lower():
            clean = clean.split()[0]  # Take first word (usually city name)
        
        # Remove extra spaces
        clean = ' '.join(clean.split())
        
        return clean

    def get_coords_by_city(self, city_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates by city name"""
        if not city_name:
            return None
        
        # ✅ Clean city name
        clean_name = self._clean_city_name(city_name)
        
        params = {'where': f'com_nom="{clean_name}"', 'limit': 1}
        data = self._request(params)

        if not data or 'results' not in data or not data['results']:
            return None
        
        record = data['results'][0]
        if not record:
            return None
            
        geo = record.get('geolocalisation')
        if not geo:
            return None
        
        lat = geo.get('lat')
        lon = geo.get('lon')
        
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
        
        return None

    def get_coords_by_department(self, dep_code: str) -> Optional[Tuple[float, float]]:
        """Get coordinates by department"""
        if not dep_code:
            return None
        
        # ✅ Utiliser le mapping des régions pour les coordonnées
        if dep_code in COMPLETE_REGION_MAPPING:
            region_name, _ = COMPLETE_REGION_MAPPING[dep_code]
            
            # Coordonnées approximatives des chefs-lieux de région
            region_coords = {
                'Île-de-France': (48.8566, 2.3522),  # Paris
                'Auvergne-Rhône-Alpes': (45.7640, 4.8357),  # Lyon
                "Provence-Alpes-Côte d'Azur": (43.2965, 5.3698),  # Marseille
                'Nouvelle-Aquitaine': (44.8378, -0.5792),  # Bordeaux
                'Occitanie': (43.6047, 1.4442),  # Toulouse
                'Hauts-de-France': (50.6292, 3.0573),  # Lille
                'Grand Est': (48.5734, 7.7521),  # Strasbourg
                'Bretagne': (48.1173, -1.6778),  # Rennes
                'Pays de la Loire': (47.2184, -1.5536),  # Nantes
                'Normandie': (49.4432, 1.0993),  # Rouen
                'Bourgogne-Franche-Comté': (47.2380, 6.0243),  # Besançon
                'Centre-Val de Loire': (47.9029, 1.9093),  # Orléans
                'Corse': (41.9270, 8.7369),  # Ajaccio
            }
            
            if region_name in region_coords:
                return region_coords[region_name]
        
        # Fallback: API request
        params = {'where': f'dep_code="{dep_code}"', 'limit': 1}
        data = self._request(params)

        if not data or 'results' not in data or not data['results']:
            return None
        
        record = data['results'][0]
        if not record:
            return None
            
        geo = record.get('geolocalisation')
        if not geo:
            return None
        
        lat = geo.get('lat')
        lon = geo.get('lon')
        
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
        
        return None

    def search_city(self, partial_name: str) -> Optional[Tuple[float, float]]:
        """Search city with fuzzy match"""
        if not partial_name or len(partial_name) < 3:
            return None
        
        # ✅ Clean name first
        clean_name = self._clean_city_name(partial_name)
        
        params = {'where': f'search(com_nom, "{clean_name}")', 'limit': 1}
        data = self._request(params)

        if not data or 'results' not in data or not data['results']:
            return None
        
        record = data['results'][0]
        if not record:
            return None
            
        geo = record.get('geolocalisation')
        if not geo:
            return None
        
        lat = geo.get('lat')
        lon = geo.get('lon')
        
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
        
        return None


def enrich_locations_with_coordinates(df_locations: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich with coordinates - WITH PROGRESS BAR
    """
    print("STEP 9.2b: Enriching locations with geographic coordinates...")
    print("  ⚠️ This may take 10-15 minutes for 1000+ locations...")

    geo_api = GeoRefFranceV2()
    enriched_count = 0
    skipped_count = 0
    error_count = 0
    
    total = len(df_locations)

    for idx, row in df_locations.iterrows():
        ville = row.get('ville')
        departement = row.get('departement')
        
        # Progress indicator
        if idx % 50 == 0 and idx > 0:
            pct = (idx / total) * 100
            print(f"  📊 Progress: {idx}/{total} ({pct:.1f}%) - Enriched: {enriched_count}")
        
        # Skip UNKNOWN
        if ville == 'UNKNOWN':
            skipped_count += 1
            continue
        
        coords = None
        
        try:
            # ✅ Strategy 0: Arrondissements de Paris (hardcodé)
            if ville and "arrondissement" in str(ville).lower() and departement == '75':
                # Extraire le numéro d'arrondissement
                import re
                match = re.search(r'\b(\d{1,2})[eè]?\b', str(ville))
                if match:
                    arr_num = match.group(1).zfill(2)  # "14" → "14", "1" → "01"
                    postal_code = f"750{arr_num}"
                    if postal_code in PARIS_ARRONDISSEMENTS:
                        coords = PARIS_ARRONDISSEMENTS[postal_code]
                        print(f"  ✅ {ville} (arrondissement) → {coords}")
            
            # Strategy 1: City name
            if coords is None and ville and isinstance(ville, str) and ville.strip():
                coords = geo_api.get_coords_by_city(ville.strip())
            
            # Strategy 2: Fuzzy
            if coords is None and ville and isinstance(ville, str) and len(ville.strip()) > 3:
                coords = geo_api.search_city(ville.strip())
            
            # Strategy 3: Department
            if coords is None and departement and isinstance(departement, str) and departement.strip():
                coords = geo_api.get_coords_by_department(departement.strip())
            
            # Update
            if coords and len(coords) == 2:
                df_locations.at[idx, 'latitude'] = coords[0]
                df_locations.at[idx, 'longitude'] = coords[1]
                enriched_count += 1
            
            # ✅ Rate limiting: 1 seconde entre requêtes
            time.sleep(1)
                
        except Exception as e:
            error_count += 1
            continue

    print(f"\n  ✅ Geocoding complete:")
    print(f"     - Enriched: {enriched_count}/{total}")
    print(f"     - Skipped (UNKNOWN): {skipped_count}")
    print(f"     - Errors: {error_count}")

    return df_locations


def geocode_single_location(city: str, department: str = None) -> Optional[Tuple[float, float]]:
    """Test single location"""
    if not city:
        return None
        
    geo_api = GeoRefFranceV2()
    
    try:
        coords = geo_api.get_coords_by_city(city)
        if coords is None and len(city) > 3:
            coords = geo_api.search_city(city)
        if coords is None and department:
            coords = geo_api.get_coords_by_department(department)
        return coords
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    """Tests"""
    print("Testing Geolocation Module")
    print("=" * 60)
    
    # Test Paris 14e
    result = geocode_single_location("Paris 14e Arrondissement", "75")
    print(f"Paris 14e: {result}")
    
    # Test villes normales
    result = geocode_single_location("Lyon", "69")
    print(f"Lyon: {result}")
    
    result = geocode_single_location("Marseille", "13")
    print(f"Marseille: {result}")
    
    print("\n" + "=" * 60)