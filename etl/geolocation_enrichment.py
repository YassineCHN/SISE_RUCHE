"""
Geolocation Enrichment Module
==============================
Author: RUCHE's teams 
Date: 2026-01-05

Module for enriching French locations with geographic coordinates
using the official French geographic reference API.
"""

import pandas as pd 
import requests
from typing import Optional, Tuple

# ==================================================================================
# French Geographic Reference API Configuration
# ==================================================================================

class GeoRefFranceV2:
    """
    Client for the French Geographic Reference API (GeoRef France V2)
    to enrich locations with geographic coordinates.
    """
    BASE_URL = "https://data.enseignementsup-recherche.gouv.fr/api/explore/v2.1/catalog/datasets/fr-esr-referentiel-geographique/records"

    def _request(self, params: dict) -> Optional[dict]:
        """API request with error handling"""
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"  API request error: {e}")
            return None

    def get_coords_by_city(self, city_name: str) -> Optional[Tuple[float, float]]:
        """
        Get geographic coordinates by exact city name

        Args:
            city_name: Name of the city to search for
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        params = {
            'where': f'com_nom="{city_name}"',
            'limit': 1
        }

        data = self._request(params)

        if data and data.get('total_count', 0) > 0:
            record = data['results'][0]
            geo = record.get('geolocalisation', {})
            lat = geo.get('lat')
            lon = geo.get('lon')
            
            if lat is not None and lon is not None:
                return (lat, lon)
        return None

    def get_coords_by_department(self, dep_code: str) -> Optional[Tuple[float, float]]:
        """
        Get geographic coordinates by department code
        
        Args:
            dep_code: Department code to search for
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        params = {
            'where': f'dep_code="{dep_code}"',
        'limit': 1
        }

        data = self._request(params)
        if data and data.get('total_count', 0) > 0:
            record = data['results'][0]
            geo = record.get('geolocalisation')
        
        # *** CORRECTION ICI : Vérifier que geo n'est pas None ***
        if geo is not None:
            lat = geo.get('lat')
            lon = geo.get('lon')
            
            if lat is not None and lon is not None:
                return (lat, lon)

        return None

    def search_city(self, partial_name: str) -> Optional[Tuple[float, float]]:
        """
        Search city with partial/fuzzy name match
        
        Args:
            partial_name: Partial name of the city to search for
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        params = {
            'where': f'search(com_nom, "{partial_name}")',
            'limit': 1
        }

        data = self._request(params)

        if data and data.get('total_count', 0) > 0:
            record = data['results'][0]
            geo = record.get('geolocalisation', {})
            lat = geo.get('lat')
            lon = geo.get('lon')
            
            if lat is not None and lon is not None:
                return (lat, lon)
        return None


# ==================================================================================
# Main Enrichment Function (OUTSIDE CLASS!)
# ==================================================================================

def enrich_locations_with_coordinates(df_locations: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich location DataFrame with latitude and longitude from French API

    Args:
        df_locations: DataFrame with columns ['ville', 'departement']
    
    Returns:
        DataFrame enriched with 'latitude' and 'longitude' columns
    """
    print("STEP 9.2b: Enriching locations with geographic coordinates...")

    geo_api = GeoRefFranceV2()
    enriched_count = 0

    for idx, row in df_locations.iterrows():
        ville = row['ville']
        departement = row['departement']
        
        coords = None
        
        # Strategy 1: Exact city name match
        if ville and isinstance(ville, str) and ville.strip():
            coords = geo_api.get_coords_by_city(ville.strip())
        
        # Strategy 2: Fuzzy search (if exact match failed)
        if coords is None and ville and isinstance(ville, str) and len(ville.strip()) > 3:
            coords = geo_api.search_city(ville.strip())
        
        # Strategy 3: Fallback to department centroid
        if coords is None and departement and isinstance(departement, str) and departement.strip():
            coords = geo_api.get_coords_by_department(departement.strip())
        
        # Update DataFrame
        if coords and coords[0] is not None and coords[1] is not None:
            df_locations.at[idx, 'latitude'] = coords[0]
            df_locations.at[idx, 'longitude'] = coords[1]
            enriched_count += 1

    print(f"  Geocoding complete: {enriched_count}/{len(df_locations)} locations enriched")

    return df_locations


# ==================================================================================
# Utility Functions (OUTSIDE CLASS!)
# ==================================================================================

def geocode_single_location(city: str, department: str = None) -> Optional[Tuple[float, float]]:
    """
    Geocode a single location (utility function for testing)

    Args:
        city: City name
        department: Department code (optional)

    Returns:
        Tuple of (latitude, longitude) or None
    """
    geo_api = GeoRefFranceV2()

    # Try exact match
    coords = geo_api.get_coords_by_city(city)

    # Try fuzzy search
    if coords is None and len(city) > 3:
        coords = geo_api.search_city(city)

    # Try department fallback
    if coords is None and department:
        coords = geo_api.get_coords_by_department(department)

    return coords


# ==================================================================================
# Testing
# ==================================================================================

# ==================================================================================
# Testing
# ==================================================================================

if __name__ == "__main__":
    """Test the geocoding functions"""
    print("Testing Geolocation Enrichment Module")
    print("=" * 60)
    
    geo_api = GeoRefFranceV2()
    
    # Test 1: Noms exacts
    print("\n1. TEST: Noms exacts de villes")
    print("-" * 60)
    coords = geo_api.get_coords_by_city("Bordeaux")
    print(f"   'Bordeaux' → {coords}")
    
    coords = geo_api.get_coords_by_city("Lyon")
    print(f"   'Lyon' → {coords}")
    
    coords = geo_api.get_coords_by_city("Paris")
    print(f"   'Paris' → {coords}")
    
    coords = geo_api.get_coords_by_city("Marseille")
    print(f"   'Marseille' → {coords}")
    
    # Test 2: Fuzzy search
    print("\n2. TEST: Fuzzy search (noms partiels)")
    print("-" * 60)
    coords = geo_api.search_city("Bordeaux")
    print(f"   'Bordeaux' (complet) → {coords}")
    
    coords = geo_api.search_city("Bord")
    print(f"   'Bord' (partiel) → {coords} ⚠️ Peut ne pas être Bordeaux")
    
    # Test 3: Recherche par CODE département
    print("\n3. TEST: Recherche par CODE de département")
    print("-" * 60)
    
    test_depts = [
        ("33", "Gironde"),
        ("11", "Paris"),
        ("69", "Rhône"),
        ("13", "Bouches-du-Rhône"),
        ("31", "Haute-Garonne"),
        ("59", "Nord"),
        ("44", "Loire-Atlantique")
    ]
    
    for code, nom in test_depts:
        coords = geo_api.get_coords_by_department(code)
        print(f"   Code '{code}' ({nom}) → {coords}")
    
    # Test 4: Recherche par NOM département
    print("\n4. TEST: Recherche par NOM de département")
    print("-" * 60)
    
    def search_by_dept_name(dept_name):
        """Helper pour tester la recherche par nom"""
        params = {
            'where': f'dep_nom="{dept_name}"',
            'limit': 1
        }
        data = geo_api._request(params)
        if data and data.get('total_count', 0) > 0:
            record = data['results'][0]
            geo = record.get('geolocalisation')
            if geo is not None:
                lat = geo.get('lat')
                lon = geo.get('lon')
                if lat and lon:
                    return (lat, lon)
        return None
    
    for code, nom in test_depts:
        coords = search_by_dept_name(nom)
        print(f"   Nom '{nom}' → {coords}")
    
    # Test 5: DataFrame enrichment
    print("\n5. TEST: DataFrame enrichment complet")
    print("-" * 60)
    test_df = pd.DataFrame({
        'id_ville': [1, 2, 3, 4, 5],
        'ville': ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nantes'],
        'departement': ['11', '13', '69', '31', '44'],
        'latitude': [None, None, None, None, None],
        'longitude': [None, None, None, None, None]
    })
    
    print("\nBefore enrichment:")
    print(test_df)
    
    enriched_df = enrich_locations_with_coordinates(test_df)
    
    print("\nAfter enrichment:")
    print(enriched_df[['ville', 'latitude', 'longitude']])
    
    print("\n" + "=" * 60)
    print("Testing complete!")