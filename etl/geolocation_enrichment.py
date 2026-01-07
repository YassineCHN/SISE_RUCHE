"""
Geolocation Enrichment - VERSION FINALE
========================================
Avec get_full_location_info() qui retourne (lat, lon, département)
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
    
    # Auvergne-Rhône-Alpes
    '01': ('Auvergne-Rhône-Alpes', '84'), '03': ('Auvergne-Rhône-Alpes', '84'),
    '07': ('Auvergne-Rhône-Alpes', '84'), '15': ('Auvergne-Rhône-Alpes', '84'),
    '26': ('Auvergne-Rhône-Alpes', '84'), '38': ('Auvergne-Rhône-Alpes', '84'),
    '42': ('Auvergne-Rhône-Alpes', '84'), '43': ('Auvergne-Rhône-Alpes', '84'),
    '63': ('Auvergne-Rhône-Alpes', '84'), '69': ('Auvergne-Rhône-Alpes', '84'),
    '73': ('Auvergne-Rhône-Alpes', '84'), '74': ('Auvergne-Rhône-Alpes', '84'),
    
    # Provence-Alpes-Côte d'Azur
    '04': ("Provence-Alpes-Côte d'Azur", '93'), '05': ("Provence-Alpes-Côte d'Azur", '93'),
    '06': ("Provence-Alpes-Côte d'Azur", '93'), '13': ("Provence-Alpes-Côte d'Azur", '93'),
    '83': ("Provence-Alpes-Côte d'Azur", '93'), '84': ("Provence-Alpes-Côte d'Azur", '93'),
    
    # Nouvelle-Aquitaine
    '16': ('Nouvelle-Aquitaine', '75'), '17': ('Nouvelle-Aquitaine', '75'),
    '19': ('Nouvelle-Aquitaine', '75'), '23': ('Nouvelle-Aquitaine', '75'),
    '24': ('Nouvelle-Aquitaine', '75'), '33': ('Nouvelle-Aquitaine', '75'),
    '40': ('Nouvelle-Aquitaine', '75'), '47': ('Nouvelle-Aquitaine', '75'),
    '64': ('Nouvelle-Aquitaine', '75'), '79': ('Nouvelle-Aquitaine', '75'),
    '86': ('Nouvelle-Aquitaine', '75'), '87': ('Nouvelle-Aquitaine', '75'),
    
    # Occitanie
    '09': ('Occitanie', '76'), '11': ('Occitanie', '76'),
    '12': ('Occitanie', '76'), '30': ('Occitanie', '76'),
    '31': ('Occitanie', '76'), '32': ('Occitanie', '76'),
    '34': ('Occitanie', '76'), '46': ('Occitanie', '76'),
    '48': ('Occitanie', '76'), '65': ('Occitanie', '76'),
    '66': ('Occitanie', '76'), '81': ('Occitanie', '76'),
    '82': ('Occitanie', '76'),
    
    # Hauts-de-France
    '02': ('Hauts-de-France', '32'), '59': ('Hauts-de-France', '32'),
    '60': ('Hauts-de-France', '32'), '62': ('Hauts-de-France', '32'),
    '80': ('Hauts-de-France', '32'),
    
    # Grand Est
    '08': ('Grand Est', '44'), '10': ('Grand Est', '44'),
    '51': ('Grand Est', '44'), '52': ('Grand Est', '44'),
    '54': ('Grand Est', '44'), '55': ('Grand Est', '44'),
    '57': ('Grand Est', '44'), '67': ('Grand Est', '44'),
    '68': ('Grand Est', '44'), '88': ('Grand Est', '44'),
    
    # Bretagne
    '22': ('Bretagne', '53'), '29': ('Bretagne', '53'),
    '35': ('Bretagne', '53'), '56': ('Bretagne', '53'),
    
    # Pays de la Loire
    '44': ('Pays de la Loire', '52'), '49': ('Pays de la Loire', '52'),
    '53': ('Pays de la Loire', '52'), '72': ('Pays de la Loire', '52'),
    '85': ('Pays de la Loire', '52'),
    
    # Normandie
    '14': ('Normandie', '28'), '27': ('Normandie', '28'),
    '50': ('Normandie', '28'), '61': ('Normandie', '28'),
    '76': ('Normandie', '28'),
    
    # Bourgogne-Franche-Comté
    '21': ('Bourgogne-Franche-Comté', '27'), '25': ('Bourgogne-Franche-Comté', '27'),
    '39': ('Bourgogne-Franche-Comté', '27'), '58': ('Bourgogne-Franche-Comté', '27'),
    '70': ('Bourgogne-Franche-Comté', '27'), '71': ('Bourgogne-Franche-Comté', '27'),
    '89': ('Bourgogne-Franche-Comté', '27'), '90': ('Bourgogne-Franche-Comté', '27'),
    
    # Centre-Val de Loire
    '18': ('Centre-Val de Loire', '24'), '28': ('Centre-Val de Loire', '24'),
    '36': ('Centre-Val de Loire', '24'), '37': ('Centre-Val de Loire', '24'),
    '41': ('Centre-Val de Loire', '24'), '45': ('Centre-Val de Loire', '24'),
    
    # Corse
    '2A': ('Corse', '94'), '2B': ('Corse', '94'),
}


class GeoRefFranceV2:
    """Client with get_full_location_info() method"""
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

    def get_full_location_info(self, city_name: str) -> Optional[Tuple[float, float, str]]:
        """
        🆕 NOUVELLE MÉTHODE: Récupère (latitude, longitude, département) depuis l'API
        
        Args:
            city_name: Nom de ville nettoyé
            
        Returns:
            Tuple (lat, lon, code_département) ou None
            
        Example:
            >>> api.get_full_location_info("Paris")
            (48.8566, 2.3522, '75')
        """
        if not city_name or city_name == 'UNKNOWN':
            return None
       
        # Essai 1: Recherche exacte
        params = {'where': f'com_nom="{city_name}"', 'limit': 1}
        data = self._request(params)

        # Essai 2: Recherche fuzzy si échec
        if not data or 'results' not in data or not data['results']:
            params = {'where': f'search(com_nom, "{city_name}")', 'limit': 1}
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
        dept_code = record.get('dep_code')  # ← DÉPARTEMENT DEPUIS L'API
        
        if lat is not None and lon is not None and dept_code:
            return (float(lat), float(lon), str(dept_code))
        
        return None

    def get_coords_by_city(self, city_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates by city name (legacy - utilise get_full_location_info)"""
        result = self.get_full_location_info(city_name)
        if result:
            return (result[0], result[1])
        return None

    def get_coords_by_department(self, dep_code: str) -> Optional[Tuple[float, float]]:
        """Get coordinates by department"""
        if not dep_code:
            return None
        
        # Utiliser le mapping des régions pour les coordonnées
        if dep_code in COMPLETE_REGION_MAPPING:
            region_name, _ = COMPLETE_REGION_MAPPING[dep_code]
            
            # Coordonnées approximatives des chefs-lieux de région
            region_coords = {
                'Île-de-France': (48.8566, 2.3522),
                'Auvergne-Rhône-Alpes': (45.7640, 4.8357),
                "Provence-Alpes-Côte d'Azur": (43.2965, 5.3698),
                'Nouvelle-Aquitaine': (44.8378, -0.5792),
                'Occitanie': (43.6047, 1.4442),
                'Hauts-de-France': (50.6292, 3.0573),
                'Grand Est': (48.5734, 7.7521),
                'Bretagne': (48.1173, -1.6778),
                'Pays de la Loire': (47.2184, -1.5536),
                'Normandie': (49.4432, 1.0993),
                'Bourgogne-Franche-Comté': (47.2380, 6.0243),
                'Centre-Val de Loire': (47.9029, 1.9093),
                'Corse': (41.9270, 8.7369),
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
        
        params = {'where': f'search(com_nom, "{partial_name}")', 'limit': 1}
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


if __name__ == "__main__":
    """Tests de la nouvelle méthode"""
    print("=" * 60)
    print("TEST: get_full_location_info()")
    print("=" * 60 + "\n")
    
    api = GeoRefFranceV2()
    
    tests = ["Paris", "Lyon", "Marseille", "Bordeaux","Nanc", "Saint Etienne"]
    
    for city in tests:
        result = api.get_full_location_info(city)
        if result:
            lat, lon, dept = result
            print(f"✅ {city:15s} → lat={lat:.4f}, lon={lon:.4f}, dept={dept}")
        else:
            print(f"❌ {city:15s} → Not found")
        time.sleep(1)
    
    print("\n" + "=" * 60)