import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
import csv
import hashlib
import re
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class ServicePublicJobScraperWithLLM:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
        })
        self.all_jobs = []
        self.seen_hashes = set()
        self.mistral_api_key = os.getenv('MISTRAL_API_KEY')
        self.base_url = "https://choisirleservicepublic.gouv.fr"
        
        if not self.mistral_api_key:
            raise ValueError("⚠️ MISTRAL_API_KEY non trouvée dans le fichier .env")

    def get_total_pages(self, search_url):
        """Détermine le nombre total de pages de résultats via les classes DSFR"""
        try:
            response = self.session.get(search_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            page_numbers = []
            # On cherche les liens de pagination (standard DSFR)
            pagination_links = soup.find_all('a', class_='fr-pagination__link')
            
            for link in pagination_links:
                text = link.get_text(strip=True)
                if text.isdigit():
                    page_numbers.append(int(text))
            
            total = max(page_numbers) if page_numbers else 1
            return total
        except Exception as e:
            print(f"⚠️ Erreur lors de la détection du nombre de pages : {e}")
            return 1

    def generate_job_hash(self, job_data):
        """Génère un hash unique pour détecter les doublons"""
        key = f"{job_data.get('titre', '').lower().strip()}_" \
              f"{job_data.get('collectivite', '').lower().strip()}_" \
              f"{job_data.get('lieu', '').lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def is_duplicate(self, job_data):
        """Vérifie si l'offre est un doublon"""
        job_hash = self.generate_job_hash(job_data)
        if job_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(job_hash)
        return False
    
    def extract_with_llm(self, html_content, url):
        """Utilise Mistral AI pour extraire les données de manière intelligente"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            text_content = soup.get_text(separator='\n', strip=True)[:8000]
            
            response = self.session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.mistral_api_key}"
                },
                json={
                    "model": "mistral-large-latest",
                    "messages": [{
                        "role": "user",
                        "content": f"""Analyse cette offre d'emploi et extrait les informations suivantes au format JSON strict.
Si une information n'est pas disponible, utilise une chaîne vide "".

Texte de l'offre:
{text_content}

Réponds UNIQUEMENT avec un objet JSON valide contenant ces champs (rien d'autre, pas de markdown):
{{
  "ID": "numéro de référence de l'annonce"
  "titre": "titre du poste",
  "collectivite": "nom de l'employeur/organisme",
  "departement": "département",
  "lieu": "ville et lieu de travail",
  "grade": "grade ou catégorie",
  "type_emploi": "type de contrat",
  "salaire": "rémunération ou salaire prévu",
  "date_limite": "date limite de candidature",
  "competences": "liste des compétences techniques exigées (uniquement langages, méthodes et outils informatiques, max 300 caractères)",
  "experience": "nombre d'années d'expérience requis",
  "avantages": "liste des avantages en nature du poste",
  "teletravail": "le poste inclut-il du télétravail?",
  "description": "brève description du poste (max 300 caractères)"
}}"""
                    }],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code == 200:
                text_response = response.json()['choices'][0]['message']['content']
                text_response = re.sub(r'```json\n?|\n?```', '', text_response).strip()
                job_data = json.loads(text_response)
                job_data['url'] = url
                return job_data
            else:
                print(f"    ⚠️  Erreur API Mistral: {response.status_code}")
                print(f"    Détails: {response.text}")
                return None
            
        except Exception as e:
            print(f"    ⚠️ Erreur LLM: {e}")
            return None

    def extract_job_links_with_dates(self, page_soup):
        """Extrait les liens et dates depuis le 'soup' d'une page de résultats"""
        job_entries = []
        # On cible les cartes d'offres (structure DSFR typique du site)
        offre_blocks = page_soup.find_all('div', class_='fr-card') or page_soup.find_all('article')
        
        for block in offre_blocks:
            link_tag = block.find('a', href=True)
            if link_tag and '/offre-emploi/' in link_tag['href']:
                full_url = urljoin(self.base_url, link_tag['href'])
                
                text_content = block.get_text(separator=' ', strip=True)
                date_match = re.search(r'En ligne depuis le (\d{1,2} \w+ \d{4})', text_content, re.IGNORECASE)
                date_pub = date_match.group(1) if date_match else ""
                
                job_entries.append({'url': full_url, 'date_publication': date_pub})
        return job_entries

    def scrape_service_public_data(self, max_jobs=None):
        """Boucle itérative pour le mot-clé 'data'"""
        self._scrape_keyword_loop("data", max_jobs)

    def scrape_service_public_ia(self, max_jobs=None):
        """Boucle itérative pour le mot-clé 'ia'"""
        self._scrape_keyword_loop("ia", max_jobs)

    def _scrape_keyword_loop(self, keyword, max_jobs):
        """Logique générique de boucle pour parcourir les pages par mot-clé"""
        print(f"\n=== 🔎 Scraping '{keyword.upper()}' (avec pagination) ===")
        start_url = f"{self.base_url}/nos-offres/filtres/mot-cles/{keyword}/"
        
        total_pages = self.get_total_pages(start_url)
        print(f"📄 Nombre de pages à parcourir : {total_pages}")
        
        jobs_count = 0

        for p in range(1, total_pages + 1):
            if max_jobs and jobs_count >= max_jobs:
                break

            # Construction de l'URL de la page
            current_url = start_url if p == 1 else f"{start_url}page/{p}/"
            print(f"\n🔄 Page {p}/{total_pages} : {current_url}")
            
            try:
                response = self.session.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                job_entries = self.extract_job_links_with_dates(soup)
                
                for entry in job_entries:
                    if max_jobs and jobs_count >= max_jobs:
                        break
                        
                    link = entry['url']
                    print(f"  [{jobs_count + 1}] Analyse : {link}")
                    
                    # Récupérer le contenu détaillé
                    res_detail = self.session.get(link)
                    if res_detail.status_code == 200:
                        job_data = self.extract_with_llm(res_detail.content, link)
                        
                        if job_data and not self.is_duplicate(job_data):
                            job_data['source'] = 'choisirleservicepublic.gouv.fr'
                            job_data['mot_cle'] = keyword
                            job_data['date_publication'] = entry['date_publication']
                            self.all_jobs.append(job_data)
                            jobs_count += 1
                            print(f"    ✅ Extrait : {job_data.get('titre')}")
                        elif job_data:
                            print(f"    ⚠️ Doublon ignoré")
                    
                    time.sleep(1.5) # Délai entre deux offres (Mistral + Serveur)
                
            except Exception as e:
                print(f"  ❌ Erreur sur la page {p} : {e}")

    def scrape_all(self):
        """Lance le scraping pour DATA et IA"""
        print("🚀 Démarrage global...")
        self.scrape_service_public_data(max_jobs=None) # Limite pour test
        self.scrape_service_public_ia(max_jobs=None)   # Limite pour test
        return self.all_jobs

    def save_to_json(self, filename='offres_service_public.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_jobs, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Sauvegardé : {filename}")

    def save_to_csv(self, filename='offres_service_public.csv'):
        if not self.all_jobs: return
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.all_jobs[0].keys())
            writer.writeheader()
            writer.writerows(self.all_jobs)
        print(f"💾 Sauvegardé : {filename}")

# ==================== EXÉCUTION ====================
if __name__ == "__main__":
    try:
        scraper = ServicePublicJobScraperWithLLM()
        scraper.scrape_all()
        scraper.save_to_json()
        scraper.save_to_csv()
        print(f"\n✅ Terminé ! Total : {len(scraper.all_jobs)} offres.")
    except Exception as e:

        print(f"❌ Erreur critique : {e}")
