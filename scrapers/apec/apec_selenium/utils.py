"""
Module utilitaire pour l'affichage, la progression et les statistiques.
"""

import sys
import time
from typing import Dict


class ProgressTracker:
    """Gestionnaire de progression avec affichage en temps réel."""
    
    def __init__(self, total: int):
        self.total = total
        self.success = 0
        self.failed = 0
        self.start_time = time.time()
    
    def increment_success(self):
        """Incrémente le compteur de succès."""
        self.success += 1
    
    def increment_failed(self):
        """Incrémente le compteur d'échecs."""
        self.failed += 1
    
    def get_current(self) -> int:
        """Retourne le nombre total traité."""
        return self.success + self.failed
    
    def get_elapsed(self) -> float:
        """Retourne le temps écoulé en secondes."""
        return time.time() - self.start_time
    
    def get_eta(self) -> str:
        """Calcule et retourne l'ETA formaté."""
        current = self.get_current()
        if current == 0:
            return "..."
        
        elapsed = self.get_elapsed()
        avg = elapsed / current
        remaining = avg * (self.total - current)
        
        return f"{int(remaining//60)}m{int(remaining%60):02d}s"
    
    def get_speed(self) -> float:
        """Retourne la vitesse en offres/min."""
        elapsed = self.get_elapsed()
        if elapsed == 0:
            return 0.0
        return self.success / elapsed * 60
    
    def get_success_rate(self) -> float:
        """Retourne le taux de succès en %."""
        current = self.get_current()
        if current == 0:
            return 0.0
        return self.success / current * 100
    
    def print_progress(self):
        """Affiche la barre de progression."""
        current = self.get_current()
        progress = current / self.total
        
        # Barre
        bar_len = 50
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        # Métriques
        eta = self.get_eta()
        speed = self.get_speed()
        rate = self.get_success_rate()
        
        # Affichage
        print(
            f"\r[{bar}] {current}/{self.total} | "
            f"✅{self.success} ❌{self.failed} | "
            f"ETA:{eta} | {speed:.1f}/min | {rate:.1f}%",
            end=''
        )
        sys.stdout.flush()
    
    def print_final_stats(self):
        """Affiche les statistiques finales."""
        elapsed = self.get_elapsed()
        
        print("\n")
        print("=" * 90)
        print("🎯 RÉSULTATS FINAUX")
        print("=" * 90)
        print()
        print(f"✅ Succès           : {self.success}/{self.total} ({self.get_success_rate():.1f}%)")
        print(f"❌ Échecs           : {self.failed}/{self.total}")
        print(f"⏱️  Temps scraping   : {int(elapsed//60)}m{int(elapsed%60):02d}s")
        print(f"⚡ Vitesse moyenne  : {self.get_speed():.1f} offres/min")


def print_header():
    """Affiche l'en-tête du programme."""
    print("=" * 90)
    print("🚀 SCRAPER APEC - ARCHITECTURE MODULAIRE OPTIMISÉE")
    print("=" * 90)
    print()
    print("📋 ÉTAPE 1: Collecte des URLs (< 30 jours, max 1000/mot-clé)")
    print("⚡ ÉTAPE 2: Scraping parallèle (HTML uniquement)")
    print("🧠 ÉTAPE 3: NLP ultra-rapide (sans Selenium)")
    print()


def print_section(title: str):
    """Affiche un titre de section."""
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    print()


def get_user_keywords() -> list:
    """
    Demande les mots-clés à l'utilisateur.
    
    Returns:
        Liste de mots-clés
    """
    print("📝 Mots-clés de recherche (un par ligne, ligne vide pour terminer):")
    print("   💡 Exemples: 'machine learning', 'data scientist', 'python developer'")
    print()
    
    keywords = []
    while True:
        keyword = input(f"   Mot-clé {len(keywords)+1}: ").strip()
        if not keyword:
            break
        keywords.append(keyword)
    
    if not keywords:
        keywords = ["machine learning"]
        print(f"\n💡 Aucun mot-clé saisi, utilisation par défaut: '{keywords[0]}'")
    
    return keywords


def get_num_workers() -> int:
    """
    Demande le nombre de threads à l'utilisateur.
    
    Returns:
        Nombre de threads (entre 1 et 15)
    """
    try:
        workers = int(input("\n⚙️  Threads parallèles (défaut: 10, max: 15): ").strip() or "10")
        return min(max(workers, 1), 15)
    except:
        return 10


def print_nlp_progress(current: int, total: int):
    """Affiche la progression du NLP."""
    pct = current / total * 100
    print(f"   Traité: {current}/{total} ({pct:.1f}%)", end='\r')


def display_summary(enriched_offers: list, tracker: ProgressTracker):
    """
    Affiche un résumé des données extraites.
    
    Args:
        enriched_offers: Liste des offres enrichies
        tracker: Tracker de progression
    """
    total = len(enriched_offers)
    
    print()
    print("📊 STATISTIQUES D'EXTRACTION:")
    print("-" * 90)
    
    stats = {
        "Entreprises": sum(1 for o in enriched_offers if o.get('company_name')),
        "Localisations": sum(1 for o in enriched_offers if o.get('location')),
        "Dates publication": sum(1 for o in enriched_offers if o.get('publication_date')),
        "Salaires": sum(1 for o in enriched_offers if o.get('salary')),
        "Offres < 30j": sum(1 for o in enriched_offers if o.get('is_recent')),
        "Soft skills": sum(len(o.get('soft_skills', [])) for o in enriched_offers),
        "Hard skills": sum(len(o.get('hard_skills', [])) for o in enriched_offers),
    }
    
    for key, val in stats.items():
        if key in ["Entreprises", "Localisations", "Dates publication", "Salaires", "Offres < 30j"]:
            pct = val / total * 100 if total > 0 else 0
            print(f"   • {key:<20} : {val:>4}/{total} ({pct:>5.1f}%)")
        else:
            print(f"   • {key:<20} : {val:>4}")