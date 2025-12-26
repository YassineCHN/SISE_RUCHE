"""
Validation Script for JobTeaser Scraping Results
Analyzes the quality of scraped data
"""

import json
import sys
from collections import Counter
from pathlib import Path


def validate_jobteaser_data(filename):
    """Validate quality of scraped data"""
    
    # Load data
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return 0
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {filename}")
        return 0
    
    if not data:
        print("❌ Empty dataset")
        return 0
    
    print("="*80)
    print(f"📊 VALIDATION REPORT - {Path(filename).name}")
    print("="*80)
    print(f"\n📈 Dataset size: {len(data)} offers\n")
    
    # =========================================================================
    # CRITERION 1: Complete descriptions
    # =========================================================================
    desc_complete = []
    desc_cookie = []
    desc_short = []
    
    for job in data:
        desc = job.get('description', '')
        desc_len = len(desc)
        
        if 'cookie' in desc.lower() and desc_len < 300:
            desc_cookie.append(job['id'])
        elif desc_len < 100:
            desc_short.append(job['id'])
        elif desc_len >= 200:
            desc_complete.append(job['id'])
    
    desc_ok = len(desc_complete)
    desc_pct = desc_ok / len(data) * 100
    
    print("✅ Criterion 1: Complete Descriptions")
    print(f"   Complete (>200 chars): {desc_ok}/{len(data)} ({desc_pct:.1f}%)")
    print(f"   Cookie text detected: {len(desc_cookie)}")
    print(f"   Too short (<100 chars): {len(desc_short)}")
    
    if desc_pct < 70:
        print("   ⚠️  WARNING: Too many incomplete descriptions")
        print("   → Increase PAGE_LOAD_WAIT and CONTENT_LOAD_WAIT")
    print()
    
    # =========================================================================
    # CRITERION 2: Company identification
    # =========================================================================
    company_ok = sum(1 for job in data 
                     if job.get('entreprise') not in [None, "", "Entreprise non spécifiée"])
    company_pct = company_ok / len(data) * 100
    
    print("✅ Criterion 2: Company Identification")
    print(f"   Identified: {company_ok}/{len(data)} ({company_pct:.1f}%)")
    
    if company_pct < 50:
        print("   ⚠️  WARNING: Too many unidentified companies")
        print("   → Check CSS selectors for company extraction")
    print()
    
    # =========================================================================
    # CRITERION 3: Precise location
    # =========================================================================
    location_ok = sum(1 for job in data 
                      if job.get('lieu') not in [None, "", "France"] 
                      and ',' in job.get('lieu', ''))
    location_pct = location_ok / len(data) * 100
    
    # Location analysis
    locations = Counter(job.get('lieu', 'Non spécifié') for job in data)
    
    print("✅ Criterion 3: Precise Location")
    print(f"   Precise (city + region): {location_ok}/{len(data)} ({location_pct:.1f}%)")
    print(f"   Generic 'France': {locations.get('France', 0)}")
    
    if location_pct < 60:
        print("   ⚠️  WARNING: Too many imprecise locations")
        print("   → Check location parsing in get_job_detail_selenium()")
    
    # Top locations
    print(f"\n   Top 5 locations:")
    for loc, count in locations.most_common(5):
        print(f"      - {loc}: {count}")
    print()
    
    # =========================================================================
    # CRITERION 4: Contract type
    # =========================================================================
    contract_ok = sum(1 for job in data 
                      if job.get('type_contrat') not in [None, "", "Non spécifié"])
    contract_pct = contract_ok / len(data) * 100
    
    # Contract type distribution
    contracts = Counter(job.get('type_contrat', 'Non spécifié') for job in data)
    
    print("✅ Criterion 4: Contract Type")
    print(f"   Identified: {contract_ok}/{len(data)} ({contract_pct:.1f}%)")
    
    print(f"\n   Distribution:")
    for contract, count in contracts.most_common(10):
        print(f"      - {contract}: {count}")
    print()
    
    # =========================================================================
    # CRITERION 5: Relevance (Data/IA jobs)
    # =========================================================================
    relevant_keywords = [
        'data scientist', 'data analyst', 'data engineer', 'data architect',
        'machine learning', 'deep learning', 'intelligence artificielle',
        'ia', 'big data', 'mlops'
    ]
    
    relevant_jobs = []
    irrelevant_jobs = []
    
    for job in data:
        title = job.get('intitule', '').lower()
        if any(kw in title for kw in relevant_keywords):
            relevant_jobs.append(job)
        else:
            irrelevant_jobs.append(job)
    
    relevant_pct = len(relevant_jobs) / len(data) * 100
    
    print("✅ Criterion 5: Relevance (Data/IA)")
    print(f"   Relevant jobs: {len(relevant_jobs)}/{len(data)} ({relevant_pct:.1f}%)")
    
    if relevant_pct < 70:
        print("   ⚠️  WARNING: Too many irrelevant jobs")
        print("   → Strengthen EXCLUDE_PATTERNS filtering")
        
        # Show examples of irrelevant jobs
        print("\n   Examples of filtered jobs:")
        for job in irrelevant_jobs[:5]:
            print(f"      - {job['intitule']}")
    
    # Keyword distribution
    keywords_found = Counter()
    for job in data:
        for kw in job.get('matched_keywords', []):
            keywords_found[kw] += 1
    
    print(f"\n   Top matched keywords:")
    for kw, count in keywords_found.most_common(5):
        print(f"      - {kw}: {count}")
    print()
    
    # =========================================================================
    # CRITERION 6: Skills extraction
    # =========================================================================
    skills_ok = sum(1 for job in data if job.get('competences') and len(job['competences']) > 0)
    skills_pct = skills_ok / len(data) * 100
    
    print("✅ Criterion 6: Skills Extraction")
    print(f"   Jobs with skills: {skills_ok}/{len(data)} ({skills_pct:.1f}%)")
    
    # Most common skills
    all_skills = []
    for job in data:
        all_skills.extend(job.get('competences', []))
    
    skill_counts = Counter(all_skills)
    
    if skill_counts:
        print(f"\n   Top 10 skills:")
        for skill, count in skill_counts.most_common(10):
            print(f"      - {skill}: {count}")
    else:
        print("   ⚠️  WARNING: No skills extracted")
        print("   → This is optional but recommended for better analysis")
    print()
    
    # =========================================================================
    # OVERALL SCORE
    # =========================================================================
    weights = {
        'descriptions': (desc_pct, 0.30),      # 30% weight
        'companies': (company_pct, 0.15),      # 15% weight
        'locations': (location_pct, 0.20),     # 20% weight
        'contracts': (contract_pct, 0.15),     # 15% weight
        'relevance': (relevant_pct, 0.20),     # 20% weight
    }
    
    total_score = sum(score * weight for score, weight in weights.values())
    
    print("="*80)
    print(f"🎯 OVERALL QUALITY SCORE: {total_score:.1f}%")
    print("="*80)
    
    print("\nWeighted breakdown:")
    print(f"   Descriptions (30%): {desc_pct:.1f}%")
    print(f"   Companies (15%):    {company_pct:.1f}%")
    print(f"   Locations (20%):    {location_pct:.1f}%")
    print(f"   Contracts (15%):    {contract_pct:.1f}%")
    print(f"   Relevance (20%):    {relevant_pct:.1f}%")
    
    print("\n" + "="*80)
    
    if total_score >= 75:
        print("✅ EXCELLENT - Data ready for project")
        print("   → Proceed to next steps (cleaning, geocoding, database)")
    elif total_score >= 60:
        print("⚠️  GOOD - Minor improvements possible")
        print("   → Data usable but consider running optimization")
    elif total_score >= 45:
        print("⚠️  ACCEPTABLE - Several improvements needed")
        print("   → Review configuration and re-run with adjustments")
    else:
        print("❌ INSUFFICIENT - Major issues detected")
        print("   → Debug and fix issues before using data")
        print("\nRecommended actions:")
        if desc_pct < 70:
            print("   1. Increase PAGE_LOAD_WAIT to 7-8 seconds")
            print("   2. Increase CONTENT_LOAD_WAIT to 12-15 seconds")
        if relevant_pct < 70:
            print("   3. Strengthen EXCLUDE_PATTERNS for better filtering")
        if company_pct < 50 or location_pct < 60:
            print("   4. Review CSS selectors for metadata extraction")
    
    print("="*80)
    
    return total_score


def quick_stats(filename):
    """Quick statistics without full validation"""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        print(f"❌ Cannot read {filename}")
        return
    
    print(f"\n📊 Quick Stats - {Path(filename).name}")
    print(f"   Total offers: {len(data)}")
    
    # Average description length
    avg_desc = sum(len(job.get('description', '')) for job in data) / len(data)
    print(f"   Avg description length: {avg_desc:.0f} chars")
    
    # Companies identified
    companies = sum(1 for job in data if job.get('entreprise') != "Entreprise non spécifiée")
    print(f"   Companies identified: {companies} ({companies/len(data)*100:.1f}%)")
    
    # Unique keywords
    keywords = set()
    for job in data:
        keywords.update(job.get('matched_keywords', []))
    print(f"   Unique keywords: {len(keywords)}")
    print(f"   Keywords: {', '.join(sorted(keywords)[:10])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_results.py <json_file> [--quick]")
        print("\nExamples:")
        print("   python validate_results.py jobs_jobteaser_20251224_160000.json")
        print("   python validate_results.py jobs_jobteaser_20251224_160000.json --quick")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == '--quick':
        quick_stats(filename)
    else:
        score = validate_jobteaser_data(filename)
        sys.exit(0 if score >= 60 else 1)
