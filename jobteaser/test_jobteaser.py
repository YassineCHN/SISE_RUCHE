"""
Quick test script for JobTeaser scraper
Tests basic functionality without full scraping
"""

import requests
from bs4 import BeautifulSoup
import re

def test_connection():
    """Test if JobTeaser is accessible"""
    print("🔍 Testing connection to JobTeaser...")
    
    url = "https://www.jobteaser.com/fr/job-offers"
    params = {"q": "data", "page": 1}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Connection OK")
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_parsing():
    """Test if HTML parsing works"""
    print("\n🔍 Testing HTML parsing...")
    
    url = "https://www.jobteaser.com/fr/job-offers"
    params = {"q": "data", "page": 1}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find job cards
        cards = soup.find_all("div", {"class": "JobAdCard_main__1mTeA"})
        
        if cards:
            print(f"✅ Found {len(cards)} job cards")
            
            # Test parsing first card
            card = cards[0]
            link = card.find("a", {"class": "JobAdCard_link__LMtBN"})
            
            if link:
                title = link.get_text(strip=True)
                url = link.get('href', '')
                
                # Extract UUID
                match = re.search(r'/job-offers/([a-f0-9-]{36})', url)
                if match:
                    job_id = match.group(1)
                    print(f"✅ Extracted job ID: {job_id[:20]}...")
                    print(f"✅ Title: {title[:50]}...")
                    return True
                else:
                    print("❌ Could not extract job ID")
                    return False
            else:
                print("❌ Could not find job link")
                return False
        else:
            print("❌ No job cards found (HTML structure may have changed)")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_mongodb():
    """Test MongoDB connection"""
    print("\n🔍 Testing MongoDB connection...")
    
    try:
        import os
        from dotenv import load_dotenv
        from pymongo import MongoClient
        import certifi
        
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        
        if not mongo_uri:
            print("⚠️  MONGO_URI not found in .env file")
            print("   Create a .env file with: MONGO_URI='your_mongodb_uri'")
            return False
        
        client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        
        print("✅ MongoDB connection OK")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB error: {e}")
        return False


def test_keywords():
    """Test keyword matching function"""
    print("\n🔍 Testing keyword matching...")
    
    KEYWORD_PATTERNS = {
        'data': r'\bdata\b',
        'IA': r'\bia\b',
        'intelligence artificielle': r'\bintelligence\s+artificielle\b',
    }
    
    test_cases = [
        ("Data Scientist avec Python", True, ["data"]),
        ("Ingénieur en intelligence artificielle", True, ["intelligence artificielle", "IA"]),
        ("Update database daily", False, []),  # "update" contains "data" but not a match
        ("Media analyst", False, []),  # "media" contains "ia" but not a match
        ("Data analyst et IA", True, ["data", "IA"]),
    ]
    
    all_passed = True
    for text, expected_match, expected_keywords in test_cases:
        text_lower = text.lower()
        matched = [kw for kw, pattern in KEYWORD_PATTERNS.items() 
                   if re.search(pattern, text_lower)]
        has_match = len(matched) > 0
        
        if has_match == expected_match:
            print(f"✅ '{text[:40]}...' -> {matched}")
        else:
            print(f"❌ '{text[:40]}...' -> Expected {expected_match}, got {has_match}")
            all_passed = False
    
    return all_passed


def main():
    print("="*60)
    print("JobTeaser Scraper - Quick Test")
    print("="*60)
    
    results = []
    
    # Test 1: Connection
    results.append(("Connection", test_connection()))
    
    # Test 2: Parsing
    results.append(("HTML Parsing", test_parsing()))
    
    # Test 3: MongoDB
    results.append(("MongoDB", test_mongodb()))
    
    # Test 4: Keywords
    results.append(("Keyword Matching", test_keywords()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the scraper.")
        print("\nNext step: python utils_jt.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the scraper.")
    
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress ENTER to exit...")
