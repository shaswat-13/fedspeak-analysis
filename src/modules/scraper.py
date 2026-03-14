import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.federalreserve.gov"

def fetch_and_parse(url, selector_class='col-xs-12 col-sm-8 col-md-8'):
    """Utility to fetch page content and extract text."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_=selector_class)
        return content_div.get_text(strip=True) if content_div else None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def scrape_fomc_materials(category='St'):
    """Scrapes either Statements ('St') or Minutes ('Mn')."""
    hist_url = f"{BASE_URL}/monetarypolicy/materials/assets/final-hist.json"
    recent_url = f"{BASE_URL}/monetarypolicy/materials/assets/final-recent.json"
    
    # Use r.content and set encoding manually to handle the BOM
    def get_json(url):
        r = requests.get(url)
        r.encoding = 'utf-8-sig'
        return r.json()

    data = get_json(hist_url)['mtgitems'] + get_json(recent_url)['mtgitems']
    items = [item for item in data if item['type'] == category and item['d'] >= '2008-01-01']
    
    results = []
    for item in items:
        # ... (rest of your logic remains the same)
        link = item.get('url') or (item['files'][0]['url'] if 'files' in item else None)
        if not link: continue
        
        full_url = BASE_URL + link
        text = fetch_and_parse(full_url)
        results.append({'date': item['d'], 'meeting': item['mtg'], 'url': full_url, 'text': text})
        time.sleep(0.5)
        
    return pd.DataFrame(results)

def scrape_speeches():
    """Scrapes Fed Chair speeches."""
    url = "https://www.federalreserve.gov/json/ne-speeches.json"
    r = requests.get(url)
    r.encoding = 'utf-8-sig'  # This is the key fix
    speeches = r.json()
    
    fed_chairs = ['Powell', 'Yellen']
    filtered = [s for s in speeches if any(c in s.get('s', '') for c in fed_chairs)]
    
    # ... (rest of your logic remains the same)
    results = []
    for s in filtered:
        full_url = BASE_URL + s['l']
        results.append({'date': s['d'], 'title': s['t'], 'speaker': s['s'], 'url': full_url, 'text': fetch_and_parse(full_url)})
        time.sleep(0.5)
        
    return pd.DataFrame(results)

def run_scrapers():
    """Orchestrator for the scraping module."""
    print("Scraping FOMC Statements...")
    scrape_fomc_materials('St').to_csv('data/raw/fomc_statements.csv', index=False)
    
    print("Scraping FOMC Minutes...")
    scrape_fomc_materials('Mn').to_csv('data/raw/fomc_minutes.csv', index=False)
    
    print("Scraping Fed Speeches...")
    scrape_speeches().to_csv('data/raw/fed_speeches.csv', index=False)
    print("Scraping complete.")