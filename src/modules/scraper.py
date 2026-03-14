import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = "https://www.federalreserve.gov"

HIST_URL = "https://www.federalreserve.gov/monetarypolicy/materials/assets/final-hist.json"
RECENT_URL = "https://www.federalreserve.gov/monetarypolicy/materials/assets/final-recent.json"
SPEECH_URL = "https://www.federalreserve.gov/json/ne-speeches.json"



# Fetch JSON

def fetch_json(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()



# Fetch page content

def fetch_text(url, css_class):

    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

        div = soup.find("div", class_=css_class)

        if div:
            return div.get_text(strip=True)

        return None

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None



# Worker function for statements

def scrape_statement_worker(statement):

    if "url" in statement:
        full_url = BASE_URL + statement["url"]
    elif "files" in statement:
        full_url = BASE_URL + statement["files"][0]["url"]
    else:
        return None

    text = fetch_text(full_url, "col-xs-12 col-sm-8 col-md-8")

    return {
        "date": statement["d"],
        "meeting": statement["mtg"],
        "url": full_url,
        "text": text
    }



# Scrape statements (parallel)

def scrape_statements(hist_data, recent_data):

    hist_statements = [
        item for item in hist_data["mtgitems"]
        if item["type"] == "St" and item["d"] >= "2008-01-01"
    ]

    recent_statements = [
        item for item in recent_data["mtgitems"]
        if item["type"] == "St"
    ]

    statements = hist_statements + recent_statements

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [
            executor.submit(scrape_statement_worker, s)
            for s in statements
        ]

        for i, future in enumerate(as_completed(futures)):

            result = future.result()

            if result:
                results.append(result)

            print(f"Processed {i+1}/{len(statements)} statements")

    df = pd.DataFrame(results)

    df.to_csv("data/raw/fomc_statements.csv", index=False)

    return df



# Worker function for minutes

def scrape_minutes_worker(item):

    if "url" in item:
        full_url = BASE_URL + item["url"]
    elif "files" in item:
        full_url = BASE_URL + item["files"][0]["url"]
    else:
        return None

    text = fetch_text(full_url, "generalContentText")

    return {
        "date": item["d"],
        "meeting": item["mtg"],
        "url": full_url,
        "text": text
    }



# Scrape minutes (parallel)

def scrape_minutes(hist_data, recent_data):

    hist_minutes = [
        item for item in hist_data["mtgitems"]
        if item["type"] == "Mn" and item["d"] >= "2008-01-01"
    ]

    recent_minutes = [
        item for item in recent_data["mtgitems"]
        if item["type"] == "Mn"
    ]

    minutes = hist_minutes + recent_minutes

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [
            executor.submit(scrape_minutes_worker, m)
            for m in minutes
        ]

        for i, future in enumerate(as_completed(futures)):

            result = future.result()

            if result:
                results.append(result)

            print(f"Processed {i+1}/{len(minutes)} minutes")

    df = pd.DataFrame(results)

    df.to_csv("data/raw/fomc_minutes.csv", index=False)

    return df



# Worker function for speeches

def scrape_speech_worker(speech):

    full_url = BASE_URL + speech["l"]

    text = fetch_text(full_url, "col-xs-12 col-sm-8 col-md-8")

    return {
        "date": speech["d"],
        "title": speech["t"],
        "speaker": speech["s"],
        "url": full_url,
        "text": text
    }



# Scrape speeches (parallel)

def scrape_speeches():

    r = requests.get(SPEECH_URL)
    r.encoding = "utf-8-sig"

    speeches_data = r.json()

    speeches = [item for item in speeches_data if "d" in item]

    fed_chairs = ["Powell", "Yellen"]

    filtered = [
        item for item in speeches
        if any(chair in item["s"] for chair in fed_chairs)
        and int(item["d"].split("/")[-1].split(" ")[0]) >= 2017
    ]

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [
            executor.submit(scrape_speech_worker, s)
            for s in filtered
        ]

        for i, future in enumerate(as_completed(futures)):

            result = future.result()

            if result:
                results.append(result)

            print(f"Processed {i+1}/{len(filtered)} speeches")

    df = pd.DataFrame(results)

    df.to_csv("data/raw/fed_speeches.csv", index=False)

    return df



# Main pipeline
def run_scraper():

    hist_data = fetch_json(HIST_URL)
    recent_data = fetch_json(RECENT_URL)

    print("Scraping statements...")
    scrape_statements(hist_data, recent_data)

    print("Scraping minutes...")
    scrape_minutes(hist_data, recent_data)

    print("Scraping speeches...")
    scrape_speeches()

    print("Scraping complete.")


if __name__ == "__main__":
    run_scraper()