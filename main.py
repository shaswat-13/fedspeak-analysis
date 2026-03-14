from src.modules.scraper import run_scrapers
from src.modules.market import run_market_pipeline

if __name__ == "__main__":
    run_scrapers()      # Downloads all text data
    run_market_pipeline() # Downloads indices and calculates vol
    print("All tasks finished successfully.")