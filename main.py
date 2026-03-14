from src.modules.scraper import run_scraper
from src.modules.market import run_market_pipeline
from src.modules.align import align_market_with_fedevents
# from src.modules.abnormal_returns import calc_abnormal_returns
import time

if __name__ == "__main__":
    # Download all text data
    scraper_s = time.time()
    run_scraper()      
    scraper_e = time.time()

    print("-"*80)
    print("Fed Statements Scraped Successfully")
    print("-"*80)

    # Downloads indices and calculates vol
    market_s = time.time()
    run_market_pipeline() 
    market_e = time.time()

    print("-"*80)
    print("Market Indices Downloaded Successfully")
    print("-"*80)

    # Align fedstatements and market indices
    align_s = time.time()
    align_market_with_fedevents()
    align_e = time.time()

    print("-"*80)
    print("Alignment Done Successfully")
    print("-"*80)

    # calculate abnormal returns
    # abnorm_s = time.time()
    # calc_abnormal_returns()
    # abnorm_e = time.time()

    # print("-"*80)
    # print("Abnormality Returns Calculated Successfully")
    # print("-"*80)

    # check times
    s = scraper_e - scraper_s
    m = market_e - market_s
    a = align_e - align_s
    # ab = abnorm_e - abnorm_s

    print("-"*80)
    print("Times:")
    print(f"Fed statements: {round(s,4)}s")
    print(f"Market indices: {round(m,4)}s")
    print(f"Alignment: {round(a,4)}s")
    # print(f"Abnorm calc: {round(ab,4)}s")
    print("-"*80)
    