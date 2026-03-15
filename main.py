from src.modules.scraper import run_scraper
from src.modules.market import run_market_pipeline
from src.modules.align import align_market_with_fedevents
# from src.modules.abnormal_returns import calc_abnormal_returns
from src.modules.preprocessing import run_preprocessing_pipeline

import time

if __name__ == "__main__":

    # Download all text data
    run_scraper()      

    print("-"*80)
    print("Fed Statements Scraped Successfully")
    print("-"*80)

    # Downloads indices and calculates vol
    run_market_pipeline() 

    print("-"*80)
    print("Market Indices Downloaded Successfully")
    print("-"*80)

    # Align fedstatements and market indices
    align_market_with_fedevents()

    print("-"*80)
    print("Alignment Done Successfully")
    print("-"*80)

    # Preprocess with NLTK
    run_preprocessing_pipeline()

    print("-"*80)
    print("Pre Processing Completed")
    print("-"*80)

    # calculate abnormal returns
    # abnorm_s = time.time()
    # calc_abnormal_returns()
    # abnorm_e = time.time()

    # print("-"*80)
    # print("Abnormality Returns Calculated Successfully")
    # print("-"*80)

    