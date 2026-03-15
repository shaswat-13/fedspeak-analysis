from src.modules.s1_scraper import run_scraper
from src.modules.s2_market import run_market_pipeline
from src.modules.s3_align import align_market_with_fedevents
from src.modules.s4_preprocessing import run_preprocessing_pipeline
from src.modules.s5_sentiment_analysis import run_sentiment_analysis_pipeline
# from src.modules.abnormal_returns import calc_abnormal_returns
from src.modules.s7_statistical_evaluation import run_statistical_evaluation

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

    # Run Sentiment analysis
    run_sentiment_analysis_pipeline()

    print("-"*80)
    print("Sentiment Analysis Completed")
    print("-"*80)

    # calculate abnormal returns
    # abnorm_s = time.time()
    # calc_abnormal_returns()
    # abnorm_e = time.time()

    # print("-"*80)
    # print("Abnormality Returns Calculated Successfully")
    # print("-"*80)

    # Run Statistical Evaluation
    run_statistical_evaluation()

    print("-"*80)
    print("Statistical Evaluation Completed")
    print("-"*80)


    