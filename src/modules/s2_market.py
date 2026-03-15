import yfinance as yf
import pandas as pd
import numpy as np

# Configuration
START_DATE = "2008-01-01"
END_DATE = "2026-02-01"
TICKERS = {
    "sp500": "^GSPC",
    "vix": "^VIX",
    "10y_treasury": "^TNX",
    "2y_treasury": "^IRX"
}
OUTPUT_PATH = "data/raw/market_data.csv"

def download_market_data():
    """Downloads market data from Yahoo Finance."""
    print("Downloading market data...")
    data = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=True
    )
    return data

def process_market_data(data):
    """Processes raw yfinance data into a clean, analytical dataframe."""
    df = pd.DataFrame()

    # Mapping raw yfinance multi-index to flat columns
    df["sp500_open"] = data["Open"]["^GSPC"]
    df["sp500_high"] = data["High"]["^GSPC"]
    df["sp500_low"] = data["Low"]["^GSPC"]
    df["sp500_close"] = data["Close"]["^GSPC"]
    df["sp500_volume"] = data["Volume"]["^GSPC"]
    df["vix"] = data["Close"]["^VIX"]
    df["treasury_10y"] = data["Close"]["^TNX"]
    df["treasury_2y"] = data["Close"]["^IRX"]

    # Log returns: Essential for stationary time series analysis
    df["sp500_log_return"] = np.log(df["sp500_close"] / df["sp500_close"].shift(1))

    # Rolling Volatility (20-day)
    df["realized_volatility"] = df["sp500_log_return"].rolling(window=20).std()

    return df.reset_index()

def run_market_pipeline():
    """Execution function to be called by the main orchestrator."""
    raw_data = download_market_data()
    market_df = process_market_data(raw_data)
    
    print(f"Saving dataset to {OUTPUT_PATH}...")
    market_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Pipeline finished. Total rows: {len(market_df)}")

if __name__ == "__main__":
    run_market_pipeline()