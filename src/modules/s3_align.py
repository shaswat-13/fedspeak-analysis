import pandas as pd

# Load Fed communication datasets
def load_events(statements_path, minutes_path, speeches_path):

    statements = pd.read_csv(statements_path)
    minutes = pd.read_csv(minutes_path)
    speeches = pd.read_csv(speeches_path)

    statements["event_type"] = "statement"
    minutes["event_type"] = "minutes"
    speeches["event_type"] = "speech"

    events = pd.concat([statements, minutes, speeches], ignore_index=True)

    events["date"] = pd.to_datetime(events["date"], format="mixed")
    events["date"] = events["date"].dt.normalize()

    events = events.sort_values("date").reset_index(drop=True)

    return events



# Load market data
def load_market_data(market_path):

    market = pd.read_csv(market_path)

    market["date"] = pd.to_datetime(market["Date"])

    market = market.sort_values("date").reset_index(drop=True)

    # forward fill treasury yields
    market["treasury_10y"] = market["treasury_10y"].ffill()
    market["treasury_2y"] = market["treasury_2y"].ffill()

    return market



# Align event to next trading day
def align_to_trading_day(event_date, market_dates):

    if event_date in market_dates.values:
        return event_date

    future_dates = market_dates[market_dates > event_date]

    if len(future_dates) == 0:
        return pd.NaT

    return future_dates.iloc[0]


def align_events(events, market):

    events["aligned_date"] = events["date"].apply(
        lambda d: align_to_trading_day(d, market["date"])
    )

    events = events.dropna(subset=["aligned_date"])

    # ensure full estimation and event windows exist
    events = events[
        (events["aligned_date"] >= market["date"].iloc[65]) &
        (events["aligned_date"] <= market["date"].iloc[-31])
    ]

    return events



# Extract event and estimation windows
def extract_event_window(event_date, market):

    idx = market.index[market["date"] == event_date][0]

    estimation_window = market.iloc[idx - 65: idx - 5]
    event_window = market.iloc[idx: idx + 31]

    return estimation_window, event_window


def generate_event_windows(events, market):

    event_windows = []
    estimation_windows = []

    for i, row in events.iterrows():

        event_date = row["aligned_date"]

        try:

            est_win, evt_win = extract_event_window(event_date, market)

            evt_win = evt_win.copy()
            evt_win["t"] = range(0, len(evt_win))
            evt_win["event_id"] = i
            evt_win["event_date"] = event_date

            est_win = est_win.copy()
            est_win["event_id"] = i
            est_win["event_date"] = event_date

            event_windows.append(evt_win)
            estimation_windows.append(est_win)

        except:
            continue

    event_windows_df = pd.concat(event_windows)
    estimation_df = pd.concat(estimation_windows)

    return event_windows_df, estimation_df



# Save processed datasets
def save_outputs(events, event_windows_df, estimation_df):

    event_windows_df.to_csv("data/processed/events_window.csv", index=False)
    events.to_csv("data/processed/events_all.csv", index=False)
    estimation_df.to_csv("data/processed/estimation_window.csv", index=False)



# Main pipeline
def align_market_with_fedevents():

    events = load_events(
        "data/raw/fomc_statements.csv",
        "data/raw/fomc_minutes.csv",
        "data/raw/fed_speeches.csv"
    )

    market = load_market_data("data/raw/market_data.csv")

    events = align_events(events, market)

    event_windows_df, estimation_df = generate_event_windows(events, market)

    save_outputs(events, event_windows_df, estimation_df)

    print("Event alignment and window extraction completed.")


if __name__ == "__main__":
    align_market_with_fedevents()