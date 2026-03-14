# import necessary libraries
import pandas as pd
import datetime

# read csvs into respective dfs
statements = pd.read_csv("../src/data/raw/fomc_statements.csv")
minutes = pd.read_csv("../src/data/raw/fomc_minutes.csv")
speeches = pd.read_csv("../src/data/raw/fed_speeches.csv")

statements["event_type"] = "statement"
minutes["event_type"] = "minutes"
speeches["event_type"] = "speech"

# concatenate the dfs into a single one
events = pd.concat([statements, minutes, speeches])

# convert the date to datetime and sort the df by the date of different fed events
events["date"] = pd.to_datetime(events["date"], format="mixed")
events["date"] = events["date"].dt.normalize()
events = events.sort_values("date").reset_index(drop=True)

# load the market data
market = pd.read_csv("../src/data/raw/market_data.csv")
market["date"] = pd.to_datetime(market["Date"])
market = market.sort_values("date").reset_index(drop=True)


# align the fed event to a trading day and if it occurs on a day where market is closed, align it to the next trading day
def align_to_trading_day(event_date, market_dates):

    if event_date in market_dates.values:
        return event_date

    future_dates = market_dates[market_dates > event_date]

    return future_dates.iloc[0]

events["aligned_date"] = events["date"].apply(
    lambda d: align_to_trading_day(d, market["date"])
)

# create event windows:
# estimation window: t-65 days to t-5 days for normal market behaviour
# event window: t=0 to t+30 days for changes due to fed events
def extract_event_window(event_date, market):

    idx = market.index[market["date"] == event_date][0]

    estimation_window = market.iloc[idx-65: idx-5]
    event_window = market.iloc[idx: idx+31]

    return estimation_window, event_window


# generate a df for all the events
event_windows = []

for i, row in events.iterrows():

    event_date = row["aligned_date"]

    try:
        est_win, evt_win = extract_event_window(event_date, market)

        evt_win = evt_win.copy()

        evt_win["t"] = range(0, len(evt_win))
        evt_win["event_id"] = i
        evt_win["event_date"] = event_date

        event_windows.append(evt_win)

    except:
        continue

event_windows_df = pd.concat(event_windows)

# a df for estimation 
estimation_windows = []

for i, row in events.iterrows():

    event_date = row["aligned_date"]

    try:

        est_win, evt_win = extract_event_window(event_date, market)

        est_win = est_win.copy()

        est_win["event_id"] = i
        est_win["event_date"] = event_date

        estimation_windows.append(est_win)

    except:
        continue

estimation_df = pd.concat(estimation_windows)

# forward fill the treasury data since it doesnt change value directly
market["treasury_10y"] = market["treasury_10y"].ffill()
market["treasury_2y"] = market["treasury_2y"].ffill()

event_windows_df.to_csv('../../data/processed/events_window.csv', index=False)
events.to_csv('../../data/processed/events_all.csv', index=False)
estimation_df.to_csv('../../data/processed/estimation_window.csv', index=False)