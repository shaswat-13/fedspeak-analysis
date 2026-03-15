
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)



# 1. LOAD DATA
def load_data():
    estimation = pd.read_csv("data/processed/estimation_window.csv")
    events     = pd.read_csv("data/processed/events_window.csv")
    market     = pd.read_csv("data/raw/market_data.csv")

    tone_path = "data/finbertscores/events_finbert.csv"
    if os.path.exists(tone_path):
        tone_df = pd.read_csv(tone_path)
        print(f"  Tone scores loaded: {len(tone_df)} rows")
    else:
        print(f"  WARNING: {tone_path} not found — tone columns will be NaN.")
        tone_df = None

    # Normalise date columns across all DataFrames
    for df in [estimation, events, market]:
        col = "date" if "date" in df.columns else "Date"
        df["date"] = pd.to_datetime(df[col])

    return estimation, events, market, tone_df



# 2. PARAMETER ESTIMATION — CONSTANT MEAN RETURN MODEL
def estimate_parameters(estimation):
    params = []

    for event_id, group in estimation.groupby("event_id"):

        group = group.dropna(subset=["sp500_log_return"])

        if len(group) < 10:
            # Fewer than 10 observations makes the mean unreliable; skip.
            print(f"  SKIP event_id={event_id}: only {len(group)} estimation obs")
            continue

        params.append({
            "event_id":       event_id,
            "alpha":          group["sp500_log_return"].mean(),
            "est_std_return": group["sp500_log_return"].std(),
            "est_n":          len(group)
        })

    df = pd.DataFrame(params)
    print(f"  Parameters estimated for {len(df)} events "
          f"(mean alpha = {df['alpha'].mean():.6f})")
    return df



# 3. ABNORMAL RETURNS


def compute_abnormal_returns(events, params):
    """
    Compute abnormal returns for every row in the event window.

        AR_t = R_t - alpha_hat

    where alpha_hat is the constant mean return estimated in step 2.

    Also computes a running CAR (cumulative abnormal return) per event,
    used for the CAAR plot.

    Parameters
    ----------
    events : DataFrame
        Event window rows (t=0..30). Must contain: event_id, t,
        sp500_log_return.

    params : DataFrame
        Output of estimate_parameters(). Must contain: event_id, alpha.

    Returns
    -------
    events : DataFrame
        Input with three new columns:
            expected_return  — constant alpha for that event
            abnormal_return  — R_t minus expected_return
            CAR              — cumulative sum of abnormal_return per event
    """
    events = events.merge(
        params[["event_id", "alpha"]],
        on="event_id",
        how="left"
    )

    events["expected_return"] = events["alpha"]
    events["abnormal_return"] = (
        events["sp500_log_return"] - events["expected_return"]
    )

    events = events.sort_values(["event_id", "t"]).reset_index(drop=True)

    events["CAR"] = (
        events
        .groupby("event_id")["abnormal_return"]
        .cumsum()
    )

    n_missing = events["abnormal_return"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} abnormal_return values are NaN "
              f"(events not matched in params)")

    return events



# 4. CAR WINDOWS


def compute_car_windows(events):
    """
    Sum abnormal returns within fixed event-time windows per event.

    Windows:
        CAR_0_3   — days t=0 to t=3   (immediate reaction, ~1 week)
        CAR_0_10  — days t=0 to t=10  (short-term,  ~2 weeks)
        CAR_0_30  — days t=0 to t=30  (medium-term, ~1 month)

    Parameters
    ----------
    events : DataFrame
        Output of compute_abnormal_returns(). Must contain:
        event_id, t, abnormal_return.

    Returns
    -------
    DataFrame with columns: event_id, CAR_0_3, CAR_0_10, CAR_0_30
    """
    def _car(max_t, col):
        return (
            events[events["t"] <= max_t]
            .groupby("event_id")["abnormal_return"]
            .sum()
            .reset_index(name=col)
        )

    car = (
        _car(3,  "CAR_0_3")
        .merge(_car(10, "CAR_0_10"), on="event_id")
        .merge(_car(30, "CAR_0_30"), on="event_id")
    )

    print(f"  CAR windows computed for {len(car)} events")
    print(f"    Mean CAR(0,3)  = {car['CAR_0_3'].mean():.4f}")
    print(f"    Mean CAR(0,10) = {car['CAR_0_10'].mean():.4f}")
    print(f"    Mean CAR(0,30) = {car['CAR_0_30'].mean():.4f}")

    return car



# 5. EXCESS VIX VOLATILITY


def compute_vix_volatility(estimation, events):
    """
    Measure the change in implied volatility around each event.

        Excess VIX = mean(VIX[t=0..3]) - mean(VIX[estimation window])

    A positive excess VIX means the market became more uncertain after
    the Fed communication relative to the pre-event baseline.

    Baseline is pulled from estimation_window.csv (not from the event
    window DataFrame, which only contains t=0..30 and has no
    negative-t rows to average over).

    Parameters
    ----------
    estimation : DataFrame  — must contain: event_id, vix
    events     : DataFrame  — must contain: event_id, t, vix

    Returns
    -------
    DataFrame with columns: event_id, vix_baseline, vix_event, excess_vix
    """
    baseline = (
        estimation
        .groupby("event_id")["vix"]
        .mean()
        .reset_index(name="vix_baseline")
    )

    event_vix = (
        events[events["t"] <= 3]
        .groupby("event_id")["vix"]
        .mean()
        .reset_index(name="vix_event")
    )

    vix = baseline.merge(event_vix, on="event_id")
    vix["excess_vix"] = vix["vix_event"] - vix["vix_baseline"]
    return vix



# 6. EXCESS REALIZED VOLATILITY


def compute_realized_volatility(estimation, events):
    """
    Measure the change in realized volatility around each event.

        Excess RV = mean(RV[t=0..20]) - mean(RV[estimation window])

    realized_volatility is the 20-day rolling standard deviation of
    log returns pre-computed in Stage 2 (s2_market.py).

    Baseline comes from estimation_window.csv — not from negative-t rows
    in the event window DataFrame (those rows do not exist; event_window
    only contains t=0..30).

    Parameters
    ----------
    estimation : DataFrame  — must contain: event_id, realized_volatility
    events     : DataFrame  — must contain: event_id, t, realized_volatility

    Returns
    -------
    DataFrame with columns: event_id, rv_baseline, rv_event, excess_rv
    """
    rv_baseline = (
        estimation
        .groupby("event_id")["realized_volatility"]
        .mean()
        .reset_index(name="rv_baseline")
    )

    rv_event = (
        events[events["t"] <= 20]
        .groupby("event_id")["realized_volatility"]
        .mean()
        .reset_index(name="rv_event")
    )

    rv = rv_event.merge(rv_baseline, on="event_id")
    rv["excess_rv"] = rv["rv_event"] - rv["rv_baseline"]
    return rv



# 7. AAR AND CAAR


def compute_aar_caar(events):
    """
    Aggregate abnormal returns across all events for each event-time t.

        AAR_t  = (1/N) * sum(AR_{i,t})   — cross-sectional mean at time t
        CAAR_t = sum_{s=0}^{t} AAR_s     — running cumulative sum

    This is the main diagnostic for whether Fed communication
    systematically shifted market returns on and around the event day.

    Parameters
    ----------
    events : DataFrame  — must contain: t, abnormal_return

    Returns
    -------
    DataFrame with columns: t, AAR, CAAR
    """
    aar = (
        events
        .groupby("t")["abnormal_return"]
        .mean()
        .reset_index()
        .rename(columns={"abnormal_return": "AAR"})
        .sort_values("t")
        .reset_index(drop=True)
    )
    aar["CAAR"] = aar["AAR"].cumsum()
    return aar



# 8. SIGNIFICANCE TESTS


def t_test(events):
    """
    One-sample t-test at each event day t.

    H0: mean(AR_t) = 0   across all events at this t
    H1: mean(AR_t) != 0

    A significant result at t=0 or t=1 means the Fed communication
    produced a statistically detectable return shift beyond the
    historical baseline.

    Returns
    -------
    DataFrame with columns: t, t_stat, p_value, n, mean_ar
    """
    rows = []
    for t in sorted(events["t"].unique()):
        sample = events[events["t"] == t]["abnormal_return"].dropna()
        if len(sample) < 3:
            continue
        t_stat, p_val = stats.ttest_1samp(sample, 0)
        rows.append({
            "t":       t,
            "t_stat":  round(t_stat, 4),
            "p_value": round(p_val,  4),
            "n":       len(sample),
            "mean_ar": round(sample.mean(), 6)
        })
    return pd.DataFrame(rows)


def bootstrap_ci(events, n_bootstrap=1000, seed=42):
    """
    Non-parametric 95% confidence interval for mean AR at each event day.

    Bootstraps by resampling (with replacement) from the cross-section
    of events at each t. More robust than the t-test when the distribution
    of abnormal returns is heavy-tailed (common around crisis periods).

    Parameters
    ----------
    n_bootstrap : int   Number of bootstrap resamples (1000 is standard)
    seed        : int   Random seed for reproducibility

    Returns
    -------
    DataFrame with columns: t, ci_lower, ci_upper
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for t in sorted(events["t"].unique()):
        sample = events[events["t"] == t]["abnormal_return"].dropna().values
        if len(sample) < 3:
            continue

        boot_means = [
            rng.choice(sample, size=len(sample), replace=True).mean()
            for _ in range(n_bootstrap)
        ]
        rows.append({
            "t":        t,
            "ci_lower": round(float(np.percentile(boot_means, 2.5)),  6),
            "ci_upper": round(float(np.percentile(boot_means, 97.5)), 6)
        })

    return pd.DataFrame(rows)



# 9. TONE SCORE MERGE


def merge_tone_scores(car_df, tone_df,
                      events_all_path="data/processed/events_all.csv"):
    """
    Join FinBERT tone scores onto the per-event CAR results table.

    Stage 5 outputs: positive / negative / neutral probabilities and
    keyword_score. This function:

        1. Assigns event_id to tone_df rows if the column is missing,
           by matching row positions against events_all.csv.

        2. Computes the composite tone_score if not already present:
               tone_score = 0.6 * keyword_score + 0.4 * (positive - negative)

        3. Classifies each document:
               tone_score > +0.2  ->  hawkish
               tone_score < -0.2  ->  dovish
               otherwise          ->  neutral

        4. Aggregates by event_id using mean (handles cases where
           multiple documents — e.g. statement + speech — fall on
           the same event date).

        5. Merges onto car_df on event_id.

    Parameters
    ----------
    car_df         : DataFrame  — output of compute_car_windows()
    tone_df        : DataFrame or None
    events_all_path: str

    Returns
    -------
    car_df with tone columns appended (unchanged if tone_df is None)
    """
    if tone_df is None:
        return car_df

    tone_df = tone_df.copy()

    # Step 1: assign event_id if missing
    if "event_id" not in tone_df.columns:
        events_all = pd.read_csv(events_all_path)
        if len(tone_df) == len(events_all):
            tone_df["event_id"] = events_all.index
        else:
            print(f"  WARNING: tone_df ({len(tone_df)} rows) != "
                  f"events_all ({len(events_all)} rows). Skipping tone merge.")
            return car_df

    # Step 2: compute composite tone_score
    if "tone_score" not in tone_df.columns:
        kw = tone_df["keyword_score"].fillna(0)
        fb = tone_df["positive"].fillna(0) - tone_df["negative"].fillna(0)
        tone_df["tone_score"] = 0.6 * kw + 0.4 * fb

    # Step 3: classify tone_label
    if "tone_label" not in tone_df.columns:
        tone_df["tone_label"] = np.select(
            [tone_df["tone_score"] > 0.2,
             tone_df["tone_score"] < -0.2],
            ["hawkish", "dovish"],
            default="neutral"
        )

    # Step 4: aggregate per event_id
    numeric_cols = [c for c in
                    ["positive", "negative", "neutral",
                     "keyword_score", "tone_score"]
                    if c in tone_df.columns]

    tone_agg = (
        tone_df[["event_id"] + numeric_cols]
        .groupby("event_id")
        .mean()
        .reset_index()
    )

    # Re-classify after aggregation (mean may shift borderline documents)
    tone_agg["tone_label"] = np.select(
        [tone_agg["tone_score"] > 0.2,
         tone_agg["tone_score"] < -0.2],
        ["hawkish", "dovish"],
        default="neutral"
    )

    print(f"  Tone labels: {tone_agg['tone_label'].value_counts().to_dict()}")

    # Step 5: merge
    return car_df.merge(tone_agg, on="event_id", how="left")



# 10. REGRESSION: CAR ~ tone + volatility + rates


def run_regression(dataset, car_col="CAR_0_3"):
    """
    OLS cross-sectional regression of CAR on tone and control variables.

        CAR_i = b0 + b1*tone_score_i + b2*excess_vix_i
                   + b3*treasury_10y_i + b4*treasury_2y_i + e_i

    Tests whether tone_score has incremental explanatory power for
    abnormal returns after controlling for surprise volatility and the
    prevailing rate environment.

    treasury_10y / treasury_2y are t=0 values (rate level on the
    announcement day), giving context for whether the announcement
    occurred in a rising-rate or falling-rate regime.

    Skips gracefully if tone_score or excess_vix are missing.
    """
    required = ["tone_score", "excess_vix"]
    missing  = [c for c in required if c not in dataset.columns]
    if missing:
        print(f"  Skipping regression — missing columns: {missing}")
        return None

    rate_cols    = [c for c in ["treasury_10y", "treasury_2y"]
                    if c in dataset.columns]
    feature_cols = ["tone_score", "excess_vix"] + rate_cols
    subset       = dataset[feature_cols + [car_col]].dropna()

    if len(subset) < 20:
        print(f"  Skipping regression — only {len(subset)} complete rows")
        return None

    X     = sm.add_constant(subset[feature_cols])
    y     = subset[car_col]
    model = sm.OLS(y, X).fit()

    print("\n" + "=" * 60)
    print(f"OLS Regression: {car_col} ~ " + " + ".join(feature_cols))
    print("=" * 60)
    print(model.summary())
    return model



# 11. PLOTS


def plot_caar(events, save_path="outputs/caar_plot.png"):
    """
    Line chart of CAAR from t=-10 to t=30.

    Showing a pre-event window (t=-10 to t=-1) allows a visual check for
    anticipation effects — if CAAR trends noticeably before t=0 it
    suggests markets partially priced in the announcement beforehand.
    """
    subset = events[(events["t"] >= -10) & (events["t"] <= 30)].copy()
    aar    = compute_aar_caar(subset)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(aar["t"], aar["CAAR"], color="steelblue", linewidth=2, label="CAAR")
    ax.axvline(0, color="red",   linestyle="--", linewidth=1, label="Event (t=0)")
    ax.axhline(0, color="black", linestyle="-",  linewidth=0.5)
    ax.fill_between(aar["t"], aar["CAAR"], 0,
                    where=(aar["CAAR"] >= 0), alpha=0.1, color="green")
    ax.fill_between(aar["t"], aar["CAAR"], 0,
                    where=(aar["CAAR"] <  0), alpha=0.1, color="red")
    ax.set_title("Cumulative Average Abnormal Return (CAAR)\n"
                 "Constant Mean Return Model — All Fed Events 2008–2025",
                 fontsize=13)
    ax.set_xlabel("Event Time (Trading Days Relative to Announcement)")
    ax.set_ylabel("CAAR (log return)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  CAAR plot        -> {save_path}")


def plot_avg_ar(aar, ttest, save_path="outputs/avg_ar.png"):
    """
    Bar chart of AAR by event day with significance stars (p < 0.05).

    Blue = positive AAR, red = negative AAR.
    Black star above/below a bar means t-test p < 0.05.
    """
    merged = aar.merge(ttest[["t", "p_value"]], on="t", how="left")
    sig    = merged[merged["p_value"] < 0.05]

    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = ["steelblue" if v >= 0 else "tomato" for v in merged["AAR"]]
    ax.bar(merged["t"], merged["AAR"], color=colors, width=0.8, alpha=0.8)

    if len(sig) > 0:
        offset = merged["AAR"].abs().max() * 0.05
        ax.scatter(sig["t"],
                   sig["AAR"] + np.sign(sig["AAR"]) * offset,
                   marker="*", color="black", s=90, zorder=5, label="p<0.05")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="red",   linestyle="--", linewidth=1,
               label="Event day (t=0)")
    ax.set_title("Average Abnormal Return (AAR) by Event Day")
    ax.set_xlabel("Event Time t")
    ax.set_ylabel("AAR (log return)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  AAR plot         -> {save_path}")


def plot_car_by_tone(results, car_col="CAR_0_3",
                     save_path="outputs/car_by_tone.png"):
    """
    Box plot comparing CAR distributions for Hawkish / Neutral / Dovish.

    This is the core visual for the research question: does tone predict
    the direction and magnitude of the market reaction?
    Only produced when tone_label column is present.
    """
    if "tone_label" not in results.columns:
        print("  Skipping car_by_tone plot — tone_label not available")
        return

    order  = ["hawkish", "neutral", "dovish"]
    colors = {"hawkish": "tomato",
              "neutral": "steelblue",
              "dovish":  "mediumseagreen"}

    groups = [results[results["tone_label"] == lb][car_col].dropna()
              for lb in order]
    counts = [len(g) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(groups,
                    labels=[f"{lb}\n(n={n})" for lb, n in zip(order, counts)],
                    patch_artist=True, notch=False)
    for patch, lb in zip(bp["boxes"], order):
        patch.set_facecolor(colors[lb])
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"{car_col} Distribution by Policy Tone")
    ax.set_xlabel("Tone Label")
    ax.set_ylabel(f"{car_col} (log return)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  CAR by tone plot -> {save_path}")



# 12. MAIN PIPELINE


def run_event_study():

    print("\n" + "=" * 60)
    print("STAGE 6 — Event Study (Constant Mean Return Model)")
    print("=" * 60)

    # 1. Load
    print("\n[1/9] Loading data...")
    estimation, events, market, tone_df = load_data()

    # 2. Estimate alpha per event
    print("\n[2/9] Estimating baseline mean return per event...")
    params = estimate_parameters(estimation)

    # 3. Compute abnormal returns
    print("\n[3/9] Computing abnormal returns (AR = R - alpha)...")
    events = compute_abnormal_returns(events, params)
    events.to_csv("data/processed/events_with_ar.csv", index=False)
    print(f"  Saved -> data/processed/events_with_ar.csv")

    # 4. CAR windows
    print("\n[4/9] Computing CAR windows (0,3) (0,10) (0,30)...")
    car = compute_car_windows(events)

    # 5. Excess VIX
    print("\n[5/9] Computing excess VIX...")
    vix = compute_vix_volatility(estimation, events)

    # 6. Excess realized volatility
    print("\n[6/9] Computing excess realized volatility...")
    rv = compute_realized_volatility(estimation, events)

    # 7. Assemble results table
    print("\n[7/9] Assembling results table...")
    results = (
        car
        .merge(vix, on="event_id")
        .merge(rv,  on="event_id")
    )

    # Attach t=0 metadata (event date + rate levels on announcement day)
    t0_rows = (
        events[events["t"] == 0]
        .drop_duplicates("event_id")
    )
    keep = ["event_id", "treasury_10y", "treasury_2y"]
    if "event_date" in t0_rows.columns:
        keep.insert(1, "event_date")
    results = results.merge(t0_rows[keep], on="event_id", how="left")

    # Merge tone scores from Stage 5
    results = merge_tone_scores(results, tone_df)
    results.to_csv("data/processed/event_study_results.csv", index=False)
    print(f"  Saved -> data/processed/event_study_results.csv  "
          f"({len(results)} rows, cols: {list(results.columns)})")

    # 8. AAR / CAAR + significance tests
    print("\n[8/9] Computing AAR/CAAR and running tests...")
    aar   = compute_aar_caar(events)
    ttest = t_test(events)
    ci    = bootstrap_ci(events)

    aar.to_csv("data/processed/aar_caar.csv",        index=False)
    ttest.to_csv("data/processed/ttest_results.csv", index=False)
    ci.to_csv("data/processed/bootstrap_ci.csv",     index=False)
    params.to_csv("data/processed/cmrm_params.csv",  index=False)

    sig_days = ttest[ttest["p_value"] < 0.05]["t"].tolist()
    print(f"  Significant days (p<0.05): {sig_days if sig_days else 'none'}")

    # Regression
    run_regression(results, car_col="CAR_0_3")

    # 9. Plots
    print("\n[9/9] Generating plots...")
    plot_caar(events)
    plot_avg_ar(aar, ttest)
    plot_car_by_tone(results)

    # Summary
    print("\n" + "=" * 60)
    print("Stage 6 complete.")
    print(f"  Events processed : {len(results)}")
    print(f"  Mean CAR(0,3)    : {results['CAR_0_3'].mean():.4f}")
    print(f"  Mean CAR(0,10)   : {results['CAR_0_10'].mean():.4f}")
    print(f"  Mean CAR(0,30)   : {results['CAR_0_30'].mean():.4f}")
    if "tone_label" in results.columns:
        print(f"  Tone distribution:\n"
              f"{results['tone_label'].value_counts().to_string()}")
    print("=" * 60)

    return results, aar, ttest, ci


if __name__ == "__main__":
    run_event_study()