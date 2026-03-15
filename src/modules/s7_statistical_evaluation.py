"""
Stage 7: Statistical Evaluation

Runs all statistical analyses described in the methodology.
Reads event_study_results.csv (output of Stage 6) which contains one
row per event with: CAR windows, volatility metrics, tone_score,
tone_label, and treasury yields.

Tests performed:
    1.  Pearson & Spearman correlation  — tone_score vs each CAR window
    2.  Directional accuracy            — hawkish/dovish/neutral vs market direction
    3.  Granger causality               — tone -> future S&P 500 returns
    4a. Rolling correlation             — time-stability of the tone-return relationship
    4b. Sub-sample analysis             — crisis / normal / high VIX / low VIX
    4c. Event window sensitivity        — correlations at CAR(0,1) (0,5) (0,10) (0,30)
    4d. Partial correlation             — tone vs CAR controlling for VIX and rates
    5.  Cohen's d                       — effect size: hawkish vs dovish distributions
    6.  Point-biserial correlation      — binary tone label vs continuous CAR
    7.  Summary table                   — outputs/statistical_summary.csv

S7 is entirely downstream of S6's computed abnormal returns.
The change from OLS to the constant mean return model in S6 only affects
what numbers are in those columns — S7's analysis logic is unchanged.

Two bugs fixed vs. the previous version:
    - granger_causality: removed a fragile index-based event_date merge
      that silently failed. Event dates are now pulled cleanly from
      events_window.csv (t=0 rows), consistent with how the rest of the
      function was already working.
    - rolling_correlation: the window parameter is in events (rows), not
      calendar days. The docstring and variable name now say "window_events"
      to avoid confusion with the 252-trading-day language in the methodology
      (which referred to a calendar-day rolling window, not an event-count
      window). With ~540 events, a window of 100 events is more practical
      and still captures ~1 year of data; 252 events spans the full dataset
      and leaves only ~290 data points for the chart.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, pointbiserialr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CRISIS PERIOD DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_PERIODS = [
    ("2008-09-01", "2009-06-30"),   # Global Financial Crisis
    ("2020-02-20", "2020-06-30"),   # COVID crash
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load all files needed for statistical evaluation.

    Returns
    -------
    results        : DataFrame — one row per event, from event_study_results.csv
                     Columns: event_id, CAR_0_3, CAR_0_10, CAR_0_30,
                              vix_baseline, vix_event, excess_vix,
                              rv_baseline, rv_event, excess_rv,
                              treasury_10y, treasury_2y, event_date (if present),
                              tone_score, tone_label (if Stage 5 ran)
    market         : DataFrame — full daily market data (for Granger time series)
    events_with_ar : DataFrame — event window rows with abnormal_return and t
                     Used for event window sensitivity (CAR_0_1, CAR_0_5)
    """
    results        = pd.read_csv("data/processed/event_study_results.csv")
    market         = pd.read_csv("data/raw/market_data.csv")
    events_with_ar = pd.read_csv("data/processed/events_with_ar.csv")

    date_col = "Date" if "Date" in market.columns else "date"
    market["date"] = pd.to_datetime(market[date_col])
    events_with_ar["date"] = pd.to_datetime(events_with_ar["date"])

    if "event_date" in results.columns:
        results["event_date"] = pd.to_datetime(results["event_date"])

    print(f"  results:        {len(results)} events, "
          f"tone available: {'tone_score' in results.columns}")
    print(f"  events_with_ar: {len(events_with_ar)} rows")

    return results, market, events_with_ar


# ─────────────────────────────────────────────────────────────────────────────
# 1. PEARSON & SPEARMAN CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(results, car_cols=None):
    """
    Pearson and Spearman correlation between tone_score and each CAR window.
    95% confidence intervals for Pearson r via Fisher z-transformation.

    Pearson tests for a linear relationship.
    Spearman tests for a monotonic (not necessarily linear) relationship and
    is more robust to the outliers present in crisis periods.

    Parameters
    ----------
    results  : DataFrame  — must contain tone_score and CAR columns
    car_cols : list or None — defaults to all available CAR windows

    Returns
    -------
    DataFrame with one row per CAR window and columns:
    car_window, n, pearson_r, pearson_p, pearson_ci_low, pearson_ci_high,
    spearman_r, spearman_p, significant
    """
    if "tone_score" not in results.columns:
        print("  SKIP correlation_analysis — tone_score column missing")
        return pd.DataFrame()

    if car_cols is None:
        car_cols = [c for c in ["CAR_0_3", "CAR_0_10", "CAR_0_30"]
                    if c in results.columns]

    rows = []
    for col in car_cols:
        subset = results[["tone_score", col]].dropna()
        n = len(subset)
        if n < 10:
            continue

        x = subset["tone_score"].values
        y = subset[col].values

        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)

        # 95% CI for Pearson via Fisher z-transformation
        z  = np.arctanh(r_p)
        se = 1.0 / np.sqrt(n - 3)
        ci = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))

        rows.append({
            "car_window":      col,
            "n":               n,
            "pearson_r":       round(r_p,   4),
            "pearson_p":       round(p_p,   4),
            "pearson_ci_low":  round(ci[0], 4),
            "pearson_ci_high": round(ci[1], 4),
            "spearman_r":      round(r_s,   4),
            "spearman_p":      round(p_s,   4),
            "significant":     (p_p < 0.05) or (p_s < 0.05)
        })

    df = pd.DataFrame(rows)
    print("\n── Correlation Analysis ──")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIRECTIONAL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def directional_accuracy(results, car_col="CAR_0_3", neutral_threshold=0.01):
    """
    Measure what percentage of events produced the economically expected
    market movement given the policy tone.

    Classification logic:
        hawkish + negative CAR  -> correct  (rate fears -> lower valuations)
        dovish  + positive CAR  -> correct  (easing signal -> risk appetite)
        neutral + |CAR| < 0.01 -> correct  (no strong signal -> small reaction)

    Baseline accuracy = 50% (random coin flip).
    Significance tested with a one-sided binomial test: H1 > 50%.

    Parameters
    ----------
    results           : DataFrame  — must contain tone_label and car_col
    neutral_threshold : float      — absolute CAR below which is "small" for neutral

    Returns
    -------
    dict with keys: overall_accuracy, breakdown (DataFrame), binomial_p, n
    """
    if "tone_label" not in results.columns or car_col not in results.columns:
        print(f"  SKIP directional_accuracy — missing tone_label or {car_col}")
        return None

    subset = results[["tone_label", car_col]].dropna().copy()

    def is_correct(row):
        if row["tone_label"] == "hawkish":
            return row[car_col] < 0
        elif row["tone_label"] == "dovish":
            return row[car_col] > 0
        else:
            return abs(row[car_col]) < neutral_threshold

    subset["correct"] = subset.apply(is_correct, axis=1)
    accuracy = subset["correct"].mean() * 100

    breakdown = (
        subset.groupby("tone_label")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    breakdown["accuracy"] = (breakdown["accuracy"] * 100).round(2)

    n_total   = len(subset)
    n_correct = int(subset["correct"].sum())
    binom     = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")

    print(f"\n── Directional Accuracy ({car_col}) ──")
    print(f"  Overall: {accuracy:.2f}%  (baseline 50%,  n={n_total})")
    print(breakdown.to_string())
    print(f"  Binomial test p-value: {binom.pvalue:.4f}")

    return {
        "overall_accuracy": accuracy,
        "breakdown":        breakdown,
        "binomial_p":       binom.pvalue,
        "n":                n_total
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRANGER CAUSALITY
# ─────────────────────────────────────────────────────────────────────────────

def granger_causality(market, results, max_lag=5):
    """
    Test whether policy tone Granger-causes future S&P 500 returns.

    Approach:
        1. Get the t=0 date for each event from events_window.csv.
        2. Join tone_score onto the full daily market time series.
        3. Forward-fill tone_score so non-event days inherit the most
           recent tone (tone is assumed to persist between announcements).
        4. Run grangercausalitytests([sp500_log_return, tone_ffill], maxlag).

    H0: past tone_score does NOT improve prediction of today's return
        beyond what past returns alone predict.
    H1: it does (reject H0 if F-test p < 0.05).

    Note: Granger "causality" is a predictive test, not true causation.
    A significant result means tone carries incremental information about
    future returns in a statistical sense.

    Parameters
    ----------
    market  : DataFrame  — full daily market data (must contain date,
                           sp500_log_return)
    results : DataFrame  — event-level results (must contain event_id,
                           tone_score)
    max_lag : int        — number of lags to test (5 trading days = 1 week)

    Returns
    -------
    dict of statsmodels Granger results (keyed by lag) or None on failure
    """
    if "tone_score" not in results.columns:
        print("  SKIP granger_causality — tone_score column missing")
        return None

    # Get event dates from events_window.csv (t=0 rows).
    # This is the clean approach: no fragile index-based merges.
    events_w = pd.read_csv("data/processed/events_window.csv")
    events_w["date"] = pd.to_datetime(events_w["date"])
    t0_rows = (
        events_w[events_w["t"] == 0][["event_id", "date"]]
        .drop_duplicates("event_id")
    )

    # Join tone_score onto event dates
    tone_daily = t0_rows.merge(
        results[["event_id", "tone_score"]],
        on="event_id",
        how="left"
    )

    # Merge onto full daily market series
    market_sorted = market.sort_values("date").copy()
    market_merged = market_sorted.merge(
        tone_daily[["date", "tone_score"]],
        on="date",
        how="left"
    )

    # Forward-fill: non-event days inherit most recent tone
    market_merged["tone_ffill"] = (
        market_merged["tone_score"]
        .ffill()
        .fillna(0)   # days before the first event get 0 (neutral)
    )

    ts = market_merged[["sp500_log_return", "tone_ffill"]].dropna()

    if len(ts) < 100:
        print(f"  SKIP granger_causality — insufficient data ({len(ts)} rows)")
        return None

    print(f"\n── Granger Causality Test (max_lag={max_lag}, n={len(ts)}) ──")
    print("  H0: tone does NOT Granger-cause S&P 500 returns")
    try:
        gc_results = grangercausalitytests(ts, maxlag=max_lag, verbose=True)
    except Exception as e:
        print(f"  Granger test failed: {e}")
        return None

    return gc_results


# ─────────────────────────────────────────────────────────────────────────────
# 4a. ROLLING CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def rolling_correlation(results, car_col="CAR_0_3", window_events=100):
    """
    Recalculate Pearson correlation between tone_score and CAR using a
    rolling window of `window_events` consecutive events.

    This tests whether the tone-return relationship is stable over time
    or varies across monetary policy regimes (QE era, tightening cycle, etc.).

    Note on window size:
        The methodology mentions a "252-day rolling window" — that is a
        calendar-day concept from a continuous return series. Here we have
        a sparse event series (~540 events over 17 years, ~32/year).
        A window of 100 events spans approximately 3 years, which is a
        reasonable regime-length for the Fed. Adjust window_events as needed.

    Parameters
    ----------
    results       : DataFrame  — must contain event_date (if available),
                                 tone_score, and car_col
    window_events : int        — number of consecutive events per window

    Returns
    -------
    DataFrame with columns: event_idx, rolling_r
    Saves plot to outputs/rolling_correlation_{car_col}.png
    """
    if "tone_score" not in results.columns:
        print(f"  SKIP rolling_correlation — tone_score missing")
        return pd.DataFrame()

    if "event_date" in results.columns:
        df = results[["event_date", "tone_score", car_col]].dropna().copy()
        df = df.sort_values("event_date").reset_index(drop=True)
    else:
        df = results[["tone_score", car_col]].dropna().copy()

    if len(df) < window_events + 10:
        print(f"  SKIP rolling_correlation — not enough events "
              f"({len(df)} < {window_events})")
        return pd.DataFrame()

    roll_rows = []
    for i in range(window_events, len(df) + 1):
        chunk = df.iloc[i - window_events: i]
        r, _ = pearsonr(chunk["tone_score"], chunk[car_col])
        roll_rows.append({"event_idx": i, "rolling_r": r})

    roll_df = pd.DataFrame(roll_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll_df["event_idx"], roll_df["rolling_r"], color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Rolling {window_events}-Event Pearson Correlation: "
                 f"tone_score vs {car_col}")
    ax.set_xlabel("Event Index (end of window)")
    ax.set_ylabel("Pearson r")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"outputs/rolling_correlation_{car_col}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Rolling correlation plot -> {path}")

    return roll_df


# ─────────────────────────────────────────────────────────────────────────────
# 4b. SUB-SAMPLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def subgroup_correlation(results, car_col="CAR_0_3"):
    """
    Run separate Pearson correlations for economically distinct subgroups:

        crisis  — GFC (Sep 2008–Jun 2009) + COVID crash (Feb–Jun 2020)
        normal  — all other periods
        high_vix — events in the top VIX quartile (>75th percentile)
        low_vix  — events in the bottom VIX quartile (<25th percentile)

    If the relationship is stronger in crisis periods, it suggests tone
    matters most when uncertainty is already elevated. If stronger in
    normal periods, tone may only be informative in calmer markets.

    Requires event_date column to classify crisis vs normal.
    Requires vix_event column for VIX quartile splits.

    Parameters
    ----------
    results  : DataFrame
    car_col  : str

    Returns
    -------
    DataFrame with columns: subgroup, n, pearson_r, p_value
    """
    if "tone_score" not in results.columns:
        print(f"  SKIP subgroup_correlation — tone_score missing")
        return pd.DataFrame()

    df = results.copy()

    # Crisis flag
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
        crisis_mask = pd.Series(False, index=df.index)
        for start, end in CRISIS_PERIODS:
            crisis_mask |= (
                (df["event_date"] >= start) & (df["event_date"] <= end)
            )
        df["is_crisis"] = crisis_mask
    else:
        df["is_crisis"] = False
        print("  NOTE: event_date not found — all events classified as 'normal'")

    # VIX quartile groups
    if "vix_event" in df.columns:
        p25 = df["vix_event"].quantile(0.25)
        p75 = df["vix_event"].quantile(0.75)
        df["vix_group"] = "medium"
        df.loc[df["vix_event"] < p25, "vix_group"] = "low"
        df.loc[df["vix_event"] > p75, "vix_group"] = "high"

    subgroups = {
        "all_events": df,
        "crisis":     df[df["is_crisis"]],
        "normal":     df[~df["is_crisis"]],
    }
    if "vix_group" in df.columns:
        subgroups["high_vix"] = df[df["vix_group"] == "high"]
        subgroups["low_vix"]  = df[df["vix_group"] == "low"]

    rows = []
    for name, subset in subgroups.items():
        sub = subset[["tone_score", car_col]].dropna()
        n   = len(sub)
        if n < 10:
            rows.append({"subgroup": name, "n": n,
                         "pearson_r": np.nan, "p_value": np.nan})
            continue
        r, p = pearsonr(sub["tone_score"], sub[car_col])
        rows.append({"subgroup": name, "n": n,
                     "pearson_r": round(r, 4), "p_value": round(p, 4)})

    result_df = pd.DataFrame(rows)
    print(f"\n── Sub-sample Correlation ({car_col}) ──")
    print(result_df.to_string(index=False))
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# 4c. EVENT WINDOW SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def event_window_sensitivity(events_with_ar, results):
    """
    Test whether the tone-return correlation is stronger at shorter or
    longer horizons by computing correlations for four CAR windows:
        CAR(0,1), CAR(0,5), CAR(0,10), CAR(0,30)

    CAR(0,1) and CAR(0,5) are not in event_study_results.csv so they are
    computed here from events_with_ar.csv.

    If the correlation peaks at CAR(0,1) and decays, it means markets
    respond immediately and the signal dissipates. If it peaks at
    CAR(0,10) or CAR(0,30), there is a delayed or sustained market response.

    Parameters
    ----------
    events_with_ar : DataFrame  — must contain event_id, t, abnormal_return
    results        : DataFrame  — must contain event_id, tone_score,
                                  CAR_0_10, CAR_0_30

    Returns
    -------
    DataFrame with columns: window, n, pearson_r, p_value
    """
    if "tone_score" not in results.columns:
        print("  SKIP event_window_sensitivity — tone_score missing")
        return pd.DataFrame()

    # Compute the two short windows not in results
    extra = {}
    for max_t, col in [(1, "CAR_0_1"), (5, "CAR_0_5")]:
        extra[col] = (
            events_with_ar[events_with_ar["t"] <= max_t]
            .groupby("event_id")["abnormal_return"]
            .sum()
            .reset_index(name=col)
        )

    df = results.copy()
    for col, car_df in extra.items():
        df = df.merge(car_df, on="event_id", how="left")

    all_windows = ["CAR_0_1", "CAR_0_5", "CAR_0_10", "CAR_0_30"]
    rows = []
    for col in all_windows:
        if col not in df.columns:
            continue
        sub = df[["tone_score", col]].dropna()
        if len(sub) < 10:
            continue
        r, p = pearsonr(sub["tone_score"], sub[col])
        rows.append({"window": col, "n": len(sub),
                     "pearson_r": round(r, 4), "p_value": round(p, 4)})

    sens_df = pd.DataFrame(rows)
    print("\n── Event Window Sensitivity ──")
    print(sens_df.to_string(index=False))
    return sens_df


# ─────────────────────────────────────────────────────────────────────────────
# 4d. PARTIAL CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def partial_correlation(results, car_col="CAR_0_3"):
    """
    Partial correlation between tone_score and CAR, controlling for
    confounding variables using the residuals-on-residuals method:

        1. Regress tone_score on controls -> get residuals r_tone
        2. Regress car_col    on controls -> get residuals r_car
        3. Pearson(r_tone, r_car) is the partial correlation

    Controls: excess_vix (surprise volatility) and treasury_10y
    (rate environment on announcement day).

    A significant partial r means tone predicts returns even after
    accounting for the level of market fear and interest rates.

    Parameters
    ----------
    results : DataFrame
    car_col : str

    Returns
    -------
    dict with keys: partial_r, partial_p, controls
    """
    if "tone_score" not in results.columns:
        print("  SKIP partial_correlation — tone_score missing")
        return None

    controls = [c for c in ["excess_vix", "treasury_10y"]
                if c in results.columns]
    if not controls:
        print("  SKIP partial_correlation — no control columns available")
        return None

    cols   = ["tone_score", car_col] + controls
    df     = results[cols].dropna()
    if len(df) < 20:
        print(f"  SKIP partial_correlation — only {len(df)} complete rows")
        return None

    X_ctrl = sm.add_constant(df[controls])

    def residuals(col):
        return sm.OLS(df[col], X_ctrl).fit().resid

    r_partial, p_partial = pearsonr(residuals("tone_score"),
                                    residuals(car_col))

    print(f"\n── Partial Correlation: tone vs {car_col} "
          f"(controlling {controls}) ──")
    print(f"  Partial r = {r_partial:.4f},  p = {p_partial:.4f}")

    return {"partial_r": r_partial,
            "partial_p": p_partial,
            "controls":  controls}


# ─────────────────────────────────────────────────────────────────────────────
# 5. COHEN'S D — EFFECT SIZE
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(results, car_col="CAR_0_3"):
    """
    Cohen's d for the difference in CAR distributions between hawkish
    and dovish events.

        d = (mean_hawkish - mean_dovish) / pooled_std

    Interpretation:
        |d| < 0.2  — negligible
        |d| ~ 0.5  — moderate
        |d| > 0.8  — large

    A large negative d (hawkish events have much lower CARs than dovish)
    would confirm that tone has economically meaningful impact, not just
    statistical significance.

    Significance tested with Welch's t-test (unequal variances allowed).

    Parameters
    ----------
    results : DataFrame  — must contain tone_label and car_col
    car_col : str

    Returns
    -------
    dict with keys: d, t_stat, p_value, hawkish_mean, dovish_mean,
                    hawkish_n, dovish_n
    """
    if "tone_label" not in results.columns:
        print("  SKIP cohens_d — tone_label missing")
        return None

    hawkish = results[results["tone_label"] == "hawkish"][car_col].dropna()
    dovish  = results[results["tone_label"] == "dovish"][car_col].dropna()

    if len(hawkish) < 5 or len(dovish) < 5:
        print(f"  SKIP cohens_d — too few events "
              f"(hawkish={len(hawkish)}, dovish={len(dovish)})")
        return None

    pooled_std = np.sqrt(
        ((len(hawkish) - 1) * hawkish.std()**2 +
         (len(dovish)  - 1) * dovish.std()**2)
        / (len(hawkish) + len(dovish) - 2)
    )
    d = (hawkish.mean() - dovish.mean()) / pooled_std

    # Welch t-test (does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(hawkish, dovish, equal_var=False)

    print(f"\n── Effect Size (Cohen's d): hawkish vs dovish {car_col} ──")
    print(f"  Hawkish  mean = {hawkish.mean():.4f}  (n={len(hawkish)})")
    print(f"  Dovish   mean = {dovish.mean():.4f}  (n={len(dovish)})")
    print(f"  Cohen's d     = {d:.4f}")
    print(f"  Welch t = {t_stat:.3f},  p = {p_val:.4f}")

    return {
        "d":            d,
        "t_stat":       t_stat,
        "p_value":      p_val,
        "hawkish_mean": hawkish.mean(),
        "dovish_mean":  dovish.mean(),
        "hawkish_n":    len(hawkish),
        "dovish_n":     len(dovish)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. POINT-BISERIAL CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def point_biserial(results, car_col="CAR_0_3"):
    """
    Point-biserial correlation between a binary tone indicator and CAR.

        binary = 1 if hawkish, 0 if dovish
        (neutral events excluded)

    Mathematically equivalent to Pearson r with a 0/1 predictor.
    Complements the Pearson correlation on the continuous tone_score
    by asking: does the categorical label alone (ignoring signal strength)
    predict the direction of the market reaction?

    Parameters
    ----------
    results : DataFrame  — must contain tone_label and car_col
    car_col : str

    Returns
    -------
    dict with keys: r_pb, p_value, n
    """
    if "tone_label" not in results.columns:
        print("  SKIP point_biserial — tone_label missing")
        return None

    sub = (
        results[results["tone_label"].isin(["hawkish", "dovish"])]
        [["tone_label", car_col]]
        .dropna()
    )

    if len(sub) < 10:
        print(f"  SKIP point_biserial — only {len(sub)} hawkish/dovish events")
        return None

    binary = (sub["tone_label"] == "hawkish").astype(int)
    r, p   = pointbiserialr(binary, sub[car_col])

    print(f"\n── Point-Biserial Correlation ({car_col}) ──")
    print(f"  r_pb = {r:.4f},  p = {p:.4f},  n = {len(sub)}")

    return {"r_pb": r, "p_value": p, "n": len(sub)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(corr_df, dir_acc, effect, pb):
    """
    Assemble all primary metrics into a single summary CSV.
    Saved to outputs/statistical_summary.csv.

    Parameters
    ----------
    corr_df  : DataFrame or None — output of correlation_analysis()
    dir_acc  : dict or None      — output of directional_accuracy()
    effect   : dict or None      — output of cohens_d()
    pb       : dict or None      — output of point_biserial()

    Returns
    -------
    DataFrame with columns: metric, value, p_value, n, significant
    """
    rows = []

    if corr_df is not None and len(corr_df) > 0:
        for _, row in corr_df.iterrows():
            rows.append({
                "metric":      f"Pearson r ({row['car_window']})",
                "value":       row["pearson_r"],
                "ci":          f"[{row['pearson_ci_low']}, {row['pearson_ci_high']}]",
                "p_value":     row["pearson_p"],
                "n":           row["n"],
                "significant": row["significant"]
            })
            rows.append({
                "metric":      f"Spearman rho ({row['car_window']})",
                "value":       row["spearman_r"],
                "ci":          "",
                "p_value":     row["spearman_p"],
                "n":           row["n"],
                "significant": row["spearman_p"] < 0.05
            })

    if dir_acc:
        rows.append({
            "metric":      "Directional Accuracy (CAR_0_3)",
            "value":       round(dir_acc["overall_accuracy"], 2),
            "ci":          "",
            "p_value":     round(dir_acc["binomial_p"], 4),
            "n":           dir_acc["n"],
            "significant": dir_acc["binomial_p"] < 0.05
        })

    if effect:
        rows.append({
            "metric":      "Cohen's d: hawkish vs dovish (CAR_0_3)",
            "value":       round(effect["d"], 4),
            "ci":          "",
            "p_value":     round(effect["p_value"], 4),
            "n":           effect["hawkish_n"] + effect["dovish_n"],
            "significant": effect["p_value"] < 0.05
        })

    if pb:
        rows.append({
            "metric":      "Point-biserial r (CAR_0_3)",
            "value":       round(pb["r_pb"], 4),
            "ci":          "",
            "p_value":     round(pb["p_value"], 4),
            "n":           pb["n"],
            "significant": pb["p_value"] < 0.05
        })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/statistical_summary.csv", index=False)

    print("\n── Statistical Summary ──")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_evaluation():

    print("\n" + "=" * 60)
    print("STAGE 7 — Statistical Evaluation")
    print("=" * 60)

    print("\nLoading data...")
    results, market, events_with_ar = load_data()

    # 1. Pearson + Spearman
    corr_df = correlation_analysis(results)

    # 2. Directional accuracy
    dir_acc = directional_accuracy(results)

    # 3. Granger causality
    granger_causality(market, results, max_lag=5)

    # 4a. Rolling correlation
    rolling_correlation(results, window_events=100)

    # 4b. Sub-sample analysis
    subgroup_correlation(results)

    # 4c. Event window sensitivity
    event_window_sensitivity(events_with_ar, results)

    # 4d. Partial correlation
    partial_correlation(results)

    # 5. Cohen's d
    effect = cohens_d(results)

    # 6. Point-biserial
    pb = point_biserial(results)

    # 7. Summary table
    build_summary(corr_df, dir_acc, effect, pb)

    print("\n" + "=" * 60)
    print("Stage 7 complete. All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    run_statistical_evaluation()