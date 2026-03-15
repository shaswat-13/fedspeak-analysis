"""
Stage 7: Statistical Evaluation

Implements all statistical analyses described in the methodology:
  1. Pearson & Spearman correlation (tone_score vs CAR windows)
  2. Directional accuracy (hawkish→negative CAR, dovish→positive CAR)
  3. Granger causality test (tone → future returns)
  4. Robustness checks:
       a. Rolling-window correlations (252-day)
       b. Sub-sample analysis (crisis / normal / high VIX / low VIX)
       c. Event window sensitivity across CAR(0,1), CAR(0,5), CAR(0,10), CAR(0,30)
       d. Partial correlation controlling for confounders
  5. Effect size (Cohen's d): hawkish vs dovish return distributions
  6. Point-biserial correlation (categorical tone label vs CAR)
  7. Summary table saved to outputs/
"""

# import modules
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, pointbiserialr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)



# LOAD DATA
def load_data():
    """Load event study results (with tone scores) and raw market data."""
    results  = pd.read_csv("data/processed/event_study_results.csv")
    market   = pd.read_csv("data/raw/market_data.csv")
    events_w = pd.read_csv("data/processed/events_window.csv")

    market["date"] = pd.to_datetime(market["Date"] if "Date" in market.columns else market["date"])
    events_w["date"] = pd.to_datetime(events_w["date"])

    # Also load events_with_ar so we can do a CAR(0,1) and CAR(0,5) sensitivity check
    events_with_ar = pd.read_csv("data/processed/events_with_ar.csv")

    return results, market, events_w, events_with_ar



# 1. PEARSON & SPEARMAN CORRELATION
def correlation_analysis(results, car_cols=None):
    """
    Pearson and Spearman correlation between tone_score and each CAR window.
    Also computes 95% CI via Fisher z-transformation for Pearson r.
    """
    if "tone_score" not in results.columns:
        print("SKIP correlation_analysis — tone_score column missing")
        return pd.DataFrame()

    if car_cols is None:
        car_cols = [c for c in ["CAR_0_3", "CAR_0_10", "CAR_0_30"] if c in results.columns]

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

        # 95% CI for Pearson via Fisher z
        z   = np.arctanh(r_p)
        se  = 1 / np.sqrt(n - 3)
        ci  = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))

        rows.append({
            "car_window":      col,
            "n":               n,
            "pearson_r":       round(r_p, 4),
            "pearson_p":       round(p_p, 4),
            "pearson_ci_low":  round(ci[0], 4),
            "pearson_ci_high": round(ci[1], 4),
            "spearman_r":      round(r_s, 4),
            "spearman_p":      round(p_s, 4),
            "significant":     p_p < 0.05 or p_s < 0.05
        })

    df = pd.DataFrame(rows)
    print("\n── Correlation Analysis ──")
    print(df.to_string(index=False))
    return df



# 2. DIRECTIONAL ACCURACY
def directional_accuracy(results, car_col="CAR_0_3", neutral_threshold=0.01):
    """
    Measures % of events where market moved in economically expected direction:
      - Hawkish  → negative CAR  ✓
      - Dovish   → positive CAR  ✓
      - Neutral  → |CAR| < threshold ✓

    Baseline = 50% (random chance).
    """
    if "tone_label" not in results.columns or car_col not in results.columns:
        print(f"SKIP directional_accuracy — missing tone_label or {car_col}")
        return None

    subset = results[["tone_label", car_col]].dropna()

    def is_correct(row):
        label = row["tone_label"]
        car   = row[car_col]
        if label == "hawkish":
            return car < 0
        elif label == "dovish":
            return car > 0
        else:  # neutral
            return abs(car) < neutral_threshold

    subset = subset.copy()
    subset["correct"] = subset.apply(is_correct, axis=1)
    accuracy = subset["correct"].mean() * 100

    # Per-label breakdown
    breakdown = (
        subset.groupby("tone_label")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    breakdown["accuracy"] = (breakdown["accuracy"] * 100).round(2)

    print(f"\n── Directional Accuracy ({car_col}) ──")
    print(f"  Overall: {accuracy:.2f}%  (baseline 50%)")
    print(breakdown.to_string())

    # Binomial test: is accuracy significantly > 50%?
    n_total   = len(subset)
    n_correct = subset["correct"].sum()
    binom     = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
    print(f"  Binomial test p-value: {binom.pvalue:.4f}")

    return {
        "overall_accuracy": accuracy,
        "breakdown": breakdown,
        "binomial_p": binom.pvalue,
        "n": n_total
    }



# 3. GRANGER CAUSALITY
def granger_causality(market, results, max_lag=5):
    """
    Tests whether policy tone Granger-causes future S&P 500 returns.

    We build a daily time series by forward-filling tone_score on
    non-event days (tone is assumed to persist between events).

    H0: tone does NOT Granger-cause returns (F-test, p < 0.05 = reject H0)
    """
    if "tone_score" not in results.columns:
        print("SKIP granger — tone_score missing")
        return None

    # We need event_date in results
    if "event_date" not in results.columns:
        events_all = pd.read_csv("data/processed/events_all.csv")
        results = results.merge(
            events_all[["event_id" if "event_id" in events_all.columns else events_all.index.name]],
            left_on="event_id", right_index=True, how="left"
        ) if "event_id" in events_all.columns else results

    # Build aligned daily tone series
    market_sorted = market.sort_values("date").copy()
    market_sorted["date"] = pd.to_datetime(market_sorted["date"])

    # Get event dates from events_window
    events_w = pd.read_csv("data/processed/events_window.csv")
    events_w["date"] = pd.to_datetime(events_w["date"])
    t0_rows = events_w[events_w["t"] == 0][["event_id", "date"]].drop_duplicates("event_id")

    tone_daily = t0_rows.merge(
        results[["event_id", "tone_score"]], on="event_id", how="left"
    )

    market_merged = market_sorted.merge(
        tone_daily[["date", "tone_score"]], on="date", how="left"
    )
    market_merged["tone_ffill"] = market_merged["tone_score"].ffill().fillna(0)

    ts = market_merged[["sp500_log_return", "tone_ffill"]].dropna()

    if len(ts) < 100:
        print(f"SKIP granger — insufficient data ({len(ts)} rows)")
        return None

    print(f"\n── Granger Causality Test (max_lag={max_lag}) ──")
    print("H0: tone does NOT Granger-cause returns")
    try:
        gc_results = grangercausalitytests(ts, maxlag=max_lag, verbose=True)
    except Exception as e:
        print(f"Granger test failed: {e}")
        return None

    return gc_results



# 4a. ROLLING-WINDOW CORRELATION
def rolling_correlation(results, car_col="CAR_0_3", window=252):
    """
    Recalculates Pearson correlation in a rolling 252-day (1-year) window
    to assess time stability of the tone–return relationship.
    Results are plotted and saved.
    """
    if "tone_score" not in results.columns:
        return

    # Use event_date as time axis if available; otherwise use row index
    if "event_date" in results.columns:
        df = results[["event_date", "tone_score", car_col]].dropna().copy()
        df["event_date"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("event_date").reset_index(drop=True)
    else:
        df = results[["tone_score", car_col]].dropna().copy()

    roll_r = []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window: i]
        r, _ = pearsonr(chunk["tone_score"], chunk[car_col])
        roll_r.append({"idx": i, "rolling_r": r})

    roll_df = pd.DataFrame(roll_r)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll_df["idx"], roll_df["rolling_r"], color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Rolling {window}-event Pearson Correlation: tone vs {car_col}")
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Pearson r")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"outputs/rolling_correlation_{car_col}.png", dpi=300)
    plt.close(fig)
    print(f"Rolling correlation plot saved → outputs/rolling_correlation_{car_col}.png")
    return roll_df



# 4b. SUB-SAMPLE ANALYSIS
CRISIS_PERIODS = [
    ("2008-09-01", "2009-06-30"),   # Global Financial Crisis
    ("2020-02-20", "2020-06-30"),   # COVID crash
]

def subgroup_correlation(results, car_col="CAR_0_3"):
    """
    Separate Pearson correlation for:
      - Crisis periods (GFC + COVID)
      - Normal periods
      - High VIX (>75th percentile)
      - Low VIX  (<25th percentile)
    """
    if "tone_score" not in results.columns:
        return pd.DataFrame()

    df = results.copy()

    # Identify crisis rows
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
        crisis_mask = pd.Series(False, index=df.index)
        for start, end in CRISIS_PERIODS:
            crisis_mask |= (df["event_date"] >= start) & (df["event_date"] <= end)
        df["is_crisis"] = crisis_mask
    else:
        df["is_crisis"] = False

    # VIX percentile groups
    if "vix_event" in df.columns:
        p25 = df["vix_event"].quantile(0.25)
        p75 = df["vix_event"].quantile(0.75)
        df["vix_group"] = "medium"
        df.loc[df["vix_event"] < p25, "vix_group"] = "low"
        df.loc[df["vix_event"] > p75, "vix_group"] = "high"

    rows = []
    subgroups = {
        "crisis":    df[df.get("is_crisis", pd.Series(False, index=df.index))],
        "normal":    df[~df.get("is_crisis", pd.Series(False, index=df.index))],
    }
    if "vix_group" in df.columns:
        subgroups["high_vix"] = df[df["vix_group"] == "high"]
        subgroups["low_vix"]  = df[df["vix_group"] == "low"]

    for name, subset in subgroups.items():
        sub = subset[["tone_score", car_col]].dropna()
        if len(sub) < 10:
            rows.append({"subgroup": name, "n": len(sub), "pearson_r": np.nan, "p_value": np.nan})
            continue
        r, p = pearsonr(sub["tone_score"], sub[car_col])
        rows.append({"subgroup": name, "n": len(sub),
                     "pearson_r": round(r, 4), "p_value": round(p, 4)})

    result_df = pd.DataFrame(rows)
    print(f"\n── Sub-sample Correlation ({car_col}) ──")
    print(result_df.to_string(index=False))
    return result_df



# 4c. EVENT WINDOW SENSITIVITY
def event_window_sensitivity(events_with_ar, results):
    """
    Compute CAR(0,1) and CAR(0,5) which weren't in the original pipeline,
    then run correlation for all four windows to find the optimal horizon.
    """
    if "tone_score" not in results.columns:
        return pd.DataFrame()

    windows = [(1, "CAR_0_1"), (5, "CAR_0_5"), (10, "CAR_0_10"), (30, "CAR_0_30")]
    extra_cars = {}

    for max_t, col in [(1, "CAR_0_1"), (5, "CAR_0_5")]:
        car = (
            events_with_ar[events_with_ar["t"] <= max_t]
            .groupby("event_id")["abnormal_return"]
            .sum()
            .reset_index(name=col)
        )
        extra_cars[col] = car

    # Merge extra windows into results temporarily
    df = results.copy()
    for col, car_df in extra_cars.items():
        df = df.merge(car_df, on="event_id", how="left")

    all_car_cols = [col for _, col in windows if col in df.columns]
    sens_rows = []
    for col in all_car_cols:
        sub = df[["tone_score", col]].dropna()
        if len(sub) < 10:
            continue
        r, p = pearsonr(sub["tone_score"], sub[col])
        sens_rows.append({"window": col, "n": len(sub),
                          "pearson_r": round(r, 4), "p_value": round(p, 4)})

    sens_df = pd.DataFrame(sens_rows)
    print("\n── Event Window Sensitivity ──")
    print(sens_df.to_string(index=False))
    return sens_df



# 4d. PARTIAL CORRELATION (CONFOUNDER CONTROL)
def partial_correlation(results, car_col="CAR_0_3"):
    """
    Partial correlation between tone_score and CAR, controlling for:
      - excess_vix (surprise volatility)
      - treasury_10y change (rate environment)
    Uses residuals-on-residuals method.
    """
    if "tone_score" not in results.columns:
        return

    controls = [c for c in ["excess_vix", "treasury_10y"] if c in results.columns]
    if not controls:
        print("SKIP partial_correlation — no control columns available")
        return

    cols = ["tone_score", car_col] + controls
    df   = results[cols].dropna()

    if len(df) < 20:
        print("SKIP partial_correlation — not enough data after dropna")
        return

    X_ctrl = sm.add_constant(df[controls])

    def residuals(y_col):
        return sm.OLS(df[y_col], X_ctrl).fit().resid

    r_tone = residuals("tone_score")
    r_car  = residuals(car_col)

    r_partial, p_partial = pearsonr(r_tone, r_car)
    print(f"\n── Partial Correlation: tone vs {car_col} (controlling {controls}) ──")
    print(f"  Partial r = {r_partial:.4f},  p = {p_partial:.4f}")
    return {"partial_r": r_partial, "partial_p": p_partial, "controls": controls}



# 5. EFFECT SIZE (Cohen's d)
def cohens_d(results, car_col="CAR_0_3"):
    """
    Cohen's d for difference in CAR distributions: hawkish vs. dovish events.
    d = (mean_hawkish - mean_dovish) / pooled_std
    """
    if "tone_label" not in results.columns:
        return None

    hawkish = results[results["tone_label"] == "hawkish"][car_col].dropna()
    dovish  = results[results["tone_label"] == "dovish"][car_col].dropna()

    if len(hawkish) < 5 or len(dovish) < 5:
        print("SKIP Cohen's d — insufficient hawkish or dovish events")
        return None

    pooled_std = np.sqrt(
        ((len(hawkish) - 1) * hawkish.std()**2 + (len(dovish) - 1) * dovish.std()**2)
        / (len(hawkish) + len(dovish) - 2)
    )
    d = (hawkish.mean() - dovish.mean()) / pooled_std

    # Two-sample t-test for significance
    t_stat, p_val = stats.ttest_ind(hawkish, dovish)

    print(f"\n── Effect Size (Cohen's d): hawkish vs dovish {car_col} ──")
    print(f"  Hawkish mean: {hawkish.mean():.4f}  (n={len(hawkish)})")
    print(f"  Dovish  mean: {dovish.mean():.4f}  (n={len(dovish)})")
    print(f"  Cohen's d:    {d:.4f}")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}")

    return {"d": d, "t_stat": t_stat, "p_value": p_val,
            "hawkish_mean": hawkish.mean(), "dovish_mean": dovish.mean()}



# 6. POINT-BISERIAL CORRELATION
def point_biserial(results, car_col="CAR_0_3"):
    """
    Point-biserial correlation between binary tone (hawkish=1, dovish=0)
    and continuous CAR. (Neutral events excluded.)
    """
    if "tone_label" not in results.columns:
        return None

    sub = results[results["tone_label"].isin(["hawkish", "dovish"])][
        ["tone_label", car_col]
    ].dropna()

    if len(sub) < 10:
        return None

    binary = (sub["tone_label"] == "hawkish").astype(int)
    r, p   = pointbiserialr(binary, sub[car_col])

    print(f"\n── Point-Biserial Correlation ({car_col}) ──")
    print(f"  r = {r:.4f},  p = {p:.4f}")
    return {"r_pb": r, "p_value": p, "n": len(sub)}



# 7. SUMMARY TABLE
def build_summary(corr_df, dir_acc, effect, pb):
    """Assemble a clean summary DataFrame and save to CSV."""
    rows = []

    if corr_df is not None and len(corr_df) > 0:
        for _, row in corr_df.iterrows():
            rows.append({
                "metric":     f"Pearson r ({row['car_window']})",
                "value":      row["pearson_r"],
                "p_value":    row["pearson_p"],
                "n":          row["n"],
                "significant": row["significant"]
            })
            rows.append({
                "metric":     f"Spearman ρ ({row['car_window']})",
                "value":      row["spearman_r"],
                "p_value":    row["spearman_p"],
                "n":          row["n"],
                "significant": row["spearman_p"] < 0.05
            })

    if dir_acc:
        rows.append({
            "metric":     "Directional Accuracy (CAR_0_3)",
            "value":      round(dir_acc["overall_accuracy"], 2),
            "p_value":    round(dir_acc["binomial_p"], 4),
            "n":          dir_acc["n"],
            "significant": dir_acc["binomial_p"] < 0.05
        })

    if effect:
        rows.append({
            "metric":     "Cohen's d (hawkish vs dovish, CAR_0_3)",
            "value":      round(effect["d"], 4),
            "p_value":    round(effect["p_value"], 4),
            "n":          None,
            "significant": effect["p_value"] < 0.05
        })

    if pb:
        rows.append({
            "metric":     "Point-biserial r (CAR_0_3)",
            "value":      round(pb["r_pb"], 4),
            "p_value":    round(pb["p_value"], 4),
            "n":          pb["n"],
            "significant": pb["p_value"] < 0.05
        })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/statistical_summary.csv", index=False)
    print("\n── Statistical Summary ──")
    print(df.to_string(index=False))
    return df



# MAIN PIPELINE
def run_statistical_evaluation():
    print("Loading data...")
    results, market, events_w, events_with_ar = load_data()

    # 1. Correlation
    corr_df = correlation_analysis(results)

    # 2. Directional accuracy
    dir_acc = directional_accuracy(results)

    # 3. Granger causality
    granger_causality(market, results, max_lag=5)

    # 4a. Rolling correlation
    rolling_correlation(results)

    # 4b. Sub-sample
    subgroup_correlation(results)

    # 4c. Window sensitivity
    event_window_sensitivity(events_with_ar, results)

    # 4d. Partial correlation
    partial_correlation(results)

    # 5. Cohen's d
    effect = cohens_d(results)

    # 6. Point-biserial
    pb = point_biserial(results)

    # 7. Summary table
    build_summary(corr_df, dir_acc, effect, pb)

    print("\nStatistical evaluation complete. Outputs saved to outputs/")


if __name__ == "__main__":
    run_statistical_evaluation()