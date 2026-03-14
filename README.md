# Quantifying FedSpeak: Sentiment Analysis of Federal Reserve Communication and Market Reactions (2008–2025)

> *How does the Fed's choice of words move financial markets? This project builds a full NLP + econometrics pipeline to find out.*

---

## 🧭 Overview

**FedSpeak** refers to the deliberately calibrated — and often ambiguous — language used by Federal Reserve officials to manage investor expectations without triggering market volatility. FOMC statements, meeting minutes, and Chair speeches are not merely informational: they are instruments of monetary policy transmission.

This project systematically converts Fed communication into a numerical **policy-tone signal** (Hawkish / Dovish / Neutral), then measures how strongly that signal correlates with financial market reactions across 17 years of economic history — from the 2008 financial crisis through the post-COVID inflation cycle.

**Core question:** *Does the tone of what the Fed says predict how markets move?*

---

## 🗂️ Pipeline Overview

The project is organized into **six sequential stages**:

```
Stage 1: Data Collection (Web Scraping)
    ↓
Stage 2: Market Data Acquisition & Alignment
    ↓
Stage 3: Text Preprocessing & Sentiment Analysis (FinBERT)
    ↓
Stage 4: Policy Tone Classification (Hybrid Lexicon-Sentiment)
    ↓
Stage 5: Event Study (Abnormal Return Calculation)
    ↓
Stage 6: Statistical Evaluation & Robustness Testing
```

---

## ⚙️ Stage-by-Stage Walkthrough

---

### Stage 1 — Data Collection (Web Scraping)

**What:** Automated collection of Federal Reserve textual communications from the [Federal Reserve Board's official website](https://www.federalreserve.gov).

**Documents collected:**

| Document Type | Count | Frequency |
|---|---|---|
| FOMC Statements | ~136 | 8 per year |
| FOMC Meeting Minutes | ~136 | Released ~3 weeks post-meeting |
| Fed Chair Speeches | ~283 | Variable |
| **Total** | **~555** | **2008–2025** |

**Why these documents?** These three categories represent the primary channels through which the Fed communicates monetary policy stance to the public. Statements convey immediate policy decisions; minutes reveal deliberation depth; speeches allow the Chair to signal future direction informally.

**Tools:** `BeautifulSoup`, `Requests`

**Why scrape rather than use a pre-built dataset?** No publicly maintained, up-to-date dataset covering 2008–2025 across all three document types exists. Scraping from the primary source ensures accuracy and completeness.

---

### Stage 2 — Market Data Acquisition & Alignment

**What:** Download daily financial market data and synchronize it with each Fed communication event.

**Data source:** Yahoo Finance API (`yfinance`)

**Variables collected:**

| Category | Variables | Rationale |
|---|---|---|
| S&P 500 | Open, High, Low, Close, Volume, Log Returns | Primary equity market proxy |
| Volatility | VIX (implied), Realized Volatility (20-day rolling σ) | Measures uncertainty response |
| Interest Rates | 10Y Treasury, 2Y Treasury | Direct transmission channels |

**Why log returns?** Log returns (`r_t = ln(P_t / P_{t-1})`) are used instead of simple returns because they are time-additive, approximately normally distributed, and standard in financial econometrics.

**Why VIX alongside realized volatility?** VIX captures *forward-looking* market fear (implied by options prices), while realized volatility captures *historical* price fluctuation. Using both gives a fuller picture of how Fed communication affects uncertainty.

**Why the 2Y/10Y yield spread?** The spread between short- and long-term Treasury yields is a closely watched recession signal and a direct indicator of how the market interprets future Fed rate paths.

**Alignment logic:**
- FOMC announcements released **before market close (14:00 UTC)**: market reaction measured same day (t=0)
- Announcements released **after market close**: reaction measured from next trading day (t=1)

**Total:** 4,276 trading days across the study period, zero missing values after forward-fill imputation of Treasury yield data.

---

### Stage 3 — Text Preprocessing & Sentiment Analysis

#### 3a. Text Preprocessing Pipeline

Raw Fed documents undergo a five-stage cleaning and normalization process before NLP analysis:

| Stage | Action | Rationale |
|---|---|---|
| Cleaning | Remove headers, HTML tags, non-ASCII characters | Eliminate noise not part of policy language |
| Sentence Segmentation | `spaCy` (`en_core_web_sm`) | Enables sentence-level FinBERT scoring |
| Tokenization | `spaCy` tokenizer; preserve numbers & percentages | Figures like "4.5%" carry policy meaning |
| Lemmatization | Reduce inflected forms (`tightening` → `tighten`) | Reduces sparsity; improves keyword matching |
| Stopword Removal | Remove NLTK stopwords; **retain** domain terms | Terms like `inflation`, `rate`, `accommodative` are signals, not noise |

**Why retain financial terms from stopword removal?** Standard NLP stopword lists remove very common words. In Fed documents, frequent terms like *inflation*, *rate*, and *monetary* are exactly what carries policy signal — blindly removing them would destroy information.

**Preprocessing statistics:**
- Average document length: 1,247 tokens (SD: 634)
- Vocabulary post-preprocessing: 4,892 unique lemmatized terms

#### 3b. Sentiment Scoring — FinBERT

**Model:** [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) — a BERT transformer fine-tuned on financial corpora (SEC filings, financial news, analyst reports).

**Why FinBERT over generic BERT or lexicon methods?** Generic sentiment models (trained on movie reviews or social media) fail to capture the nuanced, domain-specific language of central banking. FinBERT's pre-training on financial text allows it to distinguish, for example, between "patient" (dovish signal) and "measured" (can be either), which lexicon methods cannot reliably do.

**Sentence-level workflow:**
1. Segment document into sentences
2. Pass each sentence through FinBERT (max 512 tokens)
3. Apply softmax normalization: `p_i = exp(logit_i) / Σ exp(logit_j)`
4. Assign label = class with maximum probability

**Output classes:**

| FinBERT Label | Policy Interpretation |
|---|---|
| Positive | Dovish / Accommodative |
| Neutral | Balanced / Uncertain |
| Negative | Hawkish / Restrictive |

**Document-level aggregation:** Arithmetic mean of all sentence-level probability scores. Sensitivity checks run with confidence-weighted aggregation (higher-confidence sentences weighted more).

**Final score range:** −1 (fully hawkish) to +1 (fully dovish), with 0 representing neutral.

**Note:** Pre-trained weights were used without fine-tuning to prevent overfitting on the limited set of labeled Fed documents.

---

### Stage 4 — Policy Tone Classification (Hybrid Approach)

**What:** Assign each document a final policy stance label — **Hawkish**, **Dovish**, or **Neutral** — using a hybrid model that combines rule-based keyword analysis with FinBERT's probabilistic output.

**Why hybrid?** Pure lexicon methods are transparent but miss context. Pure neural methods (FinBERT) capture context but can be opaque. The hybrid approach gets interpretability from keywords and semantic understanding from the transformer.

#### Keyword Lexicon (47 domain-specific terms)

**Hawkish keywords (24):** *restrictive, tightening, elevated, pressure, headwinds, above-target, inflation, rate hikes, firm, aggressive, restraint, frontloaded, warranted, urgency, appropriate, imperative, ...*

**Dovish keywords (23):** *accommodative, support, downside risks, slack, uncertainty, patient, gradual, transitory, flexibility, temporary, manageable, cautious, buffer, safeguard, insurance, ...*

Keyword frequencies normalized by document length:
```
hawkish_freq = (count of hawkish terms) / (total tokens)
dovish_freq  = (count of dovish terms)  / (total tokens)
```

#### Composite Tone Score Formula

```
Tone_Score = (0.6 × Normalized_Keyword_Score) + (0.4 × Normalized_FinBERT_Score)

where:
  Normalized_Keyword_Score = (hawkish_freq - dovish_freq) / (hawkish_freq + dovish_freq + ε)
  Normalized_FinBERT_Score = p_pos - p_neg
  ε = 0.001  (prevents division by zero)
```

**Why 60/40 weighting?** Keyword signals are more directly interpretable for monetary policy classification; FinBERT provides contextual depth. The 60/40 ratio was calibrated through manual review and achieves the best agreement with expert labels.

#### Classification Thresholds

| Tone Score | Policy Stance |
|---|---|
| > +0.2 | 🔴 Hawkish |
| < −0.2 | 🟢 Dovish |
| −0.2 to +0.2 | 🟡 Neutral |

#### Confidence Score

```
Confidence = (|Tone_Score| / max(|Tone_Score|)) × (1 − p_neutral)
```

High confidence = strong tone signal AND low model uncertainty. Used in downstream analysis to filter high-confidence events for sharper hypothesis testing.

#### Validation

- 10% random sample (54 documents) manually classified by two domain experts
- **Cohen's kappa: κ = 0.82** (p < 0.001) — indicates strong inter-rater and model–expert agreement
- Target threshold was κ > 0.75

---

### Stage 5 — Event Study (Abnormal Return Calculation)

**What:** Isolate the market's reaction to each Fed communication from the general movement of the market on the same day.

**Why event study methodology?** Markets move for many reasons simultaneously. The event study framework (Fama et al., 1969; MacKinlay, 1997) statistically controls for normal market behavior so we can attribute residual movement specifically to the Fed announcement.

#### Event Windows

| Window | Period | Purpose |
|---|---|---|
| Estimation window | t−65 to t−5 | Estimate baseline market behavior (60 trading days, ~3 months) |
| Pre-event buffer | t−5 to t−1 | Excluded to avoid anticipation effects contaminating baseline |
| Immediate reaction | t=0 to t=3 | First 3 trading days after announcement |
| Event window | t=0 to t=30 | Full one-month market adjustment |

#### Normal Return Model (OLS)

```
R_{i,t} = α_i + β_i × R_{m,t} + ε_{i,t}
```

Parameters α and β estimated on the estimation window using Ordinary Least Squares regression.

#### Abnormal Return

```
AR_t = R_t − (α̂ + β̂ × R_{m,t})
```

The difference between what the market *actually* returned and what the model *predicted* it should return based on history.

#### Cumulative Abnormal Return (CAR)

```
CAR[t1, t2] = Σ AR_t  for t in [t1, t2]
```

**CARs calculated for:**
- `CAR(0,3)` — Immediate reaction
- `CAR(0,10)` — Two-week window
- `CAR(0,30)` — One-month window

#### Excess Volatility

Two measures of volatility response calculated alongside returns:

| Measure | Definition |
|---|---|
| Excess VIX | `Mean(VIX[0,3]) − Mean(VIX[−65,−5])` |
| Excess Realized Volatility | `Mean(RV[0,20]) − Mean(RV[−65,−20])` |

---

### Stage 6 — Statistical Evaluation & Robustness Testing

**What:** Test whether policy tone scores are statistically associated with market reactions, and whether this relationship holds across different economic conditions.

**Why not a predictive ML model?** With only 8 FOMC meetings per year, supervised machine learning would be severely data-constrained. A correlation and causal inference framework is more appropriate, more interpretable, and directly answers the research question: *does tone influence markets?*

#### Primary Statistical Methods

**1. Pearson & Spearman Correlation**
Tests linear (Pearson) and monotonic (Spearman) relationships between tone scores and CARs. Spearman is preferred when outliers (crisis periods) are present.

**2. Directional Consistency Analysis**
Measures what percentage of events produced economically expected market movements:
- Hawkish tone → negative CAR ✓
- Dovish tone → positive CAR ✓
- Neutral tone → |CAR| < 1% ✓

Baseline = 50% (random). Target > 55%.

**3. Granger Causality Test**
Tests whether past policy tone scores carry *additional* predictive information for future returns beyond what past returns alone can predict. Run using `statsmodels`; significance threshold p < 0.05.

**4. Robustness Checks**

| Check | Method |
|---|---|
| Time-stability | Rolling 252-day (1-year) correlation windows |
| Crisis vs. normal markets | Separate correlations for 2008–2009 and COVID-19 periods |
| Volatility regimes | Separate analyses for VIX > 75th percentile and VIX < 25th percentile |
| Window sensitivity | Correlations for CAR(0,1), CAR(0,5), CAR(0,10), CAR(0,30) |
| Confounders | Partial correlations controlling for CPI surprises, unemployment reports, rate changes |

#### Evaluation Metrics Summary

| Metric | Purpose |
|---|---|
| Pearson / Spearman ρ | Relationship strength between tone and returns |
| p-value (< 0.05) | Statistical significance |
| Directional Accuracy | How often market moves in expected direction |
| Granger F-statistic | Whether tone adds predictive information for returns |
| Cohen's d | Effect size between hawkish vs. dovish return distributions |
| 95% Confidence Intervals | Reliability bounds on correlations (Fisher z-transform) |

---

## 🧰 Technology Stack

| Task | Library |
|---|---|
| Web scraping | `BeautifulSoup`, `Selenium` |
| Text processing | `spaCy`, `NLTK`, `scikit-learn` |
| Sentiment / NLP | `transformers` (HuggingFace), `ProsusAI/finbert` |
| Data manipulation | `pandas`, `numpy` |
| Market data | `yfinance` |
| Statistical analysis | `scipy.stats`, `statsmodels` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/fedspeak.git
cd fedspeak
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Pipeline

To run the full pipeline end-to-end:
```bash
python run_pipeline.py
```

Or run individual stages:
```bash
python src/1_scraper.py
python src/2_market_data.py
python src/3_preprocessing.py
python src/4_sentiment.py
python src/5_tone_classifier.py
python src/6_event_study.py
python src/7_statistics.py
```

> **Reproducibility:** Random seeds 42, 77, and 123 are set at pipeline initialization. All results should be exactly reproducible across runs.

---

## 🔀 Data Split Strategy

To prevent temporal leakage and simulate real-world deployment:

| Split | Period | Events |
|---|---|---|
| Training set | Jan 2008 – Dec 2018 | 374 events |
| Test set | Jan 2019 – Dec 2025 | 181 events |

Cross-validation uses **5-fold time-series CV** (chronologically sequential, no shuffling) to preserve temporal ordering and prevent look-ahead bias.

A separate **holdout set** excludes crisis periods (2008–2009 GFC, Q1-Q3 2020 COVID) to test whether findings generalize across different market volatility regimes.

---

## 📄 Citation

If you use this work, please cite:

```
Khadka, P. B., Sharma, S., & Shakya, R. (2025). Quantifying FedSpeak: Sentiment Analysis
of Federal Reserve Communication and Market Reactions from 2008 to 2025.
Department of Electronics and Computer Engineering, Thapathali Campus, Kathmandu, Nepal.
```

---

## 📚 References

- Gentzkow, M. et al. (2019). Text as data. *Journal of Economic Literature*.
- Fama, E. F. et al. (1969). Adjustment of stock prices to new information. *International Economic Review*.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*.
- Arshad, A. The role of Fed speech sentiment signals in shaping US market response. NUST Business School.
- Ahrens, M. et al. (2024). Mind your language: Market responses to central bank speeches. FRB St. Louis Working Paper 2023-013.
- Czudaj, R. (2025). ECB's central bank communication and monetary policy transmission.

---

*Built by Prayush Bikram Khadka, Shaswat Sharma, and Rajad Shakya — Thapathali Campus, Kathmandu, Nepal.*
