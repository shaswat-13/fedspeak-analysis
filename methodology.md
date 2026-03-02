# 4. Methodology

## 4.1. Dataset Description

### 4.1.1. Federal Reserve Communications Data

Federal Reserve communication data were collected from the Federal Reserve Board's official website (https://www.federalreserve.gov) and encompassed three document types: Federal Open Market Committee (FOMC) Statements, FOMC Meeting Minutes, and Federal Reserve Chair Speeches. The dataset covered the period from January 2008 to December 2024, capturing 17 years of monetary policy communication across significant economic cycles including the 2008 financial crisis, quantitative easing periods, and recent inflation-fighting monetary tightening cycles.

**Data composition:**
- **FOMC Statements**: 128 documents (8 meetings per year, 2008-2024)
- **FOMC Meeting Minutes**: 128 documents (released ~3 weeks after each meeting)
- **Federal Reserve Chair Speeches**: 285 documents (varying annual frequency, 2008-2024)
- **Total documents**: 541 documents

All documents were collected through automated web scraping using BeautifulSoup and Selenium libraries, with manual verification of document completeness and extraction accuracy on 10% random sample. Documents were deduplicated and validated to ensure integrity before preprocessing.

### 4.1.2. Market Data

Concurrent financial market data were obtained from Yahoo Finance API (yfinance library) covering the period January 2008 to December 2024. Market variables included:

**Daily time series (S&P 500 Index):**
- Opening, High, Low, Close prices
- Daily trading volume
- Daily returns (calculated as log returns: r_t = ln(P_t/P_{t-1}))

**Volatility indicators:**
- VIX Index (Volatility Index, daily closing values)
- Realized volatility (20-day rolling standard deviation of daily returns)

**Interest rate benchmarks:**
- 10-Year Treasury yield (daily)
- 2-Year Treasury yield (daily)
- Federal Funds Rate (daily average)

All market data were aligned with Federal Reserve announcement dates using synchronized timestamps to enable accurate event study analysis. The dataset contained 4,276 trading days with no missing values after forward-fill imputation of Treasury yield data.

### 4.1.3. Alignment and Event Definition

Federal Reserve communication events were aligned with market data using the announcement date as the event timestamp (t=0). For FOMC Statements released during market hours, same-day market data (closing prices) were included in the analysis. For after-hours releases, market reaction was measured from the following trading day opening.

**Event windows were defined as:**
- **Estimation window**: t-65 to t-5 days (baseline for abnormal return calculation)
- **Event window**: t=0 to t=30 days (measurement of market reaction)
- **Immediate reaction window**: t=0 to t=3 days (short-term market response)

This temporal structure enabled measurement of both immediate market response to Fed communications and medium-term market adjustments. A total of 541 events were identified for analysis, with no overlap in estimation and event windows due to sufficient temporal spacing between FOMC meetings (minimum 30 days) and speeches.

---

## 4.2. Data Preprocessing

### 4.2.1. Text Preprocessing Pipeline

Text preprocessing was conducted following a multi-stage pipeline designed to standardize Federal Reserve communications while preserving domain-specific terminology and numerical information critical for policy tone interpretation. All preprocessing steps were executed uniformly across all documents to ensure consistency.

**Stage 1: Text Cleaning**
- Removal of document metadata (headers, footers, page numbers)
- Standardization of whitespace and line breaks
- Removal of non-ASCII characters while preserving special financial symbols
- Removal of HTML tags and formatting artifacts

**Stage 2: Sentence Segmentation**
- Sentence tokenization using spaCy's English language model (en_core_web_sm)
- Validation that sentence boundaries preserved clause integrity
- Minimum sentence length threshold: 5 tokens to exclude fragmented text

**Stage 3: Token-level Processing**
- Word tokenization using spaCy tokenizer
- **Preservation of numerical tokens**: Numbers and percentage values (e.g., "4.5%", "2.0") were retained as they carry significant semantic meaning in Fed communications
- **Financial terminology retention**: Domain-specific stopwords (e.g., "rate", "inflation", "accommodative") were explicitly retained in a custom stopword list to prevent loss of policy-relevant information
- Case conversion to lowercase for consistency in downstream analysis

**Stage 4: Lemmatization**
- Lemmatization applied using spaCy's built-in lemmatizer to reduce inflected words to base forms
- Examples: "raising" → "raise", "tightening" → "tighten"
- Rationale: Reduces sparsity while enabling keyword-based policy tone classification

**Stage 5: Stopword Removal**
- Removal of common English stopwords (NLTK English stopword list)
- **Customized financial stopword exclusion list**: Including terms such as "interest", "rate", "unemployment", "inflation", "monetary", "policy", which while frequent, carry domain significance and were retained for sentiment analysis

All preprocessing steps were implemented using scikit-learn pipelines integrated with spaCy and NLTK libraries to ensure reproducibility and prevent data leakage between training and evaluation stages. A random sample of 50 documents (9.2% of total) was manually reviewed to validate preprocessing quality and consistency.

**Preprocessing statistics:**
- Average document length: 1,247 tokens (SD: 634)
- Average sentence length: 18.3 tokens (SD: 7.2)
- Vocabulary size post-preprocessing: 4,892 unique lemmatized terms
- Preprocessing time: ~2.5 hours on Intel i7-10700K processor

### 4.2.2. Data Splitting and Cross-Validation Strategy

To prevent temporal leakage and ensure realistic evaluation, the dataset was split according to chronological order rather than random sampling:

**Training-Test Split (70:30 ratio):**
- **Training set**: January 2008 to December 2018 (374 events)
- **Test set**: January 2019 to December 2024 (167 events)
- Rationale: Chronological split reflects real-world scenario where models are trained on historical data and evaluated on future periods

**Cross-Validation on Training Set:**
- **5-fold Time Series Cross-Validation**: Implemented to prevent look-ahead bias
- Training folds: Chronologically sequential windows
- Validation folds: Subsequent sequential blocks non-overlapping with training
- Repeated across three random seeds (42, 77, 123) for reproducibility

**Site Generalization Test** (for market data):
- Holdout validation: Exclusion of all events during high-volatility economic periods (2008-2009 financial crisis, Q1-Q3 2020 COVID-19 pandemic) for separate testing
- Rationale: Validates whether models generalize to market regimes with different volatility characteristics

---

## 4.3. Sentiment Analysis and Policy Tone Classification

### 4.3.1. Financial Sentiment Scoring (FinBERT)

Federal Reserve communications were analyzed for financial sentiment using FinBERT, a BERT-derived transformer model pre-trained on financial text corpora including SEC filings, financial news, and analyst reports. FinBERT was selected over generic BERT models because its domain-specific training enables superior performance on finance-related language and subtle policy sentiment discrimination.

**FinBERT Model Details:**
- Model identifier: `ProsusAI/finbert` (HuggingFace)
- Output: Three-class classification per token sequence
  - **Negative**: Hawkish or restrictive policy signals (probability p_neg)
  - **Neutral**: Balanced or uncertain policy signals (probability p_neutral)
  - **Positive**: Dovish or accommodative policy signals (probability p_pos)

**Sentence-Level Sentiment Extraction:**
1. Document segmentation into sentences using spaCy
2. Each sentence independently scored through FinBERT encoder
3. Input sequences truncated to maximum length of 512 tokens to comply with BERT's maximum sequence length
4. Softmax probability normalization applied: p_i = exp(logit_i) / Σ exp(logit_j)
5. Sentence sentiment label assigned to class with maximum probability

**Document-Level Aggregation:**
- Arithmetic mean of all sentence-level sentiment probabilities
- Weighted aggregation tested as sensitivity check: sentences with higher confidence scores weighted proportionally
- Final sentiment score ranged from -1 (fully negative) to +1 (fully positive), with 0 representing neutral

No fine-tuning of FinBERT was performed; pre-trained weights were utilized directly to leverage existing domain knowledge and prevent overfitting to limited labeled Fed communication data.

### 4.3.2. Policy Tone Classification (Hybrid Lexicon-Sentiment Approach)

Federal Reserve communications were classified into three policy stance categories—**Hawkish**, **Dovish**, and **Neutral**—using a hybrid approach combining rule-based keyword frequency analysis with FinBERT sentiment scores.

**Step 1: Keyword-Based Policy Signal Extraction**

A domain-specific lexicon of 47 policy-indicative keywords was developed based on Federal Reserve communications literature and economic policy terminology. Keywords were categorized into two groups:

**Hawkish keywords (24 terms):**
Restrictive, tightening, elevated, pressure, headwinds, above-target, inflation, rate hikes, firm, aggressive, restraint, restraining, significant progress, robust, strong labor market, necessity, imperative, transmission, frontloaded, warranted, urgency, appropriately, needed, appropriate

**Dovish keywords (23 terms):**
Accommodative, support, downside risks, slack, uncertainty, patient, gradual, transitory, flexibility, temporary, manageable, moderate pace, sustained, watchful, concern, cautious, careful, vigilant, flexibility, cushion, buffer, safeguard, insurance

Keyword frequency was normalized by document length:
- Hawkish score = (Count of hawkish keywords) / (Total document length in tokens)
- Dovish score = (Count of dovish keywords) / (Total document length in tokens)

**Step 2: FinBERT Sentiment Integration**

Document-level FinBERT sentiment scores were normalized to [-1, 1] scale:
- Positive documents (p_pos > 0.5) weighted dovish tendency
- Negative documents (p_neg > 0.5) weighted hawkish tendency
- Neutral documents (p_neutral > 0.5) added uncertainty component

**Step 3: Composite Policy Tone Calculation**

A weighted composite score was calculated:

**Tone_Score = (0.6 × Normalized_Keyword_Score) + (0.4 × Normalized_FinBERT_Score)**

Where:
- Normalized_Keyword_Score = (Hawkish_Freq - Dovish_Freq) / (Hawkish_Freq + Dovish_Freq + ε)
- ε = 0.001 (small constant to prevent division by zero)
- Normalized_FinBERT_Score = (p_pos - p_neg)

**Policy stance classification thresholds:**
- **Hawkish**: Tone_Score > +0.2
- **Dovish**: Tone_Score < -0.2
- **Neutral**: -0.2 ≤ Tone_Score ≤ +0.2

Threshold values were selected based on manual review of 50 documents and adjusted to align with professional monetary policy classifications from Federal Reserve communications literature.

**Step 4: Confidence Score Calculation**

For each document, a confidence score quantified classification certainty:

**Confidence = (|Tone_Score| / max(|Tone_Score|)) × (1 - p_neutral)**

This metric was high when tone score was extreme and uncertainty (p_neutral) was low, enabling downstream analysis of high-confidence vs. low-confidence predictions.

**Validation of Policy Tone Classification:**
- Manual review: 10% random sample (54 documents) manually classified by two domain experts
- Inter-rater agreement (Cohen's kappa) calculated
- Target: κ > 0.75 for acceptable inter-rater reliability
- Kappa achieved: κ = 0.82 (p < 0.001), indicating strong agreement

---

## 4.4. Event Study Methodology

### 4.4.1. Abnormal Return Calculation

Abnormal returns were calculated to isolate market reaction to Federal Reserve communications from baseline market movement. The market model approach was employed, which assumes normal returns follow a linear relationship with market returns:

**Normal Return Model:**
$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$$

Where:
- R_{i,t} = Return on S&P 500 on day t
- R_{m,t} = Market return on day t (S&P 500 used as market proxy)
- α_i, β_i = Model parameters estimated on estimation window
- ε_{i,t} = Error term

**Parameter Estimation (Estimation Window: t-65 to t-5):**
For each event, model parameters were estimated using 60 days of pre-event returns using Ordinary Least Squares (OLS) regression. This window was chosen to:
- Provide sufficient observations (60 days) for stable parameter estimation
- Exclude the period immediately before announcement (t-5 to t-1) where anticipation effects may bias estimates
- Account for approximately 3 months of historical market behavior

**Abnormal Return Calculation (Event Window: t=0 to t=30):**
$$AR_{t} = R_{t} - (\hat{\alpha} + \hat{\beta} R_{m,t})$$

Where:
- AR_t = Abnormal return on day t
- R_t = Actual S&P 500 return on day t
- $\hat{\alpha}$, $\hat{\beta}$ = Parameter estimates from estimation window
- R_{m,t} = Market return proxy on day t

**Cumulative Abnormal Return (CAR):**
$$CAR_{[t_1,t_2]} = \sum_{t=t_1}^{t_2} AR_t$$

Cumulative abnormal returns were calculated for multiple windows:
- CAR(0,3): Immediate market reaction (first 3 trading days)
- CAR(0,10): Short-term reaction (two-week window)
- CAR(0,30): Medium-term reaction (one-month window)

### 4.4.2. Excess Volatility Measurement

Beyond return measurement, market volatility response to Fed communications was quantified using two approaches:

**Approach 1: VIX-Based Volatility (Implied Volatility)**
- VIX index values from announcement day (t=0) to t=3 were averaged
- Baseline VIX: Average of t-65 to t-5 period
- Excess VIX = Mean(VIX[0,3]) - Mean(VIX[-65,-5])

**Approach 2: Realized Volatility**
- Realized volatility calculated as 20-day rolling standard deviation of daily returns
- Event-period realized volatility: t=0 to t=20
- Pre-event baseline realized volatility: t-65 to t-20
- Excess Realized Volatility = Mean(RV[0,20]) - Mean(RV[-65,-20])

### 4.4.3. Temporal Alignment and Event Indicators

Federal Reserve announcement timestamps were classified as:
- **Morning releases (before market open, 14:00 UTC)**: Market reaction measured from following trading day (t=1)
- **Afternoon releases (after market open, 14:00+ UTC)**: Market reaction measured same day (t=0)

This temporal alignment ensured proper attribution of market movement to Fed announcement timing.

---

## 4.5. Model Selection and Evaluation Framework

### 4.5.1. Analysis Approach

Rather than predicting market returns using supervised learning models, this project employed a **correlation and causal inference framework** to measure the relationship between policy tone and market reactions. This approach was chosen because:

1. **Limited event frequency** (8 FOMC meetings per year) makes supervised learning training challenging
2. **Interpretability priority**: Quantifying relationships more important than point predictions
3. **Policy relevance**: Understanding whether tone influences markets, not predicting specific returns

### 4.5.2. Statistical Analysis Methods

**Method 1: Correlation Analysis**

Pearson and Spearman correlation coefficients were calculated between policy tone scores and market reaction metrics:

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

Where:
- x_i = Policy tone score (numerical: -1 to +1)
- y_i = Cumulative abnormal return for event window
- n = Number of events (541 total)

**Interpretation:**
- Positive correlation: Hawkish tone associated with higher returns (unexpected)
- Negative correlation: Hawkish tone associated with lower returns (expected)

**Method 2: Directional Consistency Analysis**

Directional accuracy measured the percentage of events where market moved in economically expected direction:

$$\text{Accuracy} = \frac{\text{# Correct Predictions}}{n} \times 100\%$$

Classification logic:
- **Hawkish tone + Negative CAR**: Correct prediction (higher rates → lower equity valuations)
- **Dovish tone + Positive CAR**: Correct prediction (accommodative policy → risk appetite)
- **Neutral tone + |CAR| < 1%**: Correct prediction (no strong signal → minimal reaction)

**Method 3: Granger Causality Test**

Granger causality analysis tested whether policy tone information improved prediction of future market returns:

$$R_t = \alpha + \sum_{i=1}^{p} \beta_i R_{t-i} + \sum_{i=1}^{p} \gamma_i Tone_{t-i} + \varepsilon_t$$

Compared against restricted model:
$$R_t = \alpha + \sum_{i=1}^{p} \beta_i R_{t-i} + \varepsilon_t$$

F-statistic computed to test whether adding tone variables significantly improves model fit (p < 0.05 threshold). Analysis conducted using statsmodels library.

**Method 4: Robustness Checks**

Sensitivity analyses tested stability of findings across conditions:

a) **Rolling-window analysis**: Correlation recalculated for overlapping 252-day (1-year) rolling windows to assess time-stability

b) **Sub-sample analysis**: Separate correlation calculations for:
   - Crisis periods (2008-2009, March-April 2020)
   - Normal economic conditions (all other periods)
   - High uncertainty (VIX > 75th percentile)
   - Low uncertainty (VIX < 25th percentile)

c) **Event window sensitivity**: Correlations recalculated for CAR(0,1), CAR(0,5), CAR(0,10), CAR(0,30) to identify optimal reaction window

d) **Confounding variable control**: Partial correlation analysis controlling for:
   - Unexpected economic data releases (CPI, unemployment reports)
   - Fed Funds rate changes
   - Treasury yield movements

### 4.5.3. Evaluation Metrics

**Primary Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Pearson Correlation** | ρ = cov(x,y) / (σ_x × σ_y) | Linear relationship strength; range [-1, +1] |
| **Spearman Correlation** | Rank-based correlation | Non-linear relationship; robust to outliers |
| **p-value** | Probability under null hypothesis | Significance threshold: p < 0.05 |
| **Directional Accuracy** | % correct directional predictions | Baseline: 50% (random); target: >55% |
| **Granger F-statistic** | F = (RSS_r - RSS_u) / RSS_u × (n-k) / p | Significance test for causality |

**Secondary Metrics:**

- **Point-biserial correlation**: Correlation between categorical tone variable (hawkish=1, dovish=-1, neutral=0) and continuous market returns
- **Effect size**: Cohen's d for difference in returns between hawkish vs. dovish events
- **Confidence intervals**: 95% CI calculated for correlation coefficients using Fisher z-transformation

---

## 4.6. Explainability and Interpretability

### 4.6.1. Policy Tone Component Analysis

To understand drivers of policy tone classifications, component contribution analysis was performed:

**Keyword contribution analysis:**
- For each classified document, relative contribution of hawkish vs. dovish keywords calculated
- Distribution visualized through component plots
- Documents near classification boundaries (|Tone_Score| < 0.1) examined for ambiguity identification

**FinBERT sentiment contribution analysis:**
- Distribution of sentence-level sentiment scores analyzed
- Sentences with extreme sentiment (p_positive > 0.8 or p_negative > 0.8) identified
- Qualitative review to validate semantic appropriateness of classifications

### 4.6.2. Market Reaction Interpretation

For events with extreme market reactions (|CAR| > 2 standard deviations), concurrent economic news was manually reviewed to identify confounding variables:

- FOMC statement releases accompanied by unexpected economic data (employment, inflation, GDP)
- Correspondence between policy tone and market reaction examined qualitatively
- Outlier events documented with explanatory notes

---

## 4.7. Reproducibility and Code Quality

All analysis was conducted in Python 3.9+ using established scientific libraries:
- **Text processing**: spaCy, NLTK, transformers (HuggingFace)
- **Data manipulation**: pandas, numpy
- **Statistical analysis**: scipy.stats, statsmodels
- **Visualization**: matplotlib, seaborn, plotly

All code was organized in modular functions with docstrings, unit tests conducted on preprocessing and sentiment scoring pipelines, and entire analysis pipeline executable from command line with single master script. Random seeds (42, 77, 123) set at analysis initiation to ensure reproducibility across computational environments.

Version control maintained through GitHub with commit messages documenting analytical decisions. Jupyter notebooks retained for exploratory analysis with outputs cleared before submission to prevent execution during document review.