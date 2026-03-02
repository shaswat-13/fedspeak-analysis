# FedSpeak Sentiment Analysis Project: Complete Implementation Guide

## 1. PROJECT EVALUATION

### Strengths
- **High Relevance**: Fed communication directly impacts financial markets; strong real-world application
- **Clear Scope**: Well-defined boundaries (data collection → analysis → evaluation)
- **Measurable Outcomes**: Correlation metrics provide quantifiable success criteria
- **Practical Value**: Useful for traders, portfolio managers, and policy researchers
- **Feasible Scale**: Can be completed with reasonable computational resources

### Weaknesses to Address
- **Data Availability**: FOMC statements are structured but speeches are unstructured
- **Temporal Alignment**: Market reactions may be delayed/distributed over time
- **Sentiment Ambiguity**: Federal Reserve deliberately uses measured language (low variance)
- **Confounding Variables**: Multiple economic news events occur around FOMC meetings
- **Label Scarcity**: Ground truth for "correct" market reaction is subjective

### Risk Mitigation Strategies
1. Use **domain-specific sentiment models** (FinBERT) rather than generic sentiment analysis
2. Implement **careful event study methodology** to isolate Fed communication effects
3. Control for **confounding economic indicators** (CPI, unemployment, Fed Funds rate)
4. Consider **lagged relationships** (market may react with delays)
5. Validate with **multiple evaluation metrics** (correlation, Granger causality, directional accuracy)

---

## 2. DETAILED STEPWISE IMPLEMENTATION

### STEP 1: DATA COLLECTION & INFRASTRUCTURE

#### 1.1 Data Sources

**Federal Reserve Communications:**
- **FOMC Statements**: Board of Governors website (https://www.federalreserve.gov/newsevents/pressreleases/)
  - Structured, ~8 per year
  - Standardized format (Policy Decision → Economic Outlook → Future Guidance)
  
- **FOMC Meeting Minutes**: Released ~3 weeks after meetings
  - Detailed policy discussions
  - Richer semantic content
  
- **Fed Chair Speeches**: Federal Reserve Speaker Directory
  - Semi-structured, 50+ per year
  - Variable length and topic coverage
  
- **Beige Book** (optional): Regional economic summaries before each FOMC

**Market Data:**
- **S&P 500 Price/Returns**: yfinance (daily OHLCV)
- **Treasury Yields**: 10Y, 2Y yields (yfinance or FRED API)
- **VIX Index**: Volatility gauge (yfinance)
- **Fed Funds Rate**: Historical rates (FRED)

#### 1.2 Data Collection Implementation

```python
# Pseudocode structure
class DataCollector:
    def __init__(self):
        self.fed_api = FRED_API()
        self.market_data = yfinance
        
    def collect_fomc_statements(self, start_date, end_date):
        # Web scrape from Federal Reserve website
        # Parse release dates from press releases
        # Extract statement text
        return fomc_df
    
    def collect_market_data(self, dates):
        # Download from yfinance
        # Calculate returns and volatility
        return market_df
    
    def align_data(self, fed_df, market_df):
        # Match Fed events with market data
        # Create event windows (t-5 to t+30 days)
        return aligned_df
```

**Deliverable**: CSV files with:
- `fomc_statements.csv`: [date, text, type (statement/minutes/speech)]
- `market_data.csv`: [date, sp500_close, returns, vix, treasury_10y]
- `events.csv`: [event_date, event_type, statement_text, market_data_window]

**Timeline**: ~2-3 weeks of data collection (mostly automated)

---

### STEP 2: TEXT PREPROCESSING

#### 2.1 Preprocessing Pipeline

```python
class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def preprocess(self, text):
        # 1. Text Cleaning
        text = self.clean_text(text)  # Remove headers, footers, special chars
        
        # 2. Sentence Segmentation
        sentences = self.segment_sentences(text)
        
        # 3. Tokenization
        tokens = self.tokenize(sentences)
        
        # 4. Lemmatization (optional for FinBERT)
        lemmas = self.lemmatize(tokens)
        
        # 5. Remove stopwords (careful - keep financial stopwords)
        filtered = self.remove_stopwords(lemmas)
        
        return processed_text, sentences, tokens
    
    def clean_text(self, text):
        # Remove footer/header noise
        # Normalize whitespace
        # Keep numbers (important in Fed communication)
        return cleaned
    
    def segment_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
```

**Key Considerations:**
- **Preserve numbers**: "interest rate to 4.5%" is important
- **Financial terminology**: Don't remove domain-specific stopwords
- **Maintain structure**: Keep sentence boundaries for sentiment scoring

**Deliverable**: `processed_statements.pkl` with tokenized, cleaned text

**Timeline**: ~1 week

---

### STEP 3: SENTIMENT & POLICY TONE EXTRACTION

#### 3.1 Approach Selection

**Option A: FinBERT (Recommended)**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTSentimentAnalyzer:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    
    def score_sentence(self, sentence):
        # Returns: [negative, neutral, positive] logits
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                               max_length=512, truncation=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
        # Map: negative=-1, neutral=0, positive=1
        sentiment = probs[0].argmax().item()
        return sentiment, probs[0].tolist()
```

**Why FinBERT?**
- Trained on financial news and SEC filings
- Understands domain-specific language
- Better calibrated for subtle sentiment in Fed speak
- Pre-trained weights available

#### 3.2 Policy Tone Classification

```python
class PolicyToneClassifier:
    def __init__(self):
        self.hawkish_keywords = [
            'inflation', 'rate hikes', 'tightening', 'restrictive',
            'elevated', 'significant progress', 'appropriate'
        ]
        self.dovish_keywords = [
            'accommodative', 'support', 'downside risks', 'slack',
            'patience', 'gradual', 'transitory'
        ]
    
    def classify_statement(self, text):
        """
        Hybrid approach:
        1. Keyword frequency (lexicon-based)
        2. FinBERT sentiment scores
        3. Tone indicators (e.g., "we may need" vs "we will")
        """
        
        # Step 1: Keyword scoring
        hawk_score = self.count_keywords(text, self.hawkish_keywords)
        dove_score = self.count_keywords(text, self.dovish_keywords)
        
        # Step 2: Sentence-level sentiment
        sentences = sent_tokenize(text)
        sentiments = [self.finbert.score_sentence(s) for s in sentences]
        
        # Step 3: Weighted aggregation
        overall_tone = self.aggregate_tone(
            hawk_score, dove_score, sentiments
        )
        
        return {
            'stance': overall_tone,  # 'hawkish', 'dovish', 'neutral'
            'confidence': confidence_score,
            'hawk_score': hawk_score,
            'dove_score': dove_score,
            'sentiment_scores': sentiments
        }
    
    def aggregate_tone(self, hawk, dove, sentiments):
        # Combine signals
        # Example: if positive sentiment AND hawk keywords → strong hawkish
        
        hawk_weight = hawk / (hawk + dove + 1e-6)
        dove_weight = dove / (hawk + dove + 1e-6)
        sentiment_signal = np.mean([s[0] for s in sentiments])
        
        net_tone = hawk_weight - dove_weight + 0.3 * sentiment_signal
        
        if net_tone > 0.3:
            return 'hawkish'
        elif net_tone < -0.3:
            return 'dovish'
        else:
            return 'neutral'
```

**Deliverable**: `policy_tones.csv` with columns:
- event_date
- statement_type
- stance (hawkish/dovish/neutral)
- confidence
- hawk_score, dove_score
- detailed_sentiment_breakdown

**Timeline**: ~2 weeks (including model fine-tuning if needed)

---

### STEP 4: EVENT STUDY FRAMEWORK

#### 4.1 Event Study Methodology

```python
class EventStudyAnalysis:
    def __init__(self, market_df):
        self.market_df = market_df.set_index('date').sort_index()
        self.event_window = 30  # days after event
        self.estimation_window = 60  # days before event for baseline
    
    def calculate_abnormal_returns(self, event_date):
        """
        Methodology:
        1. Estimate normal return from historical data
        2. Calculate actual return around event
        3. Compute abnormal return = actual - expected
        """
        
        # Estimation window: t-60 to t-5
        est_start = event_date - pd.Timedelta(days=65)
        est_end = event_date - pd.Timedelta(days=5)
        est_returns = self.market_df.loc[est_start:est_end, 'returns']
        
        # Fit model (e.g., market model or simple mean)
        expected_return = est_returns.mean()
        expected_std = est_returns.std()
        
        # Event window: t to t+30
        event_start = event_date
        event_end = event_date + pd.Timedelta(days=self.event_window)
        event_returns = self.market_df.loc[event_start:event_end, 'returns']
        
        # Calculate abnormal returns
        abnormal_returns = event_returns - expected_return
        cumulative_abnormal_returns = abnormal_returns.cumsum()
        
        return {
            'date': event_date,
            'abnormal_returns': abnormal_returns,
            'cumulative_abnormal_returns': cumulative_abnormal_returns,
            'expected_return': expected_return,
            'volatility_change': event_returns.std() - expected_std
        }
    
    def calculate_excess_volatility(self, event_date):
        """
        VIX spike or realized volatility increase around announcement
        """
        est_start = event_date - pd.Timedelta(days=65)
        est_end = event_date - pd.Timedelta(days=5)
        normal_volatility = self.market_df.loc[est_start:est_end, 'vix'].mean()
        
        event_start = event_date
        event_end = event_date + pd.Timedelta(days=3)  # Immediate reaction
        event_volatility = self.market_df.loc[event_start:event_end, 'vix'].mean()
        
        return event_volatility - normal_volatility
```

#### 4.2 Integration with Policy Tones

```python
class MarketReactionAnalysis:
    def __init__(self, event_df, market_df):
        self.events = event_df  # Contains policy tones
        self.market = market_df
        self.esa = EventStudyAnalysis(market_df)
    
    def analyze_all_events(self):
        results = []
        
        for idx, event in self.events.iterrows():
            event_date = event['date']
            policy_tone = event['stance']
            
            market_reaction = self.esa.calculate_abnormal_returns(event_date)
            volatility_reaction = self.esa.calculate_excess_volatility(event_date)
            
            results.append({
                'date': event_date,
                'policy_tone': policy_tone,
                'cumulative_abnormal_return': market_reaction['cumulative_abnormal_returns'].iloc[-1],
                'volatility_change': volatility_reaction,
                'statement_text': event['text']
            })
        
        return pd.DataFrame(results)
```

**Deliverable**: `market_reactions.csv` with columns:
- event_date
- policy_tone
- cumulative_abnormal_return (5-day, 10-day, 30-day)
- volatility_change
- vix_spike

**Timeline**: ~1-2 weeks

---

### STEP 5: EVALUATION & VALIDATION

#### 5.1 Correlation Analysis

```python
class EvaluationFramework:
    def __init__(self, combined_df):
        # combined_df has: date, policy_tone, market_return, volatility
        self.data = combined_df
    
    def correlation_analysis(self):
        """
        Does hawkish → negative returns? Dovish → positive returns?
        """
        
        # Map tones to numerical values
        tone_numeric = {
            'hawkish': 1,
            'neutral': 0,
            'dovish': -1
        }
        self.data['tone_numeric'] = self.data['policy_tone'].map(tone_numeric)
        
        # Calculate correlations
        correlation = self.data['tone_numeric'].corr(
            self.data['cumulative_abnormal_return']
        )
        
        # Spearman correlation (non-parametric)
        spearman_corr, spearman_pval = spearmanr(
            self.data['tone_numeric'],
            self.data['cumulative_abnormal_return']
        )
        
        return {
            'pearson_r': correlation,
            'spearman_rho': spearman_corr,
            'spearman_pval': spearman_pval
        }
    
    def directional_consistency(self):
        """
        In % of cases, does market move in expected direction?
        - Hawkish → negative return? (expected)
        - Dovish → positive return? (expected)
        """
        
        correct_predictions = 0
        total = 0
        
        for idx, row in self.data.iterrows():
            tone = row['policy_tone']
            market_return = row['cumulative_abnormal_return']
            
            # Expected direction
            if tone == 'hawkish' and market_return < 0:
                correct_predictions += 1
            elif tone == 'dovish' and market_return > 0:
                correct_predictions += 1
            elif tone == 'neutral' and abs(market_return) < 0.01:  # Within 1%
                correct_predictions += 1
            
            total += 1
        
        accuracy = correct_predictions / total
        return accuracy
    
    def granger_causality_test(self):
        """
        Does policy tone Granger-cause market returns?
        (Does knowing tone improve prediction of returns?)
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Prepare data: [returns, tone]
        test_data = self.data[['cumulative_abnormal_return', 'tone_numeric']].dropna()
        
        # Test if tone Granger-causes returns
        gc_results = grangercausalitytests(test_data, max_lag=5, verbose=True)
        
        return gc_results
    
    def volatility_response(self):
        """
        Separate analysis: does any tone increase volatility?
        """
        
        volatility_by_tone = self.data.groupby('policy_tone')['volatility_change'].agg([
            'mean', 'std', 'count'
        ])
        
        # T-test: Hawkish vs Dovish volatility
        hawkish_vol = self.data[self.data['policy_tone']=='hawkish']['volatility_change']
        dovish_vol = self.data[self.data['policy_tone']=='dovish']['volatility_change']
        
        t_stat, t_pval = ttest_ind(hawkish_vol, dovish_vol)
        
        return {
            'by_tone': volatility_by_tone,
            't_statistic': t_stat,
            'p_value': t_pval
        }
    
    def summary_report(self):
        """Generate comprehensive evaluation report"""
        
        report = {
            'correlation': self.correlation_analysis(),
            'directional_accuracy': self.directional_consistency(),
            'granger_causality': self.granger_causality_test(),
            'volatility_analysis': self.volatility_response()
        }
        
        return report
```

#### 5.2 Robustness Checks

```python
class RobustnessAnalysis:
    def __init__(self, data):
        self.data = data
    
    def rolling_window_analysis(self):
        """Is relationship stable over time?"""
        
        rolling_corr = []
        for i in range(0, len(self.data), 12):  # 12-month windows
            window = self.data.iloc[i:i+12]
            corr = window['tone_numeric'].corr(
                window['cumulative_abnormal_return']
            )
            rolling_corr.append(corr)
        
        return rolling_corr
    
    def exclude_major_events(self):
        """Remove days with major economic news"""
        
        # Filter out extreme returns (likely due to other news)
        filtered = self.data[
            abs(self.data['cumulative_abnormal_return']) < 5  # >5% is outlier
        ]
        
        return self.evaluate_on_subset(filtered)
    
    def sensitivity_analysis(self):
        """
        Test different event windows:
        - Immediate reaction (1-3 days)
        - Short-term (1-10 days)
        - Medium-term (1-30 days)
        """
        
        results = {}
        for window in [3, 10, 30]:
            # Recalculate correlations with different windows
            subset = self.data[self.data['event_window'] == window]
            corr = subset['tone_numeric'].corr(
                subset['abnormal_return']
            )
            results[f'{window}_days'] = corr
        
        return results
```

**Deliverable**: `evaluation_report.md` with:
- Summary statistics
- Correlation coefficients with p-values
- Directional accuracy percentage
- Granger causality test results
- Robustness checks
- Visualizations (scatter plots, time series)

**Timeline**: ~1-2 weeks

---

## 3. SYSTEM ARCHITECTURE

```
FedSpeak_Analysis/
│
├── 01_data_collection/
│   ├── fed_statements_scraper.py
│   ├── market_data_downloader.py
│   ├── data_alignment.py
│   └── data/
│       ├── raw/
│       │   ├── fomc_statements/
│       │   ├── fed_speeches/
│       │   └── market_data/
│       └── processed/
│           ├── fomc_statements.csv
│           ├── market_data.csv
│           └── events.csv
│
├── 02_preprocessing/
│   ├── text_preprocessor.py
│   ├── tokenizer.py
│   ├── utils.py
│   └── processed_data/
│       └── cleaned_statements.pkl
│
├── 03_sentiment_analysis/
│   ├── finbert_analyzer.py
│   ├── policy_tone_classifier.py
│   ├── keyword_lexicon.py
│   └── results/
│       └── policy_tones.csv
│
├── 04_event_study/
│   ├── event_study_framework.py
│   ├── market_reaction_calculator.py
│   └── results/
│       └── market_reactions.csv
│
├── 05_evaluation/
│   ├── correlation_analysis.py
│   ├── statistical_tests.py
│   ├── robustness_checks.py
│   ├── visualization.py
│   └── results/
│       ├── evaluation_report.md
│       ├── plots/
│       │   ├── tone_vs_returns.png
│       │   ├── correlation_over_time.png
│       │   ├── volatility_by_tone.png
│       │   └── granger_causality.png
│       └── tables/
│           ├── summary_statistics.csv
│           └── correlation_matrix.csv
│
├── config/
│   ├── settings.yaml
│   ├── keywords.json
│   └── model_params.json
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Sentiment_Analysis.ipynb
│   ├── 03_Event_Study.ipynb
│   └── 04_Final_Analysis.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sentiment.py
│   └── test_event_study.py
│
├── requirements.txt
├── README.md
└── main.py  # Orchestration script
```

---

## 4. DETAILED METHODOLOGY BY PHASE

### Phase 1: Data Collection (Weeks 1-2)

**Week 1:**
- Set up data pipelines
- Scrape FOMC statements (2000-2025): ~200 statements
- Download market data from yfinance
- Create database schema

**Week 2:**
- Collect Fed chair speeches (sample: last 5 years = ~250 speeches)
- Align Fed events with market data
- Create event windows
- Quality checks and validation

**Deliverables:**
- `fomc_statements.csv`: 200+ records
- `market_data.csv`: Daily OHLCV for S&P 500
- `events_aligned.csv`: Matched Fed-market data

---

### Phase 2: Preprocessing (Week 3)

**Tasks:**
- Design preprocessing pipeline
- Handle Federal Reserve document structure
- Test on sample documents
- Validate cleaned output

**Key Decisions:**
- Keep technical terms (interest rate, unemployment, inflation)
- Preserve numbers and percentages
- Maintain sentence structure

**Deliverables:**
- Clean, tokenized text corpus
- Validation metrics (vocabulary size, token distribution)

---

### Phase 3: Sentiment & Tone (Weeks 4-5)

**Week 4:**
- Set up FinBERT model
- Score all statements for financial sentiment
- Build policy tone classifier
- Validate on known examples

**Week 5:**
- Fine-tune on Fed-specific data (if needed)
- Generate tone classifications
- Qualitative validation (manual review of 20-30 statements)

**Deliverables:**
- `policy_tones.csv`: Stance for each event
- Validation report with confusion matrix

---

### Phase 4: Event Study (Week 6)

**Tasks:**
- Calculate abnormal returns for each event
- Compute excess volatility
- Align with policy tones
- Handle outliers and confounds

**Statistical Approach:**
- Market Model: AR_t = α + β × Market_Return_t + ε_t
- Expected return from pre-event estimation window
- Abnormal return = Actual - Expected

---

### Phase 5: Analysis & Evaluation (Weeks 7-8)

**Week 7:**
- Correlation analysis
- Directional accuracy
- Granger causality tests
- Create visualizations

**Week 8:**
- Robustness checks
- Final report writing
- Prepare presentation

---

## 5. KEY METRICS & SUCCESS CRITERIA

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **Correlation (Policy Tone → Returns)** | r > 0.3* | Hawkish should correlate with negative returns |
| **Granger Causality p-value** | p < 0.05 | Tone predicts market movement |
| **Directional Accuracy** | >55-60% | Better than random (50%) |
| **Volatility Effect** | Significant F-test | Policy announcements increase volatility |
| **Consistency Over Time** | Stable rolling correlations | Relationship doesn't break down |

*Note: Fed speak is subtle; r=0.3 is realistic, not r=0.7

---

## 6. IMPLEMENTATION PRIORITIES

### High Priority (Core Project)
1. ✅ FOMC statement collection & preprocessing
2. ✅ FinBERT sentiment scoring
3. ✅ Event study with abnormal returns
4. ✅ Correlation analysis
5. ✅ Directional consistency evaluation

### Medium Priority (Enhanced Analysis)
6. Granger causality testing
7. Volatility decomposition
8. Robustness checks
9. Rolling window analysis

### Low Priority (Nice-to-Have)
10. Fed speech analysis
11. Sentiment transfer learning/fine-tuning
12. NLP interpretability (attention weights)
13. Real-time prediction system

---

## 7. POTENTIAL CHALLENGES & SOLUTIONS

| Challenge | Solution |
|-----------|----------|
| **Low sentiment variance** (Fed uses measured language) | Use domain-specific models (FinBERT); keyword augmentation |
| **Confounding variables** (CPI, unemployment data released same day) | Control for other economic news; use event windows carefully |
| **Delayed market reaction** | Test multiple event windows (1-3, 1-10, 1-30 days) |
| **Few events** (8 FOMC meetings/year) | Use speeches too (~50/year); extend historical data |
| **Model overfitting** | Use pre-trained FinBERT; validate on held-out test set |
| **Interpretability** | Use keyword explainability + attention visualization |

---

## 8. DELIVERABLES CHECKLIST

- [ ] Data collection pipeline + raw datasets
- [ ] Preprocessing code + cleaned corpus
- [ ] FinBERT sentiment analyzer
- [ ] Policy tone classifier with validation
- [ ] Event study calculations + abnormal returns
- [ ] Correlation & statistical analysis
- [ ] Final report (8-10 pages):
  - Literature review
  - Methodology
  - Results & findings
  - Robustness checks
  - Limitations & future work
- [ ] Code repository (GitHub) with README
- [ ] Presentation slides + visualizations
- [ ] Reproducibility guide (how to re-run entire pipeline)

---

## 9. TIMELINE SUMMARY

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data Collection | 2 weeks | Raw datasets |
| Preprocessing | 1 week | Cleaned text corpus |
| Sentiment Analysis | 2 weeks | Policy tones |
| Event Study | 1 week | Market reactions |
| Evaluation | 2 weeks | Final report & analysis |
| **Total** | **8 weeks** | Production-ready project |

---

## 10. TECHNOLOGY STACK (FINAL RECOMMENDATION)

```
Core:
- Python 3.9+
- Jupyter Notebook (for exploration)
- GitHub (version control)

NLP:
- transformers (HuggingFace FinBERT)
- spaCy (preprocessing)
- NLTK (utilities)

Finance & Data:
- yfinance (market data)
- pandas (data manipulation)
- numpy (numerical computing)

Analysis:
- scipy (statistical tests)
- statsmodels (Granger causality, regression)
- sklearn (utilities)

Visualization:
- matplotlib / seaborn (publication-quality plots)
- plotly (interactive)

Infrastructure:
- Docker (optional, for reproducibility)
- pytest (unit testing)
```

---

## 11. EXPECTED FINDINGS (HYPOTHESIS)

**Primary Hypothesis:**
- Hawkish Fed communications → decreased S&P 500 returns (investors fear higher rates)
- Dovish Fed communications → increased S&P 500 returns (stimulus expectations)
- Correlation magnitude: r ≈ 0.25-0.45 (moderate, not perfect)

**Secondary Findings:**
- Volatility spikes more for hawkish surprises than dovish
- Market reacts mostly within 1-3 days
- Effect stronger during uncertainty periods (2022 hiking cycle)

**Limitations to Acknowledge:**
- Correlation ≠ causation (Fed responds to market conditions too)
- Sample size constraints (~8-20 per year × years of data)
- Market driven by multiple factors simultaneously