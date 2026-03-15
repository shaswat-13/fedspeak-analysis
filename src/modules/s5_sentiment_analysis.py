import pandas as pd
import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load resources globally to avoid reloading inside loops
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("FinBERT loaded!")

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please download it.")
    nlp = None

# Keyword Lists
HAWKISH_KEYWORDS = [
    'restrictive', 'tightening', 'tighten', 'elevated', 'pressure', 
    'headwinds', 'inflation', 'rate hike', 'firm', 'aggressive', 
    'restraint', 'robust', 'strong', 'necessity', 'urgency', 
    'warranted', 'appropriate', 'frontloaded', 'above target'
]

DOVISH_KEYWORDS = [
    'accommodative', 'support', 'downside', 'slack', 'uncertainty', 
    'patient', 'gradual', 'transitory', 'flexible', 'temporary', 
    'manageable', 'moderate', 'sustained', 'watchful', 'cautious', 
    'careful', 'cushion', 'buffer', 'insurance'
]

def get_finbert_sentiment(text):
    """Calculates sentiment probabilities using FinBERT."""
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    return {'positive': probs[0].item(), 'negative': probs[1].item(), 'neutral': probs[2].item()}

def score_document(text):
    """Splits document into sentences and averages FinBERT sentiment scores."""
    if not isinstance(text, str) or nlp is None:
        return None
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    if not sentences:
        return None
    
    scores = [get_finbert_sentiment(s) for s in sentences]
    return {
        'positive': np.mean([s['positive'] for s in scores]),
        'negative': np.mean([s['negative'] for s in scores]),
        'neutral': np.mean([s['neutral'] for s in scores])
    }

def keyword_score(text):
    """Calculates a normalized hawkish/dovish score."""
    if not isinstance(text, str):
        return 0
    text = text.lower()
    total_tokens = len(text.split())

    hawk_count = sum(text.count(word) for word in HAWKISH_KEYWORDS)
    dove_count = sum(text.count(word) for word in DOVISH_KEYWORDS)

    hawk_freq = hawk_count / (total_tokens + 1e-6)
    dove_freq = dove_count / (total_tokens + 1e-6)
    return (hawk_freq - dove_freq) / (hawk_freq + dove_freq + 1e-6)

def run_sentiment_analysis_pipeline():
    """Applies sentiment and keyword scoring to all datasets."""
    # file_configs = {
    #     "statements": "data/raw/fomc_statements.csv",
    #     "minutes": "data/raw/fomc_minutes.csv",
    #     "speeches": "data/raw/fed_speeches.csv"
    # }
    name = "events"
    path = "data/processed/events_all.csv"
    
    # for name, path in file_configs.items():
    print(f"Scoring {name}...")
    df = pd.read_csv(path)
    
    # FinBERT Scoring
    scores = df['text'].apply(score_document)
    df['positive'] = scores.apply(lambda x: x['positive'] if x else None)
    df['negative'] = scores.apply(lambda x: x['negative'] if x else None)
    df['neutral'] = scores.apply(lambda x: x['neutral'] if x else None)
    
    # Keyword Scoring (using processed text if available)
    preproc_path = "data/preprocessed/events_processed.csv"
    df_proc = pd.read_csv(preproc_path)
    df['keyword_score'] = df_proc['processed_text'].apply(keyword_score)
    
    df.to_csv(f"data/finbertscores/{name}_finbert.csv", index=False)
    print(f"Finished {name}.")

if __name__ == "__main__":
    run_sentiment_analysis_pipeline()