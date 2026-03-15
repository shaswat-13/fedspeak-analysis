import pandas as pd
import spacy
import nltk
import re
from nltk.corpus import stopwords

# Configuration
FINANCIAL_TERMS = {
    'inflation', 'rate', 'unemployment', 'monetary', 'policy', 
    'interest', 'economic', 'growth', 'market', 'federal', 
    'reserve', 'committee', 'financial', 'price', 'stable',
    'employment', 'target', 'increase', 'decrease', 'raise',
    'cut', 'hike', 'tighten', 'accommodate', 'neutral'
}

def get_stop_words():
    """Initializes and returns the filtered stop words set."""
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    return stop_words - FINANCIAL_TERMS

# Initialize global resources
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

STOP_WORDS = get_stop_words()

def preprocess_text(text):
    """Cleans, tokenizes, lemmatizes, and removes stopwords."""
    if not isinstance(text, str) or nlp is None:
        return None
    
    # Cleaning
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\%\,]', '', text)
    text = text.strip()
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extraction & Lemmatization
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct 
        and not token.is_space 
        and token.lemma_.lower() not in STOP_WORDS 
        and len(token.text) > 2
    ]
    
    return ' '.join(tokens)

def run_preprocessing_pipeline():
    """Loads raw data, processes it, and saves the output."""
    
    name = "events"
    path = "data/processed/events_all.csv"
    

    print(f"Processing {name}...")
    df = pd.read_csv(path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    output_path = f"data/preprocessed/{name}_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing_pipeline()