"""
SupportSense NLP - Text Preprocessing & Tokenization Module
Handles regex sanitization, contraction expansion, stopword filtering,
and dual-head TF-IDF vectorization.
"""

import re
import string
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Standard English contractions mapping
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "it's": "it is", "he's": "he is",
    "she's": "she is", "that's": "that is", "what's": "what is",
    "where's": "where is", "there's": "there is", "how's": "how is",
    "'ve": " have", "'re": " are", "'d": " would", "'ll": " will"
}

# Domain-specific stopwords to prune generic fillers while retaining operational intent
DOMAIN_STOPWORDS = {
    "hello", "hi", "hey", "dear", "team", "support", "customer", "service", "please",
    "thanks", "thank", "you", "regards", "fyi", "attention", "greetings", "help",
    "look", "into", "this", "let", "know", "awaiting", "reply", "quick", "asap",
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "was",
    "are", "were", "be", "been", "of", "from", "by", "about", "it", "its"
}

def clean_ticket_text(text: str) -> str:
    """
    Cleans raw customer ticket text:
    - Lowercases text
    - Strips URLs and email patterns
    - Expands colloquial contractions
    - Removes punctuation and extra whitespace
    - Filters filler stopwords while keeping domain signal
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text)
    
    # Expand contractions
    for cont, exp in CONTRACTIONS.items():
        text = text.replace(cont, exp)
        
    # Remove punctuation & special characters (preserve alphanumerics)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove numeric noise / ticket IDs if isolated
    text = re.sub(r"\b\d+\b", "", text)
    
    # Tokenize & filter stopwords
    tokens = text.split()
    filtered_tokens = [t for t in tokens if len(t) > 2 and t not in DOMAIN_STOPWORDS]
    
    return " ".join(filtered_tokens)

def build_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> TfidfVectorizer:
    """Instantiates a production-tuned TF-IDF Vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode"
    )
