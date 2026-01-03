"""
Configuration file for depression text analysis project.
"""

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
LABELS_DIR = "data/labels"
PROCESSED_LABELS_DIR = "data/labels/processed"

# Output directories
OUTPUT_DIR = "output"
FIGURES_DIR = "output/figures"
RESULTS_DIR = "output/results"

# Text preprocessing parameters
REMOVE_STOPWORDS = True
LEMMATIZE = False
MIN_WORD_LENGTH = 2

# Analysis parameters
MIN_WORD_FREQUENCY = 2  # Minimum occurrences for a word to be included
TOP_N_WORDS = 20  # Number of top words to display in visualizations

# Correlation analysis
CORRELATION_METHOD = 'point-biserial'  # or 'spearman'
TOP_CORRELATED_WORDS = 25

# Visualization parameters
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FIGURE_SIZE = (12, 6)

# Random seed for reproducibility
RANDOM_SEED = 42

# Advanced analysis features
ENABLE_TFIDF = True
ENABLE_NGRAMS = True
ENABLE_SENTIMENT = True
ENABLE_WORDCLOUDS = True

# TF-IDF parameters
TFIDF_MAX_FEATURES = 100
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8

# N-gram parameters
NGRAM_RANGE = (2, 3)  # (min_n, max_n) - bigrams and trigrams
TOP_NGRAMS = 20

# Sentiment analysis
SENTIMENT_METHOD = 'lexicon'  # 'lexicon' or 'vader' (if available)

