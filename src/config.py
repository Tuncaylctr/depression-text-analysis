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
