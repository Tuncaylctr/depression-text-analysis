# Depression Text Analysis - Project Overview

## What This Project Does

Analyzes interview transcripts to find correlations between word usage and depression levels. Follows a simple-to-advanced progression: starts with word frequency analysis, then finds depression-associated words using correlation, establishing foundation for machine learning.

**Real-world application**: Understand which speech patterns are associated with depression in clinical interviews.

## The Data

| Item | Details |
|------|---------|
| Interviews | 189 transcripts from AVEC 2017 Depression Challenge |
| Participants | 57 depressed (30%), 132 non-depressed (70%) |
| Depression Scale | PHQ-8 scores converted to binary classification |
| Text Format | Interview transcripts with speaker turns, timestamps |
| Total Words | ~8,474 unique words after preprocessing |

## Analysis Pipeline (7 Steps)

1. **Load Data** - Read 189 transcripts and depression labels
2. **Preprocess** - Clean text, remove stopwords, tokenize (~8,500 unique words)
3. **Word Frequencies** - Find most common words overall
4. **Group Analysis** - Compare word usage: depressed vs non-depressed speakers
5. **Correlation** - Statistical analysis: which words correlate with depression
6. **Visualizations** - Create 5+ publication-quality plots
7. **Export Results** - Save CSV files with findings

## Key Findings

Words most associated with depression (from testing):
- "couldn't" (correlation: +0.278)
- "depressed" (correlation: +0.277)
- "tools" (correlation: +0.276)
- "forget" (correlation: +0.272)
- "psychiatrist" (correlation: +0.255)

(Positive correlation = more common in depressed participants)

## Project Structure

```
src/                          # Python modules (production code)
├── data_loader.py           # DataLoader: loads transcripts + labels
├── text_preprocessing.py     # TextPreprocessor: cleans & prepares text
├── frequency_analysis.py     # WordFrequencyAnalyzer & CorrelationAnalyzer
├── visualization.py          # DataVisualizer: creates plots
├── config.py                # Configuration (all settings in one place)
└── main.py                  # DepressionTextAnalysis: 7-step pipeline

notebooks/
└── 00_analysis.ipynb        # Interactive notebook: explore, visualize, learn

scripts/
├── quick_start.py           # Test script: verifies all components work
└── setup_nltk.py            # Setup: downloads required NLTK data

output/                       # Auto-generated results
├── figures/                 # PNG plots
└── results/                 # CSV files with analysis results

data/                        # Dataset
├── raw/                     # 189 interview transcripts
└── labels/processed/        # PHQ-8 depression scores
```

## Technologies Used

| Component | Purpose |
|-----------|---------|
| **pandas** | Data manipulation & analysis |
| **NLTK** | Natural language processing (tokenization, stopwords) |
| **scikit-learn** | TF-IDF, correlation utilities |
| **scipy** | Statistical functions (point-biserial correlation) |
| **matplotlib/seaborn** | Publication-quality visualizations |
| **Jupyter** | Interactive notebook environment |

## How to Run

**Minimal (verify it works):**
```bash
source .venv/bin/activate
python3 quick_start.py
```

**Full analysis:**
```bash
source .venv/bin/activate
python3 src/main.py
```

**Interactive:**
```bash
source .venv/bin/activate
jupyter notebook notebooks/00_analysis.ipynb
```

See `RUN_PROJECT.md` for detailed instructions, troubleshooting, and customization options.

## Customization

All settings in `src/config.py`:

```python
# Text processing
REMOVE_STOPWORDS = True      # Remove "the", "a", "is"?
LEMMATIZE = False            # Convert to base forms?
MIN_WORD_LENGTH = 2          # Minimum word length

# Analysis
MIN_WORD_FREQUENCY = 2       # Ignore rare words
TOP_N_WORDS = 20             # How many top words to display
TOP_CORRELATED_WORDS = 25    # Depression-associated words

# Correlation
CORRELATION_METHOD = 'point-biserial'  # With binary depression
RANDOM_SEED = 42             # For reproducibility
```

Change any setting and re-run `python3 src/main.py` to see effects.

## Output Files

Running the analysis creates:

| File | Contains |
|------|----------|
| `top_words.csv` | Most frequent words across all interviews |
| `word_frequencies_by_group.csv` | Word frequencies separated by depression status |
| `correlations.csv` | Statistical correlation of each word with depression |
| `corpus_analysis.csv` | Text statistics (length, unique words) per participant |
| `analysis_metadata.txt` | Configuration & summary statistics |
| `figures/*.png` | 5 plots: top words, group comparison, correlations, etc. |

## Code Design

**Modular architecture** - Each class handles one responsibility:
- `DataLoader` - Data I/O
- `TextPreprocessor` - Text cleaning
- `WordFrequencyAnalyzer` - Frequency computation
- `CorrelationAnalyzer` - Statistical analysis
- `DataVisualizer` - Plotting
- `DepressionTextAnalysis` - Orchestration

**Easy to extend** - Add new analysis methods without changing existing code:
```python
# Example: add a new analysis method
class MyAnalyzer:
    def __init__(self, corpus_df):
        self.corpus_df = corpus_df
    
    def my_analysis(self):
        # your code here
        pass

# Use it
from src.data_loader import DataLoader
loader = DataLoader()
corpus = loader.create_corpus_with_labels()
analyzer = MyAnalyzer(corpus)
analyzer.my_analysis()
```

## For Your Semester Project

**What to include:**
1. ✅ Dataset description (189 participants, text format)
2. ✅ Methodology (preprocessing steps, analysis approach)
3. ✅ 5 generated plots from `output/figures/`
4. ✅ Key findings (top depression-associated words)
5. ✅ Discussion (interpret findings, limitations)
6. ✅ Code (include `src/main.py` or notebook)

**Deliverables:**
- Written report (PDF)
- Python code or Jupyter notebook
- Generated figures (PNGs)
- CSV result files
- README with how to run

The project is production-ready: clean architecture, documented, tested, and generates publication-quality outputs suitable for academic submission.

## Next Steps

1. **Quick test**: `python3 quick_start.py` (1-2 minutes)
2. **Explore**: Open `notebooks/00_analysis.ipynb` in Jupyter
3. **Full run**: `python3 src/main.py` (generates all outputs)
4. **Customize**: Edit `src/config.py`, re-run analysis
5. **Submit**: Copy outputs to your report

Questions? See `RUN_PROJECT.md` for troubleshooting and detailed instructions.
