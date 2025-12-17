# How to Run This Project

## Quick Start (2 minutes)

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify everything works
python3 quick_start.py

# 3. Open interactive notebook (optional)
jupyter notebook notebooks/00_analysis.ipynb

# 4. Or run full analysis
python3 src/main.py
```

## What You Get

After running, you'll find:
- **output/figures/** - 5 PNG plots showing analysis results
- **output/results/** - CSV files with findings
- **Console output** - Statistical results printed to terminal

## Key Commands

| Command | What It Does |
|---------|-------------|
| `python3 quick_start.py` | Tests all components (1-2 min) - good for verification |
| `python3 src/main.py` | Full analysis pipeline (2-3 min) |
| `jupyter notebook notebooks/00_analysis.ipynb` | Interactive notebook with explanations |

## Customization

Edit `src/config.py` to change:
- `REMOVE_STOPWORDS` - Remove common words like "the", "a", "is"
- `LEMMATIZE` - Convert words to base form (better/best → good)
- `MIN_WORD_FREQUENCY` - Ignore words appearing less than N times
- `TOP_N_WORDS` - How many top words to show
- `CORRELATION_METHOD` - 'point-biserial' or 'spearman'

Example:
```python
# src/config.py
REMOVE_STOPWORDS = False  # Keep all words
LEMMATIZE = True          # Convert to base form
MIN_WORD_FREQUENCY = 5    # Only words appearing 5+ times
```

Then run: `python3 src/main.py`

## Troubleshooting

**"NLTK data not found"**
```bash
python3 setup_nltk.py
```

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"Python command not found"**
```bash
python3 src/main.py  # Use python3 instead of python
```

## File Structure

```
src/
  ├── data_loader.py      # Loads transcripts & labels
  ├── text_preprocessing.py # Cleans text
  ├── frequency_analysis.py # Analyzes words & correlations
  ├── visualization.py     # Creates plots
  ├── config.py           # All settings in one place
  └── main.py             # Runs complete pipeline

notebooks/
  └── 00_analysis.ipynb   # Interactive exploration

data/
  ├── raw/                # Interview transcripts
  └── labels/processed/   # PHQ depression scores
```

## Dataset

- **189 interview transcripts** from AVEC 2017 Depression Challenge
- **PHQ-8 scores** for each participant (depression assessment)
- **Binary classification**: 57 depressed, 132 non-depressed
- **Text**: Cleaned interview conversations

## Output Explained

The analysis produces:

1. **top_words.csv** - Most frequent words overall
2. **word_frequencies_by_group.csv** - Words by depression status
3. **correlations.csv** - Words most associated with depression
4. **corpus_analysis.csv** - Text statistics per participant
5. **analysis_metadata.txt** - Summary of analysis settings

## For Your Semester Project

Include in your report:
- The 5 generated plots from `output/figures/`
- Top correlation findings from `output/results/correlations.csv`
- Dataset description (189 participants, PHQ-8 scores)
- Methodology (text preprocessing steps used)
- Discussion of depression-associated words found

Copy the generated files into your project report folder and reference them in your writeup.
