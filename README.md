# Depression Text Analysis

A comprehensive text analysis project exploring linguistic patterns in psychologist-patient conversations to understand correlations between word usage and depression levels.

## 📊 Project Overview

This project analyzes **189 interview transcripts** from the AVEC 2017 Depression Recognition Challenge to identify words and linguistic patterns associated with depression using both simple and advanced text analysis techniques.

**Dataset:**
- 189 participants (57 depressed, 132 non-depressed)
- PHQ-8 depression scores with binary classification
- Interview transcripts with psychologist-patient conversations

**Analysis Approach** (following professor's recommendations):
1. ✅ Start with simple word frequency distributions
2. ✅ Analyze correlations between word frequencies and depression levels
3. ✅ Progress to advanced text processing techniques

---

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Quick verification (1-2 minutes)
python3 quick_start.py

# 3. Run full analysis pipeline
python3 src/main.py

# 4. Explore interactively (optional)
jupyter notebook notebooks/00_analysis.ipynb
```

---

## 📁 Project Structure

```
depression-text-analysis/
├── data/
│   ├── raw/                    # 189 interview transcripts
│   └── labels/processed/       # PHQ-8 depression scores
│
├── src/                        # Analysis modules
│   ├── data_loader.py          # Load transcripts & labels
│   ├── text_preprocessing.py   # Clean & tokenize text
│   ├── frequency_analysis.py   # Word frequencies & correlations
│   ├── advanced_analysis.py    # TF-IDF, N-grams, sentiment
│   ├── visualization.py        # Create plots
│   ├── config.py              # Configuration settings
│   └── main.py                # Main analysis pipeline
│
├── notebooks/
│   └── 00_analysis.ipynb      # Interactive exploration
│
├── output/                     # Generated results
│   ├── figures/               # PNG visualizations
│   └── results/               # CSV data files
│
├── scripts/
│   └── clean_labels.py        # Data preprocessing utilities
│
├── quick_start.py             # Quick verification script
├── setup_nltk.py              # NLTK data setup
└── requirements.txt           # Python dependencies
```

---

## 🔬 Analysis Pipeline

The project implements a **13-step analysis pipeline**:

### Basic Analysis (Steps 1-7)
1. **Load Data** - Read transcripts and depression labels
2. **Preprocess** - Clean text, remove stopwords, tokenize
3. **Word Frequencies** - Compute most common words
4. **Group Comparison** - Compare depressed vs non-depressed
5. **Correlation Analysis** - Find depression-associated words
6. **Visualizations** - Create publication-quality plots
7. **Export Results** - Save CSV files

### Advanced Analysis (Steps 8-13)
8. **TF-IDF Analysis** - Weight words by document importance
9. **N-grams** - Analyze word sequences (bigrams, trigrams)
10. **Sentiment Analysis** - Measure emotional tone
11. **Linguistic Features** - Extract pronouns, negations, readability
12. **Statistical Testing** - Chi-square, effect sizes, significance
13. **Report Generation** - Auto-generate findings summary

---

## 📈 Key Findings

**Words Most Associated with Depression** (positive correlation):
- "couldnt" (r=+0.278)
- "depressed" (r=+0.277)
- "tools" (r=+0.276)
- "forget" (r=+0.272)
- "psychiatrist" (r=+0.255)

**Dataset Statistics:**
- Total unique words: ~8,474
- Average tokens per participant: 756 words
- Depression prevalence: 30% (57/189)

---

## ⚙️ Customization

Edit `src/config.py` to customize analysis:

```python
# Text preprocessing
REMOVE_STOPWORDS = True      # Remove common words
LEMMATIZE = False            # Convert to base forms
MIN_WORD_LENGTH = 2          # Minimum word length

# Analysis parameters
MIN_WORD_FREQUENCY = 2       # Ignore rare words
TOP_N_WORDS = 20             # Top words to display
CORRELATION_METHOD = 'point-biserial'

# Advanced features
ENABLE_TFIDF = True          # TF-IDF analysis
ENABLE_NGRAMS = True         # N-grams analysis
ENABLE_SENTIMENT = True      # Sentiment analysis
NGRAM_RANGE = (2, 3)         # Bigrams and trigrams
```

---

## 📊 Output Files

Running the analysis generates:

| File | Description |
|------|-------------|
| `top_words.csv` | Most frequent words across all interviews |
| `word_frequencies_by_group.csv` | Word frequencies by depression status |
| `correlations.csv` | Statistical correlations with depression |
| `tfidf_features.csv` | TF-IDF importance scores |
| `ngrams_analysis.csv` | Top n-grams by group |
| `sentiment_scores.csv` | Sentiment analysis results |
| `corpus_analysis.csv` | Text statistics per participant |
| `figures/*.png` | 10+ publication-quality visualizations |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| **pandas** | Data manipulation & analysis |
| **NLTK** | Natural language processing |
| **scikit-learn** | TF-IDF, machine learning utilities |
| **scipy** | Statistical tests & correlations |
| **matplotlib/seaborn** | Visualizations |
| **wordcloud** | Word cloud generation |
| **Jupyter** | Interactive notebooks |

---

## 🎓 For Semester Project Submission

**Include in your report:**

1. **Dataset Description**
   - 189 participants from AVEC 2017 Challenge
   - PHQ-8 depression scores
   - Binary classification (depressed/non-depressed)

2. **Methodology**
   - Text preprocessing steps
   - Word frequency analysis
   - Correlation analysis (point-biserial)
   - Advanced techniques (TF-IDF, N-grams, sentiment)

3. **Results**
   - Top depression-associated words
   - Statistical significance (p-values, effect sizes)
   - Visualizations from `output/figures/`

4. **Discussion**
   - Interpretation of findings
   - Limitations (sample size, binary classification)
   - Future work (machine learning, deep learning)

5. **Code & Outputs**
   - Python source code or Jupyter notebook
   - Generated CSV files
   - Publication-quality figures

---

## 🔧 Troubleshooting

**"NLTK data not found"**
```bash
python3 setup_nltk.py
```

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"No output directory"**
The `output/` directory is created automatically when running `src/main.py`

**SSL Certificate Error (NLTK)**
This is a warning and doesn't affect analysis. The project works without WordNet lemmatization.

---

## 📚 How It Works

### 1. Data Loading
```python
from src.data_loader import DataLoader
loader = DataLoader()
corpus_df, metadata = loader.create_corpus_with_labels()
```

### 2. Text Preprocessing
```python
from src.text_preprocessing import TextPreprocessor
preprocessor = TextPreprocessor(remove_stopwords=True)
tokens = preprocessor.process_batch(texts)
```

### 3. Frequency Analysis
```python
from src.frequency_analysis import WordFrequencyAnalyzer
analyzer = WordFrequencyAnalyzer()
word_freq = analyzer.compute_frequencies(tokens)
top_words = analyzer.get_top_words(n=20)
```

### 4. Correlation Analysis
```python
from src.frequency_analysis import CorrelationAnalyzer
corr_analyzer = CorrelationAnalyzer(tokens, phq_scores)
corr_analyzer.compute_correlations(method='point-biserial')
top_correlated = corr_analyzer.get_top_correlated_words(n=20)
```

### 5. Advanced Analysis
```python
from src.advanced_analysis import TFIDFAnalyzer, NGramAnalyzer, SentimentAnalyzer

# TF-IDF
tfidf = TFIDFAnalyzer()
tfidf.fit_transform(texts)
top_features = tfidf.get_top_features_by_group(labels)

# N-grams
ngram = NGramAnalyzer(n=2)
ngram.compute_ngrams_by_group(tokens, labels)
top_bigrams = ngram.get_top_ngrams(n=20)

# Sentiment
sentiment = SentimentAnalyzer()
sentiment_scores = sentiment.analyze_batch(tokens)
```

---

## 🎯 Next Steps

1. **Machine Learning**: Train classifiers (Logistic Regression, SVM, Random Forest)
2. **Deep Learning**: Experiment with LSTM, BERT, or GPT models
3. **Feature Engineering**: Extract more linguistic features (POS tags, dependency parsing)
4. **Cross-validation**: Implement k-fold validation for robust evaluation
5. **Explainability**: Use SHAP or LIME to interpret model predictions

---

## 📄 License

This project uses the AVEC 2017 Depression Recognition Challenge dataset. Please cite the original dataset if using this code for research.

---

## 🤝 Contributing

This is a semester project for data analysis class. Feel free to fork and extend for your own research!

---

## 📞 Support

For questions about running the project:
1. Check the troubleshooting section above
2. Review `quick_start.py` for a working example
3. Explore `notebooks/00_analysis.ipynb` for interactive examples

**Happy Analyzing! 📊🔬**
