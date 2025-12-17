#!/usr/bin/env python3
"""
QUICK START SCRIPT - Run this first!
This script demonstrates the basic analysis workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_banner(text):
    """Print formatted banner"""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def main():
    print_banner("DEPRESSION TEXT ANALYSIS - QUICK START")
    
    # Step 1: Check data
    print("\n✓ Step 1: Checking data files...")
    data_path = Path("data/labels/processed/all_participants_phq_binary.csv")
    if data_path.exists():
        print(f"  ✓ Labels file found: {data_path}")
    else:
        print(f"  ✗ Labels file NOT found at {data_path}")
        return False
    
    raw_path = Path("data/raw")
    if raw_path.exists():
        transcript_count = len(list(raw_path.glob("*_TRANSCRIPT.csv")))
        print(f"  ✓ Found {transcript_count} transcript files")
    else:
        print(f"  ✗ Raw data folder NOT found at {raw_path}")
        return False
    
    # Step 2: Load data
    print("\n✓ Step 2: Loading data...")
    try:
        from data_loader import DataLoader
        loader = DataLoader()
        corpus_df, metadata = loader.create_corpus_with_labels()
        
        print(f"  ✓ Loaded {metadata['n_total_participants']} participants")
        print(f"    - Depressed: {metadata['n_depressed']}")
        print(f"    - Non-depressed: {metadata['n_non_depressed']}")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return False
    
    # Step 3: Preprocess
    print("\n✓ Step 3: Text preprocessing...")
    try:
        from text_preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor(remove_stopwords=True)
        processed_tokens = preprocessor.process_batch(corpus_df['text'].values)
        
        avg_tokens = sum(len(t) for t in processed_tokens) / len(processed_tokens)
        print(f"  ✓ Processed {len(processed_tokens)} texts")
        print(f"    - Average tokens per text: {avg_tokens:.0f}")
    except Exception as e:
        print(f"  ✗ Error preprocessing: {e}")
        return False
    
    # Step 4: Analyze frequencies
    print("\n✓ Step 4: Word frequency analysis...")
    try:
        from frequency_analysis import WordFrequencyAnalyzer
        analyzer = WordFrequencyAnalyzer()
        freq = analyzer.compute_frequencies(processed_tokens)
        top_words = analyzer.get_top_words(n=10)
        
        print(f"  ✓ Unique words found: {len(freq)}")
        print(f"  Top 5 words:")
        for i, (word, count) in enumerate(top_words[:5], 1):
            print(f"    {i}. {word:15s} ({count} occurrences)")
    except Exception as e:
        print(f"  ✗ Error analyzing frequencies: {e}")
        return False
    
    # Step 5: Correlation
    print("\n✓ Step 5: Correlation analysis...")
    try:
        from frequency_analysis import CorrelationAnalyzer
        phq_scores = corpus_df['PHQ_Score'].values
        phq_binary = corpus_df['PHQ_Binary'].values
        
        corr_analyzer = CorrelationAnalyzer(processed_tokens, phq_scores)
        corr_analyzer.build_frequency_matrix()
        corr_analyzer.compute_correlations(method='point-biserial', phq_binary=phq_binary)
        
        top_corr = corr_analyzer.get_top_correlated_words(n=5, positive=True)
        print(f"  ✓ Computed correlations")
        print(f"  Top words associated with depression:")
        for i, (word, corr) in enumerate(top_corr, 1):
            print(f"    {i}. {word:15s} (r={corr:+.4f})")
    except Exception as e:
        print(f"  ✗ Error computing correlations: {e}")
        return False
    
    # Success!
    print_banner("QUICK START COMPLETED SUCCESSFULLY!")
    
    print("""
Next steps:

1. INTERACTIVE ANALYSIS (Recommended):
   jupyter notebook notebooks/00_analysis.ipynb
   
2. FULL ANALYSIS SCRIPT:
   python src/main.py
   
3. READ FULL GUIDE:
   cat PROJECT_GUIDE.md
   
4. EXPLORE MODULES:
   - src/data_loader.py          : Load data
   - src/text_preprocessing.py   : Clean text
   - src/frequency_analysis.py   : Analyze words
   - src/visualization.py         : Create plots
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
