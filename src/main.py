"""
Main analysis script - Depression Text Analysis Project
Follows the professor's recommendations:
1. Start with simple word frequency distributions
2. Look for correlations between word frequencies and depression levels
3. Gradually move to more advanced techniques
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from text_preprocessing import TextPreprocessor, CustomStopwords
from frequency_analysis import WordFrequencyAnalyzer, CorrelationAnalyzer
from visualization import DataVisualizer
import config


class DepressionTextAnalysis:
    """Main analysis pipeline."""
    
    def __init__(self):
        """Initialize analysis pipeline."""
        self.data_loader = DataLoader(config.DATA_DIR)
        self.preprocessor = TextPreprocessor(
            remove_stopwords=config.REMOVE_STOPWORDS,
            lemmatize=config.LEMMATIZE
        )
        self.freq_analyzer = WordFrequencyAnalyzer(config.MIN_WORD_FREQUENCY)
        self.corr_analyzer = None
        
        self.corpus_df = None
        self.processed_tokens = None
        self.metadata = None
        
        # Create output directories
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        Path(config.FIGURES_DIR).mkdir(exist_ok=True)
        Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    
    def step1_load_and_explore_data(self):
        """
        Step 1: Load data and compute basic statistics.
        """
        print("\n" + "="*60)
        print("STEP 1: Loading and Exploring Data")
        print("="*60)
        
        # Load labels
        labels = self.data_loader.load_labels(use_processed=True)
        print(f"\nLoaded {len(labels)} participant labels")
        print(f"Columns: {list(labels.columns)}")
        
        # Create corpus with labels
        self.corpus_df, self.metadata = self.data_loader.create_corpus_with_labels()
        
        print("\n--- Corpus Statistics ---")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
        
        # Show sample
        print("\n--- Sample Participants ---")
        sample_indices = [0, 1, 2]
        for idx in sample_indices:
            if idx < len(self.corpus_df):
                row = self.corpus_df.iloc[idx]
                text_preview = row['text'][:100] + "..."
                print(f"\n  Participant {row['Participant_ID']} (PHQ Score: {row['PHQ_Score']}, " 
                      f"Binary: {row['PHQ_Binary']})")
                print(f"  Text: {text_preview}")
        
        return self.corpus_df
    
    def step2_preprocess_text(self):
        """
        Step 2: Preprocess all texts.
        """
        print("\n" + "="*60)
        print("STEP 2: Text Preprocessing")
        print("="*60)
        
        print(f"\nPreprocessing {len(self.corpus_df)} texts...")
        print(f"  - Remove stopwords: {config.REMOVE_STOPWORDS}")
        print(f"  - Lemmatize: {config.LEMMATIZE}")
        
        # Process all texts
        self.processed_tokens = self.preprocessor.process_batch(
            self.corpus_df['text'].values
        )
        
        # Show statistics
        token_counts = [len(tokens) for tokens in self.processed_tokens]
        print(f"\n--- Preprocessing Results ---")
        print(f"  Total documents: {len(self.processed_tokens)}")
        print(f"  Avg tokens per document: {np.mean(token_counts):.1f}")
        print(f"  Max tokens in a document: {np.max(token_counts)}")
        print(f"  Min tokens in a document: {np.min(token_counts)}")
        
        # Show sample processing
        print(f"\n--- Sample Preprocessing ---")
        sample_idx = 0
        print(f"\n  Original text:")
        print(f"    {self.corpus_df.iloc[sample_idx]['text'][:150]}...")
        print(f"\n  Processed tokens (first 20):")
        print(f"    {self.processed_tokens[sample_idx][:20]}")
        
        return self.processed_tokens
    
    def step3_word_frequency_analysis(self):
        """
        Step 3: Analyze word frequencies (professor's first recommendation).
        """
        print("\n" + "="*60)
        print("STEP 3: Word Frequency Analysis")
        print("="*60)
        print("\nThis is the simplest and recommended starting point")
        
        # Compute frequencies
        print("\nComputing word frequencies...")
        freq_counter = self.freq_analyzer.compute_frequencies(self.processed_tokens)
        
        # Filter by minimum frequency
        filtered_freq = self.freq_analyzer.filter_by_frequency(config.MIN_WORD_FREQUENCY)
        
        print(f"\nVocabulary size (min frequency {config.MIN_WORD_FREQUENCY}): "
              f"{len(filtered_freq)} unique words")
        print(f"Total word occurrences: {sum(filtered_freq.values())}")
        
        # Get top words
        top_words = self.freq_analyzer.get_top_words(n=config.TOP_N_WORDS)
        
        print(f"\n--- Top {config.TOP_N_WORDS} Most Frequent Words ---")
        for i, (word, freq) in enumerate(top_words, 1):
            print(f"  {i:2d}. {word:15s} - {freq:4d} occurrences")
        
        # Save figure
        fig_path = Path(config.FIGURES_DIR) / "01_top_words.png"
        DataVisualizer.plot_top_words(top_words, 
                                     title=f"Top {config.TOP_N_WORDS} Most Frequent Words",
                                     save_path=str(fig_path))
        print(f"\nSaved: {fig_path}")
        
        return top_words
    
    def step4_frequency_by_depression_status(self):
        """
        Step 4: Compare word frequencies between depressed and non-depressed groups.
        """
        print("\n" + "="*60)
        print("STEP 4: Word Frequency by Depression Status")
        print("="*60)
        
        # Compute frequencies by group
        phq_binary = self.corpus_df['PHQ_Binary'].values
        self.freq_analyzer.compute_frequencies_by_group(
            self.processed_tokens, 
            phq_binary
        )
        
        # Get top words for each group
        print("\n--- Top Words in Non-Depressed Group (PHQ_Binary=0) ---")
        top_non_depressed = self.freq_analyzer.get_top_words(n=10, group=0)
        for i, (word, freq) in enumerate(top_non_depressed, 1):
            print(f"  {i:2d}. {word:15s} - {freq:4d}")
        
        print("\n--- Top Words in Depressed Group (PHQ_Binary=1) ---")
        top_depressed = self.freq_analyzer.get_top_words(n=10, group=1)
        for i, (word, freq) in enumerate(top_depressed, 1):
            print(f"  {i:2d}. {word:15s} - {freq:4d}")
        
        # Find differentiating words
        print("\n--- Words that Differentiate Groups ---")
        diff_words = self.freq_analyzer.words_differentiating_groups(0, 1, n=10)
        
        print(f"\nMore common in Non-Depressed:")
        for word in diff_words['more_in_group0']:
            ratio = self.freq_analyzer.get_word_frequency_ratio(word, 0, 1)
            print(f"  {word:15s} (ratio: {ratio:.2f}x)")
        
        print(f"\nMore common in Depressed:")
        for word in diff_words['more_in_group1']:
            ratio = self.freq_analyzer.get_word_frequency_ratio(word, 1, 0)
            print(f"  {word:15s} (ratio: {ratio:.2f}x)")
        
        # Create comparison plot
        top_all = self.freq_analyzer.get_top_words(n=15)
        words = [w for w, _ in top_all]
        freq_group0 = [self.freq_analyzer.word_freq_by_group[0].get(w, 0) for w in words]
        freq_group1 = [self.freq_analyzer.word_freq_by_group[1].get(w, 0) for w in words]
        
        fig_path = Path(config.FIGURES_DIR) / "02_frequency_comparison.png"
        DataVisualizer.plot_word_frequency_comparison(
            words, freq_group0, freq_group1,
            group1_label="Non-Depressed (PHQ=0)",
            group2_label="Depressed (PHQ=1)",
            save_path=str(fig_path)
        )
        print(f"\nSaved: {fig_path}")
    
    def step5_correlation_analysis(self):
        """
        Step 5: Analyze correlations between word frequencies and depression levels.
        (Professor's second recommendation)
        """
        print("\n" + "="*60)
        print("STEP 5: Correlation Analysis")
        print("="*60)
        
        print(f"\nAnalyzing correlations with {config.CORRELATION_METHOD} method...")
        
        # Initialize correlation analyzer
        phq_scores = self.corpus_df['PHQ_Score'].values
        phq_binary = self.corpus_df['PHQ_Binary'].values
        
        self.corr_analyzer = CorrelationAnalyzer(self.processed_tokens, phq_scores)
        
        # Build frequency matrix
        self.corr_analyzer.build_frequency_matrix(config.MIN_WORD_FREQUENCY)
        print(f"Built frequency matrix: {self.corr_analyzer.word_freq_matrix.shape}")
        
        # Compute correlations
        self.corr_analyzer.compute_correlations(
            method=config.CORRELATION_METHOD,
            phq_binary=phq_binary
        )
        
        # Get top correlated words
        top_corr_positive = self.corr_analyzer.get_top_correlated_words(
            n=config.TOP_CORRELATED_WORDS, 
            positive=True
        )
        top_corr_negative = self.corr_analyzer.get_top_correlated_words(
            n=config.TOP_CORRELATED_WORDS, 
            positive=False
        )
        
        print(f"\n--- Top Words Positively Correlated with Depression ---")
        print("(Higher frequency = more likely depressed)")
        for i, (word, corr) in enumerate(top_corr_positive, 1):
            print(f"  {i:2d}. {word:15s} - correlation: {corr:+.4f}")
        
        print(f"\n--- Top Words Negatively Correlated with Depression ---")
        print("(Higher frequency = less likely depressed)")
        for i, (word, corr) in enumerate(top_corr_negative, 1):
            print(f"  {i:2d}. {word:15s} - correlation: {corr:+.4f}")
        
        # Create visualization
        all_words = [w for w, _ in top_corr_positive + top_corr_negative]
        all_corr = [c for _, c in top_corr_positive + top_corr_negative]
        
        fig_path = Path(config.FIGURES_DIR) / "03_correlations.png"
        DataVisualizer.plot_correlations(
            all_words, all_corr,
            title=f"Word-Depression Correlations ({config.CORRELATION_METHOD})",
            save_path=str(fig_path)
        )
        print(f"\nSaved: {fig_path}")
    
    def step6_additional_visualizations(self):
        """
        Step 6: Create additional exploratory visualizations.
        """
        print("\n" + "="*60)
        print("STEP 6: Additional Visualizations")
        print("="*60)
        
        phq_scores = self.corpus_df['PHQ_Score'].values
        phq_binary = self.corpus_df['PHQ_Binary'].values
        
        # PHQ distribution
        print("\nCreating PHQ distribution plot...")
        fig_path = Path(config.FIGURES_DIR) / "04_phq_distribution.png"
        DataVisualizer.plot_phq_distribution(phq_scores, phq_binary, 
                                            save_path=str(fig_path))
        print(f"Saved: {fig_path}")
        
        # Text length distribution
        print("Creating text length distribution plot...")
        text_lengths = [len(tokens) for tokens in self.processed_tokens]
        fig_path = Path(config.FIGURES_DIR) / "05_text_length_distribution.png"
        DataVisualizer.plot_text_length_distribution(text_lengths, phq_binary,
                                                    save_path=str(fig_path))
        print(f"Saved: {fig_path}")
    
    def step7_save_results(self):
        """
        Step 7: Save detailed results to CSV files.
        """
        print("\n" + "="*60)
        print("STEP 7: Saving Results")
        print("="*60)
        
        # Save corpus with processed tokens
        results_df = self.corpus_df.copy()
        results_df['tokens_count'] = [len(t) for t in self.processed_tokens]
        results_df['processed_tokens'] = [' '.join(t) for t in self.processed_tokens]
        
        results_path = Path(config.RESULTS_DIR) / "corpus_with_tokens.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved corpus: {results_path}")
        
        # Save word frequencies
        if self.freq_analyzer.word_freq:
            freq_df = pd.DataFrame(
                self.freq_analyzer.word_freq.most_common(),
                columns=['word', 'frequency']
            )
            freq_path = Path(config.RESULTS_DIR) / "word_frequencies.csv"
            freq_df.to_csv(freq_path, index=False)
            print(f"Saved word frequencies: {freq_path}")
        
        # Save correlations
        if self.corr_analyzer and self.corr_analyzer.word_correlation:
            corr_df = pd.DataFrame(
                sorted(self.corr_analyzer.word_correlation.items(), 
                      key=lambda x: abs(x[1]), reverse=True),
                columns=['word', 'correlation']
            )
            corr_path = Path(config.RESULTS_DIR) / "word_correlations.csv"
            corr_df.to_csv(corr_path, index=False)
            print(f"Saved correlations: {corr_path}")
        
        # Save metadata
        metadata_df = pd.DataFrame([self.metadata])
        metadata_path = Path(config.RESULTS_DIR) / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Saved metadata: {metadata_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*70)
        print("  DEPRESSION TEXT ANALYSIS - WORD FREQUENCY & CORRELATION ANALYSIS")
        print("="*70)
        print("\nApproach: Following professor's recommendations")
        print("  1. Start with simple word frequency distributions")
        print("  2. Look for correlations with depression levels")
        print("  3. Gradually move to advanced techniques")
        
        try:
            # Execute all steps
            self.step1_load_and_explore_data()
            self.step2_preprocess_text()
            self.step3_word_frequency_analysis()
            self.step4_frequency_by_depression_status()
            self.step5_correlation_analysis()
            self.step6_additional_visualizations()
            self.step7_save_results()
            
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE!")
            print("="*70)
            print(f"\nOutput files saved to:")
            print(f"  Figures: {Path(config.FIGURES_DIR).absolute()}")
            print(f"  Results: {Path(config.RESULTS_DIR).absolute()}")
            
        except Exception as e:
            print(f"\nERROR during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


if __name__ == "__main__":
    analysis = DepressionTextAnalysis()
    success = analysis.run_full_analysis()
    sys.exit(0 if success else 1)
