"""
Word frequency analysis module.
Analyzes word distributions and correlations with depression levels.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from scipy.stats import pointbiserialr, spearmanr
import warnings

warnings.filterwarnings('ignore')


class WordFrequencyAnalyzer:
    """Analyze word frequencies in text data."""
    
    def __init__(self, min_frequency: int = 2):
        """
        Initialize analyzer.
        
        Args:
            min_frequency: Minimum occurrences for a word to be included
        """
        self.min_frequency = min_frequency
        self.word_freq = Counter()
        self.word_freq_by_group = defaultdict(Counter)
        
    def compute_frequencies(self, token_lists: List[List[str]]) -> Counter:
        """
        Compute word frequencies across all texts.
        
        Args:
            token_lists: List of token lists
            
        Returns:
            Counter with word frequencies
        """
        self.word_freq = Counter()
        
        for tokens in token_lists:
            self.word_freq.update(tokens)
        
        return self.word_freq
    
    def compute_frequencies_by_group(self, 
                                     token_lists: List[List[str]], 
                                     groups: List[int]) -> Dict[int, Counter]:
        """
        Compute word frequencies for each group.
        
        Args:
            token_lists: List of token lists
            groups: Group label for each text (e.g., depression binary: 0 or 1)
            
        Returns:
            Dictionary mapping group -> Counter of word frequencies
        """
        self.word_freq_by_group = defaultdict(Counter)
        
        for tokens, group in zip(token_lists, groups):
            self.word_freq_by_group[group].update(tokens)
        
        return self.word_freq_by_group
    
    def get_top_words(self, n: int = 20, group: int = None) -> List[Tuple[str, int]]:
        """
        Get top N most frequent words.
        
        Args:
            n: Number of top words to return
            group: If specified, get top words for that group only
            
        Returns:
            List of (word, frequency) tuples
        """
        if group is not None:
            freq_counter = self.word_freq_by_group.get(group, Counter())
        else:
            freq_counter = self.word_freq
        
        return freq_counter.most_common(n)
    
    def filter_by_frequency(self, min_freq: int = None) -> Counter:
        """
        Filter words by minimum frequency.
        
        Args:
            min_freq: Minimum frequency threshold
            
        Returns:
            Filtered Counter
        """
        min_freq = min_freq or self.min_frequency
        return Counter({word: freq for word, freq in self.word_freq.items() 
                       if freq >= min_freq})
    
    def get_word_frequency_ratio(self, word: str, 
                                 group1: int, 
                                 group2: int) -> float:
        """
        Compute ratio of word frequency between two groups.
        
        Args:
            word: Word to analyze
            group1: First group label
            group2: Second group label
            
        Returns:
            Ratio of frequencies (group1/group2)
        """
        freq1 = self.word_freq_by_group[group1].get(word, 0)
        freq2 = self.word_freq_by_group[group2].get(word, 0)
        
        if freq2 == 0:
            return freq1 if freq1 == 0 else float('inf')
        
        return freq1 / freq2
    
    def words_differentiating_groups(self, 
                                     group1: int, 
                                     group2: int,
                                     n: int = 20,
                                     min_freq_total: int = 3) -> Dict[str, Dict]:
        """
        Find words that best differentiate between two groups.
        Uses normalized frequencies (per participant) to account for class imbalance.
        
        Args:
            group1: First group label
            group2: Second group label
            n: Number of differentiating words to return
            min_freq_total: Minimum total frequency across both groups
            
        Returns:
            Dictionary with 'more_in_group#' lists
        """
        freq1 = self.word_freq_by_group[group1]
        freq2 = self.word_freq_by_group[group2]
        
        # Get group sizes for normalization
        # Estimate from total word counts
        total1 = sum(freq1.values())
        total2 = sum(freq2.values())
        
        all_words = set(freq1.keys()) | set(freq2.keys())
        
        # Calculate normalized difference for each word
        word_scores = {}
        for word in all_words:
            f1 = freq1.get(word, 0)
            f2 = freq2.get(word, 0)
            
            # Skip very rare words
            if f1 + f2 < min_freq_total:
                continue
            
            # Normalize by group size (proportional frequency)
            norm_f1 = f1 / total1 if total1 > 0 else 0
            norm_f2 = f2 / total2 if total2 > 0 else 0
            
            # Calculate normalized difference
            # Positive = more in group1, Negative = more in group2
            diff = norm_f1 - norm_f2
            
            # Also track ratio for ranking
            if f2 > 0:
                ratio = f1 / f2
            elif f1 > 0:
                ratio = float('inf')
            else:
                ratio = 1.0
            
            word_scores[word] = {
                'diff': diff,
                'ratio': ratio,
                'freq1': f1,
                'freq2': f2
            }
        
        # Sort by absolute difference (most distinctive words)
        sorted_by_diff = sorted(word_scores.items(), key=lambda x: x[1]['diff'])
        
        # Words more common in group2 (negative diff)
        group2_words = sorted_by_diff[:n]
        # Words more common in group1 (positive diff)  
        group1_words = sorted_by_diff[-n:][::-1]
        
        return {
            f'more_in_group{group1}': [w[0] for w in group1_words],
            f'more_in_group{group2}': [w[0] for w in group2_words],
        }


class CorrelationAnalyzer:
    """Analyze correlations between word frequencies and depression levels."""
    
    def __init__(self, token_lists: List[List[str]], phq_scores: List[float]):
        """
        Initialize analyzer.
        
        Args:
            token_lists: List of token lists
            phq_scores: PHQ depression scores for each text
        """
        self.token_lists = token_lists
        self.phq_scores = np.array(phq_scores)
        self.word_freq_matrix = None
        self.word_correlation = {}
    
    def build_frequency_matrix(self, min_frequency: int = 2) -> pd.DataFrame:
        """
        Build word frequency matrix for correlation analysis.
        
        Args:
            min_frequency: Minimum word frequency
            
        Returns:
            DataFrame with words as columns and documents as rows
        """
        # Count word frequencies across all documents
        all_words = Counter()
        for tokens in self.token_lists:
            all_words.update(tokens)
        
        # Filter by frequency
        words = [w for w, f in all_words.items() if f >= min_frequency]
        
        # Build frequency matrix
        matrix = []
        for tokens in self.token_lists:
            token_counter = Counter(tokens)
            row = [token_counter.get(word, 0) for word in words]
            matrix.append(row)
        
        self.word_freq_matrix = pd.DataFrame(matrix, columns=words)
        return self.word_freq_matrix
    
    def compute_correlations(self, method: str = 'point-biserial',
                           phq_binary: List[int] = None) -> Dict[str, float]:
        """
        Compute correlation between word frequency and PHQ score.
        
        Args:
            method: 'point-biserial' (with binary PHQ) or 'spearman' (with continuous PHQ)
            phq_binary: Binary depression labels (0/1) if using point-biserial
            
        Returns:
            Dictionary mapping word -> correlation coefficient
        """
        if self.word_freq_matrix is None:
            self.build_frequency_matrix()
        
        self.word_correlation = {}
        
        if method == 'point-biserial' and phq_binary is not None:
            phq_data = np.array(phq_binary)
            for word in self.word_freq_matrix.columns:
                word_freq = self.word_freq_matrix[word].values
                try:
                    corr, _ = pointbiserialr(phq_data, word_freq)
                    self.word_correlation[word] = corr
                except:
                    self.word_correlation[word] = 0.0
        
        elif method == 'spearman':
            for word in self.word_freq_matrix.columns:
                word_freq = self.word_freq_matrix[word].values
                try:
                    corr, _ = spearmanr(self.phq_scores, word_freq)
                    self.word_correlation[word] = corr if not np.isnan(corr) else 0.0
                except:
                    self.word_correlation[word] = 0.0
        
        return self.word_correlation
    
    def get_top_correlated_words(self, n: int = 20, 
                                positive: bool = True) -> List[Tuple[str, float]]:
        """
        Get top N words correlated with depression.
        
        Args:
            n: Number of words
            positive: If True, get positive correlations; if False, negative
            
        Returns:
            List of (word, correlation) tuples
        """
        if not self.word_correlation:
            raise ValueError("Must run compute_correlations first")
        
        sorted_words = sorted(self.word_correlation.items(), 
                            key=lambda x: x[1], 
                            reverse=positive)
        
        return sorted_words[:n]


if __name__ == "__main__":
    # Test with sample data
    sample_tokens = [
        ['sad', 'lonely', 'tired', 'depressed'],
        ['happy', 'excited', 'energetic', 'sad'],
        ['anxious', 'worried', 'sad', 'tired'],
    ]
    sample_phq = [1, 0, 1]
    
    # Test word frequency
    analyzer = WordFrequencyAnalyzer()
    freq = analyzer.compute_frequencies(sample_tokens)
    print("Top words:", analyzer.get_top_words(5))
    
    # Test correlation
    corr_analyzer = CorrelationAnalyzer(sample_tokens, sample_phq)
    corr_analyzer.build_frequency_matrix()
    corr_analyzer.compute_correlations(method='spearman')
    print("\nTop correlated words:", corr_analyzer.get_top_correlated_words(5, positive=True))
