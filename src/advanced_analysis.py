"""
Advanced text analysis module.
Provides TF-IDF, N-grams, sentiment analysis, and statistical testing.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')


class TFIDFAnalyzer:
    """Analyze text using TF-IDF (Term Frequency-Inverse Document Frequency)."""
    
    def __init__(self, max_features: int = 100, min_df: int = 2, max_df: float = 0.8):
        """
        Initialize TF-IDF analyzer.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as proportion)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def get_top_features_by_group(self, labels: np.ndarray, n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top TF-IDF features for each group.
        
        Args:
            labels: Group labels for each document
            n: Number of top features to return
            
        Returns:
            Dictionary mapping group -> list of (feature, score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        results = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get mean TF-IDF scores for this group
            group_mask = labels == label
            group_tfidf = self.tfidf_matrix[group_mask].mean(axis=0).A1
            
            # Get top features
            top_indices = np.argsort(group_tfidf)[-n:][::-1]
            top_features = [(self.feature_names[i], group_tfidf[i]) for i in top_indices]
            results[label] = top_features
            
        return results
    
    def get_feature_importance_df(self, labels: np.ndarray) -> pd.DataFrame:
        """
        Create DataFrame with TF-IDF scores for each group.
        
        Args:
            labels: Group labels
            
        Returns:
            DataFrame with features and scores by group
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        unique_labels = np.unique(labels)
        data = {'feature': self.feature_names}
        
        for label in unique_labels:
            group_mask = labels == label
            group_tfidf = self.tfidf_matrix[group_mask].mean(axis=0).A1
            data[f'group_{label}_tfidf'] = group_tfidf
            
        df = pd.DataFrame(data)
        df['tfidf_diff'] = df[f'group_{unique_labels[1]}_tfidf'] - df[f'group_{unique_labels[0]}_tfidf']
        return df.sort_values('tfidf_diff', ascending=False)


class NGramAnalyzer:
    """Analyze n-grams (word sequences) in text."""
    
    def __init__(self, n: int = 2):
        """
        Initialize n-gram analyzer.
        
        Args:
            n: Size of n-grams (2=bigrams, 3=trigrams)
        """
        self.n = n
        self.ngram_freq = Counter()
        self.ngram_freq_by_group = {}
        
    def extract_ngrams(self, tokens: List[str]) -> List[str]:
        """
        Extract n-grams from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of n-grams as strings
        """
        if len(tokens) < self.n:
            return []
        return [' '.join(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]
    
    def compute_ngrams(self, token_lists: List[List[str]]) -> Counter:
        """
        Compute n-gram frequencies across all documents.
        
        Args:
            token_lists: List of token lists
            
        Returns:
            Counter with n-gram frequencies
        """
        all_ngrams = []
        for tokens in token_lists:
            ngrams = self.extract_ngrams(tokens)
            all_ngrams.extend(ngrams)
        
        self.ngram_freq = Counter(all_ngrams)
        return self.ngram_freq
    
    def compute_ngrams_by_group(self, token_lists: List[List[str]], groups: List[int]) -> Dict[int, Counter]:
        """
        Compute n-gram frequencies for each group.
        
        Args:
            token_lists: List of token lists
            groups: Group labels
            
        Returns:
            Dictionary mapping group -> Counter of n-grams
        """
        self.ngram_freq_by_group = defaultdict(Counter)
        
        for tokens, group in zip(token_lists, groups):
            ngrams = self.extract_ngrams(tokens)
            self.ngram_freq_by_group[group].update(ngrams)
        
        return dict(self.ngram_freq_by_group)
    
    def get_top_ngrams(self, n: int = 20, group: int = None) -> List[Tuple[str, int]]:
        """
        Get top n-grams.
        
        Args:
            n: Number of top n-grams
            group: If specified, get for specific group only
            
        Returns:
            List of (ngram, frequency) tuples
        """
        if group is not None:
            if group not in self.ngram_freq_by_group:
                return []
            return self.ngram_freq_by_group[group].most_common(n)
        return self.ngram_freq.most_common(n)
    
    def get_distinctive_ngrams(self, group1: int, group2: int, n: int = 20) -> Tuple[List, List]:
        """
        Get n-grams that distinguish between two groups.
        
        Args:
            group1: First group
            group2: Second group
            n: Number of distinctive n-grams per group
            
        Returns:
            Tuple of (group1_distinctive, group2_distinctive)
        """
        if group1 not in self.ngram_freq_by_group or group2 not in self.ngram_freq_by_group:
            return ([], [])
        
        freq1 = self.ngram_freq_by_group[group1]
        freq2 = self.ngram_freq_by_group[group2]
        
        # Calculate ratio for each n-gram
        ratios = {}
        all_ngrams = set(freq1.keys()) | set(freq2.keys())
        
        for ngram in all_ngrams:
            f1 = freq1.get(ngram, 0) + 1  # Add 1 smoothing
            f2 = freq2.get(ngram, 0) + 1
            ratios[ngram] = f1 / f2
        
        # Get distinctive n-grams for each group
        sorted_ngrams = sorted(ratios.items(), key=lambda x: x[1])
        
        group2_distinctive = [(ng, freq2.get(ng, 0)) for ng, _ in sorted_ngrams[:n]]
        group1_distinctive = [(ng, freq1.get(ng, 0)) for ng, _ in sorted_ngrams[-n:][::-1]]
        
        return (group1_distinctive, group2_distinctive)


class SentimentAnalyzer:
    """Analyze sentiment in text using simple lexicon-based approach."""
    
    def __init__(self):
        """Initialize sentiment analyzer with basic lexicons."""
        # Simple positive/negative word lists
        self.positive_words = {
            'good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'fantastic',
            'amazing', 'positive', 'better', 'best', 'enjoy', 'enjoyed', 'fun',
            'nice', 'well', 'fine', 'glad', 'excited', 'pleasure'
        }
        
        self.negative_words = {
            'bad', 'sad', 'hate', 'terrible', 'awful', 'horrible', 'negative',
            'worse', 'worst', 'depressed', 'depression', 'anxious', 'anxiety',
            'worried', 'stress', 'stressed', 'pain', 'hurt', 'difficult', 'hard',
            'problem', 'problems', 'wrong', 'sick', 'tired', 'lonely', 'alone'
        }
        
    def analyze_tokens(self, tokens: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment of tokenized text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary with sentiment scores
        """
        if not tokens:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}
        
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        total = len(tokens)
        positive_score = positive_count / total
        negative_score = negative_count / total
        compound = (positive_count - negative_count) / total
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': 1 - (positive_score + negative_score),
            'compound': compound
        }
    
    def analyze_batch(self, token_lists: List[List[str]]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple documents.
        
        Args:
            token_lists: List of token lists
            
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        for tokens in token_lists:
            scores = self.analyze_tokens(tokens)
            results.append(scores)
        
        return pd.DataFrame(results)
    
    def compare_by_group(self, token_lists: List[List[str]], groups: List[int]) -> pd.DataFrame:
        """
        Compare sentiment between groups.
        
        Args:
            token_lists: List of token lists
            groups: Group labels
            
        Returns:
            DataFrame with mean sentiment by group
        """
        sentiment_df = self.analyze_batch(token_lists)
        sentiment_df['group'] = groups
        
        return sentiment_df.groupby('group').mean()


class StatisticalTester:
    """Perform statistical tests on text features."""
    
    @staticmethod
    def chi_square_test(word_freq_group1: int, word_freq_group2: int,
                       total_group1: int, total_group2: int) -> Tuple[float, float]:
        """
        Perform chi-square test for word frequency difference.
        
        Args:
            word_freq_group1: Word frequency in group 1
            word_freq_group2: Word frequency in group 2
            total_group1: Total words in group 1
            total_group2: Total words in group 2
            
        Returns:
            Tuple of (chi2_statistic, p_value)
        """
        # Create contingency table
        observed = np.array([
            [word_freq_group1, total_group1 - word_freq_group1],
            [word_freq_group2, total_group2 - word_freq_group2]
        ])
        
        try:
            chi2, p_value, _, _ = chi2_contingency(observed)
            return chi2, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def mann_whitney_test(group1_values: np.ndarray, group2_values: np.ndarray) -> Tuple[float, float]:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            group1_values: Values for group 1
            group2_values: Values for group 2
            
        Returns:
            Tuple of (u_statistic, p_value)
        """
        try:
            u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
            return u_stat, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: Values for group 1
            group2: Values for group 2
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std


if __name__ == "__main__":
    # Test the analyzers
    print("Testing Advanced Analysis Module")
    print("=" * 60)
    
    # Sample data
    texts = [
        "I feel sad and depressed today",
        "I am happy and excited about life",
        "Everything is terrible and awful",
        "Life is wonderful and amazing"
    ]
    labels = np.array([1, 0, 1, 0])
    
    # Test TF-IDF
    print("\n1. TF-IDF Analysis:")
    tfidf = TFIDFAnalyzer(max_features=20)
    tfidf.fit_transform(texts)
    top_features = tfidf.get_top_features_by_group(labels, n=5)
    for group, features in top_features.items():
        print(f"   Group {group}: {[f[0] for f in features]}")
    
    # Test N-grams
    print("\n2. N-gram Analysis:")
    tokens_list = [text.lower().split() for text in texts]
    ngram = NGramAnalyzer(n=2)
    ngram.compute_ngrams_by_group(tokens_list, labels)
    print(f"   Top bigrams: {ngram.get_top_ngrams(n=5)}")
    
    # Test Sentiment
    print("\n3. Sentiment Analysis:")
    sentiment = SentimentAnalyzer()
    sent_df = sentiment.compare_by_group(tokens_list, labels)
    print(f"   Sentiment by group:\n{sent_df}")
    
    print("\n✓ All tests completed successfully")
