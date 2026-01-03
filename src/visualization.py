"""
Visualization module for exploratory data analysis.
Creates plots for understanding word frequencies and depression patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DataVisualizer:
    """Create visualizations for depression text analysis."""
    
    @staticmethod
    def plot_top_words(top_words: List[Tuple[str, int]], 
                       title: str = "Top 20 Most Frequent Words",
                       save_path: str = None):
        """
        Plot top words as horizontal bar chart.
        
        Args:
            top_words: List of (word, frequency) tuples
            title: Plot title
            save_path: Path to save figure (if provided)
        """
        words, freqs = zip(*top_words)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(words)), freqs, color='steelblue')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Frequency')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_word_frequency_comparison(words: List[str],
                                       freq_group1: List[int],
                                       freq_group2: List[int],
                                       group1_label: str = "Group 1",
                                       group2_label: str = "Group 2",
                                       save_path: str = None):
        """
        Compare word frequencies between two groups.
        
        Args:
            words: List of words
            freq_group1: Frequencies in group 1
            freq_group2: Frequencies in group 2
            group1_label: Label for group 1
            group2_label: Label for group 2
            save_path: Path to save figure
        """
        x = np.arange(len(words))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, freq_group1, width, label=group1_label, color='lightblue')
        ax.bar(x + width/2, freq_group2, width, label=group2_label, color='coral')
        
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Word Frequency Comparison: {group1_label} vs {group2_label}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_correlations(words: List[str],
                         correlations: List[float],
                         title: str = "Word-Depression Correlations",
                         threshold: float = None,
                         save_path: str = None):
        """
        Plot word correlations with depression levels.
        
        Args:
            words: List of words
            correlations: Correlation coefficients
            title: Plot title
            threshold: Color threshold for significance
            save_path: Path to save figure
        """
        # Sort by absolute correlation
        sorted_idx = np.argsort(np.abs(correlations))[::-1]
        sorted_words = [words[i] for i in sorted_idx]
        sorted_corr = [correlations[i] for i in sorted_idx]
        
        # Color based on positive/negative
        colors = ['crimson' if c > 0 else 'steelblue' for c in sorted_corr]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(sorted_words)), sorted_corr, color=colors)
        ax.set_yticks(range(len(sorted_words)))
        ax.set_yticklabels(sorted_words)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='crimson', label='Positive (More in depressed)'),
            Patch(facecolor='steelblue', label='Negative (More in non-depressed)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_phq_distribution(phq_scores: List[float],
                             phq_binary: List[int] = None,
                             save_path: str = None):
        """
        Plot PHQ score distribution.
        
        Args:
            phq_scores: PHQ continuous scores
            phq_binary: Binary depression labels (optional)
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # Histogram of continuous scores
        axes[0].hist(phq_scores, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('PHQ Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of PHQ Scores (Continuous)')
        axes[0].axvline(np.mean(phq_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(phq_scores):.2f}')
        axes[0].legend()
        
        # Binary distribution
        if phq_binary is not None:
            counts = pd.Series(phq_binary).value_counts()
            labels = ['Non-Depressed (0)', 'Depressed (1)']
            colors = ['lightgreen', 'lightcoral']
            axes[1].bar([labels[i] for i in counts.index], counts.values, 
                       color=[colors[i] for i in counts.index])
            axes[1].set_ylabel('Number of Participants')
            axes[1].set_title('Binary Depression Distribution')
            
            # Add percentages
            for i, (label, count) in enumerate(zip([labels[j] for j in counts.index], counts.values)):
                pct = 100 * count / len(phq_binary)
                axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    @staticmethod
    def plot_word_cloud_data(words: List[str],
                            frequencies: List[float],
                            save_path: str = None,
                            max_words: int = 50):
        """
        Create word cloud-like visualization using scatter plot.
        
        Args:
            words: List of words
            frequencies: Frequency or importance scores
            save_path: Path to save figure
            max_words: Maximum words to display
        """
        # Take top words
        sorted_idx = np.argsort(frequencies)[-max_words:]
        top_words = [words[i] for i in sorted_idx]
        top_freq = [frequencies[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        scatter = ax.scatter(range(len(top_words)), top_freq, 
                           s=[f*10 for f in top_freq],  # Size by frequency
                           c=top_freq,
                           cmap='viridis',
                           alpha=0.6,
                           edgecolors='black')
        
        ax.set_xticks(range(len(top_words)))
        ax.set_xticklabels(top_words, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top {max_words} Words by Frequency', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_text_length_distribution(text_lengths: List[int],
                                     phq_binary: List[int] = None,
                                     save_path: str = None):
        """
        Plot distribution of text lengths (words per participant).
        
        Args:
            text_lengths: List of text lengths
            phq_binary: Binary depression labels (optional, for comparison)
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if phq_binary is not None:
            depressed_lengths = [l for l, b in zip(text_lengths, phq_binary) if b == 1]
            non_depressed_lengths = [l for l, b in zip(text_lengths, phq_binary) if b == 0]
            
            ax.hist([non_depressed_lengths, depressed_lengths], 
                   bins=15, 
                   label=['Non-Depressed', 'Depressed'],
                   color=['lightgreen', 'lightcoral'],
                   alpha=0.7)
        else:
            ax.hist(text_lengths, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Text Lengths')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_wordcloud(word_freq: Dict[str, int],
                      title: str = "Word Cloud",
                      save_path: str = None):
        """
        Create word cloud visualization.
        
        Args:
            word_freq: Dictionary of word frequencies
            title: Plot title
            save_path: Path to save figure
        """
        try:
            from wordcloud import WordCloud
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(word_freq)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig, ax
        except ImportError:
            print("Warning: wordcloud library not available. Skipping word cloud visualization.")
            return None, None
    
    @staticmethod
    def plot_ngram_comparison(ngrams_group1: List[Tuple[str, int]],
                             ngrams_group2: List[Tuple[str, int]],
                             group1_label: str = "Group 1",
                             group2_label: str = "Group 2",
                             n: int = 15,
                             save_path: str = None):
        """
        Compare n-grams between two groups.
        
        Args:
            ngrams_group1: List of (ngram, frequency) for group 1
            ngrams_group2: List of (ngram, frequency) for group 2
            group1_label: Label for group 1
            group2_label: Label for group 2
            n: Number of top n-grams to show
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Group 1
        ng1, freq1 = zip(*ngrams_group1[:n])
        axes[0].barh(range(len(ng1)), freq1, color='lightblue')
        axes[0].set_yticks(range(len(ng1)))
        axes[0].set_yticklabels(ng1, fontsize=9)
        axes[0].set_xlabel('Frequency')
        axes[0].set_title(f'Top N-grams: {group1_label}', fontweight='bold')
        axes[0].invert_yaxis()
        
        # Group 2
        ng2, freq2 = zip(*ngrams_group2[:n])
        axes[1].barh(range(len(ng2)), freq2, color='coral')
        axes[1].set_yticks(range(len(ng2)))
        axes[1].set_yticklabels(ng2, fontsize=9)
        axes[1].set_xlabel('Frequency')
        axes[1].set_title(f'Top N-grams: {group2_label}', fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    @staticmethod
    def plot_tfidf_heatmap(tfidf_df: pd.DataFrame,
                          n_features: int = 20,
                          save_path: str = None):
        """
        Create heatmap of TF-IDF scores by group.
        
        Args:
            tfidf_df: DataFrame with TF-IDF scores
            n_features: Number of top features to show
            save_path: Path to save figure
        """
        # Get top features by absolute difference
        top_features = tfidf_df.nlargest(n_features, 'tfidf_diff')
        
        # Prepare data for heatmap
        group_cols = [col for col in tfidf_df.columns if col.startswith('group_') and col.endswith('_tfidf')]
        heatmap_data = top_features[group_cols].T
        heatmap_data.index = [col.replace('group_', 'Group ').replace('_tfidf', '') for col in heatmap_data.index]
        heatmap_data.columns = top_features['feature'].values
        
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'TF-IDF Score'})
        ax.set_title('TF-IDF Scores by Group (Top Features)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Group')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_sentiment_distribution(sentiment_df: pd.DataFrame,
                                    group_col: str = 'group',
                                    save_path: str = None):
        """
        Plot sentiment score distributions by group.
        
        Args:
            sentiment_df: DataFrame with sentiment scores and group labels
            group_col: Column name for group labels
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        sentiment_cols = ['positive', 'negative', 'neutral', 'compound']
        colors = ['green', 'red', 'gray', 'blue']
        
        for idx, (col, color) in enumerate(zip(sentiment_cols, colors)):
            ax = axes[idx // 2, idx % 2]
            
            for group in sentiment_df[group_col].unique():
                group_data = sentiment_df[sentiment_df[group_col] == group][col]
                ax.hist(group_data, alpha=0.6, label=f'Group {group}', bins=15, color=color)
            
            ax.set_xlabel(f'{col.capitalize()} Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col.capitalize()} Sentiment Distribution', fontweight='bold')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    @staticmethod
    def plot_sentiment_comparison(sentiment_by_group: pd.DataFrame,
                                  save_path: str = None):
        """
        Compare mean sentiment scores between groups.
        
        Args:
            sentiment_by_group: DataFrame with mean sentiment by group
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiment_by_group.plot(kind='bar', ax=ax, 
                               color=['green', 'red', 'gray', 'blue'],
                               alpha=0.7)
        
        ax.set_xlabel('Group')
        ax.set_ylabel('Mean Sentiment Score')
        ax.set_title('Mean Sentiment Scores by Depression Group', fontsize=14, fontweight='bold')
        ax.legend(title='Sentiment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax


if __name__ == "__main__":
    # Test visualizations with sample data
    sample_words = ['sad', 'happy', 'tired', 'excited', 'lonely']
    sample_freqs = [45, 20, 35, 15, 40]
    sample_corr = [0.35, -0.20, 0.28, -0.15, 0.42]
    
    DataVisualizer.plot_top_words(list(zip(sample_words, sample_freqs)))
    DataVisualizer.plot_correlations(sample_words, sample_corr)
    
    plt.show()
