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


if __name__ == "__main__":
    # Test visualizations with sample data
    sample_words = ['sad', 'happy', 'tired', 'excited', 'lonely']
    sample_freqs = [45, 20, 35, 15, 40]
    sample_corr = [0.35, -0.20, 0.28, -0.15, 0.42]
    
    DataVisualizer.plot_top_words(list(zip(sample_words, sample_freqs)))
    DataVisualizer.plot_correlations(sample_words, sample_corr)
    
    plt.show()
