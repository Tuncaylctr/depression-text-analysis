"""
Depression Text Analysis Package
Analyzes linguistic patterns in interview transcripts to understand
correlations between word usage and depression levels.
"""

__version__ = "1.0.0"
__author__ = "Data Analysis Project"

from .data_loader import DataLoader
from .text_preprocessing import TextPreprocessor, CustomStopwords
from .frequency_analysis import WordFrequencyAnalyzer, CorrelationAnalyzer
from .visualization import DataVisualizer

__all__ = [
    'DataLoader',
    'TextPreprocessor',
    'CustomStopwords',
    'WordFrequencyAnalyzer',
    'CorrelationAnalyzer',
    'DataVisualizer',
]
