"""
Text preprocessing module.
Handles cleaning, tokenization, and text normalization.
"""

import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Preprocess text for analysis."""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            remove_stopwords: Whether to remove common stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text (lowercase, punctuation removed)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_fn(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token not in self.stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.lemmatize or self.lemmatizer is None:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text
            
        Returns:
            List of processed tokens
        """
        # Clean
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords_fn(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Filter empty strings
        tokens = [t for t in tokens if len(t) > 0]
        
        return tokens
    
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of processed token lists
        """
        return [self.process(text) for text in texts]


class CustomStopwords:
    """Manage custom stopword lists."""
    
    # Words common in interview setting that are not informative
    # This includes fillers, conversational noise, and generic verbs/pronouns
    INTERVIEW_STOPWORDS = {
        # Fillers and speech disfluencies
        'um', 'uh', 'yeah', 'like', 'really', 'kind', 'thing', 'things',
        'lot', 'way', 'days', 'time', 'day', 'week', 'ellie', 'l_a',
        'sigh', 'mm', 'hmm', 'hm', 'mhm', 'uh-huh', 'mm-hmm',
        
        # Laughter and non-verbal sounds
        'laughter', 'laugh', 'chuckle',
        
        # Contractions and fragments
        'wan', 'na', 'gon', 'ta', 'gonna', 'wanna', 'gotta',
        'bout', 'cause', 'em',
        
        # Generic conversational words
        'guess', 'say', 'said', 'mean', 'means', 'saying',
        'one', 'two', 'something', 'anything', 'everything',
        'someone', 'anyone', 'everybody',
        
        # Generic pronouns that don't add meaning
        'she', 'he', 'it', 'they', 'them', 'their',
        
        # Too generic to be meaningful
        'get', 'got', 'getting', 'go', 'going', 'went', 'gone',
        'make', 'making', 'made', 'take', 'taking', 'took',
        'come', 'coming', 'came',
        
        # Interview-specific
        'know', 'dont', 'im', 'ive', 'id', 'ill', 'youre', 'thats',
        'think', 'thought', 'thinking', 'feel', 'feeling', 'felt',
        
        # Generic time/quantity
        'little', 'much', 'many', 'bit', 'stuff',
        'years', 'year', 'ago', 'back', 'right', 'now',
    }
    
    @staticmethod
    def get_extended_stopwords(include_interview: bool = True) -> Set[str]:
        """
        Get extended stopwords set.
        
        Args:
            include_interview: Whether to include interview-specific stopwords
            
        Returns:
            Set of stopwords
        """
        stopwords_set = set(stopwords.words('english'))
        
        if include_interview:
            stopwords_set.update(CustomStopwords.INTERVIEW_STOPWORDS)
        
        return stopwords_set
    
    @staticmethod
    def get_meaningful_stopwords() -> Set[str]:
        """
        Get comprehensive stopwords for visualization filtering.
        Includes ALL generic, filler, and meaningless words.
        
        Returns:
            Set of stopwords for aggressive filtering
        """
        return {
            # Fillers and disfluencies
            'um', 'uh', 'yeah', 'like', 'really', 'kind', 'thing', 'things',
            'sigh', 'mm', 'hmm', 'hm', 'mhm', 'uh-huh', 'mm-hmm',
            'laughter', 'laugh', 'chuckle',
            
            # Fragments and contractions
            'wan', 'na', 'gon', 'ta', 'gonna', 'wanna', 'gotta',
            'bout', 'cause', 'em',
            
            # Too generic conversational
            'one', 'two', 'get', 'got', 'getting', 'go', 'going', 'went', 'gone',
            'come', 'coming', 'came',
            'make', 'making', 'made', 'take', 'taking', 'took', 'taken',
            'say', 'saying', 'said', 'tell', 'telling', 'told', 'talked',
            'know', 'knowing', 'knew', 'known',
            'think', 'thinking', 'thought', 'feel', 'feeling', 'felt',
            'want', 'wanting', 'wanted', 'need', 'needing', 'needed',
            'would', 'could', 'should', 'might', 'must',
            
            # Generic descriptors
            'well', 'good', 'bad', 'nice', 'fine', 'okay', 'alright',
            'just', 'maybe', 'probably', 'actually', 'basically',
            
            # People/pronouns
            'people', 'person', 'someone', 'anyone', 'everyone',
            'something', 'anything', 'everything',
            'she', 'he', 'it', 'they', 'them', 'their', 'theyre',
            'thats', 'dont', 'im', 'ive', 'id', 'ill', 'youre',
            
            # Time/quantity
            'time', 'times', 'day', 'days', 'week', 'weeks', 
            'year', 'years', 'ago', 'back', 'now', 'then',
            'lot', 'lots', 'much', 'many', 'more', 'most',
            'little', 'bit', 'few',
            
            # Generic verbs
            'asked', 'ask', 'asking',
            'guess', 'guessing', 'guessed',
            'mean', 'meaning', 'meant',
            'stuff', 'give', 'gave', 'given',
            
            # Interview artifacts
            'ellie', 'l_a',
            
            # Add all English stopwords
            *stopwords.words('english'),
        }


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "I'm really feeling um like I don't know what to do. It's been a long week."
    
    processor = TextPreprocessor(remove_stopwords=True, lemmatize=False)
    tokens = processor.process(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
