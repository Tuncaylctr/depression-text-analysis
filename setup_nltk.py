#!/usr/bin/env python3
"""
Setup script for NLTK data download (bypassing SSL issues on macOS)
Run this once to set up the project.
"""

import ssl
import nltk
import sys

# Bypass SSL certificate verification for macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading required NLTK data...")
print("(This may take a minute)\n")

try:
    print("  Downloading 'punkt' tokenizer...")
    nltk.download('punkt', quiet=False)
    
    print("\n  Downloading 'punkt_tab' tokenizer...")
    nltk.download('punkt_tab', quiet=False)
    
    print("\n  Downloading 'stopwords'...")
    nltk.download('stopwords', quiet=False)
    
    print("\n  Downloading 'wordnet'...")
    nltk.download('wordnet', quiet=False)
    
    print("\n✓ All NLTK data downloaded successfully!")
    sys.exit(0)
except Exception as e:
    print(f"\n✗ Error downloading NLTK data: {e}")
    print("  You may need to run this script again or check your internet connection")
    sys.exit(1)
