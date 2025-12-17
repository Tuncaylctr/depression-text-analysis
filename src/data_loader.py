"""
Data loading and preprocessing module.
Handles loading transcripts and depression labels from CSV files.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict


class DataLoader:
    """Load and manage depression text analysis data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Root directory containing data/ folder
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.labels_dir = self.data_dir / "labels"
        self.processed_labels_dir = self.labels_dir / "processed"
        
    def load_labels(self, use_processed: bool = True) -> pd.DataFrame:
        """
        Load depression labels (PHQ scores).
        
        Args:
            use_processed: If True, load processed binary labels. If False, load train split.
            
        Returns:
            DataFrame with Participant_ID, PHQ_Score, and PHQ_Binary columns
        """
        if use_processed:
            labels_file = self.processed_labels_dir / "all_participants_phq_binary.csv"
        else:
            labels_file = self.labels_dir / "train_split_Depression_AVEC2017.csv"
            
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
            
        return pd.read_csv(labels_file)
    
    def load_transcript(self, participant_id: int) -> pd.DataFrame:
        """
        Load transcript for a specific participant.
        
        Args:
            participant_id: Participant ID (e.g., 300, 301)
            
        Returns:
            DataFrame with columns: start_time, stop_time, speaker, value (text)
        """
        transcript_file = self.raw_dir / f"{participant_id}_TRANSCRIPT.csv"
        
        if not transcript_file.exists():
            return None
            
        return pd.read_csv(transcript_file, sep='\t')
    
    def load_all_transcripts(self) -> Dict[int, pd.DataFrame]:
        """
        Load all available transcripts.
        
        Returns:
            Dictionary mapping participant_id -> transcript DataFrame
        """
        transcripts = {}
        labels = self.load_labels(use_processed=True)
        
        for participant_id in labels['Participant_ID'].values:
            transcript = self.load_transcript(participant_id)
            if transcript is not None:
                transcripts[participant_id] = transcript
                
        return transcripts
    
    def get_participant_text(self, participant_id: int, speaker: str = None) -> str:
        """
        Get concatenated text for a participant.
        
        Args:
            participant_id: Participant ID
            speaker: Filter by speaker ('Participant', 'Ellie', or None for all)
            
        Returns:
            Concatenated text string
        """
        transcript = self.load_transcript(participant_id)
        
        if transcript is None:
            return ""
        
        if speaker:
            transcript = transcript[transcript['speaker'] == speaker]
        
        return " ".join(transcript['value'].astype(str).values)
    
    def create_corpus_with_labels(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Create a corpus combining transcripts with labels.
        
        Returns:
            Tuple of (combined_df, metadata_dict)
            - combined_df: DataFrame with Participant_ID, PHQ_Score, PHQ_Binary, text
            - metadata_dict: Information about data (n_participants, n_depressed, etc.)
        """
        labels = self.load_labels(use_processed=True)
        texts = []
        valid_ids = []
        
        for idx, row in labels.iterrows():
            participant_id = row['Participant_ID']
            text = self.get_participant_text(participant_id, speaker='Participant')
            
            if text.strip():  # Only include if we have non-empty text
                texts.append(text)
                valid_ids.append(participant_id)
        
        # Filter labels to only valid IDs
        combined_df = labels[labels['Participant_ID'].isin(valid_ids)].copy()
        combined_df['text'] = texts
        
        # Create metadata
        metadata = {
            'n_total_participants': len(combined_df),
            'n_depressed': (combined_df['PHQ_Binary'] == 1).sum(),
            'n_non_depressed': (combined_df['PHQ_Binary'] == 0).sum(),
            'mean_phq_score': combined_df['PHQ_Score'].mean(),
            'max_phq_score': combined_df['PHQ_Score'].max(),
            'min_phq_score': combined_df['PHQ_Score'].min(),
        }
        
        return combined_df, metadata


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Load labels
    labels = loader.load_labels()
    print(f"Loaded {len(labels)} labels")
    print(labels.head())
    
    # Create corpus
    corpus_df, metadata = loader.create_corpus_with_labels()
    print(f"\nCorpus Statistics:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
