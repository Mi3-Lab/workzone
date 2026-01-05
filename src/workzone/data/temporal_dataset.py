"""
Phase 2.1: Training Data Pipeline

Generates temporal sequences from video processing results for training
the attention-weighted temporal aggregator.

Input: CSV timelines from process_video_fusion.py
Output: PyTorch dataset of temporal sequences with labels
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TemporalSequenceDataset(Dataset):
    """
    Dataset of temporal sequences for Phase 2.1 training.
    
    Each sample is a sliding window of frames with:
    - Input: Multi-stream features (cues, scene, scores)
    - Target: Work zone label + state label
    """
    
    def __init__(
        self,
        csv_files: List[Path],
        window_size: int = 90,
        stride: int = 15,
        augment: bool = True,
    ):
        """
        Args:
            csv_files: List of timeline CSV files from process_video_fusion.py
            window_size: Sequence length (frames)
            stride: Sliding window stride for sequence extraction
            augment: Apply temporal augmentation (random crops, etc.)
        """
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        
        # Load and process all CSVs
        self.sequences = []
        for csv_file in csv_files:
            self._process_csv(csv_file)
        
        logger.info(f"Loaded {len(self.sequences)} sequences from {len(csv_files)} videos")
    
    def _process_csv(self, csv_file: Path):
        """Extract sequences from a single CSV file."""
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
            return
        
        # Required columns
        required = ['yolo_score', 'clip_score', 'state']
        if not all(col in df.columns for col in required):
            logger.warning(f"Missing required columns in {csv_file}")
            return
        
        # Extract sequences with sliding window
        n_frames = len(df)
        for start_idx in range(0, n_frames - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = df.iloc[start_idx:end_idx]
            
            # Build feature dict
            features = self._extract_features(window)
            
            # Label: work zone present if any INSIDE/APPROACHING/EXITING in window
            states = window['state'].values
            has_workzone = np.any([s in ['INSIDE', 'APPROACHING', 'EXITING'] for s in states])
            
            # State label: most common state in window
            state_counts = pd.Series(states).value_counts()
            dominant_state = state_counts.index[0] if len(state_counts) > 0 else 'OUT'
            
            self.sequences.append({
                'features': features,
                'label': 1.0 if has_workzone else 0.0,
                'state': dominant_state,
                'video': csv_file.stem,
            })
    
    def _extract_features(self, window: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract multi-stream features from window."""
        seq_len = len(window)
        
        # Cue scores (5 types)
        # If Phase 2.1 columns exist, use per-cue CLIP confidences
        # Otherwise, fall back to normalized counts
        cue_scores = np.zeros((seq_len, 5))
        
        if 'cue_conf_channelization' in window.columns:
            # Phase 2.1: Use per-cue text verification scores
            cue_scores[:, 0] = window['cue_conf_channelization'].fillna(0.0).values
            cue_scores[:, 1] = window['cue_conf_workers'].fillna(0.0).values
            cue_scores[:, 2] = window['cue_conf_vehicles'].fillna(0.0).values
            cue_scores[:, 3] = window['cue_conf_signs'].fillna(0.0).values
            cue_scores[:, 4] = window['cue_conf_equipment'].fillna(0.0).values
        else:
            # Fallback: Normalize counts
            if 'count_channelization' in window.columns:
                cue_scores[:, 0] = np.clip(window['count_channelization'].values / 10.0, 0, 1)
            if 'count_workers' in window.columns:
                cue_scores[:, 1] = np.clip(window['count_workers'].values / 5.0, 0, 1)
            # TODO: Add count_vehicles, count_signs, count_equipment when available
        
        # Scene context logits (3 types)
        scene_logits = np.zeros((seq_len, 3))
        if 'scene_context' in window.columns:
            for i, ctx in enumerate(window['scene_context'].values):
                if ctx == 'highway':
                    scene_logits[i, 0] = 1.0
                elif ctx == 'urban':
                    scene_logits[i, 1] = 1.0
                elif ctx == 'suburban':
                    scene_logits[i, 2] = 1.0
        
        # YOLO and CLIP scores
        yolo_scores = window['yolo_score'].values.reshape(-1, 1)
        clip_scores = window['clip_score'].fillna(0.0).values.reshape(-1, 1) if 'clip_score' in window.columns else np.zeros((seq_len, 1))
        
        # Motion plausibility (Phase 2.1 if available, otherwise default to 1.0)
        if 'motion_plausibility' in window.columns:
            motion_plausibility = window['motion_plausibility'].fillna(1.0).values.reshape(-1, 1)
        else:
            motion_plausibility = np.ones((seq_len, 1))
        
        # Frame deltas (normalized position in sequence)
        frame_deltas = np.linspace(0, 1, seq_len).reshape(-1, 1)
        
        return {
            'cue_scores': cue_scores.astype(np.float32),
            'scene_logits': scene_logits.astype(np.float32),
            'yolo_scores': yolo_scores.astype(np.float32),
            'clip_scores': clip_scores.astype(np.float32),
            'motion_plausibility': motion_plausibility.astype(np.float32),
            'frame_deltas': frame_deltas.astype(np.float32),
        }
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        features = seq['features']
        
        # Augmentation: random temporal crop if enabled
        if self.augment and np.random.rand() < 0.3:
            crop_size = np.random.randint(self.window_size // 2, self.window_size)
            start = np.random.randint(0, self.window_size - crop_size + 1)
            for key in features:
                features[key] = features[key][start:start + crop_size]
        
        # Convert to tensors
        return {
            'cue_scores': torch.from_numpy(features['cue_scores']),
            'scene_logits': torch.from_numpy(features['scene_logits']),
            'yolo_scores': torch.from_numpy(features['yolo_scores']),
            'clip_scores': torch.from_numpy(features['clip_scores']),
            'motion_plausibility': torch.from_numpy(features['motion_plausibility']),
            'frame_deltas': torch.from_numpy(features['frame_deltas']),
            'label': torch.tensor(seq['label'], dtype=torch.float32),
            'state': self._state_to_idx(seq['state']),
        }
    
    @staticmethod
    def _state_to_idx(state: str) -> int:
        """Convert state string to index."""
        state_map = {'OUT': 0, 'APPROACHING': 1, 'INSIDE': 2, 'EXITING': 3}
        return state_map.get(state, 0)


def collate_temporal_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Handles variable-length sequences with padding.
    """
    # Find max length in batch
    max_len = max(item['cue_scores'].size(0) for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    cue_scores = torch.zeros(batch_size, max_len, 5)
    scene_logits = torch.zeros(batch_size, max_len, 3)
    yolo_scores = torch.zeros(batch_size, max_len, 1)
    clip_scores = torch.zeros(batch_size, max_len, 1)
    motion_plausibility = torch.zeros(batch_size, max_len, 1)
    frame_deltas = torch.zeros(batch_size, max_len, 1)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    labels = []
    states = []
    
    for i, item in enumerate(batch):
        seq_len = item['cue_scores'].size(0)
        cue_scores[i, :seq_len] = item['cue_scores']
        scene_logits[i, :seq_len] = item['scene_logits']
        yolo_scores[i, :seq_len] = item['yolo_scores']
        clip_scores[i, :seq_len] = item['clip_scores']
        motion_plausibility[i, :seq_len] = item['motion_plausibility']
        frame_deltas[i, :seq_len] = item['frame_deltas']
        mask[i, :seq_len] = False  # False = not masked
        
        labels.append(item['label'])
        states.append(item['state'])
    
    return {
        'cue_scores': cue_scores,
        'scene_logits': scene_logits,
        'yolo_scores': yolo_scores,
        'clip_scores': clip_scores,
        'motion_plausibility': motion_plausibility,
        'frame_deltas': frame_deltas,
        'mask': mask,
        'labels': torch.stack(labels),
        'states': torch.tensor(states, dtype=torch.long),
    }


def create_dataloaders(
    train_csvs: List[Path],
    val_csvs: List[Path],
    window_size: int = 90,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = TemporalSequenceDataset(
        csv_files=train_csvs,
        window_size=window_size,
        stride=15,
        augment=True,
    )
    
    val_dataset = TemporalSequenceDataset(
        csv_files=val_csvs,
        window_size=window_size,
        stride=30,  # Less overlap for validation
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_temporal_sequences,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_temporal_sequences,
        pin_memory=True,
    )
    
    return train_loader, val_loader
