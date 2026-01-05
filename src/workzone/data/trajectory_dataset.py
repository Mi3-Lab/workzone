"""
Phase 2.1: Trajectory-based Dataset Loader

Loads temporal sequences from ROADWork dataset annotations.
Uses real annotated trajectories with:
- Text descriptions of scenes (NLP for scene understanding)
- Ground truth object categories
- Temporal sequences (sparse: 5-frame gaps)
- Binary labels: work zone present/absent
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    Dataset from ROADWork trajectory annotations.
    
    Uses real annotated sequences with ground truth labels.
    Handles both sparse (5-frame gaps) and dense (every frame) trajectories.
    
    Dataset Stats (ROADWork):
    - 5,430 total frames (3,117 train + 2,313 val)
    - 5,178 frames with objects, 252 empty frames
    - 45 unique object categories
    - 40,409 total object instances
    - Top categories: Cone (21.5%), Tubular Marker (14.1%), Work Vehicle (13.5%)
    """
    
    # Map 45 categories to 5 cue types (Phase 2.1)
    CATEGORY_TO_CUE_TYPE = {
        # Channelization (markers, barriers, signs, panels) - CUE 0
        'Cone': 0,
        'Drum': 0,
        'Barrier': 0,
        'Fence': 0,
        'Barricade': 0,
        'Vertical Panel': 0,
        'Bike Lane': 0,
        
        # Workers - CUE 1
        'Worker': 1,
        'Police Officer': 1,
        
        # Vehicles - CUE 2
        'Work Vehicle': 2,
        'Police Vehicle': 2,
        
        # Signs (traffic control) - CUE 3
        'Arrow Board': 3,
        'Temporary Traffic Control Message Board': 3,
        'Temporary Traffic Control Sign': 3,
        'Temporary Traffic Control Sign: Other': 3,
        'Temporary Traffic Control Sign: bent left arrow': 3,
        'Temporary Traffic Control Sign: bent right arrow': 3,
        'Temporary Traffic Control Sign: bi-directional arrow': 3,
        'Temporary Traffic Control Sign: bicycle': 3,
        'Temporary Traffic Control Sign: curved left arrow': 3,
        'Temporary Traffic Control Sign: curved left arrow, curved right arrow': 3,
        'Temporary Traffic Control Sign: curved right arrow': 3,
        'Temporary Traffic Control Sign: do not enter sign': 3,
        'Temporary Traffic Control Sign: down diagonal left arrow': 3,
        'Temporary Traffic Control Sign: flagger': 3,
        'Temporary Traffic Control Sign: lane shift arrow': 3,
        'Temporary Traffic Control Sign: left arrow': 3,
        'Temporary Traffic Control Sign: left chevron': 3,
        'Temporary Traffic Control Sign: left lane ends sign': 3,
        'Temporary Traffic Control Sign: no left turn': 3,
        'Temporary Traffic Control Sign: pedestrian: left arrow': 3,
        'Temporary Traffic Control Sign: pedestrian: right arrow': 3,
        'Temporary Traffic Control Sign: right arrow': 3,
        'Temporary Traffic Control Sign: right chevron': 3,
        'Temporary Traffic Control Sign: right lane ends sign': 3,
        'Temporary Traffic Control Sign: two downward diagonal arrows': 3,
        'Temporary Traffic Control Sign: two lane shift arrows': 3,
        'Temporary Traffic Control Sign: two upward diagonal arrows': 3,
        'Temporary Traffic Control Sign: up arrow': 3,
        'Temporary Traffic Control Sign: up diagonal left arrow': 3,
        'Temporary Traffic Control Sign: up diagonal right arrow': 3,
        'Temporary Traffic Control Sign: work vehicle': 3,
        
        # Equipment - CUE 4
        'Work Equipment': 4,
        'Other Roadwork Objects': 4,
    }
    
    def __init__(
        self,
        annotation_file: Path,
        image_dir: Path,
        window_size: int = 5,  # Sparse trajectories have 5 frames
        text_encoder=None,
        augment: bool = True,
    ):
        """
        Args:
            annotation_file: Path to trajectories_train/val_equidistant.json
            image_dir: Path to images directory
            window_size: Frames per sequence (5 for sparse, more for dense)
            text_encoder: Optional encoder for text descriptions (e.g., CLIP text encoder)
            augment: Apply data augmentation
        """
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.window_size = window_size
        self.text_encoder = text_encoder
        self.augment = augment
        
        # Load annotations
        with open(self.annotation_file) as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} annotated frames from {annotation_file.name}")
        
        # Group by trajectory for sequence extraction
        self.trajectories = self._group_by_trajectory()
        logger.info(f"Grouped into {len(self.trajectories)} trajectories")
    
    def _group_by_trajectory(self) -> Dict[str, List[Dict]]:
        """Group frames by trajectory ID."""
        trajectories = defaultdict(list)
        for frame in self.data:
            traj_id = frame.get('trajectory', 'default')
            trajectories[traj_id].append(frame)
        
        # Sort frames within each trajectory
        for traj_id in trajectories:
            trajectories[traj_id].sort(key=lambda x: x['id'])
        
        return trajectories
    
    def _extract_features_from_frame(self, frame: Dict) -> Dict[str, np.ndarray]:
        """Extract features from a single annotated frame."""
        seq_len = 1
        
        # Cue scores (5 types) based on detected objects
        cue_counts = [0] * 5
        for obj in frame.get('objects', []):
            category = obj.get('category_id', '')
            
            # Handle sign subtypes (all start with "Temporary Traffic Control Sign")
            if category.startswith('Temporary Traffic Control Sign'):
                cue_type = 3  # Signs
            else:
                cue_type = self.CATEGORY_TO_CUE_TYPE.get(category, 4)
            
            cue_counts[cue_type] += 1
        
        # Normalize counts to [0, 1]
        cue_scores = np.array([min(c / 10.0, 1.0) for c in cue_counts], dtype=np.float32).reshape(1, 5)
        
        # Scene context logits (can be inferred from description or set to uniform)
        # For now, use uniform prior - could use text embedding of description
        scene_logits = np.ones((1, 3), dtype=np.float32) / 3.0
        
        # Binary label: work zone present if any objects detected
        has_workzone = len(frame.get('objects', [])) > 0
        label = 1.0 if has_workzone else 0.0
        
        # Text embedding of description (if encoder available)
        text_embedding = None
        if self.text_encoder is not None:
            try:
                description = frame.get('description', '')
                text_embedding = self._encode_text(description)
            except Exception as e:
                logger.warning(f"Text encoding failed: {e}")
        
        return {
            'cue_scores': cue_scores,
            'scene_logits': scene_logits,
            'label': label,
            'text_embedding': text_embedding,
            'description': frame.get('description', ''),
            'frame_id': frame.get('id', ''),
            'image_path': frame.get('image', ''),
        }
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text description using provided encoder."""
        if self.text_encoder is None:
            return np.zeros(512, dtype=np.float32)  # Default embedding size
        
        # Assume text_encoder is CLIP or similar with tokenize + encode
        tokens = self.text_encoder['tokenizer']([text])
        with torch.no_grad():
            embeddings = self.text_encoder['model'].encode_text(tokens)
            embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        return embeddings[0].cpu().numpy().astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single frame with extracted features."""
        frame = self.data[idx]
        features = self._extract_features_from_frame(frame)
        
        return {
            'cue_scores': torch.from_numpy(features['cue_scores']),
            'scene_logits': torch.from_numpy(features['scene_logits']),
            'label': torch.tensor(features['label'], dtype=torch.float32),
            'text_embedding': torch.from_numpy(features['text_embedding']) if features['text_embedding'] is not None else None,
            'description': features['description'],
            'frame_id': features['frame_id'],
        }


class TrajectorySequenceDataset(Dataset):
    """
    Dataset that extracts temporal sequences from trajectory data.
    
    Takes consecutive frames and packages them as sequences for temporal modeling.
    """
    
    def __init__(
        self,
        annotation_file: Path,
        image_dir: Path,
        window_size: int = 5,
        stride: int = 1,
        text_encoder=None,
    ):
        """
        Args:
            annotation_file: trajectories_train/val_equidistant.json
            image_dir: images directory
            window_size: Sequence length (frames)
            stride: Sliding window stride
            text_encoder: Optional text encoder
        """
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.window_size = window_size
        self.stride = stride
        self.text_encoder = text_encoder
        
        # Load annotations
        with open(self.annotation_file) as f:
            data = json.load(f)
        
        # Group by trajectory and extract sequences
        self.sequences = []
        trajectories = defaultdict(list)
        
        for frame in data:
            traj_id = frame.get('trajectory', 'default')
            trajectories[traj_id].append(frame)
        
        # Sort and extract sequences
        for traj_id, frames in trajectories.items():
            frames.sort(key=lambda x: x['id'])
            
            # Extract sliding windows
            for start_idx in range(0, len(frames) - window_size + 1, stride):
                end_idx = start_idx + window_size
                seq_frames = frames[start_idx:end_idx]
                
                # Label: any frame has work zone objects
                has_workzone = any(
                    len(f.get('objects', [])) > 0 for f in seq_frames
                )
                
                self.sequences.append({
                    'frames': seq_frames,
                    'traj_id': traj_id,
                    'label': 1.0 if has_workzone else 0.0,
                })
        
        logger.info(f"Extracted {len(self.sequences)} sequences from {len(trajectories)} trajectories")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        frames = seq['frames']
        seq_len = len(frames)
        
        # Build feature tensors
        cue_scores = np.zeros((seq_len, 5), dtype=np.float32)
        scene_logits = np.ones((seq_len, 3), dtype=np.float32) / 3.0
        descriptions = []
        
        for i, frame in enumerate(frames):
            # Extract cue counts
            cue_counts = [0] * 5
            for obj in frame.get('objects', []):
                category = obj.get('category_id', '')
                if category.startswith('Temporary Traffic Control Sign'):
                    cue_type = 3
                else:
                    cue_type = TrajectoryDataset.CATEGORY_TO_CUE_TYPE.get(category, 4)
                cue_counts[cue_type] += 1
            
            cue_scores[i] = np.array([min(c / 10.0, 1.0) for c in cue_counts])
            descriptions.append(frame.get('description', ''))
        
        # Concatenate descriptions for overall scene understanding
        combined_description = ' | '.join(descriptions)
        
        return {
            'cue_scores': torch.from_numpy(cue_scores),
            'scene_logits': torch.from_numpy(scene_logits),
            'label': torch.tensor(seq['label'], dtype=torch.float32),
            'descriptions': descriptions,
            'combined_description': combined_description,
            'traj_id': seq['traj_id'],
        }


def create_trajectory_dataloaders(
    train_annotation: Path,
    val_annotation: Path,
    image_dir: Path,
    window_size: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    text_encoder=None,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders from ROADWork trajectory annotations."""
    
    train_dataset = TrajectorySequenceDataset(
        annotation_file=train_annotation,
        image_dir=image_dir,
        window_size=window_size,
        stride=1,
        text_encoder=text_encoder,
    )
    
    val_dataset = TrajectorySequenceDataset(
        annotation_file=val_annotation,
        image_dir=image_dir,
        window_size=window_size,
        stride=2,  # Less overlap for validation
        text_encoder=text_encoder,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_trajectory_sequences,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_trajectory_sequences,
    )
    
    return train_loader, val_loader


def collate_trajectory_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length sequences."""
    
    # Find max sequence length
    max_len = max(item['cue_scores'].size(0) for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    cue_scores = torch.zeros(batch_size, max_len, 5)
    scene_logits = torch.zeros(batch_size, max_len, 3)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    labels = []
    descriptions_list = []
    
    for i, item in enumerate(batch):
        seq_len = item['cue_scores'].size(0)
        cue_scores[i, :seq_len] = item['cue_scores']
        scene_logits[i, :seq_len] = item['scene_logits']
        mask[i, :seq_len] = False
        
        labels.append(item['label'])
        descriptions_list.append(item['combined_description'])
    
    return {
        'cue_scores': cue_scores,
        'scene_logits': scene_logits,
        'mask': mask,
        'labels': torch.stack(labels),
        'descriptions': descriptions_list,
    }
