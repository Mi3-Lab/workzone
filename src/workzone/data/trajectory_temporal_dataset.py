"""
Phase 2.1: Dataset Loader for Real Trajectory Annotations

Uses the ROADWork dataset trajectories with:
- Temporal sequences of images (sparse/dense)
- Text descriptions of scenes
- Object annotations
- Bounding boxes

This is the CORRECT approach - training on real annotated data,
not on inference outputs.

Structure: Each JSON is a list of frame annotations with:
{
  "id": "frame_id",
  "image": "relative/path/to/image.jpg",
  "description": "Text description of scene",
  "objects": [{"bbox": [...], "category_id": "...", "score": ...}, ...]
}
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)


WORK_CATEGORIES = {
    # Channelization / delineation
    "Cone",
    "Tubular Marker",
    "Vertical Panel",
    "Barrier",
    "Barricade",
    "Drum",
    "Fence",
    # Personnel / vehicles / equipment
    "Worker",
    "Police Officer",
    "Work Vehicle",
    "Police Vehicle",
    "Work Equipment",
    # Signs / boards
    "Temporary Traffic Control Sign",
    "Temporary Traffic Control Sign: Other",
    "Temporary Traffic Control Sign: up arrow",
    "Temporary Traffic Control Sign: left chevron",
    "Temporary Traffic Control Sign: two lane shift arrows",
    "Temporary Traffic Control Sign: left arrow",
    "Temporary Traffic Control Message Board",
    "Arrow Board",
}


class TrajectoryTemporalDataset(Dataset):
    """
    Dataset loader for ROADWork trajectory sequences.
    
    Loads frames from trajectory JSON and creates sliding window
    sequences for temporal training.
    
    Each trajectory annotation is a single frame with:
    - Image path
    - Text description
    - Object annotations with bboxes
    """
    
    def __init__(
        self,
        annotations_file: Path,
        images_base_dir: Path,
        window_size: int = 30,
        stride: int = 10,
        max_sequences: Optional[int] = None,
    ):
        """
        Args:
            annotations_file: Path to trajectories_*.json (list of frames)
            images_base_dir: Base directory for images
            window_size: Temporal window size (frames)
            stride: Stride for sliding window
            max_sequences: Max sequences to load (for debugging)
        """
        self.annotations_file = Path(annotations_file)
        self.images_base_dir = Path(images_base_dir)
        self.window_size = window_size
        self.stride = stride
        self.max_sequences = max_sequences
        
        # Load all frame annotations
        logger.info(f"Loading frame annotations from {self.annotations_file}")
        with open(self.annotations_file, 'r') as f:
            self.frames = json.load(f)
        
        logger.info(f"Loaded {len(self.frames)} frame annotations")
        
        # Build sliding window sequences
        self.sequences = []
        self._build_sequences()
        
        logger.info(f"Created {len(self.sequences)} temporal sequences")
    
    def _build_sequences(self):
        """Build sliding window sequences from frames."""
        num_frames = len(self.frames)
        
        if num_frames < self.window_size:
            logger.warning(f"Not enough frames ({num_frames}) for window size {self.window_size}")
            return
        
        # Create sliding windows
        for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
            if self.max_sequences and len(self.sequences) >= self.max_sequences:
                break
            
            end_idx = start_idx + self.window_size
            window_frame_indices = list(range(start_idx, end_idx))
            
            self.sequences.append({
                'start_idx': start_idx,
                'frame_indices': window_frame_indices,
            })
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        frame_indices = seq['frame_indices']
        
        # Extract features for each frame in the sequence
        images = []
        texts = []
        object_counts = []
        
        for frame_idx in frame_indices:
            frame_data = self.frames[frame_idx]

            # Load image
            img_tensor = self._load_image(frame_data)
            images.append(img_tensor)

            # Get text description
            text = frame_data.get('description', 'work zone')
            texts.append(text)

            # Count only work-related objects
            objs = frame_data.get('objects', [])
            work_objs = [o for o in objs if o.get('category_id') in WORK_CATEGORIES]
            object_counts.append(len(work_objs))
        
        # Stack images
        images_tensor = torch.stack(images)  # (seq_len, 3, H, W)
        object_counts_tensor = torch.tensor(object_counts, dtype=torch.float32)
        
        # Compute per-frame cue strength using work-related categories only
        positive_frames = []
        for text, count in zip(texts, object_counts):
            has_kw = any(kw in text.lower() for kw in ['work', 'construction', 'roadwork'])
            frame_positive = has_kw or count > 0
            positive_frames.append(frame_positive)

        num_positive = sum(positive_frames)
        # Binary label: require at least a few positive frames to mark the window as containing work
        min_pos_frames = max(1, int(self.window_size * 0.1))
        label = torch.tensor(1.0 if num_positive >= min_pos_frames else 0.0, dtype=torch.float32)

        # Derive coarse state label (0=OUT, 1=APPROACHING, 2=INSIDE, 3=EXITING)
        if num_positive < min_pos_frames:
            state_label = 0  # OUT
        else:
            first_third = object_counts[: self.window_size // 3]
            mid_third = object_counts[self.window_size // 3 : 2 * self.window_size // 3]
            last_third = object_counts[2 * self.window_size // 3 :]

            mean_first = float(np.mean(first_third)) if len(first_third) > 0 else 0.0
            mean_mid = float(np.mean(mid_third)) if len(mid_third) > 0 else 0.0
            mean_last = float(np.mean(last_third)) if len(last_third) > 0 else 0.0

            inside_thresh = 4.0
            approach_thresh = 2.0

            if mean_mid >= inside_thresh and mean_first >= approach_thresh and mean_last >= approach_thresh:
                state_label = 2  # INSIDE
            elif mean_mid >= inside_thresh and mean_last >= approach_thresh and mean_first < approach_thresh:
                state_label = 1  # APPROACHING
            elif mean_mid >= inside_thresh and mean_first >= approach_thresh and mean_last < approach_thresh:
                state_label = 3  # EXITING
            else:
                # Fall back: use coverage to decide between INSIDE and OUT
                coverage = num_positive / float(self.window_size)
                state_label = 2 if coverage >= 0.3 else 0
        
        return {
            'images': images_tensor,
            'texts': texts,
            'object_counts': object_counts_tensor,
            'label': label,
            'state_label': torch.tensor(state_label, dtype=torch.long),
            'frame_id': self.frames[frame_indices[0]].get('id'),
        }
    
    def _load_image(self, frame_data: Dict) -> torch.Tensor:
        """Load and preprocess image."""
        image_path_rel = frame_data.get('image', '')
        
        # JSON paths already include 'images/' prefix, so use parent dir
        # Path structure: data/01_raw/trajectories/sparse/images/city_uuid_frame.jpg
        # JSON stores: "images/city_uuid_frame.jpg"
        # We need: images_base_dir.parent / image_path_rel
        img_path = self.images_base_dir.parent / image_path_rel
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            return torch.zeros(3, 224, 224)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            return img_tensor
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224)


class TextualTemporalDataset(Dataset):
    """
    Dataset that combines images WITH text descriptions
    for multimodal Phase 2.1 training.
    
    Uses CLIP text embeddings + visual features.
    """
    
    def __init__(
        self,
        annotations_file: Path,
        images_dir: Path,
        clip_model = None,
        window_size: int = 30,
        stride: int = 5,
    ):
        """
        Args:
            annotations_file: Path to trajectories_*.json
            images_dir: Path to images directory
            clip_model: Pre-loaded CLIP model for text embedding
            window_size: Temporal window size
            stride: Sliding window stride
        """
        self.base_dataset = TrajectoryTemporalDataset(
            annotations_file=annotations_file,
            images_dir=images_dir,
            window_size=window_size,
            stride=stride,
        )
        self.clip_model = clip_model
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.base_dataset[idx]
        
        # Embed text descriptions with CLIP if available
        if self.clip_model is not None:
            text_embeddings = self._embed_texts(sample['texts'])
            sample['text_embeddings'] = text_embeddings
        
        return sample
    
    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed text descriptions with CLIP."""
        try:
            from open_clip import tokenizer as open_clip_tokenizer
            
            tokens = open_clip_tokenizer(texts)
            with torch.no_grad():
                embeddings = self.clip_model.encode_text(tokens)
                embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
            
            return embeddings
        except Exception as e:
            logger.warning(f"Error embedding texts: {e}")
            return torch.zeros(len(texts), 512)


def collate_trajectory_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for trajectory dataset.
    Handles variable-length sequences with padding.
    """
    # Stack images (all same length)
    images = torch.stack([item['images'] for item in batch])
    
    # Stack object counts (already tensors)
    object_counts = torch.stack([item['object_counts'] for item in batch])
    
    # Stack labels
    labels = torch.stack([item['label'] for item in batch])
    state_labels = torch.stack([item['state_label'] for item in batch])
    
    # Collect texts (keep as list)
    texts = [item['texts'] for item in batch]
    
    return {
        'images': images,
        'object_counts': object_counts,
        'labels': labels,
        'state_labels': state_labels,
        'texts': texts,
    }


class NegativeImageSequenceDataset(Dataset):
    """Creates OUT-only sequences from standalone negative images."""

    def __init__(
        self,
        images_dir: Path,
        window_size: int = 30,
        max_sequences: int = 512,
    ):
        self.images_dir = Path(images_dir)
        self.window_size = window_size
        self.max_sequences = max_sequences

        self.image_paths = [p for p in self.images_dir.rglob('*.jpg') if p.is_file()]
        self.image_paths += [p for p in self.images_dir.rglob('*.png') if p.is_file()]

        if not self.image_paths:
            logger.warning(f"No negative images found in {self.images_dir}")
        else:
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[: max_sequences * window_size]

    def __len__(self) -> int:
        if not self.image_paths:
            return 0
        return min(self.max_sequences, len(self.image_paths) // max(1, self.window_size))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.image_paths:
            raise IndexError("No negative images available")

        start = (idx * self.window_size) % len(self.image_paths)
        chosen = self.image_paths[start : start + self.window_size]
        if len(chosen) < self.window_size:
            chosen = chosen + self.image_paths[: self.window_size - len(chosen)]

        images = []
        for img_path in chosen:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                logger.debug(f"Error loading negative image {img_path}: {e}")
                img_tensor = torch.zeros(3, 224, 224)
            images.append(img_tensor)

        images_tensor = torch.stack(images)
        object_counts_tensor = torch.zeros(self.window_size, dtype=torch.float32)

        return {
            'images': images_tensor,
            'texts': ['negative sample'] * self.window_size,
            'object_counts': object_counts_tensor,
            'label': torch.tensor(0.0, dtype=torch.float32),
            'state_label': torch.tensor(0, dtype=torch.long),
            'frame_id': f"negative_{idx}",
        }


class SyntheticTransitionDataset(Dataset):
    """Synthesizes APPROACHING or EXITING sequences by blending negatives."""

    def __init__(
        self,
        base_dataset: TrajectoryTemporalDataset,
        negative_paths: List[Path],
        window_size: int,
        mode: str = "approach",
    ):
        assert mode in {"approach", "exit"}, "mode must be 'approach' or 'exit'"
        self.base_dataset = base_dataset
        self.negative_paths = negative_paths
        self.window_size = window_size
        self.mode = mode
        self.transition_len = max(3, window_size // 3)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _load_neg_image(self, path: Path) -> torch.Tensor:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224))
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            logger.debug(f"Error loading synthetic negative {path}: {e}")
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]

        images = sample['images'].clone()
        object_counts = sample['object_counts'].clone()
        texts = list(sample['texts'])

        neg_imgs = [self._load_neg_image(random.choice(self.negative_paths)) for _ in range(self.transition_len)]
        neg_stack = torch.stack(neg_imgs)

        if self.mode == 'approach':
            images[: self.transition_len] = neg_stack
            object_counts[: self.transition_len] = 0
            texts[: self.transition_len] = ['approach negative'] * self.transition_len
            state_label = 1  # APPROACHING
        else:
            images[-self.transition_len :] = neg_stack
            object_counts[-self.transition_len :] = 0
            texts[-self.transition_len :] = ['exit negative'] * self.transition_len
            state_label = 3  # EXITING

        return {
            'images': images,
            'texts': texts,
            'object_counts': object_counts,
            'label': torch.tensor(1.0, dtype=torch.float32),
            'state_label': torch.tensor(state_label, dtype=torch.long),
            'frame_id': f"synthetic_{self.mode}_{idx}",
        }


def create_trajectory_dataloaders(
    trajectories_dir: Path,
    images_dir: Path,
    batch_size: int = 16,
    window_size: int = 30,
    num_workers: int = 4,
    negative_images_dir: Optional[Path] = None,
    max_negative_sequences: int = 512,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train and val dataloaders from trajectory annotations.
    
    Args:
        trajectories_dir: Path to sparse/dense trajectories directory
        images_dir: Path to trajectory images
        batch_size: Batch size
        window_size: Temporal window size
        num_workers: Number of workers for data loading
    
    Returns:
        (train_loader, val_loader) or (None, None) if files not found
    """
    trajectories_dir = Path(trajectories_dir)
    images_dir = Path(images_dir)
    
    # Try different possible locations for annotations
    possible_paths = [
        trajectories_dir / "annotations" / "trajectories_train_equidistant.json",
        trajectories_dir / "trajectories_train_equidistant.json",
        trajectories_dir / "annotations_train.json",
    ]
    
    train_annotations = None
    for p in possible_paths:
        if p.exists():
            train_annotations = p
            break
    
    if not train_annotations:
        logger.error(f"Could not find train annotations in {trajectories_dir}")
        logger.error(f"Searched: {possible_paths}")
        return None, None
    
    possible_val_paths = [
        trajectories_dir / "annotations" / "trajectories_val_equidistant.json",
        trajectories_dir / "trajectories_val_equidistant.json",
        trajectories_dir / "annotations_val.json",
    ]
    
    val_annotations = None
    for p in possible_val_paths:
        if p.exists():
            val_annotations = p
            break
    
    if not val_annotations:
        logger.warning(f"Could not find val annotations, using last 20% of train data")
        # Will handle this below
    
    logger.info(f"Train annotations: {train_annotations}")
    logger.info(f"Val annotations: {val_annotations}")
    logger.info(f"Images directory: {images_dir}")
    
    # Create datasets
    train_dataset = TrajectoryTemporalDataset(
        annotations_file=train_annotations,
        images_base_dir=images_dir,
        window_size=window_size,
        stride=max(1, window_size // 3),  # 33% overlap
    )
    
    if val_annotations and val_annotations.exists():
        val_dataset = TrajectoryTemporalDataset(
            annotations_file=val_annotations,
            images_base_dir=images_dir,
            window_size=window_size,
            stride=window_size,  # No overlap for val
        )
    else:
        logger.warning("Creating val set from train data (80/20 split)")
        # Use last 20% of train sequences for validation
        split_idx = int(len(train_dataset) * 0.8)
        train_indices = list(range(split_idx))
        val_indices = list(range(split_idx, len(train_dataset)))
        
        train_dataset.sequences = [train_dataset.sequences[i] for i in train_indices]
        val_dataset = type(train_dataset)(
            annotations_file=train_annotations,
            images_base_dir=images_dir,
            window_size=window_size,
            stride=window_size,
        )
        val_dataset.sequences = [val_dataset.sequences[i] for i in val_indices]
    
    # Optionally add negative-only sequences to train loader for OUT coverage
    if negative_images_dir:
        neg_ds = NegativeImageSequenceDataset(
            images_dir=negative_images_dir,
            window_size=window_size,
            max_sequences=max_negative_sequences,
        )
        if len(neg_ds) > 0:
            from torch.utils.data import ConcatDataset
            concat_parts = [train_dataset, neg_ds]

            # Use the same negatives to synthesize APPROACHING/EXITING transitions
            synth_approach = SyntheticTransitionDataset(
                base_dataset=train_dataset,
                negative_paths=list(neg_ds.image_paths),
                window_size=window_size,
                mode="approach",
            )
            synth_exit = SyntheticTransitionDataset(
                base_dataset=train_dataset,
                negative_paths=list(neg_ds.image_paths),
                window_size=window_size,
                mode="exit",
            )
            concat_parts.extend([synth_approach, synth_exit])

            train_dataset = ConcatDataset(concat_parts)
            logger.info(f"Added {len(neg_ds)} negative sequences from {negative_images_dir} and synthetic transitions")
        else:
            logger.warning(f"No negative sequences added (no images in {negative_images_dir})")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_trajectory_sequences,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_trajectory_sequences,
        pin_memory=True,
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches of {batch_size}")
    logger.info(f"Val loader: {len(val_loader)} batches of {batch_size}")
    
    return train_loader, val_loader
