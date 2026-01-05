#!/usr/bin/env python3
"""
Validate trajectory dataset loader.

Tests that the TrajectoryTemporalDataset can:
1. Load trajectory JSON annotations
2. Create sliding window sequences
3. Load images and texts correctly
4. Generate proper tensor shapes for training
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workzone.data.trajectory_temporal_dataset import (
    TrajectoryTemporalDataset,
    create_trajectory_dataloaders,
)


def main():
    # Paths
    workspace_root = Path("/home/wesleyferreiramaia/data/workzone")
    trajectories_dir = workspace_root / "data/01_raw/trajectories/sparse"
    images_dir = trajectories_dir / "images"
    
    print(f"Workspace root: {workspace_root}")
    print(f"Trajectories dir: {trajectories_dir}")
    print(f"Images dir: {images_dir}")
    print(f"\nDirectories exist:")
    print(f"  trajectories_dir: {trajectories_dir.exists()}")
    print(f"  images_dir: {images_dir.exists()}")
    
    # Check for annotation files
    annotations_dir = trajectories_dir / "annotations"
    if annotations_dir.exists():
        print(f"\nAnnotation files:")
        for f in annotations_dir.glob("trajectories_*.json"):
            print(f"  {f.name}")
    else:
        print(f"\nNo annotations directory found at {annotations_dir}")
    
    # Try to create dataset
    print(f"\n{'='*60}")
    print("Creating TrajectoryTemporalDataset...")
    print(f"{'='*60}\n")
    
    try:
        train_annotations = trajectories_dir / "annotations/trajectories_train_equidistant.json"
        
        dataset = TrajectoryTemporalDataset(
            annotations_file=train_annotations,
            images_base_dir=images_dir,
            window_size=30,
            stride=15,
            max_sequences=5,  # Just test with 5 for now
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Number of sequences: {len(dataset)}")
        
        # Try to get a sample
        if len(dataset) > 0:
            print(f"\nLoading sample 0...")
            sample = dataset[0]
            
            print(f"  images shape: {sample['images'].shape}")
            print(f"  object_counts shape: {sample['object_counts'].shape}")
            print(f"  object_counts values: {sample['object_counts'].tolist()}")
            print(f"  label: {sample['label'].item()}")
            print(f"  frame_id: {sample['frame_id']}")
            print(f"  texts[0]: {sample['texts'][0][:80]}...")
            
            # Check shapes for Phase 2.1 model
            seq_len = sample['images'].shape[0]
            print(f"\n✓ Sample valid for Phase 2.1 training:")
            print(f"  Sequence length: {seq_len}")
            print(f"  Image shape per frame: {sample['images'].shape[1:]}")
            print(f"  Object counts sequence: {sample['object_counts'].shape}")
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Try to create dataloaders
    print(f"\n{'='*60}")
    print("Creating dataloaders...")
    print(f"{'='*60}\n")
    
    try:
        train_loader, val_loader = create_trajectory_dataloaders(
            trajectories_dir=trajectories_dir,
            images_dir=images_dir,
            batch_size=8,
            window_size=30,
            num_workers=0,  # 0 for testing
        )
        
        if train_loader is None:
            print(f"✗ Failed to create dataloaders")
            sys.exit(1)
        
        print(f"✓ Dataloaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Load a batch
        print(f"\nLoading batch from train loader...")
        batch = next(iter(train_loader))
        
        print(f"  images shape: {batch['images'].shape}")
        print(f"  object_counts shape: {batch['object_counts'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  texts: {len(batch['texts'])} sequences")
        
        print(f"\n✓ Batch shapes valid for training:")
        batch_size, seq_len, channels, h, w = batch['images'].shape
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Image shape: {channels}x{h}x{w}")
        
    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ All validations passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
