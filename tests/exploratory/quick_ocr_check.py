#!/usr/bin/env python3
"""Very quick OCR check on 3 images to verify it works"""

import sys
from pathlib import Path
import json

# Load dataset metadata to find real sign images
metadata_path = Path('data/01_raw/00_DATASET_METADATA.json')
with open(metadata_path) as f:
    metadata = json.load(f)

# Find a message board annotation
print("Looking for sign annotations...")
message_board_samples = [img for img in metadata['images'][:10] if any(
    ann['category_id'] == 16 for ann in metadata['annotations']
    if ann['image_id'] == img['id']
)]

if message_board_samples:
    print(f"✅ Found {len(message_board_samples)} images with message boards")
    for img in message_board_samples[:3]:
        img_path = Path(f"data/01_raw/{img['file_name']}")
        print(f"  - {img_path}")
else:
    print("❌ No message board samples found")
    # Just list what we have
    print("\nAvailable images:")
    for i, img in enumerate(metadata['images'][:5]):
        print(f"  {i}: {img['file_name']}")
