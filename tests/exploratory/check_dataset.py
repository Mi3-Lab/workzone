#!/usr/bin/env python3
"""Load COCO dataset and show stats"""
import json
from pathlib import Path

coco_path = Path('data/01_raw/00_DATASET_METADATA.json')
with open(coco_path) as f:
    data = json.load(f)

print(f"Images: {len(data['images'])}")
print(f"Annotations: {len(data['annotations'])}")
print(f"Categories: {len(data['categories'])}")

# Count by category
from collections import Counter
cat_ids = Counter(a['category_id'] for a in data['annotations'])
print(f"\nTop 10 categories:")
for cat_id, count in sorted(cat_ids.items(), key=lambda x: x[1], reverse=True)[:10]:
    cat_name = next((c['name'] for c in data['categories'] if c['id'] == cat_id), f"ID{cat_id}")
    print(f"  {cat_id:2d}: {cat_name:30s} - {count:5d}")
