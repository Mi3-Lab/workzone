#!/usr/bin/env python3
"""
Consolidate candidate JPEGs from GPU0 and GPU1 into a single browsable directory.
Creates symlinks to avoid duplicating ~9.6 GB of storage.
"""

import pathlib
import shutil
import os

def consolidate_candidates(gpu0_root: pathlib.Path, gpu1_root: pathlib.Path, 
                          output_root: pathlib.Path, use_symlinks: bool = True):
    """Create unified candidate directory with GPU0 + GPU1 JPEGs."""
    
    candidates_dir = output_root / "candidates_unified"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    skipped = 0
    
    for gpu_idx, gpu_root in enumerate([gpu0_root, gpu1_root], 1):
        candidates_gpu = gpu_root / "candidates"
        if not candidates_gpu.exists():
            print(f"GPU{gpu_idx} candidates dir not found: {candidates_gpu}")
            continue
        
        # Each GPU has subdirs per video (e.g., boston_042e1caf93114d3286c11ba14ddaa759_000001_02790_snippet/)
        for video_subdir in sorted(candidates_gpu.iterdir()):
            if not video_subdir.is_dir():
                continue
            
            for jpg in video_subdir.glob("*.jpg"):
                # Create destination with GPU prefix to avoid collisions
                dest_name = f"gpu{gpu_idx}_{jpg.name}"
                dest = candidates_dir / dest_name
                
                if dest.exists():
                    skipped += 1
                    continue
                
                if use_symlinks:
                    os.symlink(jpg, dest)
                else:
                    shutil.copy2(jpg, dest)
                
                total += 1
                if total % 1000 == 0:
                    print(f"  Processed {total} JPEGs...")
    
    print(f"\n✓ Consolidated {total} JPEGs into {candidates_dir}")
    print(f"  Skipped: {skipped} (duplicates)")
    print(f"  Total size (symlinks): ~{total * 0.55:.1f} MB metadata")
    print(f"\nYou can now browse:")
    print(f"  ls {candidates_dir} | head")
    print(f"  feh {candidates_dir} &")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidate GPU candidate directories")
    parser.add_argument('--gpu0', default='outputs/hardneg_mining_gpu0',
                        help='GPU0 mining output directory')
    parser.add_argument('--gpu1', default='outputs/hardneg_mining_gpu1',
                        help='GPU1 mining output directory')
    parser.add_argument('--output', default='outputs/hardneg_mining',
                        help='Output consolidation directory')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of creating symlinks')
    
    args = parser.parse_args()
    
    gpu0_root = pathlib.Path(args.gpu0)
    gpu1_root = pathlib.Path(args.gpu1)
    output_root = pathlib.Path(args.output)
    
    consolidate_candidates(gpu0_root, gpu1_root, output_root, 
                          use_symlinks=not args.copy)

if __name__ == '__main__':
    main()
