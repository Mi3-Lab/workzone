import cv2
import json
import time
import os
import argparse
from vlm_vllm_verifier import VLMVllmVerifier

def main():
    parser = argparse.ArgumentParser(description="Process video with VLM (Qwen2.5-VL) and save results to JSON.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename (saved in results/)")
    parser.add_argument("--hz", type=float, default=2.0, help="Sampling frequency in Hz (default: 2.0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    # Prepare output path
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_filename = args.output if args.output else f"{video_name}_vlm_results.json"
    if not output_filename.endswith(".json"):
        output_filename += ".json"
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    verifier = VLMVllmVerifier()
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    interval_frames = int(fps / args.hz) if fps > 0 else 1
    
    results = []
    frame_idx = 0
    processed = 0
    
    print(f"\n--- VLM Video Processing ---")
    print(f"Input: {args.video}")
    print(f"Video FPS: {fps:.2f} | Duration: {duration:.2f}s")
    print(f"Sampling: {args.hz} Hz (every {interval_frames} frames)")
    print(f"Output: {output_path}")
    print(f"----------------------------\n")
    
    try:
        while cap.isOpened():
            if args.limit and processed >= args.limit:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            curr_time = frame_idx / fps
            print(f"[{processed+1}] Processing frame at {curr_time:.2f}s...", end="\r")
            
            result = verifier.analyze_frame(frame)
            
            if result:
                entry = {
                    "frame_idx": frame_idx,
                    "time_sec": round(curr_time, 2),
                    "state": result.get('state'),
                    "reasoning": result.get('reasoning'),
                    "latency": result.get('latency')
                }
                results.append(entry)
                
                # Print summary of detection
                state_color = "\033[92m" if entry['state'] == "OUT" else "\033[93m" if entry['state'] == "APPROACHING" else "\033[91m"
                print(f"[{entry['time_sec']}s] State: {state_color}{entry['state']}\033[0m | Latency: {entry['latency']}s")
            
            frame_idx += interval_frames
            processed += 1
            
            if frame_idx >= total_frames:
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
    finally:
        cap.release()
        
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "video": args.video,
                    "hz": args.hz,
                    "total_processed": len(results),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": results
            }, f, indent=2)
            
        print(f"\n\nProcessing complete! {len(results)} frames saved to {output_path}")

if __name__ == "__main__":
    main()
