#!/usr/bin/env python3
"""
TensorRT Optimization Script for Workzone Detection
Converts models to TensorRT for maximum performance on Jetson Orin / RTX GPUs
"""

import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def check_gpu():
    """Check GPU availability and specs."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ Memory: {gpu_memory:.1f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    
    # Check if Jetson
    is_jetson = "tegra" in gpu_name.lower() or "orin" in gpu_name.lower()
    if is_jetson:
        print("🤖 Detected Jetson device")
    else:
        print("🖥️  Detected desktop/server GPU")
    
    return True

def export_yolo_tensorrt(
    model_path: str,
    output_dir: str = "weights",
    half: bool = True,
    int8: bool = False,
    imgsz: int = 1280,
    workspace: int = 4,
    dla_core: int = None,
):
    """
    Export YOLO model to TensorRT engine.
    
    Args:
        model_path: Path to .pt model
        output_dir: Output directory
        half: Use FP16 precision (recommended)
        int8: Use INT8 precision (requires calibration)
        imgsz: Input image size
        workspace: TensorRT workspace size in GB
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Exporting: {model_path.name}")
    print(f"{'='*70}")
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(str(model_path))
    
    # Export settings
    precision = "INT8" if int8 else ("FP16" if half else "FP32")
    print(f"  Precision: {precision}")
    print(f"  Image size: {imgsz}")
    print(f"  Workspace: {workspace}GB")
    
    # Export
    try:
        # Step 1: Export to ONNX first (required for trtexec DLA conversion)
        onnx_path = Path(model_path).with_suffix('.onnx')
        if not onnx_path.exists():
            print("Exporting to ONNX...")
            onnx_path = model.export(format='onnx', imgsz=imgsz, half=half, opset=12)
        else:
            print(f"✓ ONNX model already exists at {onnx_path}")
        
        # Step 2: Use trtexec for DLA if requested
        if dla_core is not None:
            engine_path = Path(onnx_path).with_suffix('.engine')
            print(f"🚀 Using trtexec to build for DLA Core {dla_core}...")
            import subprocess
            
            # Common Jetson path for trtexec
            trtexec_bin = "/usr/src/tensorrt/bin/trtexec"
            if not Path(trtexec_bin).exists():
                trtexec_bin = "trtexec" # Fallback to path

            cmd = [
                trtexec_bin,
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                f"--useDLACore={dla_core}",
                "--allowGPUFallback",
                f"--memPoolSize=workspace:{workspace * 1024}",
            ]
            if half: cmd.append("--fp16")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ trtexec failed: {result.stderr}")
                return False
            export_path = str(engine_path)
        else:
            # Standard GPU Export
            export_path = model.export(
                format='engine',
                device=0,
                half=half,
                int8=int8,
                imgsz=imgsz,
                workspace=workspace,
                verbose=True,
            )
        
        engine_path = Path(export_path)
        engine_size = engine_path.stat().st_size / 1e6
        
        print(f"\n✅ Export successful!")
        print(f"   Engine: {engine_path}")
        print(f"   Size: {engine_size:.1f} MB")
        
        # Benchmark
        print(f"\n🔥 Benchmarking...")
        benchmark_model(export_path, imgsz)
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def benchmark_model(model_path: str, imgsz: int = 1280, n_runs: int = 100):
    """Benchmark model inference speed."""
    import time
    import numpy as np
    
    model = YOLO(model_path)
    
    # Create dummy input
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup
    print("   Warming up...")
    for _ in range(10):
        model.predict(dummy_frame, verbose=False, device=0)
    
    # Benchmark
    print(f"   Running {n_runs} iterations...")
    times = []
    
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy_frame, verbose=False, device=0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    mean_time = np.mean(times) * 1000
    fps = 1 / np.mean(times)
    
    print(f"\n   Results:")
    print(f"   ⏱️  Mean: {mean_time:.1f}ms")
    print(f"   📊 FPS: {fps:.1f}")
    print(f"   📈 Throughput: {fps * imgsz * imgsz / 1e6:.1f} Mpix/s")

def optimize_all_models(weights_dir: str = "weights", half: bool = True):
    """Optimize all .pt models in weights directory."""
    weights_path = Path(weights_dir)
    
    if not weights_path.exists():
        print(f"❌ Weights directory not found: {weights_dir}")
        return
    
    pt_models = list(weights_path.glob("*.pt"))
    
    if not pt_models:
        print(f"❌ No .pt models found in {weights_dir}")
        return
    
    print(f"Found {len(pt_models)} models:")
    for model in pt_models:
        print(f"  - {model.name}")
    
    print(f"\n{'='*70}")
    print("Starting batch conversion...")
    print(f"{'='*70}\n")
    
    results = []
    for model_path in pt_models:
        success = export_yolo_tensorrt(
            str(model_path),
            output_dir=str(weights_path),
            half=half,
        )
        results.append((model_path.name, success))
        print()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\n{successful}/{len(results)} models converted successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT for optimal performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to specific .pt model to convert"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Directory containing .pt models (default: weights/)"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision (default: FP16)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 precision (requires calibration data)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size (default: 1280)"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace size in GB (default: 4)"
    )
    parser.add_argument(
        "--dla",
        type=int,
        choices=[0, 1],
        help="Use DLA core (0 or 1) instead of GPU"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("YOLO TensorRT Converter")
    print("="*70)
    
    # Check GPU
    if not check_gpu():
        sys.exit(1)
    
    # Convert
    half = not args.fp32
    
    if args.model:
        # Single model
        success = export_yolo_tensorrt(
            args.model,
            half=half,
            int8=args.int8,
            imgsz=args.imgsz,
            workspace=args.workspace,
            dla_core=args.dla
        )
        sys.exit(0 if success else 1)
    else:
        # All models in directory
        optimize_all_models(args.weights_dir, half=half)

if __name__ == "__main__":
    main()
