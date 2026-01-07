# Jetson Orin Optimization Guide

## Hardware Overview

### NVIDIA Jetson Orin 64GB
- **GPU**: 2048 CUDA cores, 64 Tensor Cores
- **Memory**: 64GB unified memory (shared CPU/GPU)
- **Compute**: 275 TOPS (INT8), 8.8 TFLOPS (FP16)
- **Power**: 15W-60W modes

### RTX 4090 (Development)
- **GPU**: 16384 CUDA cores, 512 Tensor Cores (4th gen)
- **Memory**: 24GB GDDR6X
- **Compute**: 1321 TOPS (INT8), 82.6 TFLOPS (FP16)

## Optimization Strategy

### 1. TensorRT Conversion (CRITICAL)
**Impact**: 3-5x speedup, uses Tensor Cores

```python
# Convert YOLO to TensorRT
from ultralytics import YOLO

# On RTX 4090 (development)
model = YOLO('weights/yolo12s_hardneg_1280.pt')
model.export(format='engine', device=0, half=True, imgsz=1280)
# → weights/yolo12s_hardneg_1280.engine

# On Jetson (inference)
model = YOLO('weights/yolo12s_hardneg_1280.engine')
results = model.predict(frame, device=0, half=True)
```

**Benefits**:
- Automatic kernel fusion
- Tensor Core utilization (FP16)
- Memory optimization
- Dynamic shape optimization

### 2. Precision Optimization

#### FP16 (Recommended for Jetson)
```python
# YOLO with FP16
results = model.predict(frame, half=True, device=0)

# CLIP with FP16
clip_model = clip_model.half()  # Convert to FP16
with torch.cuda.amp.autocast():
    features = clip_model.encode_image(images)
```

**Expected performance**:
- RTX 4090: 2x faster than FP32
- Jetson Orin: 3x faster than FP32

#### INT8 (Maximum Speed)
```python
# Export with INT8 calibration
model.export(
    format='engine',
    device=0,
    half=False,
    int8=True,
    data='path/to/calibration_data.yaml'  # 100-500 images
)
```

**Expected performance**:
- RTX 4090: 3-4x faster than FP32
- Jetson Orin: 5-8x faster than FP32

### 3. Model Size Reduction

#### Current Models
- `yolo12s_hardneg_1280.pt`: ~22MB, 11M params
- CLIP ViT-B/32: ~350MB

#### Jetson-Optimized
```python
# Use smaller YOLO variant
yolo_nano = YOLO('yolov8n.pt')  # 6MB, 3M params
yolo_nano.export(format='engine', half=True, imgsz=640)

# Prune YOLO (advanced)
from torch.nn.utils import prune
# ... pruning code
```

### 4. CUDA Streams (Parallel Processing)

```python
import torch

# Create streams for parallel execution
stream_yolo = torch.cuda.Stream()
stream_clip = torch.cuda.Stream()
stream_ocr = torch.cuda.Stream()

# Parallel inference
with torch.cuda.stream(stream_yolo):
    yolo_results = yolo_model(frame)

with torch.cuda.stream(stream_clip):
    clip_features = clip_model(frame)

with torch.cuda.stream(stream_ocr):
    ocr_text = ocr_detector(frame)

# Synchronize
torch.cuda.synchronize()
```

**Speedup**: 30-50% for multi-model pipelines

### 5. Memory Optimization

#### Current Memory Usage (Estimated)
- YOLO FP32: ~2GB
- CLIP: ~1.5GB
- EasyOCR: ~1GB
- Frame buffers: ~500MB
- **Total**: ~5GB

#### Jetson-Optimized
```python
# Enable unified memory optimizations
torch.cuda.set_per_process_memory_fraction(0.7)  # Reserve 70% max
torch.backends.cudnn.benchmark = True  # Auto-tune kernels

# Clear cache periodically
if frame_idx % 100 == 0:
    torch.cuda.empty_cache()

# Use pinned memory for faster CPU↔GPU transfers
frame_tensor = torch.from_numpy(frame).pin_memory()
```

### 6. OCR Optimization

#### Problem: EasyOCR is slow
- Current: ~300ms per frame on GPU
- Target: <50ms per frame

#### Solutions:

**Option A: PaddleOCR Lite (Recommended)**
```bash
pip install paddlepaddle-gpu paddleocr
```

```python
from paddleocr import PaddleOCR

# Lightweight model
ocr = PaddleOCR(
    use_angle_cls=False,  # Disable rotation detection
    lang='en',
    det_model_dir='en_PP-OCRv3_det_slim',  # Slim model
    rec_model_dir='en_PP-OCRv4_rec_slim',
    use_gpu=True,
    enable_mkldnn=True
)
```

**Option B: TensorRT OCR**
```python
# Convert PaddleOCR to TensorRT
import paddle2onnx
import onnx
# ... conversion pipeline
```

**Option C: Skip frames aggressively**
```python
# Only run OCR every 10-30 frames
ocr_every_n = 30  # 1 FPS OCR on 30 FPS video
```

### 7. Video Pipeline Optimization

#### Current Pipeline
```
Read Frame → YOLO → CLIP → OCR → Fusion → State Machine → Display
```

#### Optimized Pipeline
```
Read Frame (async) → [YOLO + CLIP + OCR (parallel)] → Fusion → State Machine → Display
```

```python
import cv2
import threading
from queue import Queue

class AsyncVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.q = Queue(maxsize=3)
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self._reader, daemon=True).start()
        return self
    
    def _reader(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put(frame)
    
    def read(self):
        return self.q.get()
```

### 8. Jetson Power Modes

```bash
# Maximum performance (60W)
sudo nvpmodel -m 0
sudo jetson_clocks

# Balanced (30W)
sudo nvpmodel -m 2

# Power efficient (15W)
sudo nvpmodel -m 3
```

### 9. Batch Processing

```python
# Process multiple frames at once (GPU utilization)
batch_size = 4
frames_batch = []

for i in range(batch_size):
    ret, frame = cap.read()
    if ret:
        frames_batch.append(frame)

# Batch inference
results = model.predict(frames_batch, batch=True)
```

## Implementation Checklist

### Phase 1: TensorRT Conversion (High Priority)
- [ ] Export YOLO to TensorRT (.engine)
- [ ] Test inference speed (should be 3-5x faster)
- [ ] Verify accuracy (should be same as .pt)
- [ ] Update app to use .engine when available

### Phase 2: FP16 Optimization
- [ ] Enable FP16 for YOLO
- [ ] Enable FP16 for CLIP
- [ ] Benchmark memory usage
- [ ] Validate outputs

### Phase 3: CUDA Streams
- [ ] Implement parallel inference
- [ ] Benchmark pipeline throughput
- [ ] Profile with `nvprof` or `nsys`

### Phase 4: OCR Optimization
- [ ] Test PaddleOCR Lite
- [ ] Increase `ocr_every_n` to 20-30
- [ ] Consider disabling OCR for real-time

### Phase 5: Memory Optimization
- [ ] Set memory limits
- [ ] Implement cache clearing
- [ ] Use pinned memory
- [ ] Profile with `nvidia-smi`

### Phase 6: Video Pipeline
- [ ] Implement async frame reading
- [ ] Add frame dropping logic
- [ ] Benchmark end-to-end FPS

## Benchmarking

### Current Performance (Estimated)
- **RTX 4090**: ~30 FPS (full pipeline)
- **Jetson Orin**: ~5-10 FPS (unoptimized)

### Target Performance
- **RTX 4090**: ~60+ FPS
- **Jetson Orin**: ~30 FPS (with optimizations)

### Benchmark Script
```python
import time
import torch
from ultralytics import YOLO

model = YOLO('weights/yolo12s_hardneg_1280.engine')
model.to('cuda')

# Warmup
for _ in range(10):
    model.predict(dummy_frame)

# Benchmark
times = []
for _ in range(100):
    t0 = time.perf_counter()
    results = model.predict(frame, verbose=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

print(f"Mean: {np.mean(times)*1000:.1f}ms")
print(f"FPS: {1/np.mean(times):.1f}")
```

## Jetson-Specific Tips

### 1. Monitor Temperature
```bash
sudo tegrastats
```

### 2. Increase Swap (if needed)
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. Docker Container (Recommended)
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN pip install ultralytics paddleocr opencv-python
```

### 4. JetPack Version
- Ensure JetPack 5.1+ for best TensorRT support
- Check: `sudo apt-cache show nvidia-jetpack`

## Expected Results

| Component | RTX 4090 (FP32) | RTX 4090 (TRT) | Jetson (FP32) | Jetson (TRT) |
|-----------|----------------|----------------|---------------|--------------|
| YOLO      | 8ms            | 2ms            | 60ms          | 15ms         |
| CLIP      | 12ms           | 5ms            | 80ms          | 25ms         |
| OCR       | 300ms          | 300ms          | 400ms         | 400ms        |
| **Total** | 320ms (3 FPS)  | 307ms (3 FPS)  | 540ms (2 FPS) | 440ms (2 FPS)|

**With optimizations**:
| Component | RTX 4090 | Jetson Orin |
|-----------|----------|-------------|
| YOLO (TRT)| 2ms      | 15ms        |
| CLIP (TRT)| 5ms      | 25ms        |
| OCR (skip)| 10ms*    | 33ms*       |
| **Total** | 17ms (58 FPS) | 33ms (30 FPS) |

*OCR runs every 30 frames = effective 10-33ms amortized

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Jetson Orin Developer Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit)
- [Ultralytics TensorRT Export](https://docs.ultralytics.com/modes/export/#arguments)
