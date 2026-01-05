# OCR Real-Time Strategy (Jetson Orin 64GB)

## Target Hardware
- **Device**: NVIDIA Jetson Orin 64GB
- **GPU**: 2048 CUDA cores (Ampere architecture)
- **RAM**: 64GB unified memory
- **Power**: 15-60W (configurable)
- **TensorRT**: Native support for optimized inference

## Problem
PaddleOCR takes ~3 seconds per detection on CPU (too slow for 30 FPS video)

## Solution: Smart Sampling + TensorRT Optimization

### 1. **Trigger-Based OCR** (RECOMMENDED)
```python
# Only run OCR when YOLO detects sign with high confidence
if yolo_score >= 0.6 and category in [11, 16, 17]:  # Signs with text
    if frame_idx % 30 == 0:  # Every 1 second at 30fps
        text_confidence = run_ocr(crop)
    else:
        text_confidence = cached_text_confidence  # Reuse last value
```

**Performance:**
- OCR frequency: 1x per second (vs 30x per second)
- 30x speedup → **0.1 seconds per OCR** (viable!)
- Signs are visible for 2-5 seconds → 2-5 OCR runs per sign

### 2. **State-Based Caching**
```python
if state == "INSIDE_WORKZONE":
    # Already inside, no need to keep running OCR
    text_confidence = last_high_confidence_value
elif state == "APPROACHING":
    # Critical phase - run OCR more frequently
    if frame_idx % 15 == 0:  # Every 0.5 seconds
        text_confidence = run_ocr(crop)
```

### 3. **TensorRT Optimization (CRITICAL for Jetson)**
- Current: CPU-based PaddleOCR (~3s)
- With Jetson GPU (FP32): **~0.2-0.3s per OCR**
- With TensorRT (FP16): **~0.05-0.1s per OCR** ⚡
- With TensorRT (INT8): **~0.03-0.05s per OCR** ⚡⚡

**Key considerations:**
- YOLO + OCR share same GPU → need memory management
- Thermal throttling at 60W power mode → prefer 30W mode
- Batch processing: Run OCR on multiple crops at once (2-4x speedup)

### 4. **Parallel Processing**
```python
# Run OCR in background thread/process
ocr_queue = Queue()
ocr_results = Queue()

# Main thread: YOLO detection + state machine (30 FPS)
# Background thread: OCR processing (1 FPS)
```

## Recommended Implementation

### Phase 1: Trigger-Based OCR (Immediate - works on CPU)
- Run OCR only when YOLO score ≥ 0.6
- Sample every 30 frames (1 Hz)
- Cache result for intermediate frames
- **Expected latency: < 0.2 seconds per OCR (with Jetson GPU)**

### Phase 2: TensorRT Conversion (Before deployment)
- Convert PaddleOCR models to TensorRT format
- Use FP16 precision (balance speed/accuracy)
- Optimize for Jetson Orin architecture
- **Expected latency: < 0.08 seconds per OCR**
- **Tools**: `trtexec`, `onnx2trt`, PaddlePaddle TensorRT plugin

### Phase 3: Memory & Power Optimization (Deployment)
- Unified memory management (YOLO + OCR share 64GB)
- Power mode: 30W (sustained performance without throttling)
- Asynchronous OCR processing (CUDA streams)
- **Expected latency: < 0.05 seconds per OCR**

### Phase 4: State-Aware Sampling (Production)
- APPROACHING state: OCR every 0.5s (15 frames)
- INSIDE state: OCR every 2s (60 frames) - maintenance only
- OUTSIDE state: OCR disabled
- **Expected average overhead: < 2%**

## Performance Targets

| Metric | Current (CPU) | Jetson (GPU) | TensorRT FP16 | Production |
|--------|---------------|--------------|---------------|------------|
| OCR latency | 3.0s | 0.2s | 0.08s | 0.05s |
| OCR frequency | N/A | 1 Hz | 2 Hz | 1-2 Hz |
| GPU memory | 0 MB | ~500 MB | ~300 MB | ~300 MB |
| Power draw | N/A | +5-8W | +3-5W | +3-5W |
| Total overhead | N/A | 20% | 8% | 2% |
| Real-time viable? | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

## Jetson-Specific Optimizations

### 1. **Shared YOLO+OCR Pipeline**
```python
# Both models share same GPU context
yolo_model = YOLO("model.pt").to('cuda')
ocr_detector = SignTextDetector(device='cuda')  # Reuse GPU

# Use unified memory efficiently
with torch.cuda.stream(stream1):
    yolo_results = yolo_model(frame)
with torch.cuda.stream(stream2):
    ocr_results = ocr_detector.extract_text(crop)
```

### 2. **TensorRT Conversion Steps**
```bash
# 1. Export PaddleOCR to ONNX
paddle2onnx --model_dir paddleocr_model \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file model.onnx

# 2. Convert ONNX to TensorRT (Jetson Orin)
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=2048 \
        --minShapes=input:1x3x32x320 \
        --optShapes=input:1x3x48x640 \
        --maxShapes=input:1x3x64x960
```

### 3. **Power Mode Selection**
```bash
# For testing (max performance)
sudo nvpmodel -m 0  # MAXN mode (60W)

# For production (sustained performance)
sudo nvpmodel -m 2  # 30W mode (no throttling)

# Monitor thermals
sudo tegrastats --interval 1000
```

## Code Changes Required

See: `src/workzone/apps/streamlit/app_phase2_1_evaluation.py`
- Line ~807: Add frame sampling condition
- Line ~534: Add cached text_confidence fallback
- Line ~323: Add GPU device support


## Deployment Checklist (Jetson Orin 64GB)

- [ ] Convert PaddleOCR models to TensorRT format (FP16)
- [ ] Test memory usage with both YOLO + OCR running
- [ ] Benchmark latency at different power modes (15W/30W/60W)
- [ ] Implement frame sampling (every 30 frames = 1 Hz)
- [ ] Add confidence caching mechanism
- [ ] Test thermal stability under continuous operation
- [ ] Validate OCR accuracy with TensorRT FP16 vs FP32

## Expected Final Performance (Jetson Orin 64GB)

✅ **Real-time processing: 30 FPS sustained**
- YOLO inference: ~15-20ms per frame
- OCR inference: ~50-80ms per detection (every 30 frames)
- State machine: ~1ms per frame  
- **Total: ~16-21ms per frame (< 33ms budget for 30 FPS)** ✅

✅ **Power efficient: 30W mode (no throttling)**
✅ **Memory efficient: < 4GB GPU memory (YOLO + OCR)**
✅ **Accurate: FP16 precision (minimal accuracy loss < 1%)**

## Risk Mitigation

### Risk 1: GPU Memory Overflow
**Impact**: OOM crashes during inference
**Mitigation**: 
- Lazy loading: Load OCR model only when needed
- Model quantization: INT8 if FP16 still too large
- Monitor: Add GPU memory tracking

### Risk 2: Thermal Throttling
**Impact**: Performance degradation over time
**Mitigation**:
- 30W power mode (sustained performance)
- Active cooling (fan at 50%+)
- Reduce OCR frequency if temp > 75°C

### Risk 3: TensorRT Conversion Issues
**Impact**: OCR accuracy degradation
**Mitigation**:
- Validate accuracy on test set before deployment
- Keep FP32 fallback if accuracy drops > 2%
- Use calibration dataset for INT8 quantization

## References

- PaddleOCR TensorRT docs: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/tensorrt/README.md
- Jetson Orin benchmarks: https://developer.nvidia.com/embedded/jetson-benchmarks
- TensorRT optimization guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
