# WorkZone Applications Guide

This document describes all interactive applications in the WorkZone project, including Streamlit web apps and Alpamayo VLA applications.

## 📁 Application Structure

```
src/workzone/apps/
├── __init__.py
├── streamlit_utils.py         # Shared Streamlit utilities
├── alpamayo_utils.py          # Shared Alpamayo/VLA utilities
│
├── streamlit/                 # Streamlit web applications
│   ├── app_basic_detection.py      # Basic YOLO detection app
│   ├── app_advanced_scoring.py     # Advanced scoring with EMA
│   └── app_semantic_fusion.py      # CLIP semantic verification
│
└── alpamayo/                  # Alpamayo VLA applications
    ├── alpamayo_10hz_inspector.py  # 10Hz real-time inference
    └── alpamayo_threaded.py        # Zero-lag threaded inference
```

## 🌐 Streamlit Applications

### 1. Basic Detection App

**File**: `src/workzone/apps/streamlit/app_basic_detection.py`

**Features**:
- Upload or select demo videos
- YOLO object detection on construction zones
- Simple work zone scoring (0-1 scale)
- Class counting
- Batch mode (save annotated video)
- Live preview mode (real-time playback)

**Usage**:
```bash
# Run from project root
streamlit run src/workzone/apps/streamlit/app_basic_detection.py

# Or with specific port
streamlit run src/workzone/apps/streamlit/app_basic_detection.py --server.port 8501
```

**Interface**:
- **Sidebar**: Model settings, confidence/IOU thresholds, device selection
- **Main**: Video upload/selection, run button
- **Results**: Annotated video, global score, class counts

**Configuration**:
- Default weights: `weights/best.pt`
- Demo videos: `data/demo/*.mp4`
- Device: CPU or CUDA (auto-detected)

---

### 2. Advanced Scoring App

**File**: `src/workzone/apps/streamlit/app_advanced_scoring.py`

**Features**:
- All basic detection features
- **Semantic grouping**: Channelization, workers, vehicles
- **Statistical normalization**: Z-score based scoring
- **Exponential Moving Average (EMA)**: Smoothed scores
- **Frame-by-frame score plots**: Matplotlib visualization
- Advanced metrics dashboard

**Usage**:
```bash
streamlit run src/workzone/apps/streamlit/app_advanced_scoring.py
```

**Scoring System**:
- Groups classes into semantic categories
- Normalizes by dataset statistics (mean/std)
- Applies weighted logistic transformation
- Smooths with EMA (alpha=0.3)

**Semantic Groups**:
- **Channelization**: Cones, drums, barricades, barriers, vertical panels, markers, fences
- **Workers**: Workers, police officers
- **Vehicles**: Work vehicles, police vehicles

**Weights**:
- Channelization: 0.9
- Workers: 0.7
- Vehicles: 0.4

---

### 3. Semantic Fusion App

**File**: `src/workzone/apps/streamlit/app_semantic_fusion.py`

**Features**:
- YOLO + CLIP semantic verification
- Anti-flicker state machine
- CLIP similarity scoring
- Temporal coherence
- Timeline visualization
- Multi-level status (entering/inside/exiting)

**Usage**:
```bash
streamlit run src/workzone/apps/streamlit/app_semantic_fusion.py
```

**How It Works**:
1. YOLO detects objects in frame
2. CLIP evaluates semantic similarity to work zone concepts
3. State machine tracks zone status over time
4. Hysteresis prevents flicker
5. Confidence boost from CLIP agreement

**State Machine**:
- **NOT_IN_ZONE**: No work zone detected
- **ENTERING**: Initial work zone detection
- **IN_ZONE**: Confirmed work zone presence
- **EXITING**: Work zone ending

---

## 🤖 Alpamayo VLA Applications

### 1. 10Hz Inspector

**File**: `src/workzone/apps/alpamayo/alpamayo_10hz_inspector.py`

**Features**:
- Real-time VLA reasoning at 10Hz cadence
- Threaded inference (non-blocking)
- Live reasoning overlay on video
- Safety report generation
- Frame mailbox architecture (zero backlog)

**Usage**:
```bash
# Run from project root
python -m src.workzone.apps.alpamayo.alpamayo_10hz_inspector \
  --video data/Construction_Data/video.mp4 \
  --output results/alpamayo_10hz_output.mp4
```

**Command-line Options**:
```
--video PATH          Input video file (required)
--model MODEL_ID      Alpamayo model ID (default: nvidia/Alpamayo-R1-10B)
--device DEVICE       Device: cuda or cpu (default: cuda)
--output PATH         Output video path (optional)
--clip-id ID          Template clip ID
```

**Architecture**:
- **Main Thread**: Video playback and display
- **AI Thread**: VLA inference at 10Hz
- **Frame Mailbox**: Latest frame communication
- **Reasoning Buffer**: Thread-safe text storage

**Safety Report Format**:
1. ZONE STATUS: Entering/inside/exiting
2. HAZARDS: List of detected objects
3. SPEED LIMIT: Detected or 'Unknown'
4. ACTION: Recommended driving action

---

### 2. Threaded Zero-Lag Player

**File**: `src/workzone/apps/alpamayo/alpamayo_threaded.py`

**Features**:
- Zero-lag architecture
- Asynchronous inference
- Real-time video playback
- No frame queue buildup

**Usage**:
```bash
python -m src.workzone.apps.alpamayo.alpamayo_threaded \
  --video data/Construction_Data/video.mp4
```

**Key Difference from 10Hz**:
- 10Hz Inspector: **Rate-limited** to 10 Hz exactly (matches paper)
- Threaded Player: **Best-effort** inference (no rate limiting)

---

## 🛠️ Shared Utilities

### Streamlit Utilities (`streamlit_utils.py`)

**Functions**:
- `load_model_cached()`: Cache uploaded model weights
- `load_model_default()`: Cache default model
- `resolve_device()`: Resolve device string
- `list_demo_videos()`: List demo videos
- `get_video_properties()`: Get video metadata
- `draw_detection_boxes()`: Draw bounding boxes
- `draw_workzone_banner()`: Draw score banner
- `compute_simple_workzone_score()`: Basic scoring
- `logistic()`: Sigmoid function

**Class Definitions**:
- `WORKZONE_CLASSES_10`: 10-class workzone taxonomy
- `CHANNELIZATION`: Semantic group
- `WORKERS`: Semantic group
- `VEHICLES`: Semantic group
- `MESSAGE_BOARDS`: Semantic group

---

### Alpamayo Utilities (`alpamayo_utils.py`)

**Classes**:
- `FrameMailbox`: Thread-safe frame communication
- `ReasoningTextBuffer`: Thread-safe text buffer
- `AlpamayoInferenceWorker`: Inference worker thread

**Functions**:
- `clean_and_wrap_text()`: Clean model output
- `create_safety_instruction()`: Generate VLA prompt
- `draw_reasoning_overlay()`: Draw text overlay

---

## 🚀 Quick Start Examples

### Run Basic Detection App
```bash
# Start Streamlit app
streamlit run src/workzone/apps/streamlit/app_basic_detection.py

# Open browser to http://localhost:8501
# Upload video or select demo
# Click "Run Detection"
```

### Run Alpamayo 10Hz Inspector
```bash
# Process video with VLA reasoning
python -m src.workzone.apps.alpamayo.alpamayo_10hz_inspector \
  --video data/Construction_Data/sample.mp4 \
  --output results/vla_output.mp4

# Press 'q' to quit during playback
```

### Run Advanced Scoring App
```bash
# Start app with advanced metrics
streamlit run src/workzone/apps/streamlit/app_advanced_scoring.py

# View semantic groups and EMA-smoothed scores
# Export score timeline plots
```

---

## ⚙️ Configuration

### Environment Variables
```bash
export DEVICE=cuda                    # or cpu
export DEFAULT_WEIGHTS=weights/best.pt
export DEMO_VIDEOS_DIR=data/demo
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "localhost"

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## 📊 Performance

### Streamlit Apps
- **FPS**: Depends on model and device (10-100 fps)
- **Latency**: <50ms per frame (YOLO inference)
- **Memory**: 2-8GB GPU memory

### Alpamayo Apps
- **FPS**: 10 Hz (fixed cadence)
- **Latency**: ~100ms per inference cycle
- **Memory**: 12-20GB GPU memory

---

## 🐛 Troubleshooting

### Streamlit Won't Start
```bash
# Check if port is in use
lsof -i :8501

# Kill existing process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8502
```

### CUDA Out of Memory
```python
# In streamlit_utils.py, reduce batch size
# or use CPU mode

device = "cpu"  # Force CPU
```

### Alpamayo Import Errors
```bash
# Make sure alpamayo is in sys.path
export PYTHONPATH=$PYTHONPATH:/path/to/workzone/alpamayo/src

# Or run from project root
cd /home/cvrr/Code/workzone
python -m src.workzone.apps.alpamayo.alpamayo_10hz_inspector ...
```

### Video Won't Open
```bash
# Check video codecs
ffprobe video.mp4

# Convert to compatible format
ffmpeg -i input.mkv -c:v libx264 -c:a aac output.mp4
```

---

## 📝 Development

### Adding New Streamlit App
1. Create file in `src/workzone/apps/streamlit/`
2. Import utilities from `streamlit_utils.py`
3. Follow PEP 8 and add type hints
4. Add docstrings
5. Test with `streamlit run`

### Adding New Alpamayo App
1. Create file in `src/workzone/apps/alpamayo/`
2. Import utilities from `alpamayo_utils.py`
3. Use `FrameMailbox` and `AlpamayoInferenceWorker`
4. Add CLI with argparse
5. Document in this guide

---

## 📚 References

- **Streamlit**: https://docs.streamlit.io/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Alpamayo R1**: https://huggingface.co/nvidia/Alpamayo-R1-10B
- **CLIP**: https://github.com/openai/CLIP

---

**For questions or issues, see the main [README.md](../../README.md)**
