# OCR CLI Usage Guide



### Basic Command with OCR

```bash
python tools/process_video_fusion.py \
  /path/to/video.mp4 \
  --output-dir outputs/test_ocr \
  --enable-ocr \
  --no-motion
```

### Full Example with All Features

```bash
python tools/process_video_fusion.py \
  data/videos_compressed/boston_2bdb5a72602342a5991b402beb8b7ab4_000001_23370_snippet.mp4 \
  --output-dir outputs/phase1_4_complete \
  --enable-ocr \
  --enable-phase1-4 \
  --scene-context-weights weights/scene_context_classifier.pt \
  --no-motion \
  --stride 2 \
  --conf 0.25 \
  --ema-alpha 0.25
```

## Command-Line Options

### OCR Options

- `--enable-ocr`: Enable OCR text extraction from message boards (requires OCR modules installed)

### Core Options

- `--no-clip`: Disable CLIP semantic verification
- `--no-motion`: Disable Phase 1.3 motion validation
- `--stride N`: Process every Nth frame (default: 1)
- `--conf CONF`: YOLO confidence threshold (default: 0.25)
- `--iou IOU`: YOLO IoU threshold (default: 0.45)

### Fusion Parameters

- `--ema-alpha ALPHA`: EMA smoothing factor (default: 0.25)
- `--clip-weight WEIGHT`: CLIP fusion weight (default: 0.35)
- `--clip-trigger-th TH`: CLIP trigger threshold (default: 0.45)
- `--orange-weight WEIGHT`: Orange boost weight (default: 0.25)
- `--context-trigger-below TH`: Apply orange boost if YOLO_ema below this (default: 0.55)

### State Machine

- `--enter-th TH`: State machine entry threshold (default: 0.70)
- `--exit-th TH`: State machine exit threshold (default: 0.45)
- `--approach-th TH`: Approaching state threshold (default: 0.55)
- `--min-inside-frames N`: Min frames inside before exit (default: 25)
- `--min-out-frames N`: Min frames outside before entry (default: 15)

### Phase 1.4 Scene Context

- `--enable-phase1-4`: Enable scene context classifier
- `--scene-context-weights PATH`: Path to scene context weights (default: weights/scene_context_classifier.pt)

### Output Options

- `--output-dir DIR`: Output directory for video and CSV (default: outputs/)
- `--no-video`: Skip annotated video output (faster)
- `--no-csv`: Skip CSV timeline output (faster)
- `--quiet`: Suppress progress prints

## Output Files

### CSV Timeline

The output CSV includes OCR columns:

| Column | Description |
|--------|-------------|
| `frame` | Frame number |
| `time_sec` | Timestamp in seconds |
| `yolo_score` | Raw YOLO detection score |
| `yolo_score_ema` | EMA-smoothed YOLO score |
| `fused_score_ema` | Final fused score (YOLO + CLIP + OCR + Orange) |
| `state` | Work zone state (OUT, APPROACHING, INSIDE, EXITING) |
| `clip_used` | Whether CLIP was active (0/1) |
| `clip_score` | Raw CLIP semantic score |
| `count_channelization` | Number of channelization objects |
| `count_workers` | Number of workers detected |
| `ocr_text` | ✨ **Extracted text from message boards** |
| `text_confidence` | ✨ **OCR confidence (0-1)** |
| `text_category` | ✨ **Text category (WORKZONE, SPEED, LANE, CAUTION, etc)** |

### Annotated Video

The output video shows:
- Detection bounding boxes
- State banner with color coding
- Fused score display
- CLIP activation indicator

## OCR Boost Logic

When `--enable-ocr` is used:

1. **Text Extraction**: Detects message boards (YOLO class 16) and extracts text using PaddleOCR
2. **Classification**: Classifies text into categories (WORKZONE, SPEED, LANE, CAUTION, etc)
3. **Score Boost**: When text confidence ≥ 0.70 and category is WORKZONE/LANE/CAUTION:
   - Boost fused score by up to 15%
   - Formula: `ocr_boost = min(text_confidence * 0.15, 0.15)`
   - Applied as: `fused = min(fused + ocr_boost, 1.0)`

## Examples

### Test on Boston Dataset Video

```bash
python tools/process_video_fusion.py \
  data/videos_compressed/boston_2bdb5a72602342a5991b402beb8b7ab4_000001_23370_snippet.mp4 \
  --output-dir outputs/boston_ocr_test \
  --enable-ocr \
  --no-motion
```

### Fast Processing (No Video Output)

```bash
python tools/process_video_fusion.py \
  data/videos_compressed/video.mp4 \
  --output-dir outputs/fast_test \
  --enable-ocr \
  --no-video \
  --stride 3 \
  --quiet
```

### Full Pipeline (All Features)

```bash
python tools/process_video_fusion.py \
  data/videos_compressed/video.mp4 \
  --output-dir outputs/full_pipeline \
  --enable-ocr \
  --enable-phase1-4 \
  --scene-context-weights weights/scene_context_classifier.pt \
  --stride 2
```

## Validation

### Check OCR Text Extractions in CSV

```bash
# Show frames with OCR text detected
grep -v "^frame," outputs/test/video_timeline_fusion.csv | \
  grep -v ',"",' | \
  cut -d',' -f1,14-16
```

### Count Text Categories

```bash
# Count detections by category
grep -v "^frame," outputs/test/video_timeline_fusion.csv | \
  cut -d',' -f16 | \
  sort | uniq -c
```

### Show High-Confidence Detections

```bash
# Show frames with confidence > 0.80
awk -F',' 'NR>1 && $15 > 0.80 {print $1, $14, $15, $16}' \
  outputs/test/video_timeline_fusion.csv
```

## Troubleshooting

### OCR Not Available

If you see `Enable OCR text extraction from message boards (available: False)`:

1. Check OCR modules are installed:
   ```bash
   python -c "from workzone.ocr.text_detector import SignTextDetector; print('OK')"
   ```

2. Install dependencies if needed:
   ```bash
   pip install paddleocr paddlepaddle
   ```

### No Text Detected

If OCR columns are empty:

1. **No message boards**: Video may not contain message boards (YOLO class 16)
2. **Low confidence**: Text extraction requires confidence > 0.65
3. **OCR not enabled**: Check `--enable-ocr` flag is used

### Performance Issues

OCR adds processing overhead:

- Use `--stride 2` or higher to skip frames
- Use `--no-video` to skip video encoding
- OCR only processes frames with detected message boards

## Integration with Streamlit

The CLI script uses the same OCR pipeline as the Streamlit app. For interactive validation:

```bash
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py
```

Then enable the "Enable OCR" checkbox in the sidebar.
