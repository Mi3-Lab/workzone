#!/bin/bash
# Quick test script for OCR integration in CLI
# Run with: bash scripts/TEST_OCR_CLI.sh

set -e

echo "=========================================="
echo "Testing OCR Integration in CLI"
echo "=========================================="

# Test video path
VIDEO_PATH="/home/wesleyferreiramaia/data/workzone/data/videos_compressed/boston_2bdb5a72602342a5991b402beb8b7ab4_000001_23370_snippet.mp4"

# Output directory
OUTPUT_DIR="outputs/ocr_cli_test"

echo ""
echo "Running video processing with OCR enabled..."
echo "Command:"
echo "  python scripts/process_video_fusion.py \\"
echo "    $VIDEO_PATH \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --enable-ocr \\"
echo "    --no-motion"
echo ""

python scripts/process_video_fusion.py \
    "$VIDEO_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --enable-ocr \
    --no-motion

echo ""
echo "=========================================="
echo "✅ Processing complete!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "CSV preview (showing OCR columns):"
head -n 5 "$OUTPUT_DIR"/*.csv | cut -d',' -f1-5,14-16
echo ""
echo "Check for OCR text extractions:"
grep -v "^frame," "$OUTPUT_DIR"/*.csv | grep -v '""' | cut -d',' -f1,14-16 | head -n 10
echo ""
echo "=========================================="
echo "Test complete! Check output in: $OUTPUT_DIR"
echo "=========================================="
