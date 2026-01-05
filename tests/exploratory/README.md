# Exploratory Tests

This directory contains development and exploratory test scripts used during the OCR module development.

## Purpose

These scripts were created to validate, benchmark, and improve the OCR text classification system. They are kept for reference and future debugging but are **not part of the production codebase**.

## Contents

### OCR Testing Scripts
- `test_ocr.py` - Initial OCR functionality test
- `test_ocr_instant.py` - Quick OCR validation
- `test_ocr_quick.py` - Fast OCR test on small sample
- `test_ocr_intensive.py` - Comprehensive test on 1,195 samples (22+ min runtime)
- `test_ocr_local.py` - Local OCR testing script

### Classification Development
- `test_classifier_improved.py` - Improved classifier validation (97.7% accuracy test)
- `analyze_ocr_results.py` - Analysis of OCR test results
- `compare_gt_improved.py` - Ground truth comparison for validation
- `reprocess_ocr_results.py` - Re-process 287 detections with improved classifier

### Integration Experiments
- `integrate_ocr_plan.py` - OCR integration planning script
- `integration_example.py` - Example integration code
- `check_dataset.py` - Dataset validation script
- `quick_ocr_check.py` - Quick dataset check

## Results

Key achievements from these tests:
- **97.7%** classifier accuracy on validation set (43/44 cases)
- **87.0%** useful classification rate on ground truth matches
- **47%** reduction in UNCLEAR cases (212 → 112)
- **63** noise cases correctly filtered

## Documentation

For production-ready OCR documentation, see:
- [docs/technical/OCR_IMPROVEMENTS.md](../../docs/technical/OCR_IMPROVEMENTS.md) - Complete improvements report
- [docs/technical/OCR_REALTIME_STRATEGY.md](../../docs/technical/OCR_REALTIME_STRATEGY.md) - Jetson deployment strategy
- [docs/archive/OCR_ANALYSIS_FINDINGS.md](../../docs/archive/OCR_ANALYSIS_FINDINGS.md) - Detailed analysis findings

## Usage

⚠️ **Warning**: These scripts are for development purposes only. They may require specific datasets, have hardcoded paths, or take significant time to run.

For production OCR usage, see the main workzone package:
```python
from workzone.ocr import TextDetector, TextClassifier
```
