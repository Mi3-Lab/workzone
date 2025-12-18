# WorkZone - Extension Guide

This document outlines what additional modules should be implemented to complete the conversion of all notebooks to professional Python code.

## ✅ Already Implemented

### Core Models & Pipelines
- ✅ `models/yolo_detector.py` - YOLO object detection
- ✅ `models/vlm.py` - Vision-Language Models (Alpamayo, CLIP)
- ✅ `pipelines/yolo_training.py` - Training pipeline
- ✅ `pipelines/video_inference.py` - Inference pipeline
- ✅ `config.py` - Configuration system
- ✅ CLI modules - Train and inference commands
- ✅ Logging utilities - Professional logging setup

## 📋 TODO - Additional Modules to Implement

### High Priority

#### 1. **State Machine Pipeline** (From Notebook 04)
```python
# src/workzone/pipelines/state_machine.py

class DetectionStateMachine:
    """
    Stateful detection with EMA filtering and hysteresis.
    
    Features:
    - Track detection states (in_zone, exiting, entered)
    - Exponential moving average smoothing
    - Hysteresis for state transitions
    - Confidence tracking per class
    """
    
    def __init__(self, ema_alpha=0.7, confidence_threshold=0.5):
        pass
    
    def update(self, detections, frame_idx):
        """Process detections and update state"""
        pass
    
    def get_state(self):
        """Get current state of all objects"""
        pass
```

#### 2. **Timeline Calibration Pipeline** (From Notebook 05)
```python
# src/workzone/pipelines/timeline_calibration.py

class TimelineCalibrator:
    """
    Multi-GPU batch processing with proximity heuristics.
    
    Features:
    - Batch process multiple videos
    - GPU workload distribution
    - Proximity-based clustering
    - Temporal synchronization
    """
    
    def __init__(self, num_gpus=4):
        pass
    
    def process_batch(self, video_paths):
        """Process multiple videos in parallel"""
        pass
```

#### 3. **Data Preparation Module** (From Notebook 01)
```python
# src/workzone/data/dataset.py

class DatasetConverter:
    """Convert COCO format to YOLO format"""
    
    def __init__(self, coco_json_path):
        pass
    
    def convert_to_yolo(self, output_dir):
        """Convert COCO annotations to YOLO format"""
        pass
    
    def create_splits(self, train_ratio=0.8, val_ratio=0.1):
        """Create train/val/test splits"""
        pass

# src/workzone/data/loaders.py

class WorkzoneDataLoader:
    """Custom DataLoader for WorkZone dataset"""
    
    def __init__(self, yaml_path, batch_size=32):
        pass
```

#### 4. **Evaluation Metrics Module**
```python
# src/workzone/evaluation/metrics.py

class DetectionMetrics:
    """Compute detection metrics (mAP, precision, recall, F1)"""
    
    def __init__(self):
        pass
    
    def compute_map(self, predictions, ground_truth):
        pass
    
    def compute_per_class_metrics(self):
        pass

class VideoMetrics:
    """Metrics for video-level evaluation"""
    
    def __init__(self):
        pass
    
    def fps_analysis(self, results):
        """Analyze frames per second performance"""
        pass
```

### Medium Priority

#### 5. **CLIP-based Semantic Verification** (From Notebook 06)
```python
# src/workzone/pipelines/semantic_verification.py

class SemanticVerificationPipeline:
    """
    Reduce false positives using CLIP semantic fusion.
    
    Features:
    - CLIP embedding computation
    - Semantic similarity matching
    - Confidence adjustment based on semantics
    - Flicker reduction
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        pass
    
    def verify_detections(self, frame, detections):
        """Filter detections using semantic understanding"""
        pass
```

#### 6. **Exploratory Analysis Module** (From Workingzone.ipynb)
```python
# src/workzone/analysis/eda.py

class ExploratoryDataAnalysis:
    """Dataset analysis and visualization"""
    
    def __init__(self, dataset_path):
        pass
    
    def class_distribution(self):
        """Analyze class distribution"""
        pass
    
    def visualize_samples(self, n_samples=10):
        pass

# src/workzone/analysis/visualization.py

class DetectionVisualizer:
    """Visualization utilities for detections"""
    
    @staticmethod
    def draw_detections(frame, detections):
        pass
    
    @staticmethod
    def create_comparison_grid(original, detected):
        pass
```

#### 7. **Alpamayo Integration Module** (From Alpamayo Smoke Test)
```python
# src/workzone/pipelines/vla_reasoning.py

class VLAReasoningPipeline:
    """
    Real-time VLA inference with threaded I/O.
    
    Features:
    - 10Hz reasoning loop
    - Threaded frame capture and processing
    - Real-time text generation
    - Visualization of reasoning
    """
    
    def __init__(self, model_id="nvidia/Alpamayo-R1-10B", cadence_hz=10):
        pass
    
    def process_frame(self, frame):
        """Generate reasoning about frame"""
        pass
    
    def start_inference_loop(self, video_path):
        """Start continuous inference loop"""
        pass
```

### Low Priority (Optimization)

#### 8. **Performance Profiling Module**
```python
# src/workzone/profiling/profiler.py

class InferenceProfiler:
    """Profile inference performance"""
    
    def profile_model(self, model, input_size):
        pass
    
    def memory_analysis(self):
        pass
    
    def generate_report(self):
        pass
```

#### 9. **Distributed Training Module**
```python
# src/workzone/training/distributed.py

class DistributedTrainer:
    """Multi-GPU distributed training"""
    
    def __init__(self, num_gpus=4):
        pass
    
    def train_distributed(self, config):
        pass
```

#### 10. **Monitoring & Logging Module**
```python
# src/workzone/monitoring/metrics_logger.py

class MetricsLogger:
    """Advanced metrics logging to W&B"""
    
    def log_inference_metrics(self, detections, fps):
        pass
    
    def log_confusion_matrix(self, predictions, ground_truth):
        pass
```

## 📦 Implementation Template

When implementing new modules, follow this template:

```python
"""Module description."""

from typing import Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path

from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModuleConfig:
    """Configuration for this module."""
    param1: str = "default"
    param2: int = 42


class MyNewModule:
    """Main class documentation."""
    
    def __init__(self, config: ModuleConfig):
        """Initialize module with config."""
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def process(self, data: List) -> Dict:
        """
        Process data.
        
        Args:
            data: Input data
            
        Returns:
            Processed results
        """
        logger.debug("Starting process")
        try:
            # Implementation
            return {}
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
```

## 🔄 Implementation Order

1. **Data Module** - Needed for training pipeline tests
2. **Evaluation Module** - Needed for training validation
3. **State Machine** - Core functionality
4. **Timeline Calibration** - Batch processing
5. **Semantic Verification** - Quality improvement
6. **VLA Reasoning** - Advanced feature
7. **Visualization** - Supporting tools
8. **Profiling** - Optimization
9. **Distributed** - Scaling
10. **Monitoring** - Production readiness

## 🧪 Testing Each Module

Each module should have corresponding tests:

```bash
tests/
├── test_data_loaders.py        # For data module
├── test_evaluation.py          # For evaluation module
├── test_state_machine.py       # For state machine
├── test_semantic_verification.py
├── test_vla_pipeline.py
└── test_analysis.py
```

## 📚 Documentation

Each module should have:
1. Docstrings for classes and methods
2. Type hints on all functions
3. Usage examples in docstrings
4. API documentation in README

## 🚀 Integration Checklist

For each new module:
- [ ] Add to `src/workzone/` package
- [ ] Include in `__init__.py`
- [ ] Add configuration to `config.py`
- [ ] Write tests in `tests/`
- [ ] Update README with usage examples
- [ ] Add type hints throughout
- [ ] Include professional logging
- [ ] Add to CLI if applicable

---

For detailed guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md)
