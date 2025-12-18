"""
Configuration module for WorkZone project.

Handles environment variables, paths, model configs, and hyperparameters
with support for environment-based overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class YOLOConfig:
    """YOLO model and training configuration."""

    model_name: str = "yolo12s"
    model_path: Optional[str] = None
    imgsz: int = 960
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    epochs: int = 300
    batch_size: int = 35
    learning_rate: float = 0.001
    workers: int = 8
    device: str = "cuda"
    pretrained: bool = True
    data_yaml: str = "data/workzone_yolo/workzone_yolo.yaml"

    @classmethod
    def from_env(cls) -> "YOLOConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("YOLO_MODEL", "yolo12s"),
            imgsz=int(os.getenv("YOLO_IMGSZ", "960")),
            confidence_threshold=float(os.getenv("YOLO_CONF", "0.5")),
            epochs=int(os.getenv("YOLO_EPOCHS", "300")),
            batch_size=int(os.getenv("YOLO_BATCH", "35")),
            device=os.getenv("DEVICE", "cuda"),
        )


@dataclass
class DataConfig:
    """Data paths and dataset configuration."""

    data_root: Path = field(default_factory=lambda: Path("data"))
    construction_data: Path = field(default_factory=lambda: Path("data/Construction_Data"))
    weights_dir: Path = field(default_factory=lambda: Path("weights"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_root, self.weights_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class VLAConfig:
    """Vision-Language Model configuration."""

    model_id: str = "nvidia/Alpamayo-R1-10B"
    inference_cadence: float = 0.1  # 10Hz
    dtype: str = "bfloat16"
    temperature: float = 0.6
    top_p: float = 0.8
    max_generation_length: int = 128
    num_traj_samples: int = 1
    device: str = "cuda"


@dataclass
class ProcessingConfig:
    """Video and frame processing configuration."""

    video_fps: int = 30
    frame_width: int = 1920
    frame_height: int = 1080
    resize_width: int = 512
    resize_height: int = 384
    frame_queue_size: int = 5
    enable_threading: bool = True
    num_workers: int = 4


@dataclass
class ProjectConfig:
    """Main project configuration container."""

    project_name: str = "workzone"
    project_dir: Path = field(default_factory=lambda: Path.cwd())
    device: str = "cuda"
    debug: bool = False
    verbose: bool = True
    seed: int = 42
    wandb_project: str = "workzone-yolo"
    wandb_enabled: bool = True

    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vla: VLAConfig = field(default_factory=VLAConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "ProjectConfig":
        """Create config from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            device=os.getenv("DEVICE", "cuda"),
            seed=int(os.getenv("SEED", "42")),
            yolo=YOLOConfig.from_env(),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "project_name": self.project_name,
            "device": self.device,
            "debug": self.debug,
            "yolo": {
                "model": self.yolo.model_name,
                "imgsz": self.yolo.imgsz,
                "batch_size": self.yolo.batch_size,
                "epochs": self.yolo.epochs,
            },
            "vla": {
                "model_id": self.vla.model_id,
                "inference_cadence": self.vla.inference_cadence,
            },
        }


# Singleton instance
_config: Optional[ProjectConfig] = None


def get_config(config_path: Optional[Path] = None) -> ProjectConfig:
    """
    Get or create global configuration.

    Args:
        config_path: Path to YAML config file. If None, uses env variables.

    Returns:
        ProjectConfig instance
    """
    global _config
    if _config is None:
        if config_path:
            _config = ProjectConfig.from_yaml(config_path)
        else:
            _config = ProjectConfig.from_env()
    return _config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None
