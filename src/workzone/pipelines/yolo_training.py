"""YOLO training pipeline with W&B integration."""

from pathlib import Path
from typing import Optional

import wandb
from ultralytics import YOLO

from src.workzone.config import YOLOConfig
from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class YOLOTrainingPipeline:
    """
    YOLO training pipeline with experiment tracking via Weights & Biases.

    Attributes:
        config: YOLO training configuration
        model: YOLO model instance
        results: Training results
    """

    def __init__(
        self,
        config: YOLOConfig,
        wandb_enabled: bool = True,
        wandb_project: str = "workzone-yolo",
    ):
        """
        Initialize training pipeline.

        Args:
            config: YOLO configuration
            wandb_enabled: Enable Weights & Biases tracking
            wandb_project: W&B project name
        """
        self.config = config
        self.wandb_enabled = wandb_enabled
        self.wandb_project = wandb_project
        self.model = None
        self.results = None

    def load_model(self):
        """Load YOLO model checkpoint."""
        model_path = self.config.model_path or self.config.model_name
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))

    def train(
        self,
        run_name: str = "yolo_workzone_baseline",
        epochs: Optional[int] = None,
    ) -> dict:
        """
        Train YOLO model.

        Args:
            run_name: Name for this training run
            epochs: Number of epochs (uses config if None)

        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.load_model()

        epochs = epochs or self.config.epochs

        # Initialize W&B if enabled
        if self.wandb_enabled:
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=self.config.__dict__,
                job_type="training",
            )
            logger.info(f"W&B tracking enabled: {self.wandb_project}/{run_name}")

        logger.info("Starting training...")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Data: {self.config.data_yaml}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Image size: {self.config.imgsz}")

        try:
            self.results = self.model.train(
                data=self.config.data_yaml,
                imgsz=self.config.imgsz,
                epochs=epochs,
                batch=self.config.batch_size,
                device=self.config.device,
                project=self.wandb_project,
                name=run_name,
                workers=self.config.workers,
                cos_lr=True,
                pretrained=self.config.pretrained,
                exist_ok=True,
            )

            logger.info("Training completed successfully")

            if self.wandb_enabled:
                wandb.finish()

            return {"status": "success", "results": self.results}

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.wandb_enabled:
                wandb.finish()
            raise

    def validate(self, data_yaml: Optional[str] = None) -> dict:
        """
        Validate trained model.

        Args:
            data_yaml: Path to data config (uses config if None)

        Returns:
            Validation results
        """
        if self.model is None:
            self.load_model()

        data_yaml = data_yaml or self.config.data_yaml
        logger.info(f"Validating model on {data_yaml}")

        results = self.model.val(data=data_yaml, imgsz=self.config.imgsz)
        return results

    def save_model(self, output_path: Path) -> Path:
        """
        Save trained model.

        Args:
            output_path: Path to save model

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_path}")
        self.model.save(str(output_path))
        return output_path
