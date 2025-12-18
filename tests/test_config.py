"""Tests for configuration module."""

import pytest
from src.workzone.config import ProjectConfig, YOLOConfig, get_config, reset_config


class TestYOLOConfig:
    """Test YOLO configuration."""

    def test_default_config(self):
        """Test default YOLO configuration."""
        config = YOLOConfig()

        assert config.model_name == "yolo12s"
        assert config.imgsz == 960
        assert config.epochs == 300
        assert config.batch_size == 35

    def test_config_with_custom_values(self):
        """Test YOLO config with custom values."""
        config = YOLOConfig(
            model_name="yolo11n",
            epochs=100,
            batch_size=64,
        )

        assert config.model_name == "yolo11n"
        assert config.epochs == 100
        assert config.batch_size == 64


class TestProjectConfig:
    """Test project configuration."""

    def test_default_config(self):
        """Test default project configuration."""
        reset_config()
        config = get_config()

        assert config.project_name == "workzone"
        assert config.device == "cuda"
        assert config.seed == 42

    def test_config_dict_conversion(self):
        """Test conversion of config to dictionary."""
        config = ProjectConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "project_name" in config_dict
        assert "device" in config_dict
