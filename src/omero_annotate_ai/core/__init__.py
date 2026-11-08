"""Configuration (``annotation_config``) and the deprecated one-call pipeline."""

from .annotation_config import AnnotationConfig, create_default_config, load_config

__all__ = ["AnnotationConfig", "load_config", "create_default_config"]
