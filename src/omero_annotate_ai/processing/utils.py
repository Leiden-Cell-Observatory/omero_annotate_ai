"""Utility functions for micro-SAM workflows."""

import logging

import pandas as pd

_LOGGER = logging.getLogger(__name__)


def validate_table_schema(df: pd.DataFrame, logger=None) -> None:
    """
    Validate that the table has the required columns and basic data integrity.

    Args:
        df: DataFrame from OMERO table
        logger: Optional logger instance for logging messages

    Raises:
        ValueError: If required columns are missing or data integrity issues found
    """
    # Required columns for training data preparation
    required_columns = {
        "image_id",
        "channel",
        "z_slice",
        "timepoint",
        "label_id",
        "train",
        "validate",
        "is_patch",
        "patch_x",
        "patch_y",
        "patch_width",
        "patch_height",
    }

    # Optional columns that enhance functionality
    optional_columns = {"is_volumetric"}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Check for completely null critical columns
    critical_columns = ["image_id", "label_id"]
    for col in critical_columns:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"Column '{col}' contains no valid data")

    # Basic data type check for image_id (should be numeric)
    if not pd.api.types.is_numeric_dtype(df["image_id"]):
        try:
            pd.to_numeric(df["image_id"], errors="raise")
        except (ValueError, TypeError) as e:
            raise ValueError("Column 'image_id' contains non-numeric data") from e

    # Log optional columns that are available
    available_optional = optional_columns.intersection(set(df.columns))
    if available_optional:
        message = f"Optional columns found: {sorted(available_optional)}"
        (logger or _LOGGER).debug(message)
