"""Intensity normalisation of model input images, defined once in the config.

The ``normalization`` block of an
:class:`~omero_annotate_ai.core.annotation_config.AnnotationConfig` states how raw
intensities become model input. It is applied in one of two places:

- at export (``apply_at_export: true``): images are written as float32 after per-image
  percentile normalisation, and the bounds used for every image are recorded in a
  ``normalization.json`` sidecar next to the training data. Trainers then use the
  identity (see :func:`biapy_normalization`).
- by the trainer (default): images are exported at their native bit depth and the
  trainer is configured from the same block (see :func:`biapy_normalization`).

Either way the definition lives in the config, so it is stored in OMERO with the
annotation set and travels with the exported dataset.
"""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

SIDECAR_NAME = "normalization.json"


def percentile_normalize(
    image: np.ndarray,
    percentile_low: float,
    percentile_high: float,
    clip: bool = True,
    channel_axis: int | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Scale an image so the low/high percentiles map to 0 and 1.

    Bounds are computed per channel over all other axes (one image, so per image).

    Args:
        image: Image array (any dtype).
        percentile_low: Lower percentile, mapped to 0.
        percentile_high: Upper percentile, mapped to 1.
        clip: Clip the result to [0, 1].
        channel_axis: Axis holding channels, normalised independently; None for a
            single-channel image.

    Returns:
        The float32 image and the (low, high) intensity bounds per channel.
    """
    data = np.asarray(image, dtype=np.float32)
    if channel_axis is None:
        channels = [data]
    else:
        channels = [
            np.take(data, i, axis=channel_axis) for i in range(data.shape[channel_axis])
        ]

    bounds = []
    normalised = []
    for channel in channels:
        low, high = np.percentile(channel, [percentile_low, percentile_high])
        scale = high - low
        out = (channel - low) / scale if scale > 0 else np.zeros_like(channel)
        if clip:
            out = np.clip(out, 0.0, 1.0)
        normalised.append(out.astype(np.float32))
        bounds.append((float(low), float(high)))

    if channel_axis is None:
        return normalised[0], bounds
    return np.stack(normalised, axis=channel_axis), bounds


def channel_axis_for(image: np.ndarray) -> int | None:
    """Channel axis of an exported model-input image.

    Model input is written channels-last (YXC or ZYXC) when several training channels
    are stacked; single-channel images are YX or ZYX. A trailing axis of at most 4 is
    taken as channels.
    """
    return image.ndim - 1 if image.ndim >= 3 and image.shape[-1] <= 4 else None


class NormalizationRecorder:
    """Applies a normalisation spec to exported images and records the bounds used."""

    def __init__(self, spec: Any):
        self.spec = spec
        self.bounds: dict[str, list[tuple[float, float]]] = {}

    def __call__(self, annotation_id: str, image: np.ndarray) -> np.ndarray:
        normalised, bounds = percentile_normalize(
            image,
            self.spec.percentile_low,
            self.spec.percentile_high,
            clip=self.spec.clip,
            channel_axis=channel_axis_for(image),
        )
        self.bounds[str(annotation_id)] = bounds
        return normalised

    def write_sidecar(self, output_dir: Path) -> Path:
        """Write the spec and per-image bounds to ``normalization.json``."""
        path = Path(output_dir) / SIDECAR_NAME
        content = {
            "method": self.spec.method,
            "percentile_low": self.spec.percentile_low,
            "percentile_high": self.spec.percentile_high,
            "clip": self.spec.clip,
            "applied_at_export": True,
            "dtype": "float32",
            "bounds_per_image": {
                annotation_id: [list(b) for b in bounds]
                for annotation_id, bounds in sorted(self.bounds.items())
            },
        }
        path.write_text(json.dumps(content, indent=2))
        return path


def write_native_sidecar(spec: Any, output_dir: Path) -> Path:
    """Record that images were exported unnormalised, and what the trainer should apply."""
    path = Path(output_dir) / SIDECAR_NAME
    content = {
        "method": spec.method,
        "percentile_low": spec.percentile_low,
        "percentile_high": spec.percentile_high,
        "clip": spec.clip,
        "applied_at_export": False,
        "dtype": "native",
    }
    path.write_text(json.dumps(content, indent=2))
    return path


def biapy_normalization(spec: Any) -> dict[str, Any]:
    """BiaPy ``DATA.NORMALIZATION`` block that realises a normalisation spec.

    BiaPy's ``div`` and ``scale_range`` types infer the bit depth from the image
    maximum, which is unreliable (e.g. 12-bit data stored as 16-bit). Neither is used:

    - Applied at export, or ``method: none``: the identity (zero mean, unit variance
      with mean 0 and std 1), so BiaPy leaves the intensities as they are. This is the
      recommended setup: the exported float32 images are the model input.
    - ``percentile`` left to BiaPy: per-image percentile clipping followed by per-image
      standardisation. The clipping bounds are the same as in the spec; the final
      scaling is to zero mean and unit variance instead of [0, 1].
    """
    if spec.apply_at_export or spec.method == "none":
        return {
            "TYPE": "zero_mean_unit_variance",
            "ZERO_MEAN_UNIT_VAR": {"MEAN_VAL": 0.0, "STD_VAL": 1.0},
            "PERC_CLIP": {"ENABLE": False},
        }
    return {
        "TYPE": "zero_mean_unit_variance",
        "MEASURE_BY": "image",
        "PERC_CLIP": {
            "ENABLE": True,
            "LOWER_PERC": spec.percentile_low,
            "UPPER_PERC": spec.percentile_high,
        },
    }


def microsam_raw_transform(spec: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Raw transform for micro-SAM training, which requires inputs in [0, 255].

    Exported images keep their native bit depth (often 16-bit), so the
    normalisation is applied when micro-SAM loads them:

    - applied at export: the images are already in [0, 1]; scale to [0, 255];
    - ``percentile``: per-image percentile normalisation, then scale to [0, 255];
    - ``none``: pass through, but refuse data outside [0, 255] with a clear error.

    Args:
        spec: A :class:`NormalizationConfig`, an equivalent dict, or the content of
            ``normalization.json`` (``applied_at_export`` is accepted as an alias).
    """
    from omero_annotate_ai.core.annotation_config import NormalizationConfig

    if isinstance(spec, dict):
        spec = dict(spec)
        if "applied_at_export" in spec:
            spec["apply_at_export"] = spec.pop("applied_at_export")
        spec = NormalizationConfig.model_validate(
            {k: v for k, v in spec.items() if k in NormalizationConfig.model_fields}
        )
    elif spec is None:
        spec = NormalizationConfig()

    if spec.apply_at_export:
        return _unit_range_to_8bit
    if spec.method == "percentile":
        return partial(
            _percentile_to_8bit, low=spec.percentile_low, high=spec.percentile_high
        )
    return _require_8bit_range


def _unit_range_to_8bit(x: np.ndarray) -> np.ndarray:
    """Images normalised at export are in [0, 1]."""
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0) * 255.0


def _percentile_to_8bit(x: np.ndarray, *, low: float, high: float) -> np.ndarray:
    normalised, _ = percentile_normalize(x, low, high, clip=True)
    return normalised * 255.0


def _require_8bit_range(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.min() < 0 or x.max() > 255:
        raise ValueError(
            "micro-SAM needs input in the 0-255 range, but these images go up to "
            f"{x.max():g}. Use normalization.method 'percentile' (the default)."
        )
    return x
