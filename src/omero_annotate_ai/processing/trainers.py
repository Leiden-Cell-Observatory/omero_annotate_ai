"""Training-data contract and trainers.

``load_training_data`` reads one exported training-data folder (``train_*``, ``val_*``,
optional ``test_*`` and ``normalization.json``) into a ``TrainingData`` that every
trainer consumes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tifffile import TiffFile

from ..errors import InvalidTrainingData
from .normalization import SIDECAR_NAME
from .training_layout import split_folders_for

_SPLIT_CATEGORY = {"train": "training", "val": "validation", "test": "test"}


@dataclass(frozen=True)
class TrainingData:
    """One exported training-data folder, validated."""

    root: Path
    train_input: Path
    train_label: Path
    val_input: Path
    val_label: Path
    test_input: Path | None
    test_label: Path | None
    normalization: dict[str, Any]
    axes: str
    shape: tuple[int, ...]
    n_channels: int
    dtype: str

    def pairs(self, split: str) -> list[tuple[Path, Path]]:
        """(image, label) paths of a split ("train", "val" or "test"), sorted by name."""
        if split not in _SPLIT_CATEGORY:
            raise ValueError(
                f"Unknown split: {split}. Expected one of train, val, test"
            )
        image_dir = getattr(self, f"{split}_input")
        label_dir = getattr(self, f"{split}_label")
        if image_dir is None or label_dir is None:
            return []
        return [(p, label_dir / p.name) for p in sorted(image_dir.glob("*.tif"))]


def _tif_names(folder: Path) -> set[str]:
    return {p.name for p in folder.glob("*.tif")}


def _require_dir(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_dir():
        raise InvalidTrainingData(f"Training data folder is missing {name}/: {path}")
    return path


def _check_pairs(image_dir: Path, label_dir: Path) -> None:
    images, labels = _tif_names(image_dir), _tif_names(label_dir)
    for name in sorted(images - labels):
        raise InvalidTrainingData(
            f"{image_dir.name}/{name} has no label in {label_dir.name}/"
        )
    for name in sorted(labels - images):
        raise InvalidTrainingData(
            f"{label_dir.name}/{name} has no image in {image_dir.name}/"
        )


def _series_info(path: Path) -> tuple[tuple[int, ...], str]:
    with TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(series.shape), str(series.dtype)


def _axes_for(shape: tuple[int, ...], path: Path) -> str:
    if len(shape) == 2:
        return "YX"
    if len(shape) == 3:
        return "YXC" if shape[-1] <= 4 else "ZYX"
    if len(shape) == 4:
        return "ZYXC"
    raise InvalidTrainingData(f"Unsupported image shape {shape} in {path}")


def load_training_data(root: str | Path) -> TrainingData:
    """Read and validate an exported training-data folder.

    Raises:
        InvalidTrainingData: A split folder is missing, files are unpaired, a split has
            no pairs (val), or the first label does not match the first image's shape.
    """
    root = Path(root)
    if not root.is_dir():
        raise InvalidTrainingData(f"Training data folder not found: {root}")

    train_input, train_label = (
        _require_dir(root, n) for n in split_folders_for("training")[:2]
    )
    val_input, val_label = (
        _require_dir(root, n) for n in split_folders_for("validation")[:2]
    )
    test_names = split_folders_for("test")[:2]
    present = [(root / n).is_dir() for n in test_names]
    if any(present) and not all(present):
        missing = test_names[present.index(False)]
        raise InvalidTrainingData(
            f"Training data has a test split but no {missing}/: {root}"
        )
    test_input, test_label = (
        (root / n for n in test_names) if all(present) else (None, None)
    )

    for image_dir, label_dir in (
        (train_input, train_label),
        (val_input, val_label),
        (test_input, test_label),
    ):
        if image_dir is not None and label_dir is not None:
            _check_pairs(image_dir, label_dir)

    if not _tif_names(val_input):
        raise InvalidTrainingData(
            f"No image/label pairs in {val_input.name}/: {val_input}"
        )

    first = next(iter(sorted(train_input.glob("*.tif"))), None) or next(
        iter(sorted(val_input.glob("*.tif")))
    )
    shape, dtype = _series_info(first)
    axes = _axes_for(shape, first)
    n_channels = shape[-1] if axes.endswith("C") else 1
    label_dir = train_label if first.parent == train_input else val_label
    label_shape, _ = _series_info(label_dir / first.name)
    expected = shape[:-1] if axes.endswith("C") else shape
    if label_shape != expected:
        raise InvalidTrainingData(
            f"Label {label_dir.name}/{first.name} has shape {label_shape}, "
            f"expected {expected} for image shape {shape}"
        )

    sidecar = root / SIDECAR_NAME
    normalization: dict[str, Any] = (
        json.loads(sidecar.read_text()) if sidecar.exists() else {}
    )
    return TrainingData(
        root=root,
        train_input=train_input,
        train_label=train_label,
        val_input=val_input,
        val_label=val_label,
        test_input=test_input,
        test_label=test_label,
        normalization=normalization,
        axes=axes,
        shape=shape,
        n_channels=n_channels,
        dtype=dtype,
    )
