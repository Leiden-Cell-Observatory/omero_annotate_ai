"""Training-data contract and trainers.

``load_training_data`` reads one exported training-data folder (``train_*``, ``val_*``,
optional ``test_*`` and ``normalization.json``) into a ``TrainingData`` that every
trainer consumes.
"""

import datetime
import importlib.metadata
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tifffile import TiffFile, imread

from ..errors import InvalidTrainingData
from .normalization import SIDECAR_NAME, microsam_raw_transform
from .training_layout import split_folders_for

logger = logging.getLogger(__name__)

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
    try:
        with TiffFile(path) as tif:
            series = tif.series[0]
            return tuple(series.shape), str(series.dtype)
    except Exception as e:
        raise InvalidTrainingData(
            f"Cannot read {path.parent.name}/{path.name}: {e}"
        ) from e


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
        InvalidTrainingData: A split folder is missing, files are unpaired, train or val
            has no pairs, a file cannot be read, or the first label does not match the
            first image's shape.
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

    for image_dir in (train_input, val_input):
        if not _tif_names(image_dir):
            raise InvalidTrainingData(
                f"No image/label pairs in {image_dir.name}/: {image_dir}"
            )

    first = sorted(train_input.glob("*.tif"))[0]
    shape, dtype = _series_info(first)
    axes = _axes_for(shape, first)
    n_channels = shape[-1] if axes.endswith("C") else 1
    label_shape, _ = _series_info(train_label / first.name)
    expected = shape[:-1] if axes.endswith("C") else shape
    if label_shape != expected:
        raise InvalidTrainingData(
            f"Label {train_label.name}/{first.name} has shape {label_shape}, "
            f"expected {expected} for image shape {shape}"
        )

    sidecar = root / SIDECAR_NAME
    try:
        normalization: dict[str, Any] = (
            json.loads(sidecar.read_text()) if sidecar.exists() else {}
        )
    except (OSError, ValueError) as e:
        raise InvalidTrainingData(f"Cannot read {sidecar}: {e}") from e
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


@dataclass(frozen=True)
class TrainingResult:
    """What a trainer produced."""

    trainer: str
    package_version: str
    model_path: Path
    output_dir: Path
    params: dict[str, Any]
    data_root: Path
    extras: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """A JSON-serialisable copy (paths as strings)."""

        def plain(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        return {
            "trainer": self.trainer,
            "package_version": self.package_version,
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir),
            "params": plain(self.params),
            "data_root": str(self.data_root),
            "extras": plain(self.extras),
        }


class Trainer(Protocol):
    """A training back-end: consumes a ``TrainingData``, writes a model to ``output_dir``."""

    name: str

    def train(
        self, data: TrainingData, output_dir: Path, **params: Any
    ) -> TrainingResult: ...


class MicroSamTrainer:
    """Fine-tune micro-SAM (``pip install omero-annotate-ai[microsam]``)."""

    name = "microsam"

    def train(
        self,
        data: TrainingData,
        output_dir: Path,
        *,
        model_name: str = "",
        model_type: str = "vit_b_lm",
        epochs: int = 50,
        n_iterations: int | None = None,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        patch_shape: tuple[int, ...] = (512, 512),
        n_objects_per_batch: int = 25,
        train_instance_segmentation: bool = True,
    ) -> TrainingResult:
        """Train micro-SAM on ``data``.

        ``n_iterations`` overrides ``epochs`` (micro-SAM's ``train_sam`` semantics).
        ``patch_shape`` is passed to micro-SAM unchanged. The best (else latest)
        checkpoint is copied to ``<output_dir>/<model_name>_final.pt``.

        Raises:
            InvalidTrainingData: ``data`` has more than one channel.
            ValueError: Neither ``epochs`` nor ``n_iterations`` is positive.
            ImportError: micro-SAM is not installed.
        """
        if data.n_channels > 1:
            raise InvalidTrainingData(
                f"micro-SAM trains on one channel, the data has {data.n_channels}"
            )
        if n_iterations is None and epochs <= 0:
            raise ValueError(
                "Set epochs > 0, or n_iterations for a fixed number of steps"
            )
        if n_iterations is not None and n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        try:
            import micro_sam
            import micro_sam.training as sam_training
            import torch
            from torch_em.data import MinInstanceSampler
        except ImportError as e:
            raise ImportError(
                "micro-SAM is not installed. Install with: "
                "pip install omero-annotate-ai[microsam]"
            ) from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not model_name.strip():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"micro_sam_training_{stamp}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "micro-SAM training %s (%s) on %s, patch shape %s",
            model_name,
            model_type,
            device,
            patch_shape,
        )

        # micro-SAM requires inputs in [0, 255]; exports keep native bit depth.
        raw_transform = microsam_raw_transform(data.normalization or None)
        sampler = MinInstanceSampler(min_size=25)

        def loader(input_dir: Path, label_dir: Path) -> Any:
            return sam_training.default_sam_loader(
                raw_paths=str(input_dir),
                raw_key="*.tif",
                label_paths=str(label_dir),
                label_key="*.tif",
                with_segmentation_decoder=train_instance_segmentation,
                patch_shape=patch_shape,
                batch_size=batch_size,
                shuffle=True,
                raw_transform=raw_transform,
                sampler=sampler,
            )

        train_loader = loader(data.train_input, data.train_label)
        val_loader = loader(data.val_input, data.val_label)

        try:
            sam_training.train_sam(
                name=model_name,
                save_root=str(output_dir),
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=epochs,
                n_iterations=n_iterations,  # overrides n_epochs when set
                n_objects_per_batch=n_objects_per_batch,
                with_segmentation_decoder=train_instance_segmentation,
                device=device,
                lr=learning_rate,
            )
        except Exception:
            logger.exception("micro-SAM training failed")
            raise

        checkpoint_dir = output_dir / model_name
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        best = checkpoint_dir / "best.pt"
        best_checkpoint = best if best.exists() else None
        source = best_checkpoint or (checkpoints[-1] if checkpoints else None)
        model_path = output_dir / f"{model_name}_final.pt"
        if source is None:
            raise RuntimeError(f"micro-SAM wrote no checkpoint in {checkpoint_dir}")
        shutil.copy2(source, model_path)
        logger.info("Final model exported to %s", model_path)

        return TrainingResult(
            trainer=self.name,
            package_version=micro_sam.__version__,
            model_path=model_path,
            output_dir=output_dir,
            params={
                "model_name": model_name,
                "model_type": model_type,
                "epochs": epochs,
                "n_iterations": n_iterations,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "patch_shape": patch_shape,
                "n_objects_per_batch": n_objects_per_batch,
                "train_instance_segmentation": train_instance_segmentation,
            },
            data_root=data.root,
            extras={
                "checkpoints": checkpoints,
                "best_checkpoint": best_checkpoint,
                "device": device,
            },
        )


class CellposeTrainer:
    """Fine-tune Cellpose (``pip install omero-annotate-ai[cellpose]``)."""

    name = "cellpose"

    def train(
        self,
        data: TrainingData,
        output_dir: Path,
        *,
        model_name: str = "",
        pretrained_model: str = "cpsam",
        epochs: int = 100,
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        weight_decay: float = 0.1,
        min_train_masks: int = 5,
    ) -> TrainingResult:
        """Train Cellpose on ``data`` with ``cellpose.train.train_seg``.

        Cellpose trains on 2D planes: ``ZYX``/``ZYXC`` volumes are sliced along the
        first axis (``extras["n_planes"]`` counts the training planes). Cellpose
        normalises images itself unless the export already did.

        Raises:
            InvalidTrainingData: ``data`` has more than 3 channels.
            ImportError: Cellpose is not installed.
        """
        if data.n_channels > 3:
            raise InvalidTrainingData(
                f"Cellpose trains on up to 3 channels, the data has {data.n_channels}"
            )
        try:
            import cellpose
            import cellpose.models
            import cellpose.train
            import torch
        except ImportError as e:
            raise ImportError(
                "Cellpose is not installed. Install with: "
                "pip install omero-annotate-ai[cellpose]"
            ) from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not model_name.strip():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"cellpose_{stamp}"
        normalize = not data.normalization.get("applied_at_export", False)
        sliced = data.axes.startswith("Z")

        def load(split: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
            images: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            for image_path, label_path in data.pairs(split):
                image, label = imread(image_path), imread(label_path)
                if sliced:
                    images.extend(image)
                    labels.extend(label)
                else:
                    images.append(image)
                    labels.append(label)
            return images, labels

        train_images, train_labels = load("train")
        val_images, val_labels = load("val")

        use_gpu = bool(torch.cuda.is_available())
        logger.info(
            "Cellpose training %s from %s on %s, %d training planes",
            model_name,
            pretrained_model,
            "gpu" if use_gpu else "cpu",
            len(train_images),
        )
        model = cellpose.models.CellposeModel(
            gpu=use_gpu, pretrained_model=pretrained_model
        )
        try:
            model_path, train_losses, test_losses = cellpose.train.train_seg(
                model.net,
                train_data=train_images,
                train_labels=train_labels,
                test_data=val_images,
                test_labels=val_labels,
                n_epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                min_train_masks=min_train_masks,
                normalize=normalize,
                save_path=str(output_dir),
                model_name=model_name,
            )
        except Exception:
            logger.exception("Cellpose training failed")
            raise

        return TrainingResult(
            trainer=self.name,
            package_version=_package_version(cellpose),
            model_path=Path(model_path),
            output_dir=output_dir,
            params={
                "model_name": model_name,
                "pretrained_model": pretrained_model,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "min_train_masks": min_train_masks,
                "normalize": normalize,
            },
            data_root=data.root,
            extras={
                "train_losses": list(train_losses),
                "test_losses": list(test_losses),
                "n_planes": len(train_images),
            },
        )


def _package_version(module: Any) -> str:
    """The module's ``version`` attribute, else its installed distribution version."""
    version = getattr(module, "version", None)
    if isinstance(version, str):
        return version
    return importlib.metadata.version(module.__name__)


# Concrete trainers take typed keyword parameters, so they are not typed as the Protocol.
TRAINERS: dict[str, type[Any]] = {
    "microsam": MicroSamTrainer,
    "cellpose": CellposeTrainer,
}


def train(
    data: TrainingData | str | Path,
    trainer: str | Trainer,
    output_dir: str | Path,
    **params: Any,
) -> TrainingResult:
    """Train a model on an exported training-data folder.

    Args:
        data: A ``TrainingData`` or the path of an export folder.
        trainer: A name in ``TRAINERS`` (``"microsam"``, ...) or a ``Trainer`` object.
        output_dir: Where the trainer writes (created if missing).
        **params: Trainer-specific parameters, see the trainer's ``train`` method.

    Raises:
        ValueError: ``trainer`` is a name that is not in ``TRAINERS``.
    """
    if isinstance(trainer, str):
        if trainer not in TRAINERS:
            raise ValueError(
                f"Unknown trainer {trainer!r}. Available: {', '.join(TRAINERS)}"
            )
        impl: Trainer = TRAINERS[trainer]()
    else:
        impl = trainer
    if not isinstance(data, TrainingData):
        data = load_training_data(data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return impl.train(data, output_dir, **params)
