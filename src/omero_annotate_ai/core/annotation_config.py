"""Configuration management for OMERO AI annotation workflows."""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)

from ..errors import ConfigError, RemovedConfigKey

logger = logging.getLogger(__name__)

# Sub-models for the configuration


class FeatureType(BaseModel):
    """An annotation class (e.g. cell, nucleus, background)."""

    name: str
    color: str  # hex color, e.g. "#FF0000"


class AuthorInfo(BaseModel):
    """Author information compatible with bioimage.io"""

    name: str | None = Field(default=None, description="Full name of the author")
    affiliation: str | None = Field(default=None, description="Institution affiliation")
    email: str | None = Field(None, description="Contact email")
    orcid: HttpUrl | None = Field(None, description="ORCID identifier")


class AnnotationTool(BaseModel):
    """Tool configuration for annotation workflows"""

    name: str = Field(description="Tool name (e.g., 'micro-sam', 'cellpose')")
    version: str | None = Field(default=None, description="Tool version")
    mode: Literal["automatic", "interactive", "human_in_the_loop"] = Field(
        default="interactive", description="Annotation mode"
    )
    parameters: dict[str, Any] | None = Field(
        default=None, description="Tool-specific parameters"
    )


class AnnotationMethodology(BaseModel):
    """MIFA-compatible annotation methodology"""

    annotation_type: Literal[
        "segmentation_mask",
        "bounding_box",
        "point",
        "classification",
        "semantic_segmentation",
    ] = "segmentation_mask"
    annotation_method: Literal["manual", "semi_automatic", "automatic"] | None = Field(
        default=None, description="How annotations were produced"
    )
    annotation_criteria: str = Field(description="Criteria used for annotation")
    annotation_coverage: Literal["all", "representative", "partial"] = "representative"
    annotation_tool: AnnotationTool | None = Field(
        default=None, description="Tool used for annotation"
    )


_CONTEXT_COLORMAP_CYCLE = ["blue", "magenta", "green", "yellow", "cyan"]


class SpatialCoverage(BaseModel):
    """Spatial scope of annotations (MIFA requirement)"""

    channels: list[int] = Field(
        default_factory=lambda: [0], description="Channel indices processed"
    )
    timepoints: list[int] = Field(
        default_factory=lambda: [0], description="Timepoints as list"
    )
    n_timepoints: int | None = Field(
        default=None,
        description="Number of timepoints to randomly select (used when timepoint_mode='random'). "
        "If not specified, uses the length of timepoints list for backward compatibility.",
    )
    timepoint_mode: Literal["all", "random", "specific"] = "specific"
    z_slices: list[int] = Field(
        default_factory=lambda: [0], description="Z-slices as list"
    )
    # Number of slices/timepoints for random mode (optional, defaults to list length)
    n_slices: int | None = Field(
        default=None,
        description="Number of z-slices to randomly select (used when z_slice_mode='random'). "
        "If not specified, uses the length of z_slices list for backward compatibility.",
    )
    z_slice_mode: Literal["all", "random", "specific"] = "specific"
    three_d: bool = Field(default=False, description="3D volumetric processing mode")

    # 3D volumetric processing fields (used when three_d=True)
    z_range_start: int | None = Field(
        default=None, description="Starting z-slice for 3D volumes (when three_d=True)"
    )
    z_range_end: int | None = Field(
        default=None, description="Ending z-slice for 3D volumes (when three_d=True)"
    )

    # Multi-channel support for separate labeling and training channels
    label_channel: int | None = Field(
        default=None,
        description="Channel index used for segmentation/labeling (e.g., fluorescence nuclei). "
        "If not specified, defaults to channels[0] for backward compatibility.",
    )
    training_channels: list[int] | None = Field(
        default=None,
        description="Channel index(es) used for model training input (e.g., brightfield). "
        "If not specified, defaults to [label_channel] for backward compatibility. "
        "Multiple channels supported if the training model permits.",
    )

    # Display-only context channels shown alongside the label channel during annotation.
    # Never enter SAM embeddings, never saved as training pairs.
    context_channels: list[int] = Field(
        default_factory=list,
        description="Display-only channel indices shown alongside the annotation "
        "channel in napari. "
        "Never enter SAM embeddings or training data. OMERO channel index space.",
    )
    context_channel_names: list[str] | None = Field(
        default=None,
        description="Display names for context_channels. "
        "Length must match context_channels when provided. Defaults to 'ch{idx}'.",
    )
    context_channel_colormaps: list[str] | None = Field(
        default=None,
        description="Napari colormap names for context_channels. "
        "Length must match context_channels when provided. "
        "Defaults to cycle ['blue', 'magenta', 'green', 'yellow', 'cyan'].",
    )

    # Patch processing settings (moved from ProcessingConfig)
    use_patches: bool = Field(
        default=False, description="Extract patches vs full images"
    )
    patch_size: list[int] = Field(
        default=[512, 512], description="Patch dimensions [width, height]"
    )
    patches_per_image: int = Field(default=1, gt=0, description="Patches per image")
    random_patches: bool = Field(
        default=True, description="Use random patch extraction"
    )

    @property
    def primary_channel(self) -> int:
        """Get the primary/first channel"""
        return self.channels[0]

    @property
    def is_single_channel(self) -> bool:
        """Check if only one channel is configured"""
        return len(self.channels) == 1

    def get_label_channel(self) -> int:
        """Get the channel to use for labeling/segmentation.

        Returns label_channel if explicitly set, otherwise falls back to
        primary_channel (channels[0]).
        """
        return (
            self.label_channel
            if self.label_channel is not None
            else self.primary_channel
        )

    def get_training_channels(self) -> list[int]:
        """Get the channel(s) to use for training input.

        Returns training_channels if explicitly set, otherwise falls back to [get_label_channel()]
        for backward compatibility (same channel for both labeling and training).
        """
        return (
            self.training_channels
            if self.training_channels is not None
            else [self.get_label_channel()]
        )

    def uses_separate_channels(self) -> bool:
        """Check if separate channels are configured for labeling and training.

        Returns True if label_channel differs from all training_channels.
        """
        if self.label_channel is None and self.training_channels is None:
            return False
        return self.get_label_channel() not in self.get_training_channels()

    def uses_context_channels(self) -> bool:
        """Return True if any context channels are configured."""
        return bool(self.context_channels)

    def get_context_channel_specs(self) -> list[tuple[int, str, str]]:
        """Return [(channel_index, display_name, colormap), ...] for each context channel."""
        specs = []
        for i, ch_idx in enumerate(self.context_channels):
            name = (
                self.context_channel_names[i]
                if self.context_channel_names is not None
                else f"ch{ch_idx}"
            )
            cmap = (
                self.context_channel_colormaps[i]
                if self.context_channel_colormaps is not None
                else _CONTEXT_COLORMAP_CYCLE[i % len(_CONTEXT_COLORMAP_CYCLE)]
            )
            specs.append((ch_idx, name, cmap))
        return specs

    def get_z_range(self) -> tuple[int, int]:
        """Get the z-range for volumetric processing"""
        if (
            self.three_d
            and self.z_range_start is not None
            and self.z_range_end is not None
        ):
            return (self.z_range_start, self.z_range_end)

        # Fallback to z_slices if 3D mode is not enabled or range is not specified
        if self.z_slices:
            return (min(self.z_slices), max(self.z_slices))

        return (0, 0)

    def get_z_length(self) -> int:
        """Get the number of z-slices for volumetric processing"""
        if self.three_d:
            z_start, z_end = self.get_z_range()
            return z_end - z_start + 1
        return 1

    def model_dump(self, **kwargs):
        """Override model_dump to exclude patch fields when use_patches is False."""
        data = super().model_dump(**kwargs)

        if not self.use_patches:
            # Remove patch-related fields when patches are not used
            data.pop("patch_size", None)
            data.pop("patches_per_image", None)
            data.pop("random_patches", None)

        return data

    @model_validator(mode="after")
    def validate_3d_settings(self):
        """Validate 3D configuration consistency"""
        if self.three_d:
            if self.z_range_start is None or self.z_range_end is None:
                if not self.z_slices:
                    raise ValueError(
                        "three_d=True requires either "
                        "z_range_start/z_range_end or z_slices"
                    )
                # Auto-set z_range from z_slices
                self.z_range_start = min(self.z_slices)
                self.z_range_end = max(self.z_slices)
            elif self.z_range_start > self.z_range_end:
                raise ValueError("z_range_start must be <= z_range_end")

        return self

    @model_validator(mode="after")
    def validate_channel_settings(self):
        """Validate channel configuration consistency.

        Ensures that label_channel and training_channels are valid indices
        that exist in the channels list.
        """
        all_channels = set(self.channels)

        # Validate label_channel is in channels list
        if self.label_channel is not None and self.label_channel not in all_channels:
            raise ValueError(
                f"label_channel ({self.label_channel}) must be in channels list ({self.channels})"
            )

        # Validate training_channels are in channels list
        if self.training_channels is not None:
            for tc in self.training_channels:
                if tc not in all_channels:
                    raise ValueError(
                        f"training_channels ({tc}) must be in channels list ({self.channels})"
                    )

        # Validate context_channels do not overlap with label_channel
        if self.context_channels:
            label_ch = self.get_label_channel()
            overlapping = [ch for ch in self.context_channels if ch == label_ch]
            if overlapping:
                raise ValueError(
                    f"context_channels {overlapping} overlap with label_channel ({label_ch}). "
                    "Context channels must be different from the label channel."
                )
            # Validate optional list lengths match
            if self.context_channel_names is not None and len(
                self.context_channel_names
            ) != len(self.context_channels):
                raise ValueError(
                    f"context_channel_names length ({len(self.context_channel_names)}) "
                    f"must match context_channels length ({len(self.context_channels)})"
                )
            if self.context_channel_colormaps is not None and len(
                self.context_channel_colormaps
            ) != len(self.context_channels):
                raise ValueError(
                    f"context_channel_colormaps length ({len(self.context_channel_colormaps)}) "
                    f"must match context_channels length ({len(self.context_channels)})"
                )

        return self


class NormalizationConfig(BaseModel):
    """Intensity normalisation of model input images.

    Defined once and applied either when exporting training data
    (``apply_at_export``) or by the trainer, which is configured from the same values
    (see :mod:`omero_annotate_ai.processing.normalization`).
    """

    method: Literal["percentile", "none"] = Field(
        default="percentile",
        description="'percentile': per-image percentile bounds mapped to [0, 1]. "
        "'none': raw intensities.",
    )
    percentile_low: float = Field(
        default=1.0, ge=0, le=100, description="Lower percentile, mapped to 0"
    )
    percentile_high: float = Field(
        default=99.8, ge=0, le=100, description="Upper percentile, mapped to 1"
    )
    clip: bool = Field(default=True, description="Clip normalised values to [0, 1]")
    apply_at_export: bool = Field(
        default=False,
        description="Write float32 normalised images (bounds recorded in normalization.json) "
        "instead of native bit depth",
    )

    @model_validator(mode="after")
    def validate_percentiles(self):
        if self.percentile_low >= self.percentile_high:
            raise ValueError("percentile_low must be below percentile_high")
        return self


class DatasetInfo(BaseModel):
    """Dataset identification and linking (both schemas)"""

    source_dataset_id: str | None = Field(
        default=None, description="BioImage Archive accession or DOI"
    )
    source_dataset_url: HttpUrl | None = Field(
        default=None, description="URL to source dataset"
    )
    source_description: str = Field(description="Human-readable source description")
    license: str = Field(default="CC-BY-4.0", description="Data license")


class StudyContext(BaseModel):
    """Biological and experimental context (MIFA emphasis)"""

    title: str = Field(description="Study/experiment title")
    description: str = Field(description="Detailed study description")
    keywords: list[str] = Field(default_factory=list, description="Study keywords/tags")
    organism: str | None = Field(default=None, description="Organism studied")
    imaging_method: str | None = Field(
        default=None, description="Microscopy technique used"
    )
    funding_statement: str | None = Field(
        default=None,
        description="How the study was funded (required by MIFA Study metadata)",
    )


class AIModelConfig(BaseModel):
    """AI model configuration (bioimage.io compatible)"""

    model_config = ConfigDict(protected_namespaces=())

    # Framework information
    framework: Literal["micro_sam", "cellpose", "web_annotation"] = Field(
        default="micro_sam", description="AI framework"
    )
    framework_version: str | None = Field(default=None, description="Framework version")

    # Model information
    model_name: str = Field(default="", description="Model name/identifier")
    model_version: str = Field(default="latest", description="Model version")
    model_url: HttpUrl | None = Field(
        default=None, description="URL to model weights/repository"
    )
    model_doi: str | None = Field(default=None, description="DOI for model citation")

    # Training configuration
    training_mode: Literal["inference", "finetuning", "from_scratch"] = Field(
        default="inference", description="Training mode"
    )
    pretrained_from: str | None = Field(
        default="vit_b_lm", description="Base model for finetuning (e.g., 'vit_b_lm')"
    )


class TrainingConfig(BaseModel):
    """Quality metrics and validation (MIFA requirement)"""

    validation_strategy: Literal[
        "random_split", "expert_review", "cross_validation"
    ] = "random_split"

    # The level at which data are assigned to a split. Every annotation of one unit
    # lands in the same split; counts and fractions below refer to units.
    split_unit: Literal["image", "well"] = Field(
        default="image",
        description=(
            "Unit of the train/validation/test split: 'image' (a field with all its "
            "timepoints/z-slices) or 'well' (all fields of a plate well)"
        ),
    )
    split_rationale: str | None = Field(
        default=None,
        description="Why this split unit was chosen, e.g. 'fields within a well do not overlap'",
    )
    stratify_by: list[str] = Field(
        default_factory=list,
        description=(
            "Well key-value pairs (e.g. ['cell_line', 'treatment']) or 'well' to split "
            "within: each combination is divided over the splits separately"
        ),
    )

    # Fraction-based splits (for segment_all=True)
    train_fraction: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of data for training (used when segment_all=True)",
    )
    validation_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of data for validation (used when segment_all=True)",
    )
    test_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of data for testing (used when segment_all=True)",
    )

    # Count-based splits (for segment_all=False)
    # Note: ge=0 allows train_n=0 at field level; the model_validator enforces
    # train_n >= 1 only when segment_all=False (count mode)
    train_n: int = Field(
        default=3,
        ge=0,
        description="Number of training images (used when segment_all=False)",
    )
    validate_n: int = Field(
        default=2, ge=0, description="Number of validation images (0=no validation)"
    )
    test_n: int = Field(
        default=0, ge=0, description="Number of test images (0=no test)"
    )

    segment_all: bool = Field(default=False, description="Process all images vs subset")
    seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Seed for every random choice (split, timepoints, z-slices, patches, "
            "extension), so the selection can be reproduced. Generated and stored "
            "when the pipeline starts if left empty."
        ),
    )
    quality_threshold: float | None = Field(
        default=None, description="Minimum quality score"
    )

    @model_validator(mode="after")
    def validate_splits(self):
        """Validate splits based on segment_all mode."""
        if self.segment_all:
            # Fraction-based mode: validate fractions sum to <= 1.0
            total = self.train_fraction + self.validation_fraction + self.test_fraction
            if total > 1.0:
                raise ValueError(
                    f"train + validation + test fractions must sum to <= 1.0 (got {total:.2f})"
                )
            # Note: train_fraction=0 is allowed (inference-only workflows)
            if self.train_n != 3 or self.validate_n != 2 or self.test_n != 0:
                warnings.warn(
                    "train_n/validate_n/test_n are set but ignored because segment_all=True. "
                    "Use train_fraction/validation_fraction/test_fraction instead.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Count-based mode: validate counts
            if self.train_n < 1:
                raise ValueError("train_n must be at least 1 when segment_all=False")
            if (
                self.train_fraction != 0.7
                or self.validation_fraction != 0.3
                or self.test_fraction != 0.0
            ):
                warnings.warn(
                    "train_fraction/validation_fraction/test_fraction are set but ignored "
                    "because segment_all=False. Use train_n/validate_n/test_n instead.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


class WorkflowConfig(BaseModel):
    """How annotation sessions run."""

    batch_size: int = Field(
        default=0, ge=0, description="Images per napari session (0 = all)"
    )


class OMEROConfig(BaseModel):
    """OMERO connection and data selection configuration"""

    container_type: str = Field(default="dataset", description="OMERO container type")
    container_id: int = Field(
        default=0, description="OMERO container ID (single container)"
    )
    container_ids: list[int] | None = Field(
        default=None,
        description="List of OMERO container IDs of the same type. "
        "Takes precedence over container_id when set.",
    )
    source_desc: str = Field(default="", description="Source description for tracking")

    # Well filtering based on key-value pairs (for plate containers)
    well_filters: dict[str, list[str]] | None = Field(
        default=None,
        description="Filter wells by map annotation key-value pairs. "
        "Example: {'cellline': ['U2OS', 'HeLa'], 'treatment': ['Control']}. "
        "All conditions must be met (AND logic). Only applies when container_type='plate'.",
    )
    well_filter_mode: Literal["include", "exclude"] = Field(
        default="include",
        description="Whether to include or exclude wells matching the filters",
    )

    def get_all_container_ids(self) -> list[int]:
        """Get all container IDs, with container_ids taking precedence over container_id.

        Returns:
            List of container IDs. Returns container_ids if set and non-empty,
            otherwise returns [container_id] if container_id is non-zero,
            otherwise returns empty list.
        """
        if self.container_ids is not None and len(self.container_ids) > 0:
            return self.container_ids
        if self.container_id != 0:
            return [self.container_id]
        return []

    def get_primary_container_id(self) -> int:
        """Get the primary (first) container ID.

        Returns:
            The first container ID from the list, or 0 if no containers configured.
        """
        ids = self.get_all_container_ids()
        return ids[0] if ids else 0

    def is_multi_container(self) -> bool:
        """Check if multiple containers are configured.

        Returns:
            True if more than one container ID is configured.
        """
        return len(self.get_all_container_ids()) > 1


class ZarrConfig(BaseModel):
    """OME-Zarr image selection (used when ``source == "zarr"``)."""

    uris: list[str] = Field(
        default_factory=list,
        description="OME-Zarr plates, wells or images (http(s) URL or local path)",
    )
    level: int = Field(default=0, ge=0, description="Multiscale level to read")
    fields: Literal["first", "all"] = Field(
        default="first", description="Fields per well: the first one, or all"
    )
    wells: list[str] | None = Field(
        default=None, description="Only these wells, e.g. ['B2']"
    )
    well_metadata: str | None = Field(
        default=None,
        description="CSV with a 'well' column (optional 'plate') and metadata columns",
    )
    well_filters: dict[str, list[str]] | None = Field(
        default=None, description="Filter wells on well_metadata values (AND logic)"
    )
    well_filter_mode: Literal["include", "exclude"] = Field(default="include")


class Identity(BaseModel):
    """Which training set this config belongs to. Written by omero-annotate-ai.

    ``table_id`` is set for sets tracked in an OMERO table (``storage: omero``),
    ``local_uuid`` for sets kept only on disk (``storage: local``). Setting
    ``table_id`` before opening a set selects that existing set.
    """

    server: str | None = Field(default=None, description="OMERO server slug")
    table_id: int | None = Field(default=None, description="OMERO tracking table ID")
    local_uuid: str | None = Field(default=None, description="ID of a local-only set")


class SessionRecord(BaseModel):
    """One annotation session: who, when, how many units, and the user's note."""

    id: int
    started: datetime
    ended: datetime | None = None
    status: Literal["open", "finished", "interrupted"] = "open"
    user: str | None = None
    software_version: str | None = None
    model: str | None = None
    batch_size: int | None = None
    n_annotated: int = 0
    note: str | None = None


SCHEMA_VERSION = 3

# Keys earlier versions had, and where their meaning went. Loading a config that
# still has one raises RemovedConfigKey naming the replacement (no silent migration).
_REMOVED_KEYS: dict[tuple[str, ...], str] = {
    ("annotations",): "rows live in tracking_table.csv and the OMERO table; use "
    "TrainingSet.units()",
    ("output",): "each training set has its own folder under the home folder; pass "
    "home= or set $OMERO_ANNOTATE_AI_HOME",
    ("config_file_path",): "the config lives in the training set's folder",
    ("version",): "schema_version",
    ("workflow", "resume_from_table"): "open the set with TrainingSet.open(table_id), "
    "or set identity.table_id",
    ("workflow", "read_only_mode"): "storage: local",
    ("omero", "table_id"): "identity.table_id",
    ("spatial_coverage", "normalization_method"): "normalization.method",
    ("spatial_coverage", "percentile_low"): "normalization.percentile_low",
    ("spatial_coverage", "percentile_high"): "normalization.percentile_high",
}


class AnnotationConfig(BaseModel):
    """A training set's definition: what to annotate, how, and its history.

    The rows (one per annotation unit) are not part of the config; they live in the
    set's ``tracking_table.csv`` and OMERO tracking table.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = Field(
        default=SCHEMA_VERSION, description="Configuration schema version"
    )

    def __setattr__(self, name: str, value: Any) -> None:
        # pydantic runs the model checks after storing the new value, so a rejected
        # assignment would otherwise leave the config invalid; put the old value back.
        previous = self.__dict__.get(name)
        try:
            super().__setattr__(name, value)
        except ValidationError:
            if name in self.__dict__:
                object.__setattr__(self, name, previous)
            raise

    # Core identification (both schemas)
    name: str = Field(description="Annotation workflow name")
    authors: list[AuthorInfo] = Field(
        default_factory=list, description="Workflow authors"
    )
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    # Study context
    study: StudyContext = Field(
        default_factory=lambda: StudyContext(title="", description="")
    )
    dataset: DatasetInfo = Field(
        default_factory=lambda: DatasetInfo(source_description="")
    )

    # Annotation specifics
    annotation_methodology: AnnotationMethodology = Field(
        default_factory=lambda: AnnotationMethodology(annotation_criteria="")
    )
    spatial_coverage: SpatialCoverage = Field(default_factory=SpatialCoverage)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)

    # Technical configuration
    ai_model: AIModelConfig = Field(default_factory=AIModelConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    omero: OMEROConfig = Field(default_factory=OMEROConfig)
    source: Literal["omero", "zarr"] = Field(
        default="omero", description="Where images come from"
    )
    zarr: ZarrConfig = Field(default_factory=ZarrConfig)
    storage: Literal["omero", "local"] = Field(
        default="omero",
        description="Where rows and masks are kept: an OMERO tracking table with "
        "uploads ('omero'), or only the set's local folder ('local'). Fixed when the "
        "set is created; OME-Zarr sets are always local.",
    )

    # Written by omero-annotate-ai
    identity: Identity = Field(default_factory=Identity)
    sessions: list[SessionRecord] = Field(
        default_factory=list, description="Annotation sessions, oldest first"
    )

    # Workflow metadata
    documentation: HttpUrl | None = Field(default=None, description="Documentation URL")
    repository: HttpUrl | None = Field(default=None, description="Code repository URL")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    feature_types: list[FeatureType] = Field(
        default_factory=list,
        description="Annotation classes (e.g. cell, nucleus) with display colors",
    )

    # Open fields for anything the structured schema does not cover
    notes: str | None = Field(
        default=None,
        description="Free-text notes about the annotation set (context, caveats, known issues)",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata not covered by the schema",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for path, replacement in _REMOVED_KEYS.items():
            node: Any = data
            for key in path[:-1]:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, dict) and path[-1] in node:
                raise RemovedConfigKey(
                    f"'{'.'.join(path)}' was removed in schema version "
                    f"{SCHEMA_VERSION}: {replacement}"
                )
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version {version!r} is not supported; this version reads "
                f"schema_version {SCHEMA_VERSION}"
            )
        return data

    @model_validator(mode="after")
    def _check_sources(self) -> "AnnotationConfig":
        if self.source == "zarr":
            if not self.zarr.uris:
                raise ValueError("source 'zarr' needs at least one entry in zarr.uris")
            if self.storage != "local":
                raise ValueError(
                    "OME-Zarr sets are kept locally: set storage: local with source: zarr"
                )
        if self.zarr.well_filters and not self.zarr.well_metadata:
            raise ValueError(
                "zarr.well_filters needs zarr.well_metadata (a CSV) to filter on"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def to_yaml(self, sort_keys: bool = False) -> str:
        """Convert configuration to a YAML string.

        Args:
            sort_keys: When True, keys are sorted alphabetically. When False (default),
                keys are emitted in the schema-defined order from the Pydantic model.

        Returns:
            YAML string representation of the configuration.
        """
        config_dict = self.model_dump()

        # Custom YAML representer for Path objects (fallback)
        def path_representer(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

        # Use SafeDumper and ensure our Path is handled
        class _Dumper(yaml.SafeDumper):
            pass

        _Dumper.add_representer(Path, path_representer)

        # Note: sort_keys=False preserves insertion order from dict, which matches
        # the field order defined on the Pydantic model (stable across runs).
        return yaml.dump(
            config_dict,
            Dumper=_Dumper,
            default_flow_style=False,
            sort_keys=sort_keys,
            indent=2,
            allow_unicode=True,
        )

    def save_yaml(self, file_path: str | Path, *, sort_keys: bool = False) -> None:
        """Save configuration to YAML file and remember the path."""
        file_path = Path(file_path)

        # Save the config using the same serialization as to_yaml to keep ordering consistent
        file_path.write_text(self.to_yaml(sort_keys=sort_keys), encoding="utf-8")

        logger.info("Configuration saved to %s", file_path)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AnnotationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_source: str | Path) -> "AnnotationConfig":
        """Create configuration from YAML string or file path."""
        yaml_path = Path(yaml_source)

        # A multi-line string is YAML text; Path.exists() raises on long ones.
        if "\n" not in str(yaml_source) and yaml_path.exists():
            config_dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            return cls.from_dict(config_dict)

        # Not a file path - treat as YAML string
        config_dict = yaml.safe_load(str(yaml_source))
        return cls.from_dict(config_dict)

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string.

        Uses the same dict representation as to_dict(), ensuring
        consistency with YAML serialization.
        """
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_json(cls, json_source: str | Path | dict) -> "AnnotationConfig":
        """Load from JSON string, file path, or dict.

        Args:
            json_source: JSON string, Path to .json file, or dict.

        Returns:
            Hydrated AnnotationConfig instance.
        """
        if isinstance(json_source, dict):
            return cls.from_dict(json_source)
        if isinstance(json_source, Path) or (
            isinstance(json_source, str) and not json_source.strip().startswith("{")
        ):
            data = json.loads(Path(json_source).read_text(encoding="utf-8"))
            return cls.from_dict(data)
        data = json.loads(json_source)
        return cls.from_dict(data)


def load_config(config_source: str | Path | dict[str, Any]) -> AnnotationConfig:
    """Load configuration from various sources."""
    if isinstance(config_source, dict):
        return AnnotationConfig.from_dict(config_source)
    if isinstance(config_source, (str, Path)):
        # from_yaml handles both file paths and YAML strings
        return AnnotationConfig.from_yaml(config_source)
    raise ValueError("config_source must be a dict, file path, or YAML string")


def load_config_from_yaml(yaml_path: str) -> AnnotationConfig:
    """Load AnnotationConfig from a YAML file.

    Convenience function for loading configuration from YAML files.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        AnnotationConfig object
    """
    path = Path(yaml_path).expanduser()  # "~/omero_annotate_ai/..." as in the docs
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    return AnnotationConfig.from_yaml(path)


def create_default_config() -> AnnotationConfig:
    """Create a default configuration."""
    return AnnotationConfig(name="default_annotation_workflow")


def get_config_template() -> str:
    """Get a YAML template with comments for all configuration options."""
    return """# omero-annotate-ai training-set configuration (schema version 3)

schema_version: 3

name: "micro_sam_nuclei_segmentation"
authors: []
created: "2025-01-14T10:30:00Z"

study:
  title: "Automated nuclei segmentation in fluorescence microscopy"
  description: "Large-scale annotation of cell nuclei using micro-SAM for training"
  keywords: ["nuclei", "segmentation", "fluorescence", "deep learning"]
  organism: "Homo sapiens"
  imaging_method: "fluorescence microscopy"

dataset:
  source_dataset_id: "S-BIAD123"
  source_dataset_url: "https://www.ebi.ac.uk/bioimaging/studies/S-BIAD123"
  source_description: "HeLa cell imaging dataset"
  license: "CC-BY-4.0"

annotation_methodology:
  annotation_type: "segmentation_mask"
  annotation_method: "automatic"
  annotation_criteria: "Complete nuclei boundaries based on DAPI staining"
  annotation_coverage: "representative"
  annotation_tool:
    name: "micro-sam"
    version: "1.0.0"
    mode: "interactive"
    parameters: null

spatial_coverage:
  channels: [0]
  timepoints: [0]
  timepoint_mode: "specific"
  z_slices: [0]
  z_slice_mode: "specific"
  three_d: false
  # 3D volumetric processing settings
  z_range_start: null          # starting z-slice for 3D volumes
  z_range_end: null            # ending z-slice for 3D volumes
  # Patch processing settings
  use_patches: true
  patch_size: [512, 512]
  patches_per_image: 4
  random_patches: true

training:
  validation_strategy: "random_split"
  train_fraction: 0.7
  train_n: 3
  validation_fraction: 0.3
  validate_n: 3
  segment_all: false
  split_unit: "image"

normalization:
  method: "percentile"         # "percentile" or "none" (raw intensities)
  percentile_low: 1.0          # mapped to 0
  percentile_high: 99.8        # mapped to 1
  clip: true
  apply_at_export: false       # true: write float32 normalised images + normalization.json

workflow:
  batch_size: 8                # images per napari session (0 = all)

storage: "omero"               # "omero": tracking table + uploads; "local": folder only

ai_model:
  framework: "micro_sam"
  framework_version: null
  model_name: ""
  model_version: "latest"
  model_url: null
  model_doi: null
  training_mode: "inference"
  pretrained_from: "vit_b_lm"

tags: ["segmentation", "nuclei", "micro-sam", "AI-ready"]
"""
