"""Configuration of a training set: what to annotate, how, and its history.

The config has two kinds of sections. The **schema sections** (name, licence,
authors, study, source dataset, methodology, classes, sampling, split, ...) are the
models of the Bioimage Annotation Schema (``bioimage_annotation_schema``), so a set's
description follows that schema and exports to MIFA, RO-Crate and the BioImage
Archive without translation. The **tool sections** (omero, zarr, storage, workflow,
training, normalization, ai_model, display, identity, sessions) hold what
omero-annotate-ai needs to run.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from bioimage_annotation_schema import datamodel as bas
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


# Tool sections of the configuration
class _Section(BaseModel):
    """A config section: unknown keys are errors, so a typo never goes unnoticed."""

    model_config = ConfigDict(extra="forbid")


_CONTEXT_COLORMAP_CYCLE = ["blue", "magenta", "green", "yellow", "cyan"]


class DisplayConfig(_Section):
    """What napari shows next to the annotated channel. Display only: context
    channels never enter SAM embeddings or training data."""

    context_channels: list[int] = Field(
        default_factory=list,
        description="Channel indices shown alongside the annotation channel "
        "(OMERO channel index space)",
    )
    context_channel_names: list[str] | None = Field(
        default=None,
        description="Display names for context_channels (same length). "
        "Defaults to 'ch{idx}'.",
    )
    context_channel_colormaps: list[str] | None = Field(
        default=None,
        description="Napari colormaps for context_channels (same length). "
        "Defaults to cycle ['blue', 'magenta', 'green', 'yellow', 'cyan'].",
    )

    @model_validator(mode="after")
    def _lengths_match(self) -> "DisplayConfig":
        for name in ("context_channel_names", "context_channel_colormaps"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.context_channels):
                raise ValueError(
                    f"{name} length ({len(values)}) must match context_channels "
                    f"length ({len(self.context_channels)})"
                )
        return self

    def specs(self) -> list[tuple[int, str, str]]:
        """``[(channel_index, display_name, colormap), ...]`` per context channel."""
        return [
            (
                ch,
                self.context_channel_names[i]
                if self.context_channel_names
                else f"ch{ch}",
                self.context_channel_colormaps[i]
                if self.context_channel_colormaps
                else _CONTEXT_COLORMAP_CYCLE[i % len(_CONTEXT_COLORMAP_CYCLE)],
            )
            for i, ch in enumerate(self.context_channels)
        ]


class NormalizationConfig(_Section):
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


class AIModelConfig(_Section):
    """AI model configuration (bioimage.io compatible)"""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

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


class TrainingConfig(_Section):
    """How many units go to each split. Which unit is split, the stratification and
    the seed are part of the description, in ``split``."""

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


class WorkflowConfig(_Section):
    """How annotation sessions run."""

    batch_size: int = Field(
        default=0, ge=0, description="Images per napari session (0 = all)"
    )
    claim_hours: float = Field(
        default=6.0,
        gt=0,
        description="OMERO sets: hours a session's claim on a unit lasts before "
        "another computer may take the unit over (see omero_annotate_ai.claims)",
    )


class OMEROConfig(_Section):
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
        "All conditions must be met (AND logic). Applies to plates and the plates "
        "of screens.",
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


class ZarrConfig(_Section):
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


class Identity(_Section):
    """Which training set this config belongs to. Written by omero-annotate-ai.

    ``table_id`` is set for sets tracked in an OMERO table (``storage: omero``),
    ``local_uuid`` for sets kept only on disk (``storage: local``). Setting
    ``table_id`` before opening a set selects that existing set.
    """

    server: str | None = Field(default=None, description="OMERO server slug")
    table_id: int | None = Field(default=None, description="OMERO tracking table ID")
    local_uuid: str | None = Field(default=None, description="ID of a local-only set")


class SessionRecord(_Section):
    """One annotation session: who, when, how many units, and the user's note."""

    id: int
    kind: Literal["annotate", "review"] = "annotate"
    started: datetime
    ended: datetime | None = None
    status: Literal["open", "finished", "interrupted"] = "open"
    user: str | None = None
    host: str | None = Field(default=None, description="Hashed ID of the computer")
    software_version: str | None = None
    model: str | None = None
    batch_size: int | None = None
    n_annotated: int = Field(
        default=0, description="Units saved (annotated, or changed in a review)"
    )
    note: str | None = None


CONFIG_VERSION = 4

# Keys earlier versions had, and where their meaning went. Loading a config that
# still has one raises RemovedConfigKey naming the replacement (no silent migration).
_REMOVED_KEYS: dict[tuple[str, ...], str] = {
    ("annotations",): "rows live in tracking_table.csv and the OMERO table; use "
    "TrainingSet.units()",
    ("output",): "each training set has its own folder under the home folder; pass "
    "home= or set $OMERO_ANNOTATE_AI_HOME",
    ("config_file_path",): "the config lives in the training set's folder",
    ("schema_version",): "config_version: 4 (see the configuration guide)",
    ("workflow", "resume_from_table"): "open the set with TrainingSet.open(table_id), "
    "or set identity.table_id",
    ("workflow", "read_only_mode"): "storage: local",
    ("omero", "table_id"): "identity.table_id",
    ("annotation_methodology",): "methodology (annotation_types, method, criteria, "
    "coverage, tools)",
    ("spatial_coverage",): "sampling (label_channel, input_channels, timepoints, "
    "z_slices, volumetric, patches); context channels moved to display",
    ("dataset",): "source_dataset (identifier, url, description) and license",
    ("feature_types",): "classes",
    ("tags",): "keywords",
    ("documentation",): "links",
    ("repository",): "links",
    ("custom_metadata",): "extensions",
    ("training", "validation_strategy"): "split.strategy",
    ("training", "split_unit"): "split.unit",
    ("training", "split_rationale"): "split.rationale",
    ("training", "stratify_by"): "split.stratify_by",
    ("training", "seed"): "split.seed",
    ("training", "quality_threshold"): "methodology.confidence_level",
    ("study", "keywords"): "keywords (top level)",
}


def _default_methodology() -> bas.Methodology:
    return bas.Methodology.model_validate(
        {"annotation_types": ["segmentation_mask"], "criteria": ""}
    )


def _default_sampling() -> bas.Sampling:
    first_plane = {"mode": "specific", "values": [0]}
    return bas.Sampling.model_validate(
        {"timepoints": first_plane, "z_slices": first_plane}
    )


class AnnotationConfig(BaseModel):
    """A training set's definition: what to annotate, how, and its history.

    The rows (one per annotation unit) are not part of the config; they live in the
    set's ``tracking_table.csv`` and OMERO tracking table.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_version: int = Field(
        default=CONFIG_VERSION, description="Version of this configuration format"
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

    # --- Schema sections (Bioimage Annotation Schema) ---
    name: str = Field(description="Short name of the training set")
    description: str | None = None
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    license: Literal["CC-BY-4.0", "CC0-1.0"] = Field(
        default="CC-BY-4.0",
        description="Licence of the annotations: CC-BY-4.0 or CC0-1.0",
    )
    version: str | None = Field(
        default=None, description="Version of the set, e.g. '1' (needed for MIFA)"
    )
    url: str | None = Field(
        default=None,
        description="Where the set is published (DOI or archive page); needed "
        "for the Croissant and bioimage.io descriptions",
    )
    changes: str | None = None
    previous_version: str | None = None
    authors: list[bas.Person] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    links: list[bas.Link] = Field(default_factory=list)
    ai_models_trained: list[str] = Field(default_factory=list)
    study: bas.Study | None = None
    source_dataset: bas.SourceDataset | None = None
    omero_server: bas.OmeroServer | None = Field(
        default=None,
        description="The OMERO server the images are on; filled in from the "
        "connection when the set is created or opened",
    )
    methodology: bas.Methodology = Field(default_factory=_default_methodology)
    classes: list[bas.AnnotationClass] = Field(
        default_factory=list,
        description="Annotation classes (e.g. nucleus) with display colors",
    )
    sampling: bas.Sampling = Field(default_factory=_default_sampling)
    split: bas.Split = Field(
        default_factory=lambda: bas.Split.model_validate(
            {"strategy": "random_split", "unit": "image"}
        )
    )
    notes: str | None = Field(
        default=None,
        description="Free-text notes about the set (context, caveats, known issues)",
    )
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata the schema does not cover, under a "
        "namespace key",
    )

    # --- Tool sections (omero-annotate-ai) ---
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    ai_model: AIModelConfig = Field(default_factory=AIModelConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
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
                    f"'{'.'.join(path)}' was removed in config version "
                    f"{CONFIG_VERSION}: {replacement}"
                )
        version = data.get("config_version", CONFIG_VERSION)
        if version != CONFIG_VERSION:
            raise ConfigError(
                f"config_version {version!r} is not supported; this version reads "
                f"config_version {CONFIG_VERSION}"
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

    @model_validator(mode="after")
    def _check_sampling(self) -> "AnnotationConfig":
        if self.split.unit not in (None, "image", "well"):
            raise ValueError("split.unit must be 'image' or 'well'")
        if self.label_channel() in self.display.context_channels:
            raise ValueError(
                f"display.context_channels overlap with the label channel "
                f"({self.label_channel()}); context channels must differ from it"
            )
        return self

    # --- Sampling, as the tool uses it ---

    def label_channel(self) -> int:
        """Channel the annotations are drawn on (default 0)."""
        channel = self.sampling.label_channel
        return 0 if channel is None else channel

    def training_channels(self) -> list[int]:
        """Channels the model is trained on (default: the label channel)."""
        return list(self.sampling.input_channels or [self.label_channel()])

    def uses_separate_channels(self) -> bool:
        """Whether the model trains on other channels than the one annotated."""
        return self.label_channel() not in self.training_channels()

    def timepoint_values(self) -> list[int]:
        return (
            list(self.sampling.timepoints.values or [0])
            if self.sampling.timepoints
            else [0]
        )

    def z_values(self) -> list[int]:
        return (
            list(self.sampling.z_slices.values or [0])
            if self.sampling.z_slices
            else [0]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(exclude_none=True)

    def to_yaml(self, sort_keys: bool = False) -> str:
        """Convert configuration to a YAML string.

        Args:
            sort_keys: When True, keys are sorted alphabetically. When False (default),
                keys are emitted in the schema-defined order from the Pydantic model.

        Returns:
            YAML string representation of the configuration.
        """
        config_dict = self.model_dump(exclude_none=True)

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
    return """# omero-annotate-ai training-set configuration (config version 4)
config_version: 4

# --- Description (Bioimage Annotation Schema sections) ---
name: "micro_sam_nuclei_segmentation"
created: "2025-01-14T10:30:00Z"
license: "CC-BY-4.0"           # CC-BY-4.0 or CC0-1.0 (what the BioImage Archive accepts)
version: "1"
authors:
  - first_name: "Jane"
    last_name: "Example"
    orcid: "https://orcid.org/0000-0002-1825-0097"
    affiliations: [{name: "Example University"}]
keywords: ["nuclei", "segmentation", "fluorescence"]
links:
  - {url: "https://github.com/example/project", description: "Analysis code"}

study:
  title: "Automated nuclei segmentation in fluorescence microscopy"
  description: "Large-scale annotation of cell nuclei using micro-SAM for training"
  organism: "Homo sapiens"
  imaging_method: "fluorescence microscopy"
  funding_statement: "Funded by ..."

source_dataset:
  identifier: "S-BIAD123"
  url: "https://www.ebi.ac.uk/bioimaging/studies/S-BIAD123"
  description: "HeLa cell imaging dataset"

methodology:
  annotation_types: ["segmentation_mask"]
  mask_kind: "instance"        # instance or semantic
  method: "semi_automatic"     # manual, semi_automatic or automatic
  criteria: "Complete nuclei boundaries based on DAPI staining"
  coverage: "all"              # all, representative or partial
  tools:
    - {name: "micro-sam", version: "1.8.0", mode: "interactive"}

classes:
  - {name: "nucleus", color: "#0FA3B1"}

sampling:
  label_channel: 0
  input_channels: [0]          # channels the model trains on
  timepoints: {mode: "specific", values: [0]}   # mode: all, random (with n) or specific
  z_slices: {mode: "specific", values: [0]}
  volumetric: false            # true: one volume from the first to the last selected z-slice
  patches: {width: 512, height: 512, per_image: 4, random: true}   # leave out for whole planes

split:
  strategy: "random_split"
  unit: "image"                # image or well

# --- omero-annotate-ai settings ---
training:
  train_n: 3
  validate_n: 3
  segment_all: false

normalization:
  method: "percentile"         # "percentile" or "none" (raw intensities)
  percentile_low: 1.0          # mapped to 0
  percentile_high: 99.8        # mapped to 1
  clip: true
  apply_at_export: false       # true: write float32 normalised images + normalization.json

workflow:
  batch_size: 8                # images per napari session (0 = all)
  claim_hours: 6               # OMERO sets: claims on shown units expire after this

storage: "omero"               # "omero": tracking table + uploads; "local": folder only

ai_model:
  framework: "micro_sam"
  pretrained_from: "vit_b_lm"
"""
