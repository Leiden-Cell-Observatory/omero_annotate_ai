"""Configuration management for OMERO AI annotation workflows."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from pydantic import BaseModel, Field, HttpUrl, model_validator, ConfigDict
from typing_extensions import Literal


# Sub-models for the configuration

class ImageAnnotation(BaseModel):
    """Individual image annotation record for tracking processing state"""
    
    # Image identification
    image_id: int = Field(description="OMERO image ID")
    image_name: str = Field(description="OMERO image name")
    annotation_id: str = Field(default="", description="Unique annotation identifier")
    
    # Processing parameters
    category: Literal["training", "validation", "test"] = Field(
        default="training",
        description="Data split category"
    )
    timepoint: int = Field(default=-1, description="Timepoint index")
    z_slice: int = Field(default=-1, description="Z-slice index")
    channel: int = Field(default=-1, description="Channel index")
    
    # 3D/volumetric processing
    is_volumetric: bool = Field(default=False, description="3D volumetric processing mode")
    z_start: int = Field(default=-1, description="Starting z-slice for 3D volumes")
    z_end: int = Field(default=-1, description="Ending z-slice for 3D volumes") 
    z_length: int = Field(default=1, description="Number of z-slices")
    
    # Patch processing
    is_patch: bool = Field(default=False, description="Whether this is a patch")
    patch_x: int = Field(default=0, description="Patch X coordinate")
    patch_y: int = Field(default=0, description="Patch Y coordinate")
    patch_width: int = Field(default=0, description="Patch width")
    patch_height: int = Field(default=0, description="Patch height")
    
    # AI model info
    model_type: str = Field(default="vit_b_lm", description="SAM model type")
    
    # Processing status
    processed: bool = Field(default=False, description="Whether processing is complete")
    annotation_creation_time: Optional[str] = Field(default=None, description="Completion timestamp in ISO format")
    annotation_type: str = Field(default="segmentation_mask", description="Type of annotation")
    
    # OMERO annotation IDs (None until uploaded)
    roi_id: Optional[int] = Field(default=None, description="OMERO ROI ID")
    label_id: Optional[int] = Field(default=None, description="OMERO label file annotation ID") 
    schema_attachment_id: Optional[int] = Field(default=None, description="OMERO schema attachment ID")


class AuthorInfo(BaseModel):
    """Author information compatible with bioimage.io"""

    name: Optional[str] = Field(default=None, description="Full name of the author")
    affiliation: Optional[str] = Field(
        default=None, description="Institution affiliation"
    )
    email: Optional[str] = Field(None, description="Contact email")
    orcid: Optional[HttpUrl] = Field(None, description="ORCID identifier")


class AnnotationMethodology(BaseModel):
    """MIFA-compatible annotation methodology"""

    annotation_type: Literal[
        "segmentation_mask", "bounding_box", "point", "classification"
    ] = "segmentation_mask"
    annotation_method: Literal["manual", "semi_automatic", "automatic"] = "automatic"
    annotation_criteria: str = Field(description="Criteria used for annotation")
    annotation_coverage: Literal["all", "representative", "partial"] = "representative"


class SpatialCoverage(BaseModel):
    """Spatial scope of annotations (MIFA requirement)"""

    channels: List[int] = Field(description="Channel indices processed")
    timepoints: List[int] = Field(description="Timepoints as list")
    timepoint_mode: Literal["all", "random", "specific"] = "specific"
    z_slices: List[int] = Field(description="Z-slices as list")
    z_slice_mode: Literal["all", "random", "specific"] = "specific"
    spatial_units: str = Field(
        default="pixels", description="Spatial measurement units"
    )
    three_d: bool = Field(default=False, description="3D volumetric processing mode")

    # 3D volumetric processing fields (used when three_d=True)
    z_range_start: Optional[int] = Field(
        default=None, description="Starting z-slice for 3D volumes (when three_d=True)"
    )
    z_range_end: Optional[int] = Field(
        default=None, description="Ending z-slice for 3D volumes (when three_d=True)"
    )

    @property
    def primary_channel(self) -> int:
        """Get the primary/first channel"""
        return self.channels[0]

    @property
    def is_single_channel(self) -> bool:
        """Check if only one channel is configured"""
        return len(self.channels) == 1

    @property
    def is_volumetric(self) -> bool:
        """Check if 3D volumetric processing is enabled"""
        return self.three_d

    def get_z_range(self) -> Tuple[int, int]:
        """Get the z-range for volumetric processing"""
        if (
            self.is_volumetric
            and self.z_range_start is not None
            and self.z_range_end is not None
        ):
            return (self.z_range_start, self.z_range_end)
        elif self.z_slices:
            return (min(self.z_slices), max(self.z_slices))
        else:
            return (0, 0)

    def get_z_length(self) -> int:
        """Get the number of z-slices for volumetric processing"""
        if self.is_volumetric:
            z_start, z_end = self.get_z_range()
            return z_end - z_start + 1
        else:
            return 1

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


class DatasetInfo(BaseModel):
    """Dataset identification and linking (both schemas)"""

    source_dataset_id: Optional[str] = Field(
        default=None, description="BioImage Archive accession or DOI"
    )
    source_dataset_url: Optional[HttpUrl] = Field(
        default=None, description="URL to source dataset"
    )
    source_description: str = Field(description="Human-readable source description")
    license: str = Field(default="CC-BY-4.0", description="Data license")


class StudyContext(BaseModel):
    """Biological and experimental context (MIFA emphasis)"""

    title: str = Field(description="Study/experiment title")
    description: str = Field(description="Detailed study description")
    keywords: List[str] = Field(default_factory=list, description="Study keywords/tags")
    organism: Optional[str] = Field(default=None, description="Organism studied")
    imaging_method: Optional[str] = Field(
        default=None, description="Microscopy technique used"
    )


class AIModelConfig(BaseModel):
    """AI model configuration (bioimage.io compatible)"""

    name: str = Field(description="Model name/identifier")
    version: str = Field(default="latest", description="Model version")
    model_type: str = Field(default="vit_b_lm", description="Model type/architecture")
    framework: str = Field(default="micro_sam", description="AI framework")


class ProcessingConfig(BaseModel):
    """Processing parameters"""

    batch_size: int = Field(default=0, ge=0, description="Batch size (0 = all)")
    use_patches: bool = Field(
        default=False, description="Extract patches vs full images"
    )
    patch_size: List[int] = Field(
        default=[512, 512], description="Patch dimensions [width, height]"
    )
    patches_per_image: int = Field(default=1, gt=0, description="Patches per image")
    random_patches: bool = Field(
        default=True, description="Use random patch extraction"
    )


class TrainingConfig(BaseModel):
    """Quality metrics and validation (MIFA requirement)"""

    validation_strategy: Literal[
        "random_split", "expert_review", "cross_validation"
    ] = "random_split"

    # Fraction-based splits (for segment_all=True)
    train_fraction: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Fraction of data for training (used when segment_all=True)"
    )
    validation_fraction: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Fraction of data for validation (used when segment_all=True)"
    )
    test_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of data for testing (used when segment_all=True)"
    )

    # Count-based splits (for segment_all=False)
    train_n: int = Field(default=3, ge=1, description="Number of training images (used when segment_all=False)")
    validate_n: int = Field(default=2, ge=0, description="Number of validation images (0=no validation)")
    test_n: int = Field(default=0, ge=0, description="Number of test images (0=no test)")

    segment_all: bool = Field(
        default=False, description="Process all images vs subset"
    )
    quality_threshold: Optional[float] = Field(
        default=None, description="Minimum quality score"
    )

    @model_validator(mode='after')
    def validate_splits(self):
        """Ensure fractions sum to <= 1.0 and at least one training image"""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if total > 1.0:
            raise ValueError(
                f"train + validation + test fractions must sum to ≤ 1.0 (got {total:.2f})"
            )
        if self.train_n < 1:
            raise ValueError("train_n must be at least 1")
        return self


class WorkflowConfig(BaseModel):
    """Workflow control and state management"""

    resume_from_table: bool = Field(
        default=False, description="Resume from existing annotation table"
    )
    read_only_mode: bool = Field(
        default=False, description="Read-only mode for viewing results"
    )


class OMEROConfig(BaseModel):
    """OMERO connection and data selection configuration"""

    container_type: str = Field(default="dataset", description="OMERO container type")
    container_id: int = Field(default=0, description="OMERO container ID")
    source_desc: str = Field(default="", description="Source description for tracking")


class OutputConfig(BaseModel):
    """Output and workflow configuration"""

    output_directory: Path = Field(
        default=Path("./annotations"), description="Output directory"
    )
    format: Literal["tif", "ome_tif", "png", "numpy"] = Field(
        default="tif", description="Output format"
    )
    compression: Optional[str] = Field(default=None, description="Compression method")
    resume_from_checkpoint: bool = Field(
        default=False, description="Resume interrupted workflow"
    )

    model_config = ConfigDict(json_encoders={Path: str})

    def model_dump(self, **kwargs):
        """Override model_dump method to convert Path to string"""
        data = super().model_dump(**kwargs)
        if isinstance(data.get("output_directory"), Path):
            data["output_directory"] = str(data["output_directory"])
        return data


class AnnotationConfig(BaseModel):
    """Unified configuration compatible with MIFA and bioimage.io standards"""

    # Schema identification
    schema_version: str = Field(
        default="1.0.0", description="Configuration schema version"
    )

    # Config file tracking for persistence
    config_file_path: Optional[Path] = Field(
        default=None, description="Path to the configuration file for persistence"
    )

    # Core identification (both schemas)
    name: str = Field(description="Annotation workflow name")
    version: str = Field(default="1.0.0", description="Configuration version")
    authors: List[AuthorInfo] = Field(
        default_factory=list, description="Workflow authors"
    )
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    # Study context (MIFA emphasis)
    study: StudyContext = Field(
        default_factory=lambda: StudyContext(title="", description="")
    )
    dataset: DatasetInfo = Field(
        default_factory=lambda: DatasetInfo(source_description="")
    )

    # Annotation specifics (MIFA requirement)
    annotation_methodology: AnnotationMethodology = Field(
        default_factory=lambda: AnnotationMethodology(annotation_criteria="")
    )
    spatial_coverage: SpatialCoverage = Field(
        default_factory=lambda: SpatialCoverage(
            channels=[0], timepoints=[0], z_slices=[0]
        )
    )
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())

    # Technical configuration
    ai_model: AIModelConfig = Field(default_factory=lambda: AIModelConfig(name=""))
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    output: OutputConfig = Field(default_factory=lambda: OutputConfig())
    omero: OMEROConfig = Field(default_factory=lambda: OMEROConfig())

    # NEW: Annotation tracking
    annotations: List[ImageAnnotation] = Field(
        default_factory=list, description="List of image annotations for tracking processing state"
    )

    # Workflow metadata (bioimage.io style)
    documentation: Optional[HttpUrl] = Field(
        default=None, description="Documentation URL"
    )
    repository: Optional[HttpUrl] = Field(
        default=None, description="Code repository URL"
    )
    tags: List[str] = Field(default_factory=list, description="Classification tags")

    def model_dump(self, **kwargs):
        """Override model_dump to handle Path serialization"""
        data = super().model_dump(**kwargs)

        # Convert Path objects to strings
        if "output" in data and "output_directory" in data["output"]:
            if isinstance(data["output"]["output_directory"], Path):
                data["output"]["output_directory"] = str(
                    data["output"]["output_directory"]
                )

        # Handle config_file_path
        if "config_file_path" in data and data["config_file_path"] is not None:
            if isinstance(data["config_file_path"], Path):
                data["config_file_path"] = str(data["config_file_path"])

        return data

    # Annotation management methods
    def add_annotation(self, annotation: ImageAnnotation) -> ImageAnnotation:
        """Add new annotation record to the configuration.

        Args:
            annotation: Pre-created ImageAnnotation object

        Returns:
            The added ImageAnnotation object
        """
        self.annotations.append(annotation)
        return annotation

    def get_unprocessed(self) -> List[ImageAnnotation]:
        """Get annotations where processed=False.
        
        Returns:
            List of unprocessed ImageAnnotation objects
        """
        return [ann for ann in self.annotations if not ann.processed]
    
    def get_processed(self) -> List[ImageAnnotation]:
        """Get annotations where processed=True.
        
        Returns:
            List of processed ImageAnnotation objects
        """
        return [ann for ann in self.annotations if ann.processed]

    def mark_completed(self, image_id: int, roi_id: Optional[int] = None, 
                      label_id: Optional[int] = None, **kwargs):
        """Mark annotation as completed with OMERO IDs.
        
        Args:
            image_id: OMERO image ID to update
            roi_id: OMERO ROI ID (optional)
            label_id: OMERO label file annotation ID (optional)
            **kwargs: Additional fields to update
        """
        for annotation in self.annotations:
            if annotation.image_id == image_id:
                annotation.processed = True
                annotation.annotation_creation_time = datetime.now().isoformat()
                if roi_id is not None:
                    annotation.roi_id = roi_id
                if label_id is not None:
                    annotation.label_id = label_id
                # Update any additional fields
                for key, value in kwargs.items():
                    if hasattr(annotation, key):
                        setattr(annotation, key, value)

    def get_progress_summary(self) -> Dict[str, Union[int, float]]:
        """Get completion statistics.
        
        Returns:
            Dictionary with progress information
        """
        total = len(self.annotations)
        completed = len(self.get_processed())
        return {
            "total_units": total,
            "completed_units": completed,
            "pending_units": total - completed,
            "progress_percent": round(100 * completed / total, 1) if total > 0 else 0
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert annotations to OMERO-compatible DataFrame.
        
        Returns:
            DataFrame matching OMERO table schema
        """
        if not self.annotations:
            return pd.DataFrame()
            
        # Convert annotations to list of dicts
        rows = []
        for annotation in self.annotations:
            # Map ImageAnnotation fields to OMERO table columns
            row = {
                "image_id": annotation.image_id,
                "image_name": annotation.image_name,
                "train": annotation.category == "training",
                "validate": annotation.category == "validation",
                "channel": annotation.channel,
                "z_slice": annotation.z_slice,
                "timepoint": annotation.timepoint,
                "sam_model": annotation.model_type,
                "label_id": str(annotation.label_id) if annotation.label_id is not None else "None",
                "roi_id": str(annotation.roi_id) if annotation.roi_id is not None else "None",
                "is_volumetric": annotation.is_volumetric,
                "processed": annotation.processed,
                "is_patch": annotation.is_patch,
                "patch_x": annotation.patch_x,
                "patch_y": annotation.patch_y,
                "patch_width": annotation.patch_width,
                "patch_height": annotation.patch_height,
                "annotation_type": annotation.annotation_type,
                "annotation_creation_time": annotation.annotation_creation_time or "None",
                "schema_attachment_id": str(annotation.schema_attachment_id) if annotation.schema_attachment_id is not None else "None",
                "z_start": annotation.z_start,
                "z_end": annotation.z_end,
                "z_length": annotation.z_length,
            }
            rows.append(row)
        
        # Create DataFrame with proper column order
        columns = [
            "image_id", "image_name", "train", "validate", "channel", "z_slice", "timepoint",
            "sam_model", "label_id", "roi_id", "is_volumetric", "processed", "is_patch",
            "patch_x", "patch_y", "patch_width", "patch_height", "annotation_type",
            "annotation_creation_time", "schema_attachment_id", "z_start", "z_end", "z_length"
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Ensure proper data types for OMERO compatibility
        numeric_columns = ["image_id", "patch_x", "patch_y", "patch_width", "patch_height",
                          "z_slice", "timepoint", "z_start", "z_end", "z_length"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        
        boolean_columns = ["train", "validate", "processed", "is_patch", "is_volumetric"]
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
                
        return df

    def from_dataframe(self, df: pd.DataFrame):
        """Load annotations from existing OMERO table DataFrame.
        
        Args:
            df: DataFrame from OMERO table
        """
        self.annotations.clear()
        
        for _, row in df.iterrows():
            # Map OMERO table columns back to ImageAnnotation fields
            annotation_data = {
                "image_id": int(row.get("image_id", -1)),
                "image_name": str(row.get("image_name", "")),
                "category": "training" if row.get("train", True) else "validation",
                "timepoint": int(row.get("timepoint", -1)),
                "z_slice": int(row.get("z_slice", -1)),
                "channel": int(row.get("channel", -1)),
                "is_volumetric": bool(row.get("is_volumetric", False)),
                "z_start": int(row.get("z_start", -1)),
                "z_end": int(row.get("z_end", -1)),
                "z_length": int(row.get("z_length", 1)),
                "is_patch": bool(row.get("is_patch", False)),
                "patch_x": int(row.get("patch_x", 0)),
                "patch_y": int(row.get("patch_y", 0)),
                "patch_width": int(row.get("patch_width", 0)),
                "patch_height": int(row.get("patch_height", 0)),
                "model_type": str(row.get("sam_model", "vit_b_lm")),
                "processed": bool(row.get("processed", False)),
                "annotation_type": str(row.get("annotation_type", "segmentation_mask")),
            }
            
            # Handle optional fields
            roi_id_str = str(row.get("roi_id", "None"))
            if roi_id_str != "None" and roi_id_str.isdigit():
                annotation_data["roi_id"] = int(roi_id_str)
                
            label_id_str = str(row.get("label_id", "None"))  
            if label_id_str != "None" and label_id_str.isdigit():
                annotation_data["label_id"] = int(label_id_str)
                
            schema_id_str = str(row.get("schema_attachment_id", "None"))
            if schema_id_str != "None" and schema_id_str.isdigit():
                annotation_data["schema_attachment_id"] = int(schema_id_str)
                
            # Handle timestamp - keep as string
            creation_time_str = str(row.get("annotation_creation_time", "None"))
            if creation_time_str != "None":
                annotation_data["annotation_creation_time"] = creation_time_str
            
            self.add_annotation(**annotation_data)

    def to_mifa_metadata(self) -> dict:
        """Export MIFA-compatible metadata"""
        return {
            "annotation_type": self.annotation_methodology.annotation_type,
            "annotation_method": self.annotation_methodology.annotation_method,
            "annotation_criteria": self.annotation_methodology.annotation_criteria,
            "spatial_coverage": self.spatial_coverage.model_dump(),
            "study_context": self.study.model_dump(),
            "quality_metrics": self.training.model_dump(),
        }

    def to_bioimage_io_rdf(self) -> dict:
        """Export bioimage.io RDF-compatible structure"""
        return {
            "format_version": "0.5.3",
            "type": "dataset",
            "name": self.name,
            "description": self.study.description,
            "authors": [author.model_dump() for author in self.authors],
            "tags": self.tags,
            "source": self.dataset.source_dataset_url,
            "documentation": self.documentation,
            "git_repo": self.repository,
        }

    def to_dict(self) -> Dict[str, Any]:
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

    def save_yaml(self, file_path: Union[str, Path], *, sort_keys: bool = False) -> None:
        """Save configuration to YAML file and remember the path."""
        file_path = Path(file_path)
        
        # Save the config using the same serialization as to_yaml to keep ordering consistent
        yaml_str = self.to_yaml(sort_keys=sort_keys)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(yaml_str)
        
        # Remember where we saved it
        self.config_file_path = file_path
        
        print(f"✅ Configuration saved to: {file_path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnnotationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_source: Union[str, Path]) -> "AnnotationConfig":
        """Create configuration from YAML string or file path."""
        config_dict = None
        source_path = None
        
        if isinstance(yaml_source, (str, Path)):
            yaml_path = Path(yaml_source)
            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    config_dict = yaml.safe_load(f)
                source_path = yaml_path
            else:
                # Assume it's a YAML string
                config_dict = yaml.safe_load(str(yaml_source))
        else:
            config_dict = yaml.safe_load(yaml_source)

        config = cls.from_dict(config_dict)
        
        # Remember source path if loaded from file
        if source_path:
            config.config_file_path = source_path
        
        return config


def parse_sequence(value: Union[str, List[int]]) -> List[int]:
    """Parse a sequence specification into a list of integers."""
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        # Handle range notation like "0:100"
        if ":" in value:
            start, end = value.split(":")
            return list(range(int(start), int(end)))
        else:
            # Handle comma-separated values
            return [int(x.strip()) for x in value.split(",")]
    else:
        return [value]


def load_config(config_source: Union[str, Path, Dict[str, Any]]) -> AnnotationConfig:
    """Load configuration from various sources."""
    if isinstance(config_source, dict):
        return AnnotationConfig.from_dict(config_source)
    elif isinstance(config_source, (str, Path)):
        if Path(config_source).exists():
            return AnnotationConfig.from_yaml(config_source)
        else:
            # Assume it's a YAML string
            return AnnotationConfig.from_yaml(config_source)
    else:
        raise ValueError("config_source must be a dict, file path, or YAML string")


def load_config_from_yaml(yaml_path: str) -> AnnotationConfig:
    """Load AnnotationConfig from a YAML file.

    This is a simple drop-in replacement for workflow_widget.get_config()
    to enable easy testing of the pipeline with YAML configuration files.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        AnnotationConfig object

    Example:
        # Instead of: config = workflow_widget.get_config()
        config = load_config_from_yaml('test_config.yaml')
    """

    config_path = Path(yaml_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    return AnnotationConfig.from_yaml(config_path)


def create_default_config() -> AnnotationConfig:
    """Create a default configuration."""
    return AnnotationConfig(name="default_annotation_workflow")


def get_config_template() -> str:
    """Get a YAML template with comments for all configuration options."""
    template = """# OMERO micro-SAM Configuration Template v1.0.0

schema_version: "1.0.0"

name: "micro_sam_nuclei_segmentation"
version: "1.0.0"
authors: []
created: "2025-01-14T10:30:00Z"

study:
  title: "Automated nuclei segmentation in fluorescence microscopy"
  description: "Large-scale annotation of cell nuclei using micro-SAM for training segmentation models"
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

spatial_coverage:
  channels: [0]
  timepoints: [0]
  timepoint_mode: "specific"
  z_slices: [0]
  z_slice_mode: "specific"
  spatial_units: "pixels"
  three_d: false
  # 3D volumetric processing settings  
  z_range_start: null          # starting z-slice for 3D volumes
  z_range_end: null            # ending z-slice for 3D volumes
  
training:
  validation_strategy: "random_split"
  train_fraction: 0.7
  train_n: 3
  validation_fraction: 0.3
  validate_n: 3
  segment_all: false  # NEW

workflow:  # NEW SECTION
  resume_from_table: false
  read_only_mode: false

ai_model:
  name: "micro-sam"
  model_type: "vit_b_lm"
  framework: "pytorch"

processing:
  batch_size: 8
  use_patches: true
  patch_size: [512, 512]
  patches_per_image: 4
  random_patches: true  # NEW
  
output:
  output_directory: "./annotations"
  format: "ome_tiff"
  resume_from_checkpoint: false

tags: ["segmentation", "nuclei", "micro-sam", "AI-ready"]
"""
    return template
