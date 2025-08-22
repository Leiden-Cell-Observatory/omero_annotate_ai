"""Configuration management for OMERO AI annotation workflows."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, HttpUrl
from typing_extensions import Literal


# Sub-models for the configuration
class AuthorInfo(BaseModel):
    """Author information compatible with bioimage.io"""

    name: Optional[str] = Field(default=None, description="Full name of the author")
    affiliation: Optional[str] = Field(default=None, description="Institution affiliation")
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
    three_d: bool = Field(default=False, description="3D processing mode")
    
    @property
    def primary_channel(self) -> int:
        """Get the primary/first channel"""
        return self.channels[0]
    
    @property
    def is_single_channel(self) -> bool:
        """Check if only one channel is configured"""
        return len(self.channels) == 1


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
    imaging_method: Optional[str] = Field(default=None, description="Microscopy technique used")


class AIModelConfig(BaseModel):
    """AI model configuration (bioimage.io compatible)"""

    name: str = Field(description="Model name/identifier")
    version: str = Field(default="latest", description="Model version")
    model_type: str = Field(default="vit_b_lm", description="Model type/architecture")
    framework: str = Field(default="pytorch", description="AI framework")


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
    train_fraction: float = Field(
        default=0.7, ge=0.1, le=0.9, description="Training data fraction"
    )
    train_n: int = Field(default=3, gt=0, description="Number of training images")
    validation_fraction: float = Field(
        default=0.3, ge=0.1, le=0.9, description="Validation data fraction"
    )
    validate_n: int = Field(default=3, gt=0, description="Number of validation images")
    segment_all: bool = Field(
        default=False, description="Segment all objects vs sample"
    ) 
    quality_threshold: Optional[float] = Field(
        default=None, description="Minimum quality score"
    )

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
    format: Literal["ome_tiff", "png", "numpy"] = Field(
        default="ome_tiff", description="Output format"
    )
    compression: Optional[str] = Field(default=None, description="Compression method")
    resume_from_checkpoint: bool = Field(
        default=False, description="Resume interrupted workflow"
    )


class AnnotationConfig(BaseModel):
    """Unified configuration compatible with MIFA and bioimage.io standards"""

    # Schema identification
    schema_version: str = Field(
        default="1.0.0", description="Configuration schema version"
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


    # Workflow metadata (bioimage.io style)
    documentation: Optional[HttpUrl] = Field(default=None, description="Documentation URL")
    repository: Optional[HttpUrl] = Field(default=None, description="Code repository URL")
    tags: List[str] = Field(default_factory=list, description="Classification tags")

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
            "authors": [author.dict() for author in self.authors],
            "tags": self.tags,
            "source": self.dataset.source_dataset_url,
            "documentation": self.documentation,
            "git_repo": self.repository,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        config_dict = self.to_dict()
        return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnnotationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_content: Union[str, Path]) -> "AnnotationConfig":
        """Create configuration from YAML string or file path."""
        if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
            # It's a file path
            with open(yaml_content, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            # It's a YAML string
            if isinstance(yaml_content, Path):
                yaml_content = yaml_content.read_text()
            config_dict = yaml.safe_load(yaml_content)

        return cls.from_dict(config_dict)

    def save_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(file_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


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
    from pathlib import Path

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



