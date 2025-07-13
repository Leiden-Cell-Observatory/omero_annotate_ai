"""Configuration management for OMERO AI annotation workflows."""

import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing parameters.
    
    batch_size: Number of images to process at once. 
                0 means process all images in one batch (default: 0)
    """
    batch_size: int = 0  # 0 = all images in one batch
    output_folder: str = "./omero_annotations"


@dataclass
class OMEROConfig:
    """Configuration for OMERO connection and data selection."""
    container_type: str = "dataset"
    container_id: int = 0
    source_desc: str = ""
    channel: int = 0


@dataclass
class MicroSAMConfig:
    """Configuration for micro-SAM model parameters."""
    model_type: str = "vit_b_lm"  # Default model: vit_b_lm. Available: vit_b, vit_l, vit_h, vit_b_lm
    timepoints: list = field(default_factory=lambda: [0])
    timepoint_mode: str = "specific"  # "all", "random", "specific"
    z_slices: list = field(default_factory=lambda: [0])
    z_slice_mode: str = "specific"  # "all", "random", "specific"
    three_d: bool = False


@dataclass
class PatchConfig:
    """Configuration for patch extraction."""
    use_patches: bool = False
    patch_size: tuple = (512, 512)
    patches_per_image: int = 1
    random_patches: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training data organization."""
    segment_all: bool = True
    train_n: int = 3
    validate_n: int = 3
    trainingset_name: str = "default_training_set"  # Required field for training set name


@dataclass
class WorkflowConfig:
    """Configuration for workflow behavior."""
    resume_from_table: bool = False
    read_only_mode: bool = False


@dataclass
class AnnotationConfig:
    """Complete configuration for micro-SAM OMERO workflows."""
    batch_processing: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    omero: OMEROConfig = field(default_factory=OMEROConfig)
    microsam: MicroSAMConfig = field(default_factory=MicroSAMConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        config_dict = self.to_dict()
        # Convert patch_size tuple to list for YAML compatibility
        if 'patches' in config_dict and 'patch_size' in config_dict['patches']:
            if isinstance(config_dict['patches']['patch_size'], tuple):
                config_dict['patches']['patch_size'] = list(config_dict['patches']['patch_size'])
        return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnnotationConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update each section if present in the dictionary
        if 'batch_processing' in config_dict:
            config.batch_processing = BatchProcessingConfig(**config_dict['batch_processing'])
        
        if 'omero' in config_dict:
            config.omero = OMEROConfig(**config_dict['omero'])
        
        if 'microsam' in config_dict:
            config.microsam = MicroSAMConfig(**config_dict['microsam'])
        elif 'ai_model' in config_dict:  # Backward compatibility
            # Map old ai_model config to new microsam config
            old_config = config_dict['ai_model']
            microsam_config = {
                'model_type': old_config.get('model_type', 'vit_l'),
                'timepoints': old_config.get('timepoints', [0]),
                'timepoint_mode': old_config.get('timepoint_mode', 'specific'),
                'z_slices': old_config.get('z_slices', [0]),
                'z_slice_mode': old_config.get('z_slice_mode', 'specific'),
                'three_d': old_config.get('three_d', False)
            }
            config.microsam = MicroSAMConfig(**microsam_config)
        elif 'image_processing' in config_dict:  # Legacy backward compatibility
            # Map old image_processing config to new microsam config
            old_config = config_dict['image_processing']
            microsam_config = {
                'model_type': old_config.get('model_type', 'vit_l'),
                'timepoints': old_config.get('timepoints', [0]),
                'timepoint_mode': old_config.get('timepoint_mode', 'specific'),
                'z_slices': old_config.get('z_slices', [0]),
                'z_slice_mode': old_config.get('z_slice_mode', 'specific'),
                'three_d': old_config.get('three_d', False)
            }
            config.microsam = MicroSAMConfig(**microsam_config)
        
        if 'patches' in config_dict:
            # Convert patch_size from list to tuple if needed
            patch_data = config_dict['patches'].copy()
            if 'patch_size' in patch_data and isinstance(patch_data['patch_size'], list):
                patch_data['patch_size'] = tuple(patch_data['patch_size'])
            config.patches = PatchConfig(**patch_data)
        
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        if 'workflow' in config_dict:
            # Handle backward compatibility for local_output_dir
            workflow_data = config_dict['workflow'].copy()
            if 'local_output_dir' in workflow_data:
                # If local_output_dir exists, use it as output_folder if not already set
                if 'batch_processing' not in config_dict:
                    config.batch_processing.output_folder = workflow_data['local_output_dir']
                # Remove local_output_dir from workflow data
                del workflow_data['local_output_dir']
            config.workflow = WorkflowConfig(**workflow_data)
        
        return config

    @classmethod
    def from_yaml(cls, yaml_content: Union[str, Path]) -> 'AnnotationConfig':
        """Create configuration from YAML string or file path."""
        if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
            # It's a file path
            with open(yaml_content, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            # It's a YAML string
            if isinstance(yaml_content, Path):
                yaml_content = yaml_content.read_text()
            config_dict = yaml.safe_load(yaml_content)
        
        return cls.from_dict(config_dict)

    def save_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Validate batch processing
        if self.batch_processing.batch_size < 0:
            errors.append("batch_size must be non-negative (0 means process all in one batch)")
        
        # Validate OMERO config
        valid_container_types = ["dataset", "plate", "project", "screen", "image"]
        if self.omero.container_type not in valid_container_types:
            errors.append(f"container_type must be one of {valid_container_types}")
        
        if self.omero.container_id <= 0:
            errors.append("container_id must be positive")
        
        if self.omero.channel < 0:
            errors.append("channel must be non-negative")
        
        # Validate micro-SAM config
        valid_models = ["vit_b", "vit_l", "vit_h", "vit_b_lm"]
        if self.microsam.model_type not in valid_models:
            errors.append(f"model_type must be one of {valid_models}")
        
        valid_modes = ["all", "random", "specific"]
        if self.microsam.timepoint_mode not in valid_modes:
            errors.append(f"timepoint_mode must be one of {valid_modes}")
        
        if self.microsam.z_slice_mode not in valid_modes:
            errors.append(f"z_slice_mode must be one of {valid_modes}")
        
        # Validate patch config
        if self.patches.use_patches:
            if len(self.patches.patch_size) != 2:
                errors.append("patch_size must be a tuple of 2 values")
            
            if any(s <= 0 for s in self.patches.patch_size):
                errors.append("patch_size values must be positive")
            
            if self.patches.patches_per_image <= 0:
                errors.append("patches_per_image must be positive")
        
        # Validate training config
        if not self.training.segment_all:
            if self.training.train_n <= 0:
                errors.append("train_n must be positive when not segmenting all")
            
            if self.training.validate_n <= 0:
                errors.append("validate_n must be positive when not segmenting all")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


    def get_microsam_params(self) -> Dict[str, Any]:
        """Get micro-SAM specific parameters."""
        return {
            'model_type': self.microsam.model_type,
            'embedding_path': f"{self.batch_processing.output_folder}/embed",
            'is_volumetric': self.microsam.three_d
        }

    def get_omero_settings(self) -> Dict[str, Any]:
        """Get OMERO-specific settings for user display."""
        return {
            'container_type': self.omero.container_type,
            'container_id': self.omero.container_id,
            'training_set_name': self.training.trainingset_name,
            'source_description': self.omero.source_desc,
            'channel': self.omero.channel
        }

    def get_annotation_settings(self) -> Dict[str, Any]:
        """Get annotation-specific settings for user display."""
        return {
            'segment_all': self.training.segment_all,
            'training_images': self.training.train_n,
            'validation_images': self.training.validate_n,
            'channel': self.omero.channel,
            'timepoint_mode': self.microsam.timepoint_mode,
            'timepoints': self.microsam.timepoints,
            'z_slice_mode': self.microsam.z_slice_mode,
            'z_slices': self.microsam.z_slices,
            'three_d': self.microsam.three_d,
            'use_patches': self.patches.use_patches,
            'patches_per_image': self.patches.patches_per_image,
            'patch_size': self.patches.patch_size,
            'random_patches': self.patches.random_patches
        }

    def get_technical_settings(self) -> Dict[str, Any]:
        """Get technical settings for user display."""
        return {
            'batch_size': self.batch_processing.batch_size,
            'sam_model': self.microsam.model_type,
            'output_folder': self.batch_processing.output_folder,
            'resume_from_table': self.workflow.resume_from_table,
            'read_only_mode': self.workflow.read_only_mode
        }

    def update_from_widget_values(self, widget_values: Dict[str, Any]) -> None:
        """Update configuration from widget values."""
        # Update OMERO settings
        if 'container_type' in widget_values:
            self.omero.container_type = widget_values['container_type']
        if 'container_id' in widget_values:
            self.omero.container_id = widget_values['container_id']
        if 'channel' in widget_values:
            self.omero.channel = widget_values['channel']
        if 'source_desc' in widget_values:
            self.omero.source_desc = widget_values['source_desc']

        # Update training settings
        if 'training_set_name' in widget_values:
            self.training.trainingset_name = widget_values['training_set_name']
        if 'segment_all' in widget_values:
            self.training.segment_all = widget_values['segment_all']
        if 'train_n' in widget_values:
            self.training.train_n = widget_values['train_n']
        if 'validate_n' in widget_values:
            self.training.validate_n = widget_values['validate_n']

        # Update micro-SAM settings
        if 'model_type' in widget_values:
            self.microsam.model_type = widget_values['model_type']
        if 'timepoint_mode' in widget_values:
            self.microsam.timepoint_mode = widget_values['timepoint_mode']
        if 'timepoints' in widget_values:
            self.microsam.timepoints = widget_values['timepoints']
        if 'z_slice_mode' in widget_values:
            self.microsam.z_slice_mode = widget_values['z_slice_mode']
        if 'z_slices' in widget_values:
            self.microsam.z_slices = widget_values['z_slices']
        if 'three_d' in widget_values:
            self.microsam.three_d = widget_values['three_d']

        # Update patch settings
        if 'use_patches' in widget_values:
            self.patches.use_patches = widget_values['use_patches']
        if 'patch_size' in widget_values:
            self.patches.patch_size = widget_values['patch_size']
        if 'patches_per_image' in widget_values:
            self.patches.patches_per_image = widget_values['patches_per_image']
        if 'random_patches' in widget_values:
            self.patches.random_patches = widget_values['random_patches']

        # Update batch processing settings
        if 'batch_size' in widget_values:
            self.batch_processing.batch_size = widget_values['batch_size']
        if 'output_folder' in widget_values:
            self.batch_processing.output_folder = widget_values['output_folder']

        # Update workflow settings
        if 'resume_from_table' in widget_values:
            self.workflow.resume_from_table = widget_values['resume_from_table']
        if 'read_only_mode' in widget_values:
            self.workflow.read_only_mode = widget_values['read_only_mode']

    def get_workflow_summary(self) -> str:
        """Get a human-readable summary of the current workflow configuration."""
        summary = []
        
        # OMERO connection info
        summary.append(f"📡 OMERO: {self.omero.container_type} ID {self.omero.container_id}")
        summary.append(f"🎯 Training Set: {self.training.trainingset_name}")
        
        # Annotation scope
        if self.training.segment_all:
            summary.append("📊 Scope: All images")
        else:
            summary.append(f"📊 Scope: {self.training.train_n} training, {self.training.validate_n} validation")
        
        # Channel and dimensions
        summary.append(f"📺 Channel: {self.omero.channel}")
        if self.microsam.three_d:
            summary.append("🧊 3D processing enabled")
        
        # Patches
        if self.patches.use_patches:
            summary.append(f"🧩 Patches: {self.patches.patches_per_image} per image ({self.patches.patch_size[0]}×{self.patches.patch_size[1]})")
        
        # Technical details
        summary.append(f"🔬 Model: {self.microsam.model_type}")
        if self.batch_processing.batch_size > 0:
            summary.append(f"⚙️ Batch size: {self.batch_processing.batch_size}")
        
        # Workflow mode
        if self.workflow.resume_from_table:
            summary.append("🔄 Resuming from existing table")
        else:
            summary.append("✨ Creating new table")
        
        return '\n'.join(summary)


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
    return AnnotationConfig()


def get_config_template() -> str:
    """Get a YAML template with comments for all configuration options."""
    template = """# OMERO micro-SAM Configuration Template

batch_processing:
  batch_size: 0                    # Number of images to process at once (0 = all in one batch)
  output_folder: "./omero_annotations"  # Single output directory for all files

omero:
  container_type: "dataset"        # Type: dataset, plate, project, screen, image
  container_id: 0                  # ID of the OMERO container
  source_desc: ""                  # Description for tracking
  channel: 0                       # Channel index to process

microsam:
  model_type: "vit_b_lm"          # Default model. Available: vit_b, vit_l, vit_h, vit_b_lm
  timepoints: [0]                 # List of timepoint indices
  timepoint_mode: "specific"      # Mode: all, random, specific
  z_slices: [0]                   # List of z-slice indices
  z_slice_mode: "specific"        # Mode: all, random, specific
  three_d: false                  # Process as 3D volumes

patches:
  use_patches: false              # Extract patches instead of full images
  patch_size: [512, 512]          # Width and height of patches
  patches_per_image: 1            # Number of non-overlapping patches per image
  random_patches: true            # Random vs grid-based patch extraction

training:
  segment_all: true               # Process all images or subset
  train_n: 3                      # Number of training images (if not segment_all)
  validate_n: 3                   # Number of validation images (if not segment_all)
  trainingset_name: "default_training_set"  # Required name for the training set

workflow:
  resume_from_table: false        # Resume from existing tracking table
  read_only_mode: false           # Save locally instead of uploading
"""
    return template