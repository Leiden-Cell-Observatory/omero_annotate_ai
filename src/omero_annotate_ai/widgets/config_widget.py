"""Interactive widget for creating and editing micro-SAM configurations in Jupyter notebooks."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import yaml
from typing import Dict, Any, Optional
from ..core.config import AnnotationConfig, create_default_config


class ConfigWidget:
    """Interactive widget for creating AI annotation configurations."""
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        """Initialize the configuration widget."""
        self.config = config or create_default_config()
        self._create_widgets()
        self._setup_observers()
        
    def _create_widgets(self):
        """Create all the widget components."""
        
        # Header
        self.header = widgets.HTML(
            value="<h3>🔬 OMERO micro-SAM Configuration</h3>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Batch Processing Section
        self.batch_section = self._create_batch_widgets()
        
        # OMERO Section
        self.omero_section = self._create_omero_widgets()
        
        # Image Processing Section
        self.image_section = self._create_image_widgets()
        
        # Patches Section
        self.patches_section = self._create_patches_widgets()
        
        # Training Section
        self.training_section = self._create_training_widgets()
        
        # Workflow Section
        self.workflow_section = self._create_workflow_widgets()
        
        # Action buttons
        self.action_buttons = self._create_action_widgets()
        
        # Output display
        self.output_display = widgets.Output()
        
        # Main container with accordion
        self.accordion = widgets.Accordion(children=[
            self.batch_section,
            self.omero_section,
            self.image_section,
            self.patches_section,
            self.training_section,
            self.workflow_section
        ])
        
        self.accordion.set_title(0, '⚙️ Batch Processing')
        self.accordion.set_title(1, '🔌 OMERO Connection')
        self.accordion.set_title(2, '🔬 Micro-SAM')
        self.accordion.set_title(3, '🧩 Patches')
        self.accordion.set_title(4, '🎓 Training')
        self.accordion.set_title(5, '🔄 Workflow')
        
        # Set default open tabs
        self.accordion.selected_index = 1  # Open OMERO section by default
        
        self.main_widget = widgets.VBox([
            self.header,
            self.accordion,
            self.action_buttons,
            self.output_display
        ])
    
    def _create_batch_widgets(self):
        """Create batch processing widgets."""
        batch_size = widgets.IntSlider(
            value=self.config.batch_processing.batch_size,
            min=0, max=10, step=1,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )
        
        batch_info = widgets.HTML(
            value="<small><i>0 = process all images in one batch (default)<br>"
                  "Batch size splits processing into smaller chunks for memory efficiency</i></small>"
        )
        
        output_folder = widgets.Text(
            value=self.config.batch_processing.output_folder,
            description='Output Folder:',
            style={'description_width': 'initial'}
        )
        
        self.batch_widgets = {
            'batch_size': batch_size,
            'output_folder': output_folder
        }
        
        return widgets.VBox([
            widgets.HTML("<b>Batch Processing Settings</b>"),
            batch_size,
            batch_info,
            output_folder
        ])
    
    def _create_omero_widgets(self):
        """Create OMERO connection widgets."""
        container_type = widgets.Dropdown(
            options=['dataset', 'plate', 'project', 'screen', 'image'],
            value=self.config.omero.container_type,
            description='Container Type:',
            style={'description_width': 'initial'}
        )
        
        container_id = widgets.IntText(
            value=self.config.omero.container_id,
            description='Container ID:',
            style={'description_width': 'initial'}
        )
        
        source_desc = widgets.Text(
            value=self.config.omero.source_desc,
            description='Description:',
            placeholder='Optional description for tracking',
            style={'description_width': 'initial'}
        )
        
        channel = widgets.IntText(
            value=self.config.omero.channel,
            description='Channel:',
            style={'description_width': 'initial'}
        )
        
        self.omero_widgets = {
            'container_type': container_type,
            'container_id': container_id,
            'source_desc': source_desc,
            'channel': channel
        }
        
        return widgets.VBox([
            widgets.HTML("<b>OMERO Connection Settings</b>"),
            container_type,
            container_id,
            source_desc,
            channel
        ])
    
    def _create_image_widgets(self):
        """Create micro-SAM model widgets."""
        model_type = widgets.Dropdown(
            options=['vit_b_lm', 'vit_b', 'vit_l', 'vit_h'],  # vit_b_lm first as default
            value=self.config.microsam.model_type,
            description='Model Type:',
            style={'description_width': 'initial'}
        )
        
        model_info = widgets.HTML(
            value="<small><i>vit_b_lm is the default model (recommended)</i></small>"
        )
        
        timepoint_mode = widgets.Dropdown(
            options=['specific', 'all', 'random'],
            value=self.config.microsam.timepoint_mode,
            description='Timepoint Mode:',
            style={'description_width': 'initial'}
        )
        
        timepoints = widgets.Text(
            value=str(self.config.microsam.timepoints)[1:-1],  # Remove brackets
            description='Timepoints:',
            placeholder='0, 1, 2',
            style={'description_width': 'initial'}
        )
        
        z_slice_mode = widgets.Dropdown(
            options=['specific', 'all', 'random'],
            value=self.config.microsam.z_slice_mode,
            description='Z-slice Mode:',
            style={'description_width': 'initial'}
        )
        
        z_slices = widgets.Text(
            value=str(self.config.microsam.z_slices)[1:-1],  # Remove brackets
            description='Z-slices:',
            placeholder='0, 1, 2',
            style={'description_width': 'initial'}
        )
        
        three_d = widgets.Checkbox(
            value=self.config.microsam.three_d,
            description='3D Processing',
            style={'description_width': 'initial'}
        )
        
        self.image_widgets = {
            'model_type': model_type,
            'timepoint_mode': timepoint_mode,
            'timepoints': timepoints,
            'z_slice_mode': z_slice_mode,
            'z_slices': z_slices,
            'three_d': three_d
        }
        
        return widgets.VBox([
            widgets.HTML("<b>Micro-SAM Model Settings</b>"),
            model_type,
            model_info,
            three_d,
            timepoint_mode,
            timepoints,
            z_slice_mode,
            z_slices
        ])
    
    def _create_patches_widgets(self):
        """Create patch extraction widgets."""
        use_patches = widgets.Checkbox(
            value=self.config.patches.use_patches,
            description='Use Patches',
            style={'description_width': 'initial'}
        )
        
        patch_width = widgets.IntText(
            value=self.config.patches.patch_size[0],
            description='Patch Width:',
            style={'description_width': 'initial'}
        )
        
        patch_height = widgets.IntText(
            value=self.config.patches.patch_size[1],
            description='Patch Height:',
            style={'description_width': 'initial'}
        )
        
        patches_per_image = widgets.IntSlider(
            value=self.config.patches.patches_per_image,
            min=1, max=20, step=1,
            description='Patches per Image:',
            style={'description_width': 'initial'}
        )
        
        random_patches = widgets.Checkbox(
            value=self.config.patches.random_patches,
            description='Random Patches',
            style={'description_width': 'initial'}
        )
        
        self.patches_widgets = {
            'use_patches': use_patches,
            'patch_width': patch_width,
            'patch_height': patch_height,
            'patches_per_image': patches_per_image,
            'random_patches': random_patches
        }
        
        # Create container for patch settings that can be disabled
        patch_settings = widgets.VBox([
            patch_width,
            patch_height,
            patches_per_image,
            random_patches
        ])
        
        # Disable patch settings if not using patches
        if not use_patches.value:
            for widget in [patch_width, patch_height, patches_per_image, random_patches]:
                widget.disabled = True
        
        return widgets.VBox([
            widgets.HTML("<b>Patch Extraction Settings</b>"),
            use_patches,
            patch_settings
        ])
    
    def _create_training_widgets(self):
        """Create training configuration widgets."""
        segment_all = widgets.Checkbox(
            value=self.config.training.segment_all,
            description='Segment All Images',
            style={'description_width': 'initial'}
        )
        
        train_n = widgets.IntSlider(
            value=self.config.training.train_n,
            min=1, max=50, step=1,
            description='Training Images:',
            style={'description_width': 'initial'}
        )
        
        validate_n = widgets.IntSlider(
            value=self.config.training.validate_n,
            min=1, max=50, step=1,
            description='Validation Images:',
            style={'description_width': 'initial'}
        )
        
        trainingset_name = widgets.Text(
            value=self.config.training.trainingset_name,
            description='Training Set Name:',
            placeholder='Enter training set name (required)',
            style={'description_width': 'initial'}
        )
        
        self.training_widgets = {
            'segment_all': segment_all,
            'train_n': train_n,
            'validate_n': validate_n,
            'trainingset_name': trainingset_name
        }
        
        # Disable subset settings if segment_all is True
        subset_settings = widgets.VBox([train_n, validate_n])
        if segment_all.value:
            train_n.disabled = True
            validate_n.disabled = True
        
        return widgets.VBox([
            widgets.HTML("<b>Training Configuration</b>"),
            segment_all,
            subset_settings,
            trainingset_name
        ])
    
    def _create_workflow_widgets(self):
        """Create workflow configuration widgets."""
        resume_from_table = widgets.Checkbox(
            value=self.config.workflow.resume_from_table,
            description='Resume from Table',
            style={'description_width': 'initial'}
        )
        
        read_only_mode = widgets.Checkbox(
            value=self.config.workflow.read_only_mode,
            description='Read-only Mode',
            style={'description_width': 'initial'}
        )
        
        local_output_dir = widgets.Text(
            value=self.config.workflow.local_output_dir,
            description='Local Output Dir:',
            style={'description_width': 'initial'}
        )
        
        self.workflow_widgets = {
            'resume_from_table': resume_from_table,
            'read_only_mode': read_only_mode,
            'local_output_dir': local_output_dir
        }
        
        return widgets.VBox([
            widgets.HTML("<b>Workflow Settings</b>"),
            resume_from_table,
            read_only_mode,
            local_output_dir
        ])
    
    def _create_action_widgets(self):
        """Create action buttons."""
        update_button = widgets.Button(
            description='Update Configuration',
            button_style='primary',
            icon='refresh'
        )
        
        show_yaml_button = widgets.Button(
            description='Show YAML',
            button_style='info',
            icon='eye'
        )
        
        validate_button = widgets.Button(
            description='Validate',
            button_style='success',
            icon='check'
        )
        
        reset_button = widgets.Button(
            description='Reset to Defaults',
            button_style='warning',
            icon='undo'
        )
        
        update_button.on_click(self._update_config)
        show_yaml_button.on_click(self._show_yaml)
        validate_button.on_click(self._validate_config)
        reset_button.on_click(self._reset_config)
        
        return widgets.HBox([
            update_button,
            show_yaml_button,
            validate_button,
            reset_button
        ])
    
    def _setup_observers(self):
        """Setup observers for dynamic widget behavior."""
        # Enable/disable patch settings based on use_patches
        self.patches_widgets['use_patches'].observe(self._toggle_patch_settings, names='value')
        
        # Enable/disable training subset settings based on segment_all
        self.training_widgets['segment_all'].observe(self._toggle_training_settings, names='value')
    
    def _toggle_patch_settings(self, change):
        """Toggle patch settings widgets based on use_patches checkbox."""
        enabled = change['new']
        for key in ['patch_width', 'patch_height', 'patches_per_image', 'random_patches']:
            self.patches_widgets[key].disabled = not enabled
    
    def _toggle_training_settings(self, change):
        """Toggle training subset settings based on segment_all checkbox."""
        segment_all = change['new']
        self.training_widgets['train_n'].disabled = segment_all
        self.training_widgets['validate_n'].disabled = segment_all
    
    def _parse_int_list(self, text_value: str) -> list:
        """Parse comma-separated integers from text widget."""
        try:
            return [int(x.strip()) for x in text_value.split(',') if x.strip()]
        except ValueError:
            return [0]  # Default fallback
    
    def _update_config(self, button):
        """Update configuration from widget values."""
        try:
            with self.output_display:
                clear_output()
                
                # Update batch processing
                self.config.batch_processing.batch_size = self.batch_widgets['batch_size'].value
                self.config.batch_processing.output_folder = self.batch_widgets['output_folder'].value
                
                # Update OMERO
                self.config.omero.container_type = self.omero_widgets['container_type'].value
                self.config.omero.container_id = self.omero_widgets['container_id'].value
                self.config.omero.source_desc = self.omero_widgets['source_desc'].value
                self.config.omero.channel = self.omero_widgets['channel'].value
                
                # Update micro-SAM model
                self.config.microsam.model_type = self.image_widgets['model_type'].value
                self.config.microsam.timepoint_mode = self.image_widgets['timepoint_mode'].value
                self.config.microsam.timepoints = self._parse_int_list(self.image_widgets['timepoints'].value)
                self.config.microsam.z_slice_mode = self.image_widgets['z_slice_mode'].value
                self.config.microsam.z_slices = self._parse_int_list(self.image_widgets['z_slices'].value)
                self.config.microsam.three_d = self.image_widgets['three_d'].value
                
                # Update patches
                self.config.patches.use_patches = self.patches_widgets['use_patches'].value
                self.config.patches.patch_size = (
                    self.patches_widgets['patch_width'].value,
                    self.patches_widgets['patch_height'].value
                )
                self.config.patches.patches_per_image = self.patches_widgets['patches_per_image'].value
                self.config.patches.random_patches = self.patches_widgets['random_patches'].value
                
                # Update training
                self.config.training.segment_all = self.training_widgets['segment_all'].value
                self.config.training.train_n = self.training_widgets['train_n'].value
                self.config.training.validate_n = self.training_widgets['validate_n'].value
                # Note: group_by_image parameter removed as it was not useful
                trainingset_name = self.training_widgets['trainingset_name'].value
                self.config.training.trainingset_name = trainingset_name if trainingset_name else None
                
                # Update workflow
                self.config.workflow.resume_from_table = self.workflow_widgets['resume_from_table'].value
                self.config.workflow.read_only_mode = self.workflow_widgets['read_only_mode'].value
                self.config.workflow.local_output_dir = self.workflow_widgets['local_output_dir'].value
                
                print("✅ Configuration updated successfully!")
                
        except Exception as e:
            with self.output_display:
                clear_output()
                print(f"❌ Error updating configuration: {e}")
    
    def _show_yaml(self, button):
        """Display the current configuration as YAML."""
        with self.output_display:
            clear_output()
            print("📄 Current Configuration (YAML):")
            print("=" * 50)
            print(self.config.to_yaml())
    
    def _validate_config(self, button):
        """Validate the current configuration."""
        with self.output_display:
            clear_output()
            try:
                self.config.validate()
                print("✅ Configuration is valid!")
            except ValueError as e:
                print(f"❌ Configuration validation failed:\n{e}")
    
    def _reset_config(self, button):
        """Reset configuration to defaults."""
        with self.output_display:
            clear_output()
            self.config = create_default_config()
            self._update_widget_values()
            print("🔄 Configuration reset to defaults!")
    
    def _update_widget_values(self):
        """Update widget values from current configuration."""
        # Update batch processing widgets
        self.batch_widgets['batch_size'].value = self.config.batch_processing.batch_size
        self.batch_widgets['output_folder'].value = self.config.batch_processing.output_folder
        
        # Update OMERO widgets
        self.omero_widgets['container_type'].value = self.config.omero.container_type
        self.omero_widgets['container_id'].value = self.config.omero.container_id
        self.omero_widgets['source_desc'].value = self.config.omero.source_desc
        self.omero_widgets['channel'].value = self.config.omero.channel
        
        # Update micro-SAM model widgets
        self.image_widgets['model_type'].value = self.config.microsam.model_type
        self.image_widgets['timepoint_mode'].value = self.config.microsam.timepoint_mode
        self.image_widgets['timepoints'].value = str(self.config.microsam.timepoints)[1:-1]
        self.image_widgets['z_slice_mode'].value = self.config.microsam.z_slice_mode
        self.image_widgets['z_slices'].value = str(self.config.microsam.z_slices)[1:-1]
        self.image_widgets['three_d'].value = self.config.microsam.three_d
        
        # Update patches widgets
        self.patches_widgets['use_patches'].value = self.config.patches.use_patches
        self.patches_widgets['patch_width'].value = self.config.patches.patch_size[0]
        self.patches_widgets['patch_height'].value = self.config.patches.patch_size[1]
        self.patches_widgets['patches_per_image'].value = self.config.patches.patches_per_image
        self.patches_widgets['random_patches'].value = self.config.patches.random_patches
        
        # Update training widgets
        self.training_widgets['segment_all'].value = self.config.training.segment_all
        self.training_widgets['train_n'].value = self.config.training.train_n
        self.training_widgets['validate_n'].value = self.config.training.validate_n
        # Note: group_by_image widget removed as parameter was not useful
        self.training_widgets['trainingset_name'].value = self.config.training.trainingset_name or ''
        
        # Update workflow widgets
        self.workflow_widgets['resume_from_table'].value = self.config.workflow.resume_from_table
        self.workflow_widgets['read_only_mode'].value = self.config.workflow.read_only_mode
        self.workflow_widgets['local_output_dir'].value = self.config.workflow.local_output_dir
    
    def display(self):
        """Display the widget."""
        display(self.main_widget)
    
    def get_config(self) -> AnnotationConfig:
        """Get the current configuration."""
        return self.config
    
    def get_yaml(self) -> str:
        """Get the current configuration as YAML."""
        return self.config.to_yaml()


def create_config_widget(config: Optional[AnnotationConfig] = None) -> ConfigWidget:
    """Create and return a configuration widget."""
    return ConfigWidget(config)