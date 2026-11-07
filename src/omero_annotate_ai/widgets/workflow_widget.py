"""Sequential workflow widget for OMERO annotation setup."""

import html
import tempfile
from pathlib import Path

import ipywidgets as widgets
from IPython.display import clear_output, display

from ..core.annotation_config import AnnotationConfig, create_default_config
from ..omero.omero_utils import get_container_info
from ..omero.omero_functions import (
    find_annotation_config_for_table,
    generate_unique_table_name,
    infer_settings_from_table,
    list_annotation_tables,
    read_tracking_table,
)

# Settings that must match the rows already in a tracking table: changing them
# on resume makes validate_annotations_against_config fail (channel, 3D, patch)
# or gives new rows a different shape than the old ones (patch size, z-range).
RESUME_LOCKED_KEYS = ("channel", "three_d", "z_range_start", "z_range_end",
                      "use_patches", "patch_size")


class WorkflowWidget:
    """Sequential workflow widget for OMERO annotation setup."""

    def __init__(self, connection=None):
        """Initialize the workflow widget."""
        self.connection = connection
        self.config = create_default_config()
        self.current_step = 0  # Always start at first tab
        self.steps = [
            "Select Working Directory",
            "Choose Container",
            "Check Existing Tables",
            "Configure Parameters",
            "Save Configuration",
        ]
        self.annotation_tables = []
        self.selected_table_id = None
        self._locked_settings = {}
        self.working_directory = None
        self._create_widgets()

    def _create_widgets(self):
        """Create all widget components."""

        # Header with progress
        self.header = widgets.HTML(
            value="<h3>🔬 OMERO Annotation Workflow</h3>",
            layout=widgets.Layout(margin="0 0 20px 0"),
        )

        # Progress indicator
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=len(self.steps) - 1,
            description="Progress:",
            bar_style="info",
            layout=widgets.Layout(width="100%"),
        )

        # Current step display
        self.step_display = widgets.HTML(
            value=f"<b>Step {self.current_step + 1}/{len(self.steps)}: {self.steps[self.current_step]}</b>",
            layout=widgets.Layout(margin="10px 0"),
        )

        # Tab widget for steps
        self.tab_widget = widgets.Tab()
        if self.connection is None:
            # Include connection tab if no connection provided
            self.steps.insert(0, "Connect to OMERO")
            self.tab_widget.children = [
                self._create_connection_tab(),
                self._create_directory_tab(),
                self._create_container_tab(),
                self._create_tables_tab(),
                self._create_config_tab(),
                self._create_save_tab(),
            ]
        else:
            # Skip connection tab if connection already provided
            self.tab_widget.children = [
                self._create_directory_tab(),
                self._create_container_tab(),
                self._create_tables_tab(),
                self._create_config_tab(),
                self._create_save_tab(),
            ]

        # Set tab titles
        for i, step in enumerate(self.steps):
            self.tab_widget.set_title(i, f"{i + 1}. {step}")

        # Navigation buttons
        self.prev_button = widgets.Button(
            description="Previous",
            button_style="",
            icon="arrow-left",
            disabled=True,
            layout=widgets.Layout(width="100px"),
        )

        self.next_button = widgets.Button(
            description="Next",
            button_style="primary",
            icon="arrow-right",
            layout=widgets.Layout(width="100px"),
        )

        # Status output
        self.status_output = widgets.Output()

        # Main container
        self.main_widget = widgets.VBox(
            [
                self.header,
                self.progress,
                self.step_display,
                self.tab_widget,
                widgets.HBox(
                    [self.prev_button, self.next_button],
                    layout=widgets.Layout(
                        justify_content="space-between", margin="10px 0"
                    ),
                ),
                self.status_output,
            ]
        )

        # Setup observers
        self.prev_button.on_click(self._on_prev_step)
        self.next_button.on_click(self._on_next_step)
        self.tab_widget.observe(self._on_tab_change, names="selected_index")

        # Enable container widgets if connection already provided
        if self.connection is not None:
            self._enable_container_widgets()

    def _create_connection_tab(self):
        """Create connection step tab."""
        connection_status = widgets.HTML(
            value="No OMERO connection", layout=widgets.Layout(margin="10px 0")
        )

        connect_info = widgets.HTML(
            value="<p>Use the OMERO connection widget to establish a connection:</p>"
            "<pre>conn_widget = create_omero_connection_widget()\n"
            "conn_widget.display()\n"
            "# After connecting:\n"
            "workflow.set_connection(conn_widget.get_connection())</pre>",
            layout=widgets.Layout(margin="10px 0"),
        )

        self.connection_widgets = {"status": connection_status, "info": connect_info}

        return widgets.VBox(
            [
                widgets.HTML("<h4>Connect to OMERO</h4>"),
                connection_status,
                connect_info,
            ]
        )

    def _create_directory_tab(self):
        """Create directory selection tab."""
        directory_info = widgets.HTML(
            value="<p>Select a local working directory for your annotation project:</p>"
        )

        # Directory selection
        self.directory_text = widgets.Text(
            value=str(Path.home() / "omero_annotate_ai" / "omero_annotations"),
            description="Working Dir:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )
        # Update Next button state when user edits the path
        self.directory_text.observe(self._on_directory_text_change, names="value")

        # Quick options
        temp_button = widgets.Button(
            description="Use Temp Directory",
            button_style="info",
            layout=widgets.Layout(width="150px"),
        )

        create_button = widgets.Button(
            description="Create Directory",
            button_style="success",
            layout=widgets.Layout(width="150px"),
        )

        # Directory status
        self.directory_status = widgets.HTML(
            value="", layout=widgets.Layout(margin="10px 0")
        )

        temp_button.on_click(self._on_temp_directory)
        create_button.on_click(self._on_create_directory)

        self.directory_widgets = {
            "text": self.directory_text,
            "status": self.directory_status,
            "temp_button": temp_button,
            "create_button": create_button,
        }

        return widgets.VBox(
            [
                widgets.HTML("<h4>Select Working Directory</h4>"),
                directory_info,
                self.directory_text,
                widgets.HBox([temp_button, create_button]),
                self.directory_status,
            ]
        )

    def _create_container_tab(self):
        """Create container selection tab."""
        container_info = widgets.HTML(
            value="<p>Choose the OMERO container (project, dataset, plate, or screen) for annotation:</p>"
        )

        # Container type selection
        self.container_type_dropdown = widgets.Dropdown(
            options=[
                ("Project", "project"),
                ("Dataset", "dataset"),
                ("Plate", "plate"),
                ("Screen", "screen"),
            ],
            value="project",
            description="Container Type:",
            style={"description_width": "initial"},
            disabled=True,
        )

        # Container selection
        self.container_dropdown = widgets.Dropdown(
            options=[("Connect to OMERO first", None)],
            value=None,
            description="Container:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
            disabled=True,
        )
        # Update Next button state when user selects a container
        self.container_dropdown.observe(
            self._on_container_selection_change, names="value"
        )

        # Refresh button
        refresh_button = widgets.Button(
            description="Refresh",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(width="100px"),
            disabled=True,
        )

        self.container_widgets = {
            "type": self.container_type_dropdown,
            "container": self.container_dropdown,
            "refresh": refresh_button,
        }

        # Container summary (filled when user proceeds to next step)
        self.container_summary = widgets.HTML(
            value="",
            layout=widgets.Layout(margin="10px 0")
        )

        self.container_type_dropdown.observe(
            self._on_container_type_change, names="value"
        )
        refresh_button.on_click(self._on_refresh_containers)

        return widgets.VBox(
            [
                widgets.HTML("<h4>Choose Container</h4>"),
                container_info,
                self.container_type_dropdown,
                widgets.HBox([self.container_dropdown, refresh_button]),
                self.container_summary,
            ]
        )

    def _create_tables_tab(self):
        """Create tables checking tab."""
        tables_info = widgets.HTML(
            value="<p>Check for existing annotation tables and decide how to proceed:</p>"
        )

        # Scan button
        scan_button = widgets.Button(
            description="Scan for Tables",
            button_style="primary",
            icon="search",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )

        # Tables display
        self.tables_display = widgets.Output()

        # Table selection
        self.table_selection = widgets.RadioButtons(
            options=[],
            description="Select:",
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="auto", max_width="100%"),
        )

        # Action buttons
        continue_button = widgets.Button(
            description="Continue Existing",
            button_style="success",
            icon="play",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )

        new_button = widgets.Button(
            description="Name New Table",
            tooltip="Pick a name for a new table. The table is created when the pipeline runs.",
            button_style="warning",
            icon="plus",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )

        # New table name
        self.new_table_name = widgets.Text(
            description="New Table Name:",
            placeholder="Leave empty for auto-generation",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        self.tables_widgets = {
            "scan": scan_button,
            "selection": self.table_selection,
            "continue": continue_button,
            "new": new_button,
            "name": self.new_table_name,
            "display": self.tables_display,
        }

        scan_button.on_click(self._on_scan_tables)
        continue_button.on_click(self._on_continue_table)
        new_button.on_click(self._on_new_table)

        return widgets.VBox(
            [
                widgets.HTML("<h4>Check Existing Tables</h4>"),
                tables_info,
                scan_button,
                self.tables_display,
                self.table_selection,
                widgets.HBox([continue_button, new_button]),
                self.new_table_name,
            ]
        )

    def _create_config_tab(self):
        """Create configuration tab with user-focused groups."""
        config_info = widgets.HTML(value="<p>Configure your annotation parameters:</p>")

        # OMERO Settings Group (display only)
        self.omero_status = widgets.HTML(
            value="<h5>📡 OMERO Settings</h5><p>Select container in previous steps first</p>"
        )

        # Create tabbed interface for configuration groups
        config_tabs = widgets.Tab()

        # Annotation Settings Tab
        self.annotation_widgets = self._create_annotation_settings()

        # Technical Settings Tab
        self.technical_widgets = self._create_technical_settings()

        # Workflow Settings Tab
        self.workflow_settings = self._create_workflow_settings()

        config_tabs.children = [
            self.annotation_widgets,
            self.technical_widgets,
            self.workflow_settings,
        ]

        config_tabs.set_title(0, "Annotation Settings")
        config_tabs.set_title(1, "Technical Settings")
        config_tabs.set_title(2, "Workflow Settings")

        # Update configuration button
        update_config_button = widgets.Button(
            description="Update Configuration",
            button_style="primary",
            icon="refresh",
            layout=widgets.Layout(width="200px", margin="10px 0"),
        )

        update_config_button.on_click(self._on_update_config)

        self.config_widgets = {
            "omero_status": self.omero_status,
            "annotation": self.annotation_widgets,
            "technical": self.technical_widgets,
            "workflow": self.workflow_settings,
            "update_button": update_config_button,
        }

        return widgets.VBox(
            [
                widgets.HTML("<h4>Configure Parameters</h4>"),
                config_info,
                self.omero_status,
                config_tabs,
                update_config_button,
            ]
        )

    def _create_annotation_settings(self):
        """Create annotation settings group."""
        # Add informative header
        training_help = widgets.HTML(
            value="""
            <div style='background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <b>💡 Data Split Guide:</b>
                <ul style='margin: 5px 0; padding-left: 20px;'>
                    <li><b>Training:</b> Images to annotate for model training (min: 1)</li>
                    <li><b>Validation:</b> Images to annotate for model validation during training (0 to skip)</li>
                    <li><b>Test:</b> Images to annotate for final model evaluation (0 to skip)</li>
                </ul>
                <small>For Cellpose: Use training+test. For BiaPy: Use all three splits.</small>
            </div>
            """
        )

        # Number of images
        segment_all = widgets.Checkbox(
            value=self.config.training.segment_all, description="Annotate all images"
        )

        # Expose sliders as instance attributes so we can adjust .max later
        self.train_n_slider = widgets.IntSlider(
            value=self.config.training.train_n,
            min=1,
            max=50,
            description="Training images:",
            style={'description_width': '150px'},
            disabled=segment_all.value,
            tooltip="Images to annotate for training the model (required, min: 1)"
        )

        self.validate_n_slider = widgets.IntSlider(
            value=self.config.training.validate_n,
            min=0,  # Changed from 1
            max=50,
            description="Validation images:",
            style={'description_width': '150px'},
            disabled=segment_all.value,
            tooltip="Images to annotate for model validation during training (0 to skip)"
        )

        self.test_n_slider = widgets.IntSlider(
            value=self.config.training.test_n,
            min=0,
            max=50,
            description="Test images:",
            style={'description_width': '150px'},
            disabled=segment_all.value,
            tooltip="Images to annotate for final model evaluation (0 to skip)"
        )

        # Channel selection
        channel = widgets.IntText(
            value=self.config.spatial_coverage.channels[0],
            description="Channel:",
            style={"description_width": "initial"},
        )

        # Time series settings
        timepoint_mode = widgets.Dropdown(
            options=[
                ("Specific timepoints", "specific"),
                ("All timepoints", "all"),
                ("Random selection", "random"),
            ],
            value=self.config.spatial_coverage.timepoint_mode,
            description="Timepoints:",
        )

        timepoints = widgets.Text(
            value=", ".join(map(str, self.config.spatial_coverage.timepoints)),
            description="Timepoint list:",
            placeholder="0, 1, 2",
            disabled=timepoint_mode.value != "specific",
        )

        # Z-stack settings
        z_slice_mode = widgets.Dropdown(
            options=[
                ("Specific slices", "specific"),
                ("All slices", "all"),
                ("Random selection", "random"),
            ],
            value=self.config.spatial_coverage.z_slice_mode,
            description="Z-slices:",
        )

        z_slices = widgets.Text(
            value=", ".join(map(str, self.config.spatial_coverage.z_slices)),
            description="Z-slice list:",
            placeholder="0, 1, 2",
            disabled=z_slice_mode.value != "specific",
        )

        three_d = widgets.Checkbox(
            value=self.config.spatial_coverage.three_d, description="3D processing"
        )

        # 3D volumetric processing settings
        z_range_start = widgets.IntText(
            value=self.config.spatial_coverage.z_range_start or 0,
            description="Z-range start:",
            disabled=not three_d.value,
            style={"description_width": "initial"},
        )

        z_range_end = widgets.IntText(
            value=self.config.spatial_coverage.z_range_end or 0,
            description="Z-range end:",
            disabled=not three_d.value,
            style={"description_width": "initial"},
        )

        # Info text for 3D modes
        three_d_info = widgets.HTML(
            value="<small><i>3D processing: specify z-range for volumetric processing</i></small>"
        )

        # Patches settings
        use_patches = widgets.Checkbox(
            value=self.config.spatial_coverage.use_patches, description="Use patches"
        )

        patches_per_image = widgets.IntSlider(
            value=self.config.spatial_coverage.patches_per_image,
            min=1,
            max=20,
            description="Patches per image:",
            disabled=not use_patches.value,
        )

        patch_size = widgets.Text(
            value=f"{self.config.spatial_coverage.patch_size[0]}, {self.config.spatial_coverage.patch_size[1]}",
            description="Patch size:",
            placeholder="512, 512",
            disabled=not use_patches.value,
        )

        # Setup observers
        segment_all.observe(
            lambda c: self._toggle_subset_settings(c, self.train_n_slider, self.validate_n_slider, self.test_n_slider),
            names="value",
        )
        timepoint_mode.observe(
            lambda c: self._toggle_list_setting(c, timepoints), names="value"
        )
        z_slice_mode.observe(
            lambda c: self._toggle_list_setting(c, z_slices), names="value"
        )
        three_d.observe(
            lambda c: self._toggle_3d_settings(c, z_range_start, z_range_end),
            names="value",
        )
        use_patches.observe(
            lambda c: self._toggle_patch_settings(c, patches_per_image, patch_size),
            names="value",
        )

        self.resume_note = widgets.HTML(value="")

        return widgets.VBox(
            [
                widgets.HTML("<h5>🎯 Annotation Settings</h5>"),
                self.resume_note,
                training_help,  # New: Add help text
                segment_all,
                self.train_n_slider,
                self.validate_n_slider,
                self.test_n_slider,  # New: Add test slider
                channel,
                timepoint_mode,
                timepoints,
                z_slice_mode,
                z_slices,
                three_d,
                z_range_start,
                z_range_end,
                three_d_info,
                use_patches,
                patches_per_image,
                patch_size,
            ]
        )

    def _create_technical_settings(self):
        """Create technical settings group."""
        # Batch processing
        batch_size = widgets.IntSlider(
            value=self.config.workflow.batch_size,
            min=0,
            max=10,
            description="Batch size:",
            style={"description_width": "initial"},
        )

        batch_info = widgets.HTML(
            value="<small><i>0 = process all images together</i></small>"
        )

        # SAM model selection
        model_type = widgets.Dropdown(
            options=[
                ("vit_b_lm (recommended)", "vit_b_lm"),
                ("vit_b", "vit_b"),
                ("vit_l", "vit_l"),
                ("vit_h", "vit_h"),
            ],
            value=self.config.ai_model.pretrained_from,
            description="SAM Model:",
            style={"description_width": "initial"},
        )

        return widgets.VBox(
            [
                widgets.HTML("<h5>⚙️ Technical Settings</h5>"),
                batch_size,
                batch_info,
                model_type,
            ]
        )

    def _create_workflow_settings(self):
        """Create workflow settings group."""
        # Read-only mode
        read_only_mode = widgets.Checkbox(
            value=self.config.workflow.read_only_mode,
            description="Read-only mode (save locally only)",
            style={"description_width": "initial"},
        )

        # Display only: set by Continue Existing / Name New Table in step 3
        resume_from_table = widgets.Checkbox(
            value=self.config.workflow.resume_from_table,
            description="Resume from existing table",
            style={"description_width": "initial"},
            disabled=True,
        )

        self.workflow_widgets = {
            "read_only_mode": read_only_mode,
            "resume_from_table": resume_from_table,
        }

        return widgets.VBox(
            [
                widgets.HTML("<h5>🔄 Workflow Settings</h5>"),
                resume_from_table,
                read_only_mode,
            ]
        )

    def _create_save_tab(self):
        """Create save configuration tab."""
        save_info = widgets.HTML(value="<p>Review and save your configuration:</p>")

        # Configuration preview
        self.config_preview = widgets.Textarea(
            value="",
            description="Configuration:",
            layout=widgets.Layout(width="100%", height="300px"),
            disabled=True,
        )

        # Save button
        save_button = widgets.Button(
            description="Save Configuration",
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="200px"),
        )

        # Save status
        self.save_status = widgets.HTML(
            value="", layout=widgets.Layout(margin="10px 0")
        )

        save_button.on_click(self._on_save_config)

        self.save_widgets = {
            "preview": self.config_preview,
            "save": save_button,
            "status": self.save_status,
        }

        return widgets.VBox(
            [
                widgets.HTML("<h4>Save Configuration</h4>"),
                save_info,
                self.config_preview,
                save_button,
                self.save_status,
            ]
        )

    def _update_progress(self):
        """Update progress indicators."""
        self.progress.value = self.current_step
        self.step_display.value = f"<b>Step {self.current_step + 1}/{len(self.steps)}: {self.steps[self.current_step]}</b>"

        # Update button states
        self.prev_button.disabled = self.current_step == 0
        self.next_button.disabled = not self._can_proceed_to_next()

        # Update tab selection
        self.tab_widget.selected_index = self.current_step

    def _can_proceed_to_next(self):
        """Check if user can proceed to next step."""
        if self.connection is None:
            # With connection tab
            if self.current_step == 0:  # Connection step
                return self.connection is not None
            elif self.current_step == 1:  # Directory step
                # Allow proceeding if a directory path is entered; we'll create/set it on Next
                try:
                    return bool(self.directory_text.value.strip())
                except Exception:
                    return False
            elif self.current_step == 2:  # Container step
                # Enable Next if a container is selected in the dropdown
                try:
                    return self.container_widgets["container"].value is not None
                except Exception:
                    return False
            elif self.current_step == 3:  # Tables step
                return (
                    self.selected_table_id is not None
                    or self.new_table_name.value.strip()
                )
            elif self.current_step == 4:  # Config step
                return True  # Always can proceed from config
        else:
            # Without connection tab (connection already provided)
            if self.current_step == 0:  # Directory step
                # Allow proceeding if a directory path is entered; we'll create/set it on Next
                try:
                    return bool(self.directory_text.value.strip())
                except Exception:
                    return False
            elif self.current_step == 1:  # Container step
                # Enable Next if a container is selected in the dropdown
                try:
                    return self.container_widgets["container"].value is not None
                except Exception:
                    return False
            elif self.current_step == 2:  # Tables step
                return (
                    self.selected_table_id is not None
                    or self.new_table_name.value.strip()
                )
            elif self.current_step == 3:  # Config step
                return True  # Always can proceed from config

        return False

    def _on_prev_step(self, button):
        """Handle previous step button."""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_progress()

    def _on_next_step(self, button):
        """Handle next step button."""
        if self.current_step < len(self.steps) - 1 and self._can_proceed_to_next():
            # If we are leaving the directory step, set/create the working directory now
            try:
                if self.steps[self.current_step] == "Select Working Directory":
                    dir_path = Path(self.directory_text.value).expanduser()
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.working_directory = str(dir_path)
                    self.directory_status.value = f"✅ Directory ready: {self.working_directory}"
            except Exception as e:
                # If directory cannot be created, show error and do not advance
                self.directory_status.value = f"❌ Error creating directory: {e}"
                return

            # If we are leaving the container selection step, fetch and apply container info once
            try:
                if self.steps[self.current_step] == "Choose Container":
                    container_id = self.container_widgets["container"].value
                    container_type = self.container_widgets["type"].value
                    if container_id and container_type and self.connection is not None:
                        info = get_container_info(self.connection, container_type, container_id)
                        if info:
                            self._apply_container_info(info)
            except Exception:
                # Silent fail; do not block navigation on summary issues
                pass

            self.current_step += 1
            self._update_progress()

            # Update config preview when reaching save step
            if self.current_step == len(self.steps) - 1:  # Save step
                self._on_update_config(button)
                self._update_config_preview()

    def _on_tab_change(self, change):
        """Handle tab change."""
        # Allow manual tab navigation but update current step
        new_step = change["new"]
        if new_step != self.current_step:
            self.current_step = new_step
            self._update_progress()

            # Update config preview when reaching save step
            if self.current_step == len(self.steps) - 1:  # Save step
                self._on_update_config(None)
                self._update_config_preview()

    def _on_temp_directory(self, button):
        """Handle temp directory button."""
        temp_dir = tempfile.mkdtemp(prefix="omero_annotations_")
        self.directory_text.value = temp_dir
        self.working_directory = temp_dir
        self.directory_status.value = f"✅ Temporary directory created: {temp_dir}"
        self._update_progress()

    def _on_create_directory(self, button):
        """Handle create directory button."""
        try:
            dir_path = Path(self.directory_text.value)
            dir_path.mkdir(parents=True, exist_ok=True)
            self.working_directory = str(dir_path)
            self.directory_status.value = (
                f"✅ Directory ready: {self.working_directory}"
            )

            # Update config output folder
            self.config.output.output_directory = Path(self.working_directory)
            self._update_progress()
        except Exception as e:
            self.directory_status.value = f"❌ Error creating directory: {e}"

    def _on_container_type_change(self, change):
        """Handle container type change."""
        if self.connection:
            self._load_containers()

    def _on_refresh_containers(self, button):
        """Handle refresh containers."""
        self._load_containers()

    def _on_directory_text_change(self, change):
        """Re-evaluate progress when directory text is edited."""
        # Do not create directory here; just update button states
        self._update_progress()

    def _on_container_selection_change(self, change):
        """Re-evaluate progress when a container is selected."""
        self._update_progress()

    def _load_containers(self):
        """Load containers from OMERO."""
        if not self.connection:
            return

        try:
            container_type = self.container_widgets["type"].value

            if container_type == "project":
                containers = list(self.connection.getObjects("Project"))
            elif container_type == "dataset":
                containers = list(self.connection.getObjects("Dataset"))
            elif container_type == "plate":
                containers = list(self.connection.getObjects("Plate"))
            elif container_type == "screen":
                containers = list(self.connection.getObjects("Screen"))
            else:
                containers = []

            options = [("Select container...", None)]
            for container in containers:
                options.append(
                    (
                        f"{container.getName()} (ID: {container.getId()})",
                        container.getId(),
                    )
                )

            self.container_widgets["container"].options = options

        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"❌ Error loading containers: {e}")

    def _on_scan_tables(self, button):
        """Handle scan tables button."""
        try:
            container_id = self.container_widgets["container"].value
            container_type = self.container_widgets["type"].value

            if not container_id:
                return

            self.annotation_tables = list_annotation_tables(
                self.connection, container_type, container_id
            )

            with self.tables_widgets["display"]:
                clear_output()
                if self.annotation_tables:
                    print(f"Found {len(self.annotation_tables)} annotation tables:")
                    for i, table in enumerate(self.annotation_tables):
                        print(
                            f"{i + 1}. {table.get('name', 'Unknown')} (ID: {table.get('id')})"
                        )

                    # Update selection options
                    options = []
                    for table in self.annotation_tables:
                        # Labels must be unique: RadioButtons matches options by label,
                        # so two tables with the same name would both show as selected.
                        options.append((f"{table.get('name', 'Unknown')} (ID: {table.get('id')})", table.get("id")))

                    self.tables_widgets["selection"].options = options
                    self.tables_widgets["selection"].disabled = False
                    self.tables_widgets["continue"].disabled = False
                else:
                    print("No annotation tables found.")

                self.tables_widgets["new"].disabled = False

        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"❌ Error scanning tables: {e}")

    def _on_continue_table(self, button):
        """Handle continue existing table."""
        self.selected_table_id = self.tables_widgets["selection"].value
        if self.selected_table_id:
            # Find table name
            for table in self.annotation_tables:
                if table.get("id") == self.selected_table_id:
                    self.config.name = table.get("name", "unknown")
                    break

            self._set_resume_from_table(True)
            self._load_resume_settings()
            self._update_progress()

    def _set_resume_from_table(self, resume: bool):
        """Resume is decided by the table choice; keep config and checkbox in step.

        The selected table's ID goes into the config, so the pipeline resumes
        that exact table even when several tables share its name.
        """
        self.config.workflow.resume_from_table = resume
        self.config.omero.table_id = self.selected_table_id if resume else None
        if not resume:
            self._unlock_resume_fields()
        if hasattr(self, "workflow_widgets"):
            self.workflow_widgets["resume_from_table"].value = resume

    def _load_resume_settings(self):
        """Fill the Configure tab from the selected table and lock what must match.

        Prefers the config YAML the earlier run attached to the container; falls
        back to what the table rows show when no saved config matches.
        """
        container_type = self.container_widgets["type"].value
        container_id = self.container_widgets["container"].value
        try:
            saved = find_annotation_config_for_table(
                self.connection, container_type, container_id, self.config.name,
                table_id=self.selected_table_id,
            )
        except Exception:
            saved = None
        try:
            rows = infer_settings_from_table(
                read_tracking_table(self.connection, self.selected_table_id)
            )
        except Exception as e:
            rows, row_error = {}, e
        else:
            row_error = None
        if saved is None and not rows:
            self.resume_note.value = (
                f"<small>⚠️ Could not load settings from the table: {row_error}</small>"
            )
            return

        settings, overridden = self._merge_resume_settings(
            self._settings_from_config(saved) if saved is not None else {}, rows
        )
        source = "the saved config" if saved is not None else "the table rows"
        if overridden:
            source += (" (it did not match the table, so the table rows were used for: "
                       + ", ".join(overridden) + ")")

        self._apply_settings_to_widgets(settings)
        self._lock_resume_fields(settings)
        # Write the loaded settings into the config now, so a run started
        # without clicking Update Configuration still matches the table.
        self._update_annotation_settings()
        self._apply_locked_settings_to_config()
        self._update_technical_settings()
        self.resume_note.value = (
            f"<small><i>Resuming: settings loaded from {source}. Channel, 3D and patch "
            "settings are locked to match the existing rows. Raise the image counts "
            "to add new images to this table.</i></small>"
        )

    @staticmethod
    def _merge_resume_settings(saved: dict, rows: dict):
        """The table rows decide the locked settings and the counts.

        A saved config may belong to another run of the same name, so it only
        fills in what the rows cannot show (model, batch size, ...). Returns
        the merged settings and the keys where the saved config disagreed.
        """
        merged = dict(saved)
        overridden = []
        for key, value in rows.items():
            if key in saved and saved[key] != value and key in RESUME_LOCKED_KEYS:
                overridden.append(key)
            merged[key] = value
        if "channel" in overridden:
            # The saved channel list describes the wrong channel.
            merged.pop("channels", None)
            merged.pop("label_channel", None)
        return merged, overridden

    @staticmethod
    def _settings_from_config(config: AnnotationConfig) -> dict:
        """Widget-level settings from a saved AnnotationConfig."""
        sc, training = config.spatial_coverage, config.training
        return {
            "channel": sc.get_label_channel(),
            "channels": list(sc.channels),
            "label_channel": sc.label_channel,
            "three_d": sc.three_d,
            "z_range_start": sc.z_range_start or 0,
            "z_range_end": sc.z_range_end or 0,
            "use_patches": sc.use_patches,
            "patch_size": list(sc.patch_size),
            "patches_per_image": sc.patches_per_image,
            "timepoint_mode": sc.timepoint_mode,
            "timepoints": list(sc.timepoints),
            "z_slice_mode": sc.z_slice_mode,
            "z_slices": list(sc.z_slices),
            "segment_all": training.segment_all,
            "train_n": training.train_n,
            "validate_n": training.validate_n,
            "test_n": training.test_n,
            "batch_size": config.workflow.batch_size,
            "model_type": config.ai_model.pretrained_from,
        }

    def _annotation_widget_map(self) -> dict:
        """Configure-tab widgets by setting name."""
        names = {
            "Annotate all images": "segment_all", "Training images": "train_n",
            "Validation images": "validate_n", "Test images": "test_n",
            "Channel": "channel", "Timepoint list": "timepoints", "Timepoints": "timepoint_mode",
            "Z-slice list": "z_slices", "Z-slices": "z_slice_mode", "3D processing": "three_d",
            "Z-range start": "z_range_start", "Z-range end": "z_range_end",
            "Use patches": "use_patches", "Patches per image": "patches_per_image",
            "Patch size": "patch_size",
        }
        found = {}
        for child in self.annotation_widgets.children:
            description = getattr(child, "description", "") or ""
            for prefix, key in names.items():
                if description.startswith(prefix):
                    found.setdefault(key, child)
                    break
        for child in self.technical_widgets.children:
            description = getattr(child, "description", "") or ""
            if description.startswith("Batch size"):
                found["batch_size"] = child
            elif description.startswith("SAM Model"):
                found["model_type"] = child
        return found

    def _apply_settings_to_widgets(self, settings: dict):
        """Set Configure-tab widgets from a settings dict (unknown keys are skipped)."""
        widgets_by_key = self._annotation_widget_map()
        for key, value in settings.items():
            widget = widgets_by_key.get(key)
            if widget is None or value is None:
                continue
            if key in ("patch_size", "timepoints", "z_slices"):
                value = ", ".join(str(v) for v in value)
            if key == "model_type" and value not in [v for _, v in widget.options]:
                continue
            if hasattr(widget, "max") and isinstance(value, int) and value > widget.max:
                widget.max = value
            widget.value = value

    def _lock_resume_fields(self, settings: dict):
        """Disable the fields that must match the table, and remember their values."""
        self._locked_settings = {k: settings[k] for k in RESUME_LOCKED_KEYS if k in settings}
        for key in ("channels", "label_channel"):
            if key in settings:
                self._locked_settings[key] = settings[key]
        widgets_by_key = self._annotation_widget_map()
        for key in RESUME_LOCKED_KEYS:
            if key in settings and key in widgets_by_key:
                widgets_by_key[key].disabled = True

    def _unlock_resume_fields(self):
        """Re-enable locked fields, keeping the usual dependent disabled states."""
        if not getattr(self, "_locked_settings", None):
            return
        self._locked_settings = {}
        widgets_by_key = self._annotation_widget_map()
        three_d = widgets_by_key.get("three_d")
        use_patches = widgets_by_key.get("use_patches")
        for key in RESUME_LOCKED_KEYS:
            widget = widgets_by_key.get(key)
            if widget is None:
                continue
            if key in ("z_range_start", "z_range_end"):
                widget.disabled = not (three_d.value if three_d else False)
            elif key == "patch_size":
                widget.disabled = not (use_patches.value if use_patches else False)
            else:
                widget.disabled = False
        if hasattr(self, "resume_note"):
            self.resume_note.value = ""

    def _apply_locked_settings_to_config(self):
        """Locked values come from the table's config, never from the widgets."""
        locked = getattr(self, "_locked_settings", None) or {}
        sc = self.config.spatial_coverage
        if "channels" in locked:
            sc.channels = list(locked["channels"])
            sc.label_channel = locked.get("label_channel")
        elif "channel" in locked:
            sc.channels[0] = locked["channel"]
            sc.label_channel = None
        if "three_d" in locked:
            sc.three_d = locked["three_d"]
        if "z_range_start" in locked:
            sc.z_range_start = locked["z_range_start"]
        if "z_range_end" in locked:
            sc.z_range_end = locked["z_range_end"]
        if "use_patches" in locked:
            sc.use_patches = locked["use_patches"]
        if "patch_size" in locked:
            sc.patch_size = list(locked["patch_size"])

    def _on_new_table(self, button):
        """Pick a name for a new table. The table itself is created at pipeline run."""
        try:
            container_id = self.container_widgets["container"].value
            container_type = self.container_widgets["type"].value
            # If user provided a custom name, use it; otherwise let function generate timestamped name
            custom_name = self.new_table_name.value.strip()

            if custom_name:
                # User provided custom name - make it unique
                unique_name = generate_unique_table_name(
                    self.connection, container_type, container_id, custom_name
                )
            else:
                # No custom name - let function generate timestamped name
                unique_name = generate_unique_table_name(
                    self.connection, container_type, container_id
                )
            self.new_table_name.value = unique_name
            self.config.name = unique_name
            self.selected_table_id = None
            self._set_resume_from_table(False)

            with self.status_output:
                clear_output()
                print(f"✅ Generated table name: {unique_name}")

            self._update_progress()

        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"❌ Error generating table name: {e}")

    def _toggle_subset_settings(self, change, train_n, validate_n, test_n):
        """Toggle subset settings based on segment_all."""
        segment_all = change["new"]
        train_n.disabled = segment_all
        validate_n.disabled = segment_all
        test_n.disabled = segment_all

    def _toggle_list_setting(self, change, text_widget):
        """Toggle list input based on mode."""
        mode = change["new"]
        text_widget.disabled = mode != "specific"

    def _toggle_patch_settings(self, change, patches_per_image, patch_size):
        """Toggle patch settings."""
        enabled = change["new"]
        patches_per_image.disabled = not enabled
        patch_size.disabled = not enabled

    def _toggle_3d_settings(self, change, z_range_start, z_range_end):
        """Toggle 3D settings based on three_d checkbox."""
        enabled = change["new"]
        z_range_start.disabled = not enabled
        z_range_end.disabled = not enabled

    def _on_update_config(self, button):
        """Update configuration from all widget values."""
        try:
            # Update OMERO settings (from previous steps)
            container_id = self.container_widgets["container"].value
            container_type = self.container_widgets["type"].value

            if container_id:
                self.config.omero.container_id = container_id
                self.config.omero.container_type = container_type

                # Get container name for description
                try:
                    if self.connection is not None:
                        container_obj = self.connection.getObject(
                            container_type.capitalize(), container_id
                        )
                        if container_obj:
                            container_name = container_obj.getName()
                            self.config.omero.source_desc = f"Workflow: {container_type} {container_name} (ID: {container_id})"
                        else:
                            self.config.omero.source_desc = (
                                f"Workflow: {container_type} (ID: {container_id})"
                            )
                    else:
                        self.config.omero.source_desc = (
                            f"Workflow: {container_type} (ID: {container_id})"
                        )
                except Exception:
                    self.config.omero.source_desc = (
                        f"Workflow: {container_type} (ID: {container_id})"
                    )

                # Update training set name from table selection
                if self.selected_table_id:
                    # Continuing existing table
                    for table in self.annotation_tables:
                        if table.get("id") == self.selected_table_id:
                            self.config.name = table.get("name", "unknown")
                            self._set_resume_from_table(True)
                            break
                elif self.new_table_name.value.strip():
                    # Creating new table
                    self.config.name = self.new_table_name.value.strip()
                    self._set_resume_from_table(False)

                self._render_omero_status(
                    container_type, container_id,
                    extra_lines=[
                        f"<b>Training Set:</b> {self.config.name}",
                        f"<b>Resume from table:</b> {self.config.workflow.resume_from_table}",
                    ],
                )

            # Update annotation settings from widgets
            self._update_annotation_settings()
            self._apply_locked_settings_to_config()

            # Update technical settings from widgets
            self._update_technical_settings()

            # Update workflow settings from widgets
            self._update_workflow_settings()

            # Update working directory
            if self.working_directory:
                self.config.output.output_directory = Path(self.working_directory)

            with self.status_output:
                clear_output()
                print("✅ Configuration updated successfully!")
                print(
                    f"📊 Container: {self.config.omero.container_type} (ID: {self.config.omero.container_id})"
                )
                print(f"🎯 Training Set: {self.config.name}")
                print(f"🔬 Model: {self.config.ai_model.pretrained_from}")

        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"❌ Error updating configuration: {e}")

    def _update_annotation_settings(self):
        """Update annotation settings from widgets."""
        # Get all annotation widgets from the VBox
        annotation_widgets = {}
        for child in self.annotation_widgets.children:
            if hasattr(child, "description") and hasattr(child, "value"):
                if "Annotate all images" in child.description:
                    annotation_widgets["segment_all"] = child
                elif "Training images" in child.description:
                    annotation_widgets["train_n"] = child
                elif "Validation images" in child.description:
                    annotation_widgets["validate_n"] = child
                elif "Test images" in child.description:
                    annotation_widgets["test_n"] = child
                elif "Channel" in child.description:
                    annotation_widgets["channel"] = child
                elif "Timepoints" in child.description and "list" in child.description:
                    annotation_widgets["timepoints"] = child
                elif "Timepoints" in child.description:
                    annotation_widgets["timepoint_mode"] = child
                elif "Z-slices" in child.description and "list" in child.description:
                    annotation_widgets["z_slices"] = child
                elif "Z-slices" in child.description:
                    annotation_widgets["z_slice_mode"] = child
                elif "3D processing" in child.description:
                    annotation_widgets["three_d"] = child
                elif "Volumetric 3D mode" in child.description:
                    annotation_widgets["volumetric_processing"] = child
                elif "Z-range start" in child.description:
                    annotation_widgets["z_range_start"] = child
                elif "Z-range end" in child.description:
                    annotation_widgets["z_range_end"] = child
                elif "Use patches" in child.description:
                    annotation_widgets["use_patches"] = child
                elif "Patches per image" in child.description:
                    annotation_widgets["patches_per_image"] = child
                elif "Patch size" in child.description:
                    annotation_widgets["patch_size"] = child

        # Update configuration from widgets
        if "segment_all" in annotation_widgets:
            self.config.training.segment_all = annotation_widgets["segment_all"].value
        if "train_n" in annotation_widgets:
            self.config.training.train_n = annotation_widgets["train_n"].value
        if "validate_n" in annotation_widgets:
            self.config.training.validate_n = annotation_widgets["validate_n"].value
        if "test_n" in annotation_widgets:
            self.config.training.test_n = annotation_widgets["test_n"].value
        if "channel" in annotation_widgets:
            self.config.spatial_coverage.channels[0] = annotation_widgets[
                "channel"
            ].value
        if "timepoint_mode" in annotation_widgets:
            self.config.spatial_coverage.timepoint_mode = annotation_widgets[
                "timepoint_mode"
            ].value
        if "timepoints" in annotation_widgets:
            timepoints_str = annotation_widgets["timepoints"].value
            try:
                self.config.spatial_coverage.timepoints = [
                    int(x.strip()) for x in timepoints_str.split(",") if x.strip()
                ]
            except Exception:
                self.config.spatial_coverage.timepoints = [0]
        if "z_slice_mode" in annotation_widgets:
            self.config.spatial_coverage.z_slice_mode = annotation_widgets[
                "z_slice_mode"
            ].value
        if "z_slices" in annotation_widgets:
            z_slices_str = annotation_widgets["z_slices"].value
            try:
                self.config.spatial_coverage.z_slices = [
                    int(x.strip()) for x in z_slices_str.split(",") if x.strip()
                ]
            except Exception:
                self.config.spatial_coverage.z_slices = [0]
        if "three_d" in annotation_widgets:
            self.config.spatial_coverage.three_d = annotation_widgets["three_d"].value

        if "z_range_start" in annotation_widgets:
            self.config.spatial_coverage.z_range_start = annotation_widgets[
                "z_range_start"
            ].value
        if "z_range_end" in annotation_widgets:
            self.config.spatial_coverage.z_range_end = annotation_widgets[
                "z_range_end"
            ].value
        if "use_patches" in annotation_widgets:
            self.config.spatial_coverage.use_patches = annotation_widgets["use_patches"].value
        if "patches_per_image" in annotation_widgets:
            self.config.spatial_coverage.patches_per_image = annotation_widgets[
                "patches_per_image"
            ].value
        if "patch_size" in annotation_widgets:
            patch_size_str = annotation_widgets["patch_size"].value
            try:
                sizes = [int(x.strip()) for x in patch_size_str.split(",") if x.strip()]
                if len(sizes) == 2:
                    self.config.spatial_coverage.patch_size = list(sizes)
            except Exception:
                pass

    def _update_technical_settings(self):
        """Update technical settings from widgets."""
        # Get all technical widgets from the VBox
        technical_widgets = {}
        for child in self.technical_widgets.children:
            if hasattr(child, "description") and hasattr(child, "value"):
                if "Batch size" in child.description:
                    technical_widgets["batch_size"] = child
                elif "SAM Model" in child.description:
                    technical_widgets["model_type"] = child
                elif "Output folder" in child.description:
                    technical_widgets["output_folder"] = child

        # Update configuration from widgets
        if "batch_size" in technical_widgets:
            self.config.workflow.batch_size = technical_widgets["batch_size"].value
        if "model_type" in technical_widgets:
            self.config.ai_model.pretrained_from = technical_widgets["model_type"].value
        if "output_folder" in technical_widgets:
            self.config.output.output_directory = technical_widgets[
                "output_folder"
            ].value

    def _update_workflow_settings(self):
        """Update workflow settings from widgets."""
        # Get all workflow widgets from the VBox
        workflow_widgets = {}
        for child in self.workflow_settings.children:
            if hasattr(child, "description") and hasattr(child, "value"):
                if "Read-only mode" in child.description:
                    workflow_widgets["read_only_mode"] = child

        # Update configuration from widgets
        if "read_only_mode" in workflow_widgets:
            self.config.workflow.read_only_mode = workflow_widgets[
                "read_only_mode"
            ].value
        # resume_from_table is not read back: the checkbox only mirrors the
        # table choice (Continue Existing / Name New Table) made in step 3.

    def _update_config_preview(self):
        """Update configuration preview."""
        self.config_preview.value = self.config.to_yaml()

    def _on_save_config(self, button):
        """Handle save configuration."""
        try:
            if not self.working_directory:
                self.save_status.value = "❌ No working directory selected"
                return

            # First update configuration from all widgets
            self._on_update_config(button)

            # Then save to file
            config_path = Path(self.working_directory) / f"annotation_config_{self.config.name}.yaml"
            
            # This will now set config.config_file_path automatically
            self.config.save_yaml(config_path)
            
            self.save_status.value = f"✅ Configuration saved to: {config_path}"

        except Exception as e:
            self.save_status.value = f"❌ Error saving configuration: {e}"

    def _render_omero_status(self, container_type, container_id, extra_lines=()):
        """Show the selected container and its image summary on the Configure tab."""
        info = getattr(self, "_container_info", None) or {}
        same_container = info.get("container_id") == container_id
        label = " ".join(p for p in (container_type, info.get("container_name") if same_container else None) if p)
        lines = [f"<b>Container:</b> {label} (ID: {container_id})"]
        lines.extend(extra_lines)
        if same_container:
            summary = f"<b>Images:</b> {info.get('total_images', 0)}"
            if container_type == "screen" and info.get("total_plates") is not None:
                summary += f" | <b>Plates:</b> {info.get('total_plates')}"
            dims = info.get("dimensions") or {}
            if dims:
                summary += (
                    f" | <b>Sample dims:</b> X={dims.get('X')}, Y={dims.get('Y')}, "
                    f"Z={dims.get('Z')}, C={dims.get('C')}, T={dims.get('T')}"
                )
            lines.append(summary)
            channels = self._format_channel_names(info)
            if channels:
                lines.append(f"<b>Channels:</b> {channels}")
        self.omero_status.value = "<h5>📡 OMERO Settings</h5>" + "".join(f"<p>{line}</p>" for line in lines)

    @staticmethod
    def _format_channel_names(info: dict) -> str:
        """Channel index and name of the sample image, e.g. '0: DAPI, 1: GFP'."""
        names = info.get("channel_names")
        if not names:
            return ""
        text = ", ".join(f"{i}: {html.escape(name)}" for i, name in enumerate(names))
        if info.get("channel_names_vary"):
            text += " <i>(from the first image; other images name channels differently)</i>"
        return text

    def _apply_container_info(self, info: dict):
        """Apply container info to UI and config-dependent widgets.

        This updates:
        - Container summary HTML in the container tab
        - OMERO status summary in the config tab
        - Max values for training/validation image sliders
        """
        self._container_info = info

        # Build summary HTML
        name = info.get("container_name", "")
        ctype = info.get("container_type", self.container_widgets["type"].value)
        total_images = info.get("total_images", 0)
        dims = info.get("dimensions") or {}
        px = info.get("pixel_sizes") or {}

        summary_lines = [
            f"<b>Container:</b> {ctype} {name} (ID: {info.get('container_id', '')})",
            f"<b>Total images:</b> {total_images}",
        ]

        # Screen-specific info
        if ctype == "screen":
            total_plates = info.get("total_plates")
            if total_plates is not None:
                summary_lines.append(f"<b>Total plates:</b> {total_plates}")
            # Optional per-plate details if available
            plates = info.get("plates")
            if isinstance(plates, dict) and plates:
                # Show a compact list of first few plates
                listed = 0
                details = []
                for pid, pinfo in plates.items():
                    count = pinfo.get("total_images") if isinstance(pinfo, dict) else pinfo
                    details.append(f"Plate {pid}: {count} images")
                    listed += 1
                    if listed >= 5:
                        break
                if details:
                    summary_lines.append("<b>Plates (sample):</b> " + ", ".join(details))

        if dims:
            summary_lines.append(
                f"<b>Sample image dims:</b> X={dims.get('X')}, Y={dims.get('Y')}, Z={dims.get('Z')}, C={dims.get('C')}, T={dims.get('T')}"
            )
        channels = self._format_channel_names(info)
        if channels:
            summary_lines.append(f"<b>Channels:</b> {channels}")
        if px:
            parts = []
            for axis in ("X", "Y", "Z"):
                v = px.get(axis)
                if v:
                    parts.append(f"{axis}={v[0]} {v[1]}")
            if parts:
                summary_lines.append("<b>Pixel sizes:</b> " + ", ".join(parts))

        self.container_summary.value = "<div>" + "<br>".join(summary_lines) + "</div>"

        self._render_omero_status(ctype, info.get("container_id", ""))

        # Update sliders' max to the number of available images
        if hasattr(self, "train_n_slider") and hasattr(self, "validate_n_slider") and hasattr(self, "test_n_slider"):
            max_imgs = max(1, int(total_images))
            self.train_n_slider.max = max_imgs
            self.validate_n_slider.max = max_imgs
            self.test_n_slider.max = max_imgs
            # Clamp current values within new range
            self.train_n_slider.value = min(self.train_n_slider.value, max_imgs)
            self.validate_n_slider.value = min(self.validate_n_slider.value, max_imgs)
            self.test_n_slider.value = min(self.test_n_slider.value, max_imgs)

    def _enable_container_widgets(self):
        """Enable container widgets when connection is available."""
        if hasattr(self, "container_widgets"):
            self.container_widgets["type"].disabled = False
            self.container_widgets["container"].disabled = False
            self.container_widgets["refresh"].disabled = False
            self.tables_widgets["scan"].disabled = False
            self._load_containers()

    def set_connection(self, connection):
        """Set OMERO connection."""
        self.connection = connection
        if connection and connection.isConnected():
            user = connection.getUser()
            user_name = user.getName() if user else "Unknown"
            self.connection_widgets["status"].value = f"✅ Connected as {user_name}"

            # Enable container widgets
            self.container_widgets["type"].disabled = False
            self.container_widgets["container"].disabled = False
            self.container_widgets["refresh"].disabled = False
            self.tables_widgets["scan"].disabled = False

            self._load_containers()
        else:
            self.connection_widgets["status"].value = "❌ No OMERO connection"

        self._update_progress()

    def display(self):
        """Display the widget."""
        self._update_progress()
        display(self.main_widget)

    def get_config(self) -> AnnotationConfig:
        """Get current configuration; locked resume settings always win."""
        self._apply_locked_settings_to_config()
        return self.config


def create_workflow_widget(connection=None):
    """Create workflow widget with connection validation.

    Args:
        connection: Optional OMERO connection object. If provided,
                   the connection step will be skipped.

    Returns:
        WorkflowWidget instance

    Raises:
        ValueError: If the provided connection is not active
    """
    if connection is not None:
        if not hasattr(connection, "isConnected") or not connection.isConnected():
            raise ValueError(
                "Provided OMERO connection is not active. Please establish a valid connection first."
            )

    return WorkflowWidget(connection=connection)
