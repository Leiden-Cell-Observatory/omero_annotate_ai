"""Interactive widget for container-level annotation management."""

import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Dict, Any, List
import pandas as pd

try:
    import ezomero
    EZOMERO_AVAILABLE = True
except ImportError:
    EZOMERO_AVAILABLE = False

from ..omero.omero_functions import (
    list_annotation_tables,
    generate_unique_table_name,
    get_table_progress_summary,
    create_roi_namespace_for_table
)
from ..core.config import create_default_config


class ContainerAnnotationWidget:
    """Interactive widget for managing annotation containers and tables."""
    
    def __init__(self, connection=None):
        """Initialize the container annotation widget.
        
        Args:
            connection: OMERO connection object
        """
        self.connection = connection
        self.selected_container_type = 'project'
        self.selected_container_id = None
        self.selected_table_id = None
        self.annotation_tables = []
        self._create_widgets()
        self._setup_observers()
        if connection is not None:
            self.set_connection(connection)
        
    def _create_widgets(self):
        """Create all widget components."""
        
        # Header
        self.header = widgets.HTML(
            value="<h3>📊 Container Annotation Management</h3>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Connection status
        self.connection_status = widgets.HTML(
            value="❌ No OMERO connection",
            layout=widgets.Layout(margin='0 0 10px 0')
        )
        
        # Container type selection
        self.container_type_dropdown = widgets.Dropdown(
            options=[('Project', 'project'), ('Dataset', 'dataset'), ('Plate', 'plate'), ('Screen', 'screen')],
            value='project',
            description='Container Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        
        # Container selection
        self.container_dropdown = widgets.Dropdown(
            options=[('Select a container...', None)],
            value=None,
            description='Container:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            disabled=True
        )
        
        # Refresh containers button
        self.refresh_containers_button = widgets.Button(
            description='Refresh Containers',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        # Scan tables button
        self.scan_tables_button = widgets.Button(
            description='Scan for Tables',
            button_style='primary',
            icon='search',
            layout=widgets.Layout(width='150px', margin='10px 0 0 0'),
            disabled=True
        )
        
        # Tables display area
        self.tables_output = widgets.Output()
        
        # Table selection
        self.table_selection_widget = widgets.RadioButtons(
            options=[],
            description='Select:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='10px 0 0 0')
        )
        
        # Action buttons
        self.continue_button = widgets.Button(
            description='Continue Existing',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='150px', margin='10px 5px 0 0'),
            disabled=True
        )
        
        self.new_table_button = widgets.Button(
            description='Create New Table',
            button_style='warning',
            icon='plus',
            layout=widgets.Layout(width='150px', margin='10px 0 0 0'),
            disabled=True
        )
        
        # New table name input
        self.new_table_name_widget = widgets.Text(
            description='Table name:',
            placeholder='Leave empty for auto-generated name',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', margin='10px 0 0 0')
        )
        
        # Export config button
        self.export_config_button = widgets.Button(
            description='Export Config YAML',
            button_style='info',
            icon='download',
            layout=widgets.Layout(width='150px', margin='10px 0 0 0'),
            disabled=True
        )
        
        # Output area for messages
        self.message_output = widgets.Output()
        
    def _setup_observers(self):
        """Set up widget event observers."""
        self.container_type_dropdown.observe(self._on_container_type_changed, names='value')
        self.refresh_containers_button.on_click(self._on_refresh_containers)
        self.container_dropdown.observe(self._on_container_changed, names='value')
        self.scan_tables_button.on_click(self._on_scan_tables)
        self.table_selection_widget.observe(self._on_table_selection_changed, names='value')
        self.continue_button.on_click(self._on_continue_existing)
        self.new_table_button.on_click(self._on_create_new_table)
        self.export_config_button.on_click(self._on_export_config)
        
    def set_connection(self, connection):
        """Set the OMERO connection.
        
        Args:
            connection: OMERO connection object
        """
        self.connection = connection
    
        if connection and connection.isConnected():
            # Get user information which is more reliable
            user = connection.getUser()
            user_name = user.getName() if user else "Unknown User"
            
            self.connection_status.value = f"✅ Connected as {user_name}"
            self.container_type_dropdown.disabled = False
            self.container_dropdown.disabled = False
            self.refresh_containers_button.disabled = False
            self._load_containers()
        else:
            self.connection_status.value = "❌ No OMERO connection"
            self.container_type_dropdown.disabled = True
            self.container_dropdown.disabled = True
            self.refresh_containers_button.disabled = True
            
    def _load_containers(self):
        """Load user's containers from OMERO based on selected type."""
        if not self.connection or not EZOMERO_AVAILABLE:
            return
            
        container_type = self.selected_container_type
        
        try:
            if container_type == 'project':
                containers = list(self.connection.getObjects("Project"))
                type_name = "project"
            elif container_type == 'dataset':
                containers = list(self.connection.getObjects("Dataset"))
                type_name = "dataset"
            elif container_type == 'plate':
                containers = list(self.connection.getObjects("Plate"))
                type_name = "plate"
            elif container_type == 'screen':
                containers = list(self.connection.getObjects("Screen"))
                type_name = "screen"
            else:
                containers = []
                type_name = container_type
            
            if containers:
                container_options = [(f'Select a {type_name}...', None)]
                for container in containers:
                    container_options.append((f"{container.getName()} (ID: {container.getId()})", container.getId()))
                
                self.container_dropdown.options = container_options
                
            else:
                self.container_dropdown.options = [(f'No {type_name}s found', None)]
                
        except Exception as e:
            with self.message_output:
                clear_output(wait=True)
                print(f"❌ Error loading {type_name}s: {e}")
    
    def _on_container_type_changed(self, change):
        """Handle container type selection change."""
        self.selected_container_type = change['new']
        
        # Reset container selection
        self.selected_container_id = None
        self.container_dropdown.value = None
        
        # Clear previous table data
        self.annotation_tables = []
        self.table_selection_widget.options = []
        self._update_action_buttons()
        
        # Load containers of the new type
        self._load_containers()
        
        with self.tables_output:
            clear_output(wait=True)
            print(f"📂 Container type changed to {self.selected_container_type}. Select a container and scan for tables.")
    
    def _on_refresh_containers(self, button):
        """Handle refresh containers button click."""
        with self.message_output:
            clear_output(wait=True)
            print(f"🔄 Refreshing {self.selected_container_type}s...")
        
        self._load_containers()
        
        with self.message_output:
            clear_output(wait=True)
            print(f"✅ {self.selected_container_type.capitalize()}s refreshed")
    
    def _on_container_changed(self, change):
        """Handle container selection change."""
        self.selected_container_id = change['new']
        
        if self.selected_container_id:
            self.scan_tables_button.disabled = False
            # Clear previous table data
            self.annotation_tables = []
            self.table_selection_widget.options = []
            self._update_action_buttons()
            
            with self.tables_output:
                clear_output(wait=True)
                print(f"📂 {self.selected_container_type.capitalize()} selected. Click 'Scan for Tables' to find existing annotation tables.")
        else:
            self.scan_tables_button.disabled = True
            self._update_action_buttons()
    
    def _on_scan_tables(self, button):
        """Handle scan tables button click."""
        if not self.selected_container_id:
            return
            
        with self.message_output:
            clear_output(wait=True)
            print("🔍 Scanning for annotation tables...")
        
        try:
            # Get annotation tables for the container
            self.annotation_tables = list_annotation_tables(
                self.connection, self.selected_container_type, self.selected_container_id
            )
            
            with self.tables_output:
                clear_output(wait=True)
                self._display_tables()
                
            with self.message_output:
                clear_output(wait=True)
                if self.annotation_tables:
                    print(f"✅ Found {len(self.annotation_tables)} annotation table(s)")
                else:
                    print("📋 No existing annotation tables found")
            
        except Exception as e:
            with self.message_output:
                clear_output(wait=True)
                print(f"❌ Error scanning tables: {e}")
    
    def _display_tables(self):
        """Display the found annotation tables."""
        if not self.annotation_tables:
            print(f"📋 No annotation tables found in this {self.selected_container_type}.")
            print("Use 'Create New Table' to start a new annotation workflow.")
            self.table_selection_widget.options = []
            self._update_action_buttons()
            return
        
        print(f"📊 Found {len(self.annotation_tables)} annotation table(s):")
        print("=" * 60)
        
        # Create table display
        table_options = []
        
        for i, table in enumerate(self.annotation_tables):
            table_name = table.get('name', 'Unknown')
            table_id = table.get('id', 0)
            created_date = table.get('created_date', 'Unknown')
            
            # Get progress summary
            try:
                progress_summary = get_table_progress_summary(self.connection, table_id)
            except:
                progress_summary = "❓ Unknown status"
            
            # Display table info
            print(f"{i+1}. {table_name}")
            print(f"   ID: {table_id}")
            print(f"   Created: {created_date}")
            print(f"   Status: {progress_summary}")
            print()
            
            # Add to selection options
            option_text = f"{table_name} - {progress_summary}"
            table_options.append((option_text, table_id))
        
        # Update selection widget
        self.table_selection_widget.options = table_options
        self._update_action_buttons()
    
    def _on_table_selection_changed(self, change):
        """Handle table selection change."""
        self.selected_table_id = change['new']
        self._update_action_buttons()
    
    def _update_action_buttons(self):
        """Update the state of action buttons."""
        # Enable/disable buttons based on current state
        has_tables = len(self.annotation_tables) > 0
        has_selection = self.selected_table_id is not None
        has_container = self.selected_container_id is not None
        
        self.continue_button.disabled = not (has_tables and has_selection)
        self.new_table_button.disabled = not has_container
        self.export_config_button.disabled = not has_container
    
    def _on_continue_existing(self, button):
        """Handle continue existing table button click."""
        if not self.selected_table_id:
            return
            
        # Find the selected table
        selected_table = None
        for table in self.annotation_tables:
            if table.get('id') == self.selected_table_id:
                selected_table = table
                break
        
        if selected_table:
            with self.message_output:
                clear_output(wait=True)
                print(f"✅ Selected table: {selected_table.get('name', 'Unknown')}")
                print(f"   Table ID: {self.selected_table_id}")
                print(f"   {self.selected_container_type.capitalize()} ID: {self.selected_container_id}")
                print("   Ready to continue annotation workflow.")
    
    def _on_create_new_table(self, button):
        """Handle create new table button click."""
        if not self.selected_container_id:
            return
            
        try:
            # Generate unique table name
            base_name = self.new_table_name_widget.value.strip()
            if not base_name:
                base_name = None  # Let the function auto-generate
                
            # Generate unique table name for the container
            unique_name = generate_unique_table_name(
                self.connection, self.selected_container_type, self.selected_container_id, base_name
            )
            
            with self.message_output:
                clear_output(wait=True)
                print(f"✅ Generated unique table name: {unique_name}")
                print(f"   {self.selected_container_type.capitalize()} ID: {self.selected_container_id}")
                print(f"   ROI Namespace: {create_roi_namespace_for_table(unique_name)}")
                print("   Ready to create new annotation workflow.")
                
            # Store the generated name for later use
            self.new_table_name_widget.value = unique_name
            
        except Exception as e:
            with self.message_output:
                clear_output(wait=True)
                print(f"❌ Error generating table name: {e}")
    
    def _on_export_config(self, button):
        """Handle export config YAML button click."""
        if not self.selected_container_id:
            return
            
        try:
            # Generate configuration YAML
            config_yaml = self.generate_config_yaml()
            
            with self.message_output:
                clear_output(wait=True)
                print("✅ Generated Configuration YAML:")
                print("=" * 50)
                print(config_yaml)
                print("=" * 50)
                print("\n📌 Instructions:")
                print("1. Copy the YAML above")
                print("2. Use it with the Config Widget:")
                print("   config = create_config_widget()")
                print("   # Or load from YAML file")
                print("   # config = load_config_from_yaml('config.yaml')")
                print("3. Continue with your annotation workflow")
                
        except Exception as e:
            with self.message_output:
                clear_output(wait=True)
                print(f"❌ Error generating config YAML: {e}")
    
    def generate_config_yaml(self) -> str:
        """Generate a complete configuration YAML for the config widget.
        
        Returns:
            YAML string compatible with AnnotationConfig
        """
        # Create default configuration
        config = create_default_config()
        
        # Update OMERO settings with current selection
        config.omero.container_type = self.selected_container_type
        config.omero.container_id = self.selected_container_id
        
        # Update workflow settings based on current state
        if self.selected_table_id:
            # Continuing existing table
            config.workflow.resume_from_table = True
            for table in self.annotation_tables:
                if table.get('id') == self.selected_table_id:
                    table_name = table.get('name')
                    if table_name:
                        # Note: table_name is not a direct config field, but we can use it in training
                        config.training.trainingset_name = table_name
                    break
        elif self.new_table_name_widget.value.strip():
            # Creating new table
            config.workflow.resume_from_table = False
            config.training.trainingset_name = self.new_table_name_widget.value.strip()
        
        # Add container-specific description
        container_name = ""
        if self.selected_container_id:
            try:
                container_obj = self.connection.getObject(self.selected_container_type.capitalize(), self.selected_container_id)
                if container_obj:
                    container_name = container_obj.getName()
            except:
                pass
        
        config.omero.source_desc = f"Auto-generated from {self.selected_container_type} {container_name} (ID: {self.selected_container_id})"
        
        return config.to_yaml()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current widget configuration.
        
        Returns:
            Dictionary with container and table configuration
        """
        config = {
            'container_type': self.selected_container_type,
            'container_id': self.selected_container_id,
            'table_id': self.selected_table_id,
            'action': None,
            'table_name': None,
            'roi_namespace': None
        }
        
        if self.selected_table_id:
            # Continuing existing table
            config['action'] = 'continue'
            for table in self.annotation_tables:
                if table.get('id') == self.selected_table_id:
                    config['table_name'] = table.get('name')
                    config['roi_namespace'] = create_roi_namespace_for_table(config['table_name'])
                    break
        elif self.new_table_name_widget.value.strip():
            # Creating new table
            config['action'] = 'new'
            config['table_name'] = self.new_table_name_widget.value.strip()
            config['roi_namespace'] = create_roi_namespace_for_table(config['table_name'])
        
        return config
    
    def display(self):
        """Display the complete widget."""
        container_type_box = widgets.HBox([
            self.container_type_dropdown
        ])
        
        container_box = widgets.HBox([
            self.container_dropdown,
            self.refresh_containers_button
        ])
        
        scan_box = widgets.HBox([
            self.scan_tables_button
        ])
        
        action_box = widgets.HBox([
            self.continue_button,
            self.new_table_button
        ])
        
        export_box = widgets.HBox([
            self.export_config_button
        ])
        
        return widgets.VBox([
            self.header,
            self.connection_status,
            container_type_box,
            container_box,
            scan_box,
            self.tables_output,
            self.table_selection_widget,
            action_box,
            self.new_table_name_widget,
            export_box,
            self.message_output
        ])


def create_container_annotation_widget(connection=None):
    """Create a container annotation management widget.
    
    Args:
        connection: Optional OMERO connection
        
    Returns:
        ContainerAnnotationWidget instance
    """
    widget = ContainerAnnotationWidget(connection)
    return widget