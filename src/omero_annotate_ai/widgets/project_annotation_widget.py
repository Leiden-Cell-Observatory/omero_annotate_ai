"""Interactive widget for project-level annotation management."""

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
    list_annotation_tables_for_project,
    generate_unique_table_name,
    get_table_progress_summary,
    create_roi_namespace_for_table
)


class ProjectAnnotationWidget:
    """Interactive widget for managing annotation projects and tables."""
    
    def __init__(self, connection=None):
        """Initialize the project annotation widget.
        
        Args:
            connection: OMERO connection object
        """
        self.connection = connection
        self.selected_project_id = None
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
            value="<h3>📊 Project Annotation Management</h3>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Connection status
        self.connection_status = widgets.HTML(
            value="❌ No OMERO connection",
            layout=widgets.Layout(margin='0 0 10px 0')
        )
        
        # Project selection
        self.project_dropdown = widgets.Dropdown(
            options=[('Select a project...', None)],
            value=None,
            description='Project:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            disabled=True
        )
        
        # Refresh projects button
        self.refresh_projects_button = widgets.Button(
            description='Refresh Projects',
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
        
        # Output area for messages
        self.message_output = widgets.Output()
        
    def _setup_observers(self):
        """Set up widget event observers."""
        self.refresh_projects_button.on_click(self._on_refresh_projects)
        self.project_dropdown.observe(self._on_project_changed, names='value')
        self.scan_tables_button.on_click(self._on_scan_tables)
        self.table_selection_widget.observe(self._on_table_selection_changed, names='value')
        self.continue_button.on_click(self._on_continue_existing)
        self.new_table_button.on_click(self._on_create_new_table)
        
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
            self.project_dropdown.disabled = False
            self.refresh_projects_button.disabled = False
            self._load_projects()
        else:
            self.connection_status.value = "❌ No OMERO connection"
            self.project_dropdown.disabled = True
            self.refresh_projects_button.disabled = True
            
    def _load_projects(self):
        """Load user's projects from OMERO."""
        if not self.connection or not EZOMERO_AVAILABLE:
            return
            
        try:
            # Get user's projects
            projects = list(self.connection.getObjects("Project"))
            
            if projects:
                project_options = [('Select a project...', None)]
                for project in projects:
                    project_options.append((f"{project.getName()} (ID: {project.getId()})", project.getId()))
                
                self.project_dropdown.options = project_options
                
            else:
                self.project_dropdown.options = [('No projects found', None)]
                
        except Exception as e:
            with self.message_output:
                clear_output(wait=True)
                print(f"❌ Error loading projects: {e}")
    
    def _on_refresh_projects(self, button):
        """Handle refresh projects button click."""
        with self.message_output:
            clear_output(wait=True)
            print("🔄 Refreshing projects...")
        
        self._load_projects()
        
        with self.message_output:
            clear_output(wait=True)
            print("✅ Projects refreshed")
    
    def _on_project_changed(self, change):
        """Handle project selection change."""
        self.selected_project_id = change['new']
        
        if self.selected_project_id:
            self.scan_tables_button.disabled = False
            # Clear previous table data
            self.annotation_tables = []
            self.table_selection_widget.options = []
            self._update_action_buttons()
            
            with self.tables_output:
                clear_output(wait=True)
                print("📂 Project selected. Click 'Scan for Tables' to find existing annotation tables.")
        else:
            self.scan_tables_button.disabled = True
            self._update_action_buttons()
    
    def _on_scan_tables(self, button):
        """Handle scan tables button click."""
        if not self.selected_project_id:
            return
            
        with self.message_output:
            clear_output(wait=True)
            print("🔍 Scanning for annotation tables...")
        
        try:
            # Get annotation tables for the project
            self.annotation_tables = list_annotation_tables_for_project(
                self.connection, self.selected_project_id
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
            print("📋 No annotation tables found in this project.")
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
        has_project = self.selected_project_id is not None
        
        self.continue_button.disabled = not (has_tables and has_selection)
        self.new_table_button.disabled = not has_project
    
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
                print(f"   Project ID: {self.selected_project_id}")
                print("   Ready to continue annotation workflow.")
    
    def _on_create_new_table(self, button):
        """Handle create new table button click."""
        if not self.selected_project_id:
            return
            
        try:
            # Generate unique table name
            base_name = self.new_table_name_widget.value.strip()
            if not base_name:
                base_name = None  # Let the function auto-generate
                
            unique_name = generate_unique_table_name(
                self.connection, self.selected_project_id, base_name
            )
            
            with self.message_output:
                clear_output(wait=True)
                print(f"✅ Generated unique table name: {unique_name}")
                print(f"   Project ID: {self.selected_project_id}")
                print(f"   ROI Namespace: {create_roi_namespace_for_table(unique_name)}")
                print("   Ready to create new annotation workflow.")
                
            # Store the generated name for later use
            self.new_table_name_widget.value = unique_name
            
        except Exception as e:
            with self.message_output:
                clear_output(wait=True)
                print(f"❌ Error generating table name: {e}")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current widget configuration.
        
        Returns:
            Dictionary with project and table configuration
        """
        config = {
            'project_id': self.selected_project_id,
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
        project_box = widgets.HBox([
            self.project_dropdown,
            self.refresh_projects_button
        ])
        
        scan_box = widgets.HBox([
            self.scan_tables_button
        ])
        
        action_box = widgets.HBox([
            self.continue_button,
            self.new_table_button
        ])
        
        return widgets.VBox([
            self.header,
            self.connection_status,
            project_box,
            scan_box,
            self.tables_output,
            self.table_selection_widget,
            action_box,
            self.new_table_name_widget,
            self.message_output
        ])


def create_project_annotation_widget(connection=None):
    """Create a project annotation management widget.
    
    Args:
        connection: Optional OMERO connection
        
    Returns:
        ProjectAnnotationWidget instance
    """
    widget = ProjectAnnotationWidget(connection)
    return widget