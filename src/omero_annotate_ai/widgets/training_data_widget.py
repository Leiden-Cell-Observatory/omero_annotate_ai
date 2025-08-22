"""Simple training data selection widget for OMERO annotation tables."""

import ipywidgets as widgets
from IPython.display import clear_output, display

from ..omero.omero_functions import list_annotation_tables


class TrainingDataWidget:
    """Simple widget for selecting existing annotation tables for training."""

    def __init__(self, connection=None):
        """Initialize the training data widget.
        
        Args:
            connection: OMERO connection object. If provided, enables container selection.
        """
        self.connection = connection
        self.selected_table_id = None
        self.annotation_tables = []
        self._create_widgets()
        self._setup_observers()
        
        # Enable widgets if connection is provided
        if self.connection is not None:
            self._enable_widgets()

    def _create_widgets(self):
        """Create all widget components."""
        
        # Header
        self.header = widgets.HTML(
            value="<h3>🎯 Training Data Selection</h3>",
            layout=widgets.Layout(margin="0 0 20px 0"),
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

        # Refresh containers button
        self.refresh_button = widgets.Button(
            description="Refresh",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(width="100px"),
            disabled=True,
        )

        # Scan for tables button
        self.scan_button = widgets.Button(
            description="Scan for Tables",
            button_style="primary",
            icon="search",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )

        # Tables display
        self.tables_output = widgets.Output()

        # Table selection dropdown
        self.table_dropdown = widgets.Dropdown(
            options=[("No tables found", None)],
            value=None,
            description="Training Table:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
            disabled=True,
        )

        # Status output
        self.status_output = widgets.Output()

        # Main container
        self.main_widget = widgets.VBox([
            self.header,
            widgets.HTML("<b>Step 1: Select Container</b>"),
            self.container_type_dropdown,
            widgets.HBox([self.container_dropdown, self.refresh_button]),
            widgets.HTML("<br><b>Step 2: Find Training Tables</b>"),
            self.scan_button,
            self.tables_output,
            widgets.HTML("<br><b>Step 3: Select Table</b>"),
            self.table_dropdown,
            self.status_output,
        ])

    def _setup_observers(self):
        """Setup widget observers."""
        self.container_type_dropdown.observe(self._on_container_type_change, names="value")
        self.container_dropdown.observe(self._on_container_change, names="value") 
        self.table_dropdown.observe(self._on_table_change, names="value")
        self.refresh_button.on_click(self._on_refresh_containers)
        self.scan_button.on_click(self._on_scan_tables)

    def _enable_widgets(self):
        """Enable widgets when connection is available."""
        if self.connection and self.connection.isConnected():
            self.container_type_dropdown.disabled = False
            self.container_dropdown.disabled = False
            self.refresh_button.disabled = False
            self.scan_button.disabled = False
            self._load_containers()

    def _on_container_type_change(self, change):
        """Handle container type change."""
        if self.connection:
            self._load_containers()

    def _on_container_change(self, change):
        """Handle container selection change."""
        # Reset table selection when container changes
        self.table_dropdown.options = [("Scan for tables first", None)]
        self.table_dropdown.value = None
        self.selected_table_id = None

    def _on_table_change(self, change):
        """Handle table selection change."""
        self.selected_table_id = change["new"]
        if self.selected_table_id:
            with self.status_output:
                clear_output()
                # Find table name for confirmation
                table_name = "Unknown"
                for table in self.annotation_tables:
                    if table.get("id") == self.selected_table_id:
                        table_name = table.get("name", "Unknown")
                        break
                print(f"✅ Selected table: {table_name} (ID: {self.selected_table_id})")

    def _on_refresh_containers(self, button):
        """Handle refresh containers button."""
        self._load_containers()

    def _on_scan_tables(self, button):
        """Handle scan tables button."""
        try:
            container_id = self.container_dropdown.value
            container_type = self.container_type_dropdown.value

            if not container_id:
                with self.status_output:
                    clear_output()
                    print("❌ Please select a container first")
                return

            with self.status_output:
                clear_output()
                print("🔍 Scanning for annotation tables...")

            # Get annotation tables
            self.annotation_tables = list_annotation_tables(
                self.connection, container_type, container_id
            )

            with self.tables_output:
                clear_output()
                if self.annotation_tables:
                    print(f"📋 Found {len(self.annotation_tables)} annotation tables:")
                    for i, table in enumerate(self.annotation_tables):
                        name = table.get('name', 'Unknown')
                        table_id = table.get('id', 'Unknown')
                        created = table.get('created', 'Unknown')
                        print(f"  {i + 1}. {name} (ID: {table_id}) - Created: {created}")

                    # Update table dropdown options
                    options = [("Select a table...", None)]
                    for table in self.annotation_tables:
                        table_name = table.get("name", "Unknown")
                        table_id = table.get("id")
                        options.append((f"{table_name} (ID: {table_id})", table_id))

                    self.table_dropdown.options = options
                    self.table_dropdown.disabled = False
                else:
                    print("❌ No annotation tables found in this container")
                    self.table_dropdown.options = [("No tables found", None)]
                    self.table_dropdown.disabled = True

            with self.status_output:
                clear_output()
                if self.annotation_tables:
                    print(f"✅ Found {len(self.annotation_tables)} annotation tables")
                else:
                    print("ℹ️ No annotation tables found")

        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"❌ Error scanning tables: {e}")

    def _load_containers(self):
        """Load containers from OMERO."""
        if not self.connection:
            return

        try:
            container_type = self.container_type_dropdown.value

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

            self.container_dropdown.options = options

        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"❌ Error loading containers: {e}")

    def set_connection(self, connection):
        """Set OMERO connection and enable widgets.
        
        Args:
            connection: OMERO BlitzGateway connection object
        """
        self.connection = connection
        if connection and connection.isConnected():
            self._enable_widgets()
            with self.status_output:
                clear_output()
                user = connection.getUser()
                user_name = user.getName() if user else "Unknown"
                print(f"✅ Connected as {user_name}")
        else:
            with self.status_output:
                clear_output()
                print("❌ No OMERO connection")

    def display(self):
        """Display the widget."""
        display(self.main_widget)

    def get_selected_table_id(self):
        """Get the selected annotation table ID.
        
        Returns:
            Selected table ID, or None if no table selected
        """
        return self.selected_table_id

    def get_selected_table_info(self):
        """Get complete information about the selected table.
        
        Returns:
            Table information including name, ID, creation date, etc.
        """
        if not self.selected_table_id:
            return None
            
        for table in self.annotation_tables:
            if table.get("id") == self.selected_table_id:
                return table
        return None


def create_training_data_widget(connection=None):
    """Create a training data selection widget.

    Args:
        connection: Optional OMERO connection object. If provided,
                   the widget will be enabled immediately.

    Returns:
        TrainingDataWidget instance
        
    Raises:
        ValueError: If the provided connection is not active
    """
    if connection is not None:
        if not hasattr(connection, "isConnected") or not connection.isConnected():
            raise ValueError(
                "Provided OMERO connection is not active. Please establish a valid connection first."
            )

    return TrainingDataWidget(connection=connection)