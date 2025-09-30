"""Interactive widget for OMERO server connections with keychain support."""

from typing import Any, Dict

import ipywidgets as widgets
from IPython.display import clear_output, display

from ..omero.simple_connection import SimpleOMEROConnection


class OMEROConnectionWidget:
    """Interactive widget for creating OMERO connections with secure password storage."""

    def __init__(self):
        """Initialize the OMERO connection widget."""
        self.connection_manager = SimpleOMEROConnection()
        self.connection = None
        self._create_widgets()
        self._setup_observers()
        self._load_existing_config()

    def _create_widgets(self):
        """Create all widget components."""

        # Header
        self.header = widgets.HTML(
            value="""
                <h3>🔌 OMERO Server Connection</h3>
                <div style='font-size:90%;color:#888;margin-top:-10px;'>
                    <b>Note:</b> If your OMERO server uses a non-default port, add it to the host as <code>host:port</code> (e.g., <code>localhost:6064</code>).
                </div>
            """,
            layout=widgets.Layout(margin="0 0 20px 0"),
        )

        # Connection history dropdown
        self.connection_dropdown = widgets.Dropdown(
            options=[("Manual entry", None)],
            value=None,
            description="Previous:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
        )

        # Connection fields
        self.host_widget = widgets.Text(
            description="Host:",
            placeholder="omero.server.edu",
            style={"description_width": "initial"},
        )

        self.username_widget = widgets.Text(
            description="Username:",
            placeholder="your_username",
            style={"description_width": "initial"},
        )

        self.password_widget = widgets.Password(
            description="Password:",
            placeholder="Enter password",
            style={"description_width": "initial"},
        )

        self.group_widget = widgets.Text(
            description="Group:",
            placeholder="Optional group name",
            style={"description_width": "initial"},
        )

        self.secure_widget = widgets.Checkbox(
            value=True,
            description="Secure connection",
            style={"description_width": "initial"},
        )

        # Password saving options
        self.save_password_widget = widgets.Checkbox(
            value=False,
            description="Save password to keychain",
            style={"description_width": "initial"},
            tooltip="When checked, your password will be securely saved and auto-loaded for future connections to this server"
        )

        self.expire_widget = widgets.Dropdown(
            options=[
                ("1 hour", 1),
                ("8 hours", 8),
                ("24 hours", 24),
                ("1 week", 168),
                ("Never expires", None),
            ],
            value=24,
            description="Remember for:",
            style={"description_width": "initial"},
            disabled=True,
        )

        # Show password toggle
        self.show_password_widget = widgets.Checkbox(
            value=False,
            description="Show password",
            style={"description_width": "initial"},
        )

        # Action buttons
        self.test_button = widgets.Button(
            description="Test Connection", button_style="info", icon="plug"
        )

        self.connect_button = widgets.Button(
            description="Connect", button_style="success", icon="check"
        )

        self.load_keychain_button = widgets.Button(
            description="Load from Keychain", button_style="", icon="key"
        )

        self.save_connection_button = widgets.Button(
            description="Save Connection", button_style="warning", icon="save"
        )

        self.delete_connection_button = widgets.Button(
            description="Delete Connection",
            button_style="danger",
            icon="trash",
            disabled=True,  # Disabled until a connection is selected
        )

        # Status display
        self.status_output = widgets.Output()

        # Config info display
        self.config_info = widgets.HTML(
            value="", layout=widgets.Layout(margin="10px 0")
        )

        # Group widgets
        connection_fields = widgets.VBox(
            [
                self.host_widget,
                self.username_widget,
                self.password_widget,
                self.group_widget,
                self.secure_widget,
            ]
        )

        password_options = widgets.VBox(
            [
                widgets.HTML("<b>Password Options</b><br><i>Note: Passwords are only saved to keychain when explicitly requested</i>"),
                self.save_password_widget, 
                self.expire_widget, 
                self.show_password_widget
            ]
        )

        buttons_row1 = widgets.HBox(
            [self.test_button, self.connect_button, self.load_keychain_button]
        )

        buttons_row2 = widgets.HBox(
            [self.save_connection_button, self.delete_connection_button]
        )

        # Main container
        self.main_widget = widgets.VBox(
            [
                self.header,
                self.config_info,
                widgets.HTML("<b>Previous Connections</b>"),
                self.connection_dropdown,
                widgets.HTML("<br><b>Connection Settings</b>"),
                connection_fields,
                widgets.HTML("<br>"),
                password_options,
                widgets.HTML("<br>"),
                buttons_row1,
                buttons_row2,
                self.status_output,
            ]
        )

    def _setup_observers(self):
        """Setup widget observers."""
        # Enable/disable expire dropdown based on save password checkbox
        self.save_password_widget.observe(self._toggle_expire_options, names="value")

        # Show/hide password
        self.show_password_widget.observe(
            self._toggle_password_visibility, names="value"
        )

        # Connection dropdown observer
        self.connection_dropdown.observe(self._on_connection_selected, names="value")

        # Button callbacks
        self.test_button.on_click(self._test_connection)
        self.connect_button.on_click(self._connect)
        self.load_keychain_button.on_click(self._load_from_keychain)
        self.save_connection_button.on_click(self._save_connection_only)
        self.delete_connection_button.on_click(self._delete_connection)

    def _toggle_expire_options(self, change):
        """Toggle expiration dropdown based on save password checkbox."""
        self.expire_widget.disabled = not change["new"]

    def _toggle_password_visibility(self, change):
        """Toggle password visibility."""
        if change["new"]:
            # Show password - convert to text widget
            password_value = self.password_widget.value
            self.password_widget.close()

            self.password_widget = widgets.Text(
                value=password_value,
                description="Password:",
                placeholder="Enter password",
                style={"description_width": "initial"},
            )
        else:
            # Hide password - convert to password widget
            password_value = self.password_widget.value
            self.password_widget.close()

            self.password_widget = widgets.Password(
                value=password_value,
                description="Password:",
                placeholder="Enter password",
                style={"description_width": "initial"},
            )

        # Update the container (find and replace the password widget)
        connection_fields = self.main_widget.children[3]  # Connection fields VBox
        children_list = list(connection_fields.children)
        children_list[2] = self.password_widget  # Password is 3rd field
        connection_fields.children = tuple(children_list)

    def _load_existing_config(self):
        """Load existing configuration and populate connection dropdown."""
        # Load connection history for dropdown
        self._populate_connection_dropdown()

        # Load default configuration
        config = self.connection_manager.load_config_files()

        if config:
            # Pre-populate fields
            if config.get("host"):
                self.host_widget.value = config["host"]
            if config.get("username"):
                self.username_widget.value = config["username"]
            if config.get("group"):
                self.group_widget.value = config["group"]

            # Automatically try to load password from keychain if host and username are available
            host = config.get("host", "").strip()
            username = config.get("username", "").strip()
            password_loaded = False

            if host and username:
                password = self.connection_manager.load_password(host, username)
                if password:
                    self.password_widget.value = password
                    password_loaded = True
                    with self.status_output:
                        clear_output()
                        print("🔐 Password automatically loaded from keychain")

            # Show config source
            source = config.get("source", "configuration files")
            sources = [source]
            if password_loaded:
                sources.append("keychain")

            self.config_info.value = (
                f"<i>Pre-populated from {' + '.join(sources)}</i>"
            )
        else:
            self.config_info.value = "<i>💡 No existing configuration found</i>"

    def _populate_connection_dropdown(self):
        """Populate the connection dropdown with saved connections."""
        connections = self.connection_manager.get_connection_list()

        # Create dropdown options
        options = [("Manual entry", None)]

        for conn in connections:
            display_text = (
                f"{conn['display_name']} (last used: {conn['last_used_display']})"
            )
            options.append((display_text, conn))

        # Update dropdown
        self.connection_dropdown.options = options

        # Check if current form fields match any saved connection
        current_host = self.host_widget.value.strip()
        current_username = self.username_widget.value.strip()

        if current_host and current_username:
            # Look for matching connection
            for display_text, conn in options[1:]:  # Skip 'Manual entry'
                if (
                    conn
                    and conn["host"] == current_host
                    and conn["username"] == current_username
                ):
                    self.connection_dropdown.value = conn
                    self.delete_connection_button.disabled = False
                    return

        # If no match found, select manual entry
        self.connection_dropdown.value = None
        self.delete_connection_button.disabled = True

    def _load_from_keychain(self, button):
        """Load password from keychain."""
        with self.status_output:
            clear_output()

            host = self.host_widget.value.strip()
            username = self.username_widget.value.strip()

            if not host or not username:
                print("Please enter host and username first")
                return

            password = self.connection_manager.load_password(host, username)
            if password:
                self.password_widget.value = password
                print("Password loaded from keychain")
            else:
                print("No password found in keychain for this host/username")

    def _test_connection(self, button):
        """Test the OMERO connection."""
        with self.status_output:
            clear_output()

            config = self._get_widget_config()
            if not self._validate_config(config):
                return

            print("🔌 Testing connection...")
            success, message = self.connection_manager.test_connection(
                config["host"],
                config["username"],
                config["password"],
                config["group"],
                config["secure"],
            )

            if success:
                print(f"{message}")
            else:
                print(f"{message}")

    def _connect(self, button):
        """Create connection and optionally save password if requested."""
        with self.status_output:
            clear_output()

            config = self._get_widget_config()
            if not self._validate_config(config):
                return

            print("🔌 Creating connection...")
            self.connection = self.connection_manager.create_connection_from_config(
                config
            )

            if self.connection:
                print("Connection created and ready to use!")
                # Show user info
                print(f"👤 User: {self.connection.getUser().getName()}")
                print(f"🏢 Group: {self.connection.getGroupFromContext().getName()}")
                print("💾 Connection details saved to history")
                
                # Show password saving status
                if config["save_password"]:
                    expire_text = f" (expires in {config['expire_hours']} hours)" if config['expire_hours'] else " (no expiration)"
                    print(f"🔐 Password saved to keychain{expire_text}")
                else:
                    print("🔓 Password not saved (keychain saving was not requested)")
            else:
                print("❌ Failed to create connection")

    def _save_and_connect(self, button):
        """Alias for _connect method for backward compatibility."""
        return self._connect(button)

    def _get_widget_config(self) -> Dict[str, Any]:
        """Get configuration from widget values."""
        return {
            "host": self.host_widget.value.strip(),
            "username": self.username_widget.value.strip(),
            "password": self.password_widget.value.strip(),
            "group": self.group_widget.value.strip(),
            "secure": self.secure_widget.value,
            "save_password": self.save_password_widget.value,
            "expire_hours": self.expire_widget.value,
        }

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        if not config["host"]:
            print("❌ Host is required")
            return False
        if not config["username"]:
            print("❌ Username is required")
            return False
        if not config["password"]:
            print("❌ Password is required")
            return False
        return True

    def _on_connection_selected(self, change):
        """Handle connection selection from dropdown."""
        selected_connection = change["new"]

        if selected_connection is None:
            # Manual entry selected
            self.delete_connection_button.disabled = True
            return

        # Populate fields from selected connection
        self.host_widget.value = selected_connection["host"]
        self.username_widget.value = selected_connection["username"]
        self.group_widget.value = selected_connection.get("group", "") or ""

        # Enable delete button
        self.delete_connection_button.disabled = False

        # Try to load password from keychain
        host = selected_connection["host"]
        username = selected_connection["username"]
        password = self.connection_manager.load_password(host, username)

        if password:
            self.password_widget.value = password
            with self.status_output:
                clear_output()
                print(f"🔐 Password loaded from keychain for {username}@{host}")
        else:
            self.password_widget.value = ""
            with self.status_output:
                clear_output()
                print(f"💡 No saved password found for {username}@{host}")

    def _save_connection_only(self, button):
        """Save connection details without creating a connection."""
        with self.status_output:
            clear_output()

            host = self.host_widget.value.strip()
            username = self.username_widget.value.strip()
            group = self.group_widget.value.strip() or None

            if not host or not username:
                print("❌ Host and username are required to save connection")
                return

            # Save connection details
            success = self.connection_manager.save_connection_details(
                host, username, group
            )

            if success:
                # Save password if requested
                if self.save_password_widget.value:
                    password = self.password_widget.value.strip()
                    if password:
                        expire_hours = self.expire_widget.value
                        self.connection_manager.save_password(
                            host, username, password, expire_hours
                        )
                        expire_text = f" (expires in {expire_hours} hours)" if expire_hours else " (no expiration)"
                        print(f"🔐 Password saved to keychain{expire_text}")
                    else:
                        print("⚠️ Password not saved - password field is empty")
                else:
                    print("🔓 Password not saved to keychain (not requested)")

                # Refresh dropdown
                self._populate_connection_dropdown()
                print("✅ Connection saved successfully!")
            else:
                print("❌ Failed to save connection")

    def _delete_connection(self, button):
        """Delete the selected connection."""
        with self.status_output:
            clear_output()

            selected_connection = self.connection_dropdown.value
            if selected_connection is None:
                print("❌ No connection selected for deletion")
                return

            host = selected_connection["host"]
            username = selected_connection["username"]

            # Confirm deletion
            print(f"🗑️ Deleting connection: {username}@{host}")

            success = self.connection_manager.delete_connection(host, username)

            if success:
                # Refresh dropdown
                self._populate_connection_dropdown()

                # Clear fields
                self.host_widget.value = ""
                self.username_widget.value = ""
                self.password_widget.value = ""
                self.group_widget.value = ""

                # Select manual entry
                self.connection_dropdown.value = None

                print("✅ Connection deleted successfully!")
            else:
                print("❌ Failed to delete connection")

    def display(self):
        """Display the widget."""
        display(self.main_widget)

    def get_connection(self):
        """Get the current OMERO connection.

        Returns:
            BlitzGateway connection object or None
        """
        return self.connection

    def get_config(self) -> Dict[str, Any]:
        """Get the current widget configuration.

        Returns:
            Configuration dictionary
        """
        return self._get_widget_config()


def create_omero_connection_widget() -> OMEROConnectionWidget:
    """Create an OMERO connection widget.

    Returns:
        OMEROConnectionWidget instance
    """
    return OMEROConnectionWidget()
