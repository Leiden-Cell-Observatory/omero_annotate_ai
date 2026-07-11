"""Interactive widget for OMERO server connections with keychain support."""

from typing import Any, Dict, Optional

import ipywidgets as widgets
from IPython.display import clear_output, display

from ..omero.simple_connection import SimpleOMEROConnection


class OMEROConnectionWidget:
    """Interactive widget for creating OMERO connections with secure password storage."""

    DEFAULT_PORT = 4064

    def __init__(self):
        """Initialize the OMERO connection widget."""
        self.connection_manager = SimpleOMEROConnection()
        self.connection = None
        self._connections_by_key: Dict[str, Dict[str, Any]] = {}
        self._suppress_dropdown_events = False
        self._create_widgets()
        self._setup_observers()
        self._load_existing_config()

    def _create_widgets(self):
        """Create all widget components."""

        # Header
        self.header = widgets.HTML(
            value="<h3>🔌 OMERO Server Connection</h3>",
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

        self.port_widget = widgets.BoundedIntText(
            value=self.DEFAULT_PORT,
            min=1,
            max=65535,
            description="Port:",
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
            tooltip="When checked, your password will be securely saved and auto-loaded for future connections to this server",
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
                self.port_widget,
                self.username_widget,
                self.password_widget,
                self.group_widget,
                self.secure_widget,
            ]
        )

        password_options = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Password Options</b><br><i>Note: Passwords are only saved to keychain when explicitly requested</i>"
                ),
                self.save_password_widget,
                self.expire_widget,
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

    @staticmethod
    def _connection_key(host: str, username: str) -> str:
        """Build the stable identifier used for a saved connection."""
        return f"{host}:{username}"

    @staticmethod
    def _expire_text(expire_hours: Optional[int]) -> str:
        """Describe a keychain expiry period for display."""
        if expire_hours:
            return f" (expires in {expire_hours} hours)"
        return " (no expiration)"

    def _load_existing_config(self):
        """Load existing configuration and pre-populate the connection fields."""
        config = self.connection_manager.load_config_files() or {}

        if not config:
            self.config_info.value = "<i>💡 No existing configuration found</i>"
            self._populate_connection_dropdown()
            return

        # `config.get(key, "")` is not enough here: a key can be present with a
        # None value when only some of the fields are set in .env.
        self.host_widget.value = (config.get("host") or "").strip()
        self.username_widget.value = (config.get("username") or "").strip()
        self.group_widget.value = (config.get("group") or "").strip()

        port = self._parse_port(config.get("port"))
        if port is not None:
            self.port_widget.value = port

        # The password is deliberately not loaded here. ipywidgets stores widget
        # values in the notebook's saved state, so auto-filling it would put a
        # password the user never typed into the .ipynb. Use "Load from Keychain".
        source = config.get("source", "configuration files")
        self.config_info.value = f"<i>Pre-populated from {source}</i>"

        self._populate_connection_dropdown()

    def _parse_port(self, value: Any) -> Optional[int]:
        """Coerce a port from a config file into a value the widget accepts."""
        try:
            port = int(value)
        except (TypeError, ValueError):
            return None
        return port if 1 <= port <= 65535 else None

    def _populate_connection_dropdown(self):
        """Populate the connection dropdown with saved connections."""
        connections = self.connection_manager.get_connection_list()

        # Dropdown values are stable "host:username" keys rather than the
        # connection dicts themselves, which are rebuilt on every refresh.
        self._connections_by_key = {
            self._connection_key(conn["host"], conn["username"]): conn
            for conn in connections
        }

        options = [("Manual entry", None)]
        for key, conn in self._connections_by_key.items():
            display_text = (
                f"{conn['display_name']} (last used: {conn['last_used_display']})"
            )
            options.append((display_text, key))

        # Re-select whichever saved connection matches the current form fields
        current_key = self._connection_key(
            self.host_widget.value.strip(), self.username_widget.value.strip()
        )
        selected = current_key if current_key in self._connections_by_key else None

        # Refreshing the dropdown is not a user action, so it must not run the
        # selection handler - that would read the keychain behind the user's back.
        self._suppress_dropdown_events = True
        try:
            self.connection_dropdown.options = options
            self.connection_dropdown.value = selected
        finally:
            self._suppress_dropdown_events = False

        self.delete_connection_button.disabled = selected is None

    def _on_connection_selected(self, change):
        """Handle connection selection from dropdown."""
        if self._suppress_dropdown_events:
            return

        selected_key = change["new"]
        connection = (
            self._connections_by_key.get(selected_key) if selected_key else None
        )

        if connection is None:
            # Manual entry selected
            self.delete_connection_button.disabled = True
            return

        # Populate fields from selected connection
        self.host_widget.value = connection["host"]
        self.username_widget.value = connection["username"]
        self.group_widget.value = connection.get("group") or ""

        port = self._parse_port(connection.get("port"))
        self.port_widget.value = port if port is not None else self.DEFAULT_PORT

        self.delete_connection_button.disabled = False

        # Picking a connection is an explicit user action, so loading its stored
        # password here is expected rather than surprising.
        self.password_widget.value = ""
        with self.status_output:
            clear_output()
            try:
                self._load_password_into_field(
                    connection["host"], connection["username"]
                )
            except Exception as e:
                print(f"❌ Error loading password from keychain: {e}")

    def _load_password_into_field(self, host: str, username: str) -> bool:
        """Load a stored password from the keychain into the password field.

        Args:
            host: OMERO server host
            username: OMERO username

        Returns:
            True if a password was found and filled in, False otherwise
        """
        password = self.connection_manager.load_password(host, username)
        if not password:
            print(f"💡 No saved password found for {username}@{host}")
            return False

        self.password_widget.value = password
        print(f"🔐 Password loaded from keychain for {username}@{host}")
        return True

    def _load_from_keychain(self, button):
        """Load password from keychain."""
        with self.status_output:
            clear_output()
            try:
                host = self.host_widget.value.strip()
                username = self.username_widget.value.strip()

                if not host or not username:
                    print("❌ Please enter host and username first")
                    return

                self._load_password_into_field(host, username)

            except Exception as e:
                print(f"❌ Error loading password from keychain: {e}")

    def _test_connection(self, button):
        """Test the OMERO connection."""
        with self.status_output:
            clear_output()
            try:
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
                    config["port"],
                )

                print(f"{'✅' if success else '❌'} {message}")

            except Exception as e:
                print(f"❌ Error testing connection: {e}")

    def _connect(self, button):
        """Create connection and optionally save password if requested."""
        with self.status_output:
            clear_output()
            try:
                config = self._get_widget_config()
                if not self._validate_config(config):
                    return

                # Don't leak the previous gateway and its keep-alive thread
                self._close_existing_connection()

                print("🔌 Creating connection...")
                self.connection = (
                    self.connection_manager.create_connection_from_config(config)
                )

                if not self.connection:
                    print("❌ Failed to create connection")
                    return

                print("✅ Connection created and ready to use!")
                self._print_connection_details()
                print("💾 Connection details saved to history")
                self._persist_password(config)

            except Exception as e:
                print(f"❌ Error creating connection: {e}")

    def _close_existing_connection(self):
        """Close a connection this widget opened earlier, if any."""
        if self.connection is None:
            return

        try:
            self.connection.close()
        except Exception as e:
            print(f"⚠️ Could not close previous connection: {e}")
        finally:
            self.connection = None

    def _print_connection_details(self):
        """Show who we connected as, without failing an otherwise good connection."""
        try:
            print(f"👤 User: {self.connection.getUser().getName()}")
            print(f"🏢 Group: {self.connection.getGroupFromContext().getName()}")
        except Exception as e:
            print(f"⚠️ Connected, but could not read user/group details: {e}")

    def _persist_password(self, config: Dict[str, Any]):
        """Save the password to the keychain if the user asked for it."""
        if not config["save_password"]:
            print("🔓 Password not saved (keychain saving was not requested)")
            return

        saved = self.connection_manager.save_password(
            config["host"],
            config["username"],
            config["password"],
            config["expire_hours"],
        )

        if saved:
            expire_text = self._expire_text(config["expire_hours"])
            print(f"🔐 Password saved to keychain{expire_text}")
        else:
            print("⚠️ Password could NOT be saved to keychain")

    def _get_widget_config(self) -> Dict[str, Any]:
        """Get configuration from widget values, including the plaintext password."""
        return {
            "host": self.host_widget.value.strip(),
            "username": self.username_widget.value.strip(),
            # Not stripped: whitespace can be a legitimate part of a password.
            "password": self.password_widget.value,
            "group": self.group_widget.value.strip(),
            "port": self.port_widget.value,
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

    def _save_connection_only(self, button):
        """Save connection details without creating a connection."""
        with self.status_output:
            clear_output()
            try:
                host = self.host_widget.value.strip()
                username = self.username_widget.value.strip()
                group = self.group_widget.value.strip() or None
                port = self.port_widget.value

                if not host or not username:
                    print("❌ Host and username are required to save connection")
                    return

                if not self.connection_manager.save_connection_details(
                    host, username, group, port
                ):
                    print("❌ Failed to save connection")
                    return

                self._save_password_if_requested(host, username)

                # Refresh dropdown
                self._populate_connection_dropdown()
                print("✅ Connection saved successfully!")

            except Exception as e:
                print(f"❌ Error saving connection: {e}")

    def _save_password_if_requested(self, host: str, username: str):
        """Save the current password to the keychain if the user asked for it."""
        if not self.save_password_widget.value:
            print("🔓 Password not saved to keychain (not requested)")
            return

        password = self.password_widget.value
        if not password:
            print("⚠️ Password not saved - password field is empty")
            return

        expire_hours = self.expire_widget.value
        if self.connection_manager.save_password(
            host, username, password, expire_hours
        ):
            print(f"🔐 Password saved to keychain{self._expire_text(expire_hours)}")
        else:
            print("⚠️ Password could NOT be saved to keychain")

    def _delete_connection(self, button):
        """Delete the selected connection."""
        with self.status_output:
            clear_output()
            try:
                selected_key = self.connection_dropdown.value
                connection = (
                    self._connections_by_key.get(selected_key) if selected_key else None
                )

                if connection is None:
                    print("❌ No connection selected for deletion")
                    return

                host = connection["host"]
                username = connection["username"]

                print(f"🗑️ Deleting connection: {username}@{host}")

                if not self.connection_manager.delete_connection(host, username):
                    print("❌ Failed to delete connection")
                    return

                # Clear fields before refreshing, so nothing re-selects the entry
                self.host_widget.value = ""
                self.username_widget.value = ""
                self.password_widget.value = ""
                self.group_widget.value = ""
                self.port_widget.value = self.DEFAULT_PORT

                self._populate_connection_dropdown()
                print("✅ Connection deleted successfully!")

            except Exception as e:
                print(f"❌ Error deleting connection: {e}")

    def display(self):
        """Display the widget."""
        display(self.main_widget)

    def get_connection(self):
        """Get the current OMERO connection.

        Returns:
            BlitzGateway connection object or None
        """
        return self.connection

    def get_config(self, include_password: bool = False) -> Dict[str, Any]:
        """Get the current widget configuration.

        The password is left out by default: printing the returned dictionary in
        a notebook would otherwise write it into the saved .ipynb.

        Args:
            include_password: Include the plaintext password in the result

        Returns:
            Configuration dictionary
        """
        config = self._get_widget_config()
        if not include_password:
            config.pop("password", None)
        return config


def create_omero_connection_widget() -> OMEROConnectionWidget:
    """Create an OMERO connection widget.

    Returns:
        OMEROConnectionWidget instance
    """
    return OMEROConnectionWidget()
