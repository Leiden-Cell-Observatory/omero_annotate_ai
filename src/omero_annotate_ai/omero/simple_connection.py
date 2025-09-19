"""Simple OMERO connection management with keychain support."""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from omero.gateway import BlitzGateway

from pathlib import Path

try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False


from dotenv import load_dotenv
from omero.gateway import BlitzGateway

class SimpleOMEROConnection:
    """Simple OMERO connection manager with keychain support for passwords."""

    SERVICE_NAME = "omero-annotate-ai"

    def __init__(self):
        """Initialize the connection manager."""
        self.last_connection = None

    def save_password(
        self,
        host: str,
        username: str,
        password: str,
        expire_hours: Optional[int] = None,
    ) -> bool:
        """Save password to keychain with optional expiration.

        Args:
            host: OMERO server host
            username: OMERO username
            password: OMERO password
            expire_hours: Hours until password expires (None = never expires)

        Returns:
            True if saved successfully, False otherwise
        """
        if not KEYRING_AVAILABLE:
            print("Warning: Keyring not available - password not saved")
            return False

        try:
            key = f"{host}:{username}"

            # Create password data with optional expiration
            password_data = {"password": password}
            if expire_hours is not None:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=expire_hours)
                password_data["expires_at"] = expires_at.isoformat()

            # Store in keychain
            keyring.set_password(self.SERVICE_NAME, key, json.dumps(password_data))

            if expire_hours:
                print(
                    f"Password saved to keychain (expires in {expire_hours} hours)"
                )
            else:
                print("Password saved to keychain (never expires)")
            return True

        except Exception as e:
            print(f"Failed to save password to keychain: {e}")
            return False

    def load_password(self, host: str, username: str) -> Optional[str]:
        """Load password from keychain and check expiration.

        Args:
            host: OMERO server host
            username: OMERO username

        Returns:
            Password if found and not expired, None otherwise
        """
        if not KEYRING_AVAILABLE:
            return None

        try:
            key = f"{host}:{username}"
            stored_data = keyring.get_password(self.SERVICE_NAME, key)

            if not stored_data:
                return None

            # Parse stored data
            password_data = json.loads(stored_data)
            password = password_data.get("password")
            expires_at = password_data.get("expires_at")

            # Check expiration
            if expires_at:
                expiry_time = datetime.fromisoformat(expires_at)
                if datetime.now(timezone.utc) > expiry_time:
                    # Password expired, remove it
                    self._delete_password(host, username)
                    print("Stored password has expired and was removed")
                    return None
                else:
                    print("Password loaded from keychain")
            else:
                print("Password loaded from keychain (no expiration)")

            return password

        except Exception as e:
            print(f"Error loading password from keychain: {e}")
            return None

    def _delete_password(self, host: str, username: str) -> bool:
        """Delete password from keychain.

        Args:
            host: OMERO server host
            username: OMERO username

        Returns:
            True if deleted successfully, False otherwise
        """
        if not KEYRING_AVAILABLE:
            return False

        try:
            key = f"{host}:{username}"
            keyring.delete_password(self.SERVICE_NAME, key)
            return True
        except Exception:
            return False

    def load_config_files(self) -> Dict[str, Any]:
        """Load configuration from connection history, .env and .ezomero files.

        Priority order: Connection history -> .env -> .ezomero

        Returns:
            Dictionary with configuration parameters
        """
        config = {}

        # Try to load .env file (higher priority than connection history for development)
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=False)
            if os.environ.get("HOST") or os.environ.get("USER_NAME"):
                config.update(
                    {
                        "host": os.environ.get("HOST"),
                        "username": os.environ.get("USER_NAME"),
                        "group": os.environ.get("GROUP"),
                        "source": ".env file",
                    }
                )
                print("Loaded configuration from .env file")

        # Try to load from connection history (if no .env file)
        if not config:
            connections = self.load_connection_history()
            if connections:
                # Use most recent connection
                recent_conn = connections[0]
                # Add display_name if not present
                if "display_name" not in recent_conn:
                    recent_conn["display_name"] = (
                        f"{recent_conn['username']}@{recent_conn['host']}"
                    )
                    if recent_conn.get("group"):
                        recent_conn["display_name"] += f" ({recent_conn['group']})"

                config.update(
                    {
                        "host": recent_conn["host"],
                        "username": recent_conn["username"],
                        "group": recent_conn.get("group"),
                        "source": f"connection history (last used: {recent_conn.get('last_used_display', 'unknown')})",
                    }
                )
                print(
                    f"Loaded configuration from connection history: {recent_conn['display_name']}"
                )

        # Try to load .ezomero file (if no other config found)
        if not config:
            try:
                # Save current environment to restore later
                original_env = {}
                env_keys = [
                    "OMERO_USER",
                    "OMERO_PASSWORD",
                    "OMERO_HOST",
                    "OMERO_GROUP",
                    "OMERO_PORT",
                ]
                for key in env_keys:
                    original_env[key] = os.environ.get(key)

                # Temporarily clear ezomero environment variables to force config file loading
                for key in env_keys:
                    if key in os.environ:
                        del os.environ[key]

                # This will attempt to load from .ezomero file and return None (no password)
                # We don't actually want to create a connection, just test config loading
                ezomero_home = Path.home() / ".ezomero"
                if ezomero_home.exists():
                    # Read .ezomero file directly
                    import configparser

                    parser = configparser.ConfigParser()
                    parser.read(ezomero_home)

                    if "default" in parser:
                        ezomero_config = dict(parser["default"])
                        # Update config with ezomero values
                        for key, value in ezomero_config.items():
                            if value and value.strip():
                                if key == "user":
                                    config["username"] = value
                                else:
                                    config[key] = value
                        config["source"] = ".ezomero file"
                        print("Loaded configuration from .ezomero file")

                # Restore original environment
                for key, value in original_env.items():
                    if value is not None:
                        os.environ[key] = value
                    elif key in os.environ:
                        del os.environ[key]

            except Exception as e:
                print(f"Could not load .ezomero file: {e}")

        return config

    def connect(
        self,
        host: str,
        username: str,
        password: str,
        group: Optional[str] = None,
        secure: bool = True,
        verbose: bool = True,
        port: Optional[int] = None,
    ) -> Optional[Any]:
        """Create OMERO connection.

        Args:
            host: OMERO server host
            username: OMERO username
            password: OMERO password
            group: OMERO group (optional)
            secure: Use secure connection (default True)
            verbose: Print connection messages (default True)

        Returns:
            BlitzGateway connection object if successful, None otherwise
        """
        try:
            if verbose:
                host_display = f"{host}:{port}" if port else host
                print(f"Connecting to OMERO server: {host_display}")

            # Create BlitzGateway connection
            bg_kwargs: Dict[str, Any] = {
                "host": host,
                "username": username,
                "passwd": password,
                "group": group,
                "secure": secure,
            }
            if port is not None:
                bg_kwargs["port"] = port

            conn = BlitzGateway(**bg_kwargs)

            # Test connection
            if conn.connect():
                if verbose:
                    print("Connected to OMERO Server")
                    print(f"User: {conn.getUser().getName()}")
                    print(f"Group: {conn.getGroupFromContext().getName()}")

                # Enable keep-alive
                conn.c.enableKeepAlive(60)

                self.last_connection = conn
                return conn
            else:
                print("Connection to OMERO Server Failed")
                return None

        except Exception as e:
            print(f"Error connecting to OMERO: {e}")
            return None

    def test_connection(
        self,
        host: str,
        username: str,
        password: str,
        group: Optional[str] = None,
        secure: bool = True,
    ) -> Tuple[bool, str]:
        """Test OMERO connection without keeping it open.

        Args:
            host: OMERO server host
            username: OMERO username
            password: OMERO password
            group: OMERO group (optional)
            secure: Use secure connection (default True)

        Returns:
            Tuple of (success, message)
        """
        try:
            conn = self.connect(host, username, password, group, secure)
            if conn:
                # Close test connection
                conn.close()
                return True, "Connection successful"
            else:
                return False, "Connection failed"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def create_connection_from_config(
        self, widget_config: Dict[str, Any]
    ) -> Optional[Any]:
        """Create connection from widget configuration.

        Args:
            widget_config: Configuration dictionary from widget

        Returns:
            BlitzGateway connection object if successful, None otherwise
        """
        host = widget_config.get("host", "").strip()
        username = widget_config.get("username", "").strip()
        password = widget_config.get("password", "").strip()
        group = widget_config.get("group", "").strip() or None
        secure = widget_config.get("secure", True)
        port = widget_config.get("port")
        save_password = widget_config.get("save_password", False)
        expire_hours = widget_config.get("expire_hours")

        # Validate required fields
        if not host or not username or not password:
            print("Host, username, and password are required")
            return None

        # Create connection first to validate it works
        connection = self.connect(
            host,
            username,
            password,
            group,
            secure,
            verbose=False,
            port=port,
        )

        if connection:
            # Save password to keychain if requested
            if save_password:
                self.save_password(host, username, password, expire_hours)

            # Always save connection details for successful connections
            self.save_connection_details(host, username, group, verbose=False)

        return connection

    def get_last_connection(self) -> Optional["BlitzGateway"]:
        """Get the last successful connection.

        Returns:
            Last BlitzGateway connection or None
        """
        return self.last_connection

    def _get_config_dir(self) -> Path:
        """Get the configuration directory for connection history.

        Returns:
            Path to configuration directory
        """
        config_dir = Path.home() / ".omero-annotate-ai"
        config_dir.mkdir(exist_ok=True)
        return config_dir

    def _get_connections_file(self) -> Path:
        """Get the path to the connections history file.

        Returns:
            Path to connections.json file
        """
        return self._get_config_dir() / "connections.json"

    def save_connection_details(
        self,
        host: str,
        username: str,
        group: Optional[str] = None,
        verbose: bool = True,
    ) -> bool:
        """Save connection details to history file.

        Args:
            host: OMERO server host
            username: OMERO username
            group: OMERO group (optional)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            connections_file = self._get_connections_file()

            # Load existing connections
            connections = []
            if connections_file.exists():
                with open(connections_file, "r") as f:
                    connections = json.load(f)

            # Create connection entry
            connection_entry = {
                "host": host,
                "username": username,
                "group": group,
                "last_used": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Check if connection already exists
            connection_key = f"{host}:{username}"
            existing_index = None
            for i, conn in enumerate(connections):
                if f"{conn['host']}:{conn['username']}" == connection_key:
                    existing_index = i
                    break

            if existing_index is not None:
                # Update existing connection
                connection_entry["created_at"] = connections[existing_index][
                    "created_at"
                ]
                connections[existing_index] = connection_entry
            else:
                # Add new connection
                connections.append(connection_entry)

            # Sort by last_used (most recent first)
            connections.sort(key=lambda x: x["last_used"], reverse=True)

            # Save to file
            with open(connections_file, "w") as f:
                json.dump(connections, f, indent=2)

            if verbose:
                print(f"Connection details saved to history")
            return True

        except Exception as e:
            print(f"Error saving connection details: {e}")
            return False

    def load_connection_history(self) -> List[Dict[str, Any]]:
        """Load connection history from file.

        Returns:
            List of connection dictionaries, sorted by last_used (most recent first)
        """
        try:
            connections_file = self._get_connections_file()

            if not connections_file.exists():
                return []

            with open(connections_file, "r") as f:
                connections = json.load(f)

            # Sort by last_used (most recent first)
            connections.sort(key=lambda x: x["last_used"], reverse=True)

            return connections

        except Exception as e:
            print(f"Error loading connection history: {e}")
            return []

    def delete_connection(self, host: str, username: str) -> bool:
        """Delete a connection from history.

        Args:
            host: OMERO server host
            username: OMERO username

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            connections_file = self._get_connections_file()

            if not connections_file.exists():
                return False

            # Load existing connections
            with open(connections_file, "r") as f:
                connections = json.load(f)

            # Find and remove connection
            connection_key = f"{host}:{username}"
            original_count = len(connections)
            connections = [
                conn
                for conn in connections
                if f"{conn['host']}:{conn['username']}" != connection_key
            ]

            if len(connections) < original_count:
                # Save updated connections
                with open(connections_file, "w") as f:
                    json.dump(connections, f, indent=2)

                # Also delete password from keychain
                self._delete_password(host, username)

                print(f"Connection deleted from history and keychain")
                return True
            else:
                print(f"Connection not found in history")
                return False

        except Exception as e:
            print(f"Error deleting connection: {e}")
            return False

    def get_connection_list(self) -> List[Dict[str, Any]]:
        """Get a formatted list of connections for display.

        Returns:
            List of connection dictionaries with display names
        """
        connections = self.load_connection_history()

        for conn in connections:
            # Create display name
            display_parts = [conn["username"], "@", conn["host"]]
            if conn.get("group"):
                display_parts.extend([" (", conn["group"], ")"])

            conn["display_name"] = "".join(display_parts)

            # Format last used date
            try:
                last_used = datetime.fromisoformat(conn["last_used"])
                conn["last_used_display"] = last_used.strftime("%Y-%m-%d %H:%M")
            except:
                conn["last_used_display"] = "Unknown"

        return connections


def create_simple_omero_connection() -> SimpleOMEROConnection:
    """Create a simple OMERO connection manager.

    Returns:
        SimpleOMEROConnection instance
    """
    return SimpleOMEROConnection()
