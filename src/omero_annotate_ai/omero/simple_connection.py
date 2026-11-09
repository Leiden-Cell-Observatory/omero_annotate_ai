"""Simple OMERO connection management.

Passwords are never stored: only host, port, username and group are kept in the
connection history.
"""

import configparser
import contextlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from omero.gateway import BlitzGateway

from ..errors import ConnectionFailed

logger = logging.getLogger(__name__)

_URL_HOST = re.compile(r"^wss?://", re.IGNORECASE)


def is_url_host(host: str | None) -> bool:
    """True for websocket URL hosts such as ``wss://server/omero-ws``."""
    return bool(host) and bool(_URL_HOST.match(host.strip()))


class SimpleOMEROConnection:
    """Simple OMERO connection manager; never stores passwords."""

    # Keys read from ~/.ezomero, mapped to their config names. A ``password``
    # entry in that file is deliberately not listed: passwords are never stored,
    # and copying one into the config would spread it to every caller.
    EZOMERO_KEYS = {
        "host": "host",
        "user": "username",
        "group": "group",
        "port": "port",
    }

    def __init__(self):
        """Initialize the connection manager."""
        self.last_connection = None

    def load_config_files(self) -> dict[str, Any]:
        """Load configuration from .env, connection history and .ezomero files.

        Priority order: .env -> connection history -> .ezomero

        The returned dictionary never contains a password: it is always
        entered by the user.

        Returns:
            Dictionary with configuration parameters
        """
        return (
            self._load_env_config()
            or self._load_history_config()
            or self._load_ezomero_config()
        )

    @staticmethod
    def _coerce_port(value: Any) -> int | None:
        """Convert a port read from a config file or environment to an int."""
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _load_env_config(self) -> dict[str, Any]:
        """Read connection settings from a .env file in the working directory.

        Returns:
            Dictionary with configuration parameters, empty if none found
        """
        env_path = Path(".env")
        if not env_path.exists():
            return {}

        try:
            # dotenv_values, not load_dotenv: the file may hold a password, and
            # load_dotenv would export every entry into os.environ where any
            # subprocess of this kernel could read it.
            values = dotenv_values(env_path)

            def lookup(key: str) -> str | None:
                return values.get(key) or os.environ.get(key)

            host = lookup("HOST")
            username = lookup("USER_NAME")
            if not host and not username:
                return {}

            config: dict[str, Any] = {
                "host": host,
                "username": username,
                "group": lookup("GROUP"),
                "source": ".env file",
            }

            port = self._coerce_port(lookup("PORT"))
            if port is not None:
                config["port"] = port

            logger.info("Loaded configuration from .env file")
            return config

        except Exception as e:  # noqa: BLE001 - optional local config must never block connecting
            logger.warning(f"Could not load .env file: {e}")
            return {}

    def _load_history_config(self) -> dict[str, Any]:
        """Read connection settings from the most recent saved connection.

        Returns:
            Dictionary with configuration parameters, empty if none found
        """
        try:
            connections = self.get_connection_list()
            if not connections:
                return {}

            recent = connections[0]
            # Only the source is printed: notebook outputs get committed, and
            # user@host would end up in the repository.
            logger.info("Loaded configuration from connection history")
            config: dict[str, Any] = {
                "host": recent["host"],
                "username": recent["username"],
                "group": recent.get("group"),
                "source": (
                    f"connection history (last used: {recent['last_used_display']})"
                ),
            }

            port = self._coerce_port(recent.get("port"))
            if port is not None:
                config["port"] = port

            return config

        except Exception as e:  # noqa: BLE001 - optional local config must never block connecting
            logger.warning(f"Could not load connection history: {e}")
            return {}

    def _load_ezomero_config(self) -> dict[str, Any]:
        """Read connection settings from ~/.ezomero.

        Only the keys in :attr:`EZOMERO_KEYS` are read, so a ``password`` entry
        in that file is ignored rather than copied into the returned config.

        Returns:
            Dictionary with configuration parameters, empty if none found
        """
        try:
            # Path.home() raises RuntimeError on Windows when the environment
            # has no USERPROFILE/HOMEPATH, so it belongs inside the guard.
            ezomero_path = Path.home() / ".ezomero"
            if not ezomero_path.exists():
                return {}

            parser = configparser.ConfigParser()
            parser.read(ezomero_path)

            if "default" not in parser:
                return {}

            config: dict[str, Any] = {}
            for file_key, config_key in self.EZOMERO_KEYS.items():
                value = parser["default"].get(file_key)
                if value and value.strip():
                    config[config_key] = value.strip()

            if not config:
                return {}

            port = self._coerce_port(config.pop("port", None))
            if port is not None:
                config["port"] = port

            config["source"] = ".ezomero file"
            logger.info("Loaded configuration from .ezomero file")
            return config

        except Exception as e:  # noqa: BLE001 - optional local config must never block connecting
            logger.warning(f"Could not load .ezomero file: {e}")
            return {}

    def connect(
        self,
        host: str,
        username: str,
        password: str,
        group: str | None = None,
        secure: bool = True,
        verbose: bool = True,
        port: int | None = None,
    ) -> Any:
        """Create OMERO connection.

        Args:
            host: OMERO server host, or a websocket URL (``wss://server/omero-ws``)
            username: OMERO username
            password: OMERO password
            group: OMERO group (optional)
            secure: Keep data transfer encrypted after login (default True).
                ``False`` switches to plain TCP after an encrypted login, which
                cannot work over a websocket, so it is ignored for URL hosts.
            verbose: Print connection messages (default True)
            port: OMERO server port. Ignored for URL hosts, which carry their own.

        Returns:
            The connected BlitzGateway.

        Raises:
            ConnectionFailed: The server cannot be reached or the login failed.
        """
        if is_url_host(host):
            secure = True
            port = None

        conn = None
        try:
            if verbose:
                host_display = f"{host}:{port}" if port else host
                logger.info(f"Connecting to OMERO server: {host_display}")

            # Create BlitzGateway connection
            bg_kwargs: dict[str, Any] = {
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
                    user, user_group = conn.getUser(), conn.getGroupFromContext()
                    logger.info("Connected to OMERO Server")
                    logger.info(f"User: {user.getName() if user else '?'}")
                    logger.info(f"Group: {user_group.getName() if user_group else '?'}")

                # Keep the session alive while the user annotates.
                if conn.c is not None:
                    conn.c.enableKeepAlive(60)

                self.last_connection = conn
                return conn
        except Exception as e:  # noqa: BLE001 - any gateway/Ice failure is a failed connection
            self._close_quietly(conn)
            raise ConnectionFailed(f"Could not connect to OMERO: {e}") from e
        self._close_quietly(conn)
        raise ConnectionFailed(
            "Could not log in to OMERO: check the server, username and password"
        )

    def join_session(
        self, host: str, session_key: str, port: int | None = None, secure: bool = True
    ) -> Any:
        """Join an existing OMERO session by its key (e.g. from OMERO.web or a token).

        Raises:
            ConnectionFailed: The session does not exist (any more) or the server
                cannot be reached.
        """
        if is_url_host(host):
            secure, port = True, None
        kwargs: dict[str, Any] = {"host": host, "secure": secure}
        if port is not None:
            kwargs["port"] = port
        conn = None
        try:
            conn = BlitzGateway(**kwargs)
            if conn.connect(sUuid=session_key):
                if conn.c is not None:
                    conn.c.enableKeepAlive(60)
                self.last_connection = conn
                return conn
        except Exception as e:  # noqa: BLE001 - any gateway/Ice failure is a failed join
            self._close_quietly(conn)
            raise ConnectionFailed(f"Could not join the OMERO session: {e}") from e
        self._close_quietly(conn)
        raise ConnectionFailed("Could not join the OMERO session: it may have expired")

    @staticmethod
    def _close_quietly(conn: Any | None) -> None:
        """Close a gateway after a failed connect, so no session is left open."""
        if conn is None:
            return
        with contextlib.suppress(
            Exception
        ):  # closing a dead gateway may raise anything
            conn.close()

    def test_connection(
        self,
        host: str,
        username: str,
        password: str,
        group: str | None = None,
        secure: bool = True,
        port: int | None = None,
    ) -> tuple[bool, str]:
        """Test OMERO connection without keeping it open.

        Args:
            host: OMERO server host
            username: OMERO username
            password: OMERO password
            group: OMERO group (optional)
            secure: Use secure connection (default True)
            port: OMERO server port (optional)

        Returns:
            Tuple of (success, message)
        """
        try:
            conn = self.connect(host, username, password, group, secure, port=port)
        except ConnectionFailed as e:
            return False, str(e)
        conn.close()
        return True, "Connection successful"

    def create_connection_from_config(self, widget_config: dict[str, Any]) -> Any:
        """Create connection from widget configuration.

        The password is only used to log in: it is never written anywhere.
        Host, username, group and port are saved to the connection history.

        Args:
            widget_config: Configuration dictionary from widget

        Returns:
            The connected BlitzGateway.

        Raises:
            ConnectionFailed: A field is missing, or connecting failed.
        """
        host = (widget_config.get("host") or "").strip()
        username = (widget_config.get("username") or "").strip()
        # Not stripped: whitespace can be part of a password.
        password = widget_config.get("password") or ""
        group = (widget_config.get("group") or "").strip() or None
        secure = widget_config.get("secure", True)
        port = widget_config.get("port")

        # Validate required fields
        if not host or not username or not password:
            raise ConnectionFailed("Host, username and password are required")

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

        # Remember host/username/group/port (never the password) for next time
        self.save_connection_details(host, username, group, port, verbose=False)

        return connection

    def get_last_connection(self) -> BlitzGateway | None:
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
        group: str | None = None,
        port: int | None = None,
        verbose: bool = True,
    ) -> bool:
        """Save connection details to history file.

        Never stores a password.

        Args:
            host: OMERO server host
            username: OMERO username
            group: OMERO group (optional)
            port: OMERO server port (optional)
            verbose: Print a confirmation message

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            connections_file = self._get_connections_file()

            # Load existing connections
            connections = []
            if connections_file.exists():
                with connections_file.open() as f:
                    connections = json.load(f)

            # Create connection entry
            connection_entry = {
                "host": host,
                "username": username,
                "group": group,
                "port": port,
                "last_used": datetime.now(UTC).isoformat(),
                "created_at": datetime.now(UTC).isoformat(),
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
            with connections_file.open("w") as f:
                json.dump(connections, f, indent=2)

            if verbose:
                logger.info("Connection details saved to history")
            return True

        except Exception as e:  # noqa: BLE001 - optional local config must never block connecting
            logger.error(f"Error saving connection details: {e}")
            return False

    def load_connection_history(self) -> list[dict[str, Any]]:
        """Load connection history from file.

        Returns:
            List of connection dictionaries, sorted by last_used (most recent first)
        """
        try:
            connections_file = self._get_connections_file()

            if not connections_file.exists():
                return []

            with connections_file.open() as f:
                connections = json.load(f)

            # Sort by last_used (most recent first)
            connections.sort(key=lambda x: x["last_used"], reverse=True)

            return connections

        except Exception as e:  # noqa: BLE001 - optional local config must never block connecting
            logger.error(f"Error loading connection history: {e}")
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
            with connections_file.open() as f:
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
                with connections_file.open("w") as f:
                    json.dump(connections, f, indent=2)

                logger.info("Connection deleted from history")
                return True
            logger.warning("Connection not found in history")
            return False

        except Exception as e:  # noqa: BLE001 - optional local config must never block connecting
            logger.error(f"Error deleting connection: {e}")
            return False

    def get_connection_list(self) -> list[dict[str, Any]]:
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
            except (ValueError, KeyError, TypeError, OSError):
                conn["last_used_display"] = "Unknown"

        return connections
