"""Tests for SimpleOMEROConnection credential and config handling.

These are unit tests: no OMERO server is contacted. They focus on the paths
where a secret could escape - the connection history, ``.env`` and
``~/.ezomero``. Passwords are never stored anywhere.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from omero_annotate_ai.errors import ConnectionFailed

try:
    import omero_annotate_ai.omero.simple_connection as sc

    CONNECTION_AVAILABLE = True
except ImportError:
    CONNECTION_AVAILABLE = False

LOGGER = "omero_annotate_ai.omero.simple_connection"


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not CONNECTION_AVAILABLE, reason="OMERO dependencies not available"
    ),
]


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """A connection manager with an isolated home directory.

    The cwd is already isolated by the autouse ``isolated_cwd`` fixture, so the
    ``.env`` files these tests write land in a throwaway directory.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    return sc.SimpleOMEROConnection()


class TestEnvConfig:
    """.env is read, but never exported into the process environment."""

    def test_password_in_env_file_is_not_leaked(self, manager, monkeypatch):
        monkeypatch.delenv("PASSWORD", raising=False)
        Path(".env").write_text(
            "HOST=omero.example.org\nUSER_NAME=alice\nPASSWORD=s3cret\n"
        )

        config = manager.load_config_files()

        assert config["host"] == "omero.example.org"
        assert config["username"] == "alice"
        # The password must reach neither the config dict nor os.environ, where
        # every subprocess of the kernel would be able to read it.
        assert "password" not in config
        assert "PASSWORD" not in os.environ

    def test_env_file_does_not_pollute_os_environ(self, manager, monkeypatch):
        monkeypatch.delenv("HOST", raising=False)
        Path(".env").write_text("HOST=omero.example.org\nUSER_NAME=alice\n")

        manager.load_config_files()

        assert "HOST" not in os.environ

    def test_port_is_parsed_to_int(self, manager):
        Path(".env").write_text("HOST=omero.example.org\nUSER_NAME=alice\nPORT=6064\n")

        assert manager.load_config_files()["port"] == 6064

    def test_username_without_host(self, manager):
        """Regression: a partial .env must not crash the caller."""
        Path(".env").write_text("USER_NAME=alice\n")

        config = manager.load_config_files()

        assert config["username"] == "alice"
        assert config["host"] is None


class TestEzomeroConfig:
    """~/.ezomero is read through a whitelist."""

    def test_password_in_ezomero_is_ignored(self, manager):
        (Path.home() / ".ezomero").write_text(
            "[default]\n"
            "host=omero.example.org\n"
            "user=alice\n"
            "group=lab\n"
            "port=6064\n"
            "password=s3cret\n"
        )

        config = manager.load_config_files()

        assert config["host"] == "omero.example.org"
        assert config["username"] == "alice"
        assert config["group"] == "lab"
        assert config["port"] == 6064
        assert "password" not in config

    def test_malformed_ezomero_preserves_the_environment(self, manager, monkeypatch):
        """Regression: a parse failure used to permanently drop OMERO_* env vars."""
        monkeypatch.setenv("OMERO_PASSWORD", "from-env")
        monkeypatch.setenv("OMERO_HOST", "omero.example.org")
        (Path.home() / ".ezomero").write_text("this is not valid ini [[[\n")

        assert manager.load_config_files() == {}

        assert os.environ["OMERO_PASSWORD"] == "from-env"
        assert os.environ["OMERO_HOST"] == "omero.example.org"

    def test_unresolvable_home_directory(self, manager, monkeypatch):
        """Regression: Path.home() raises on Windows with a cleared environment.

        tests/test_omero_integration.py wipes os.environ, which leaves Windows
        with no USERPROFILE/HOMEPATH to derive a home from. POSIX falls back to
        the pwd module, so this only ever failed on Windows CI.
        """

        def no_home():
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(Path, "home", staticmethod(no_home))

        assert manager.load_config_files() == {}


class TestConnectionHistory:
    """Connection history stores metadata only, never a password."""

    def test_history_file_holds_no_secret(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "connect", lambda *a, **kw: object())

        manager.create_connection_from_config(
            {
                "host": "omero.example.org",
                "username": "alice",
                "password": "s3cret",
                "group": "lab",
                "port": 6064,
            }
        )

        raw = (Path.home() / ".omero-annotate-ai" / "connections.json").read_text()
        assert "s3cret" not in raw
        entry = json.loads(raw)[0]
        assert "password" not in entry
        assert entry["host"] == "omero.example.org"
        assert entry["port"] == 6064
        assert all("password" not in conn for conn in manager.load_connection_history())

    def test_delete_connection_removes_the_history_entry(self, manager):
        manager.save_connection_details("omero.example.org", "alice")

        assert manager.delete_connection("omero.example.org", "alice") is True
        assert manager.load_connection_history() == []
        assert manager.delete_connection("omero.example.org", "alice") is False

    def test_history_round_trips_the_port(self, manager):
        manager.save_connection_details("omero.example.org", "alice", None, 6064)

        config = manager.load_config_files()

        assert config["port"] == 6064
        assert config["source"].startswith("connection history")

    def test_env_file_takes_priority_over_history(self, manager):
        manager.save_connection_details("history.example.org", "bob")
        Path(".env").write_text("HOST=env.example.org\nUSER_NAME=alice\n")

        assert manager.load_config_files()["host"] == "env.example.org"


class TestCreateConnectionFromConfig:
    """The manager connects, but does not quietly persist secrets."""

    def test_password_is_not_stripped(self, manager, monkeypatch):
        seen = {}

        def fake_connect(host, username, password, group=None, secure=True, **kwargs):
            seen["password"] = password
            return

        monkeypatch.setattr(manager, "connect", fake_connect)

        manager.create_connection_from_config(
            {
                "host": "omero.example.org",
                "username": "alice",
                "password": "  s3 cret  ",
            }
        )

        assert seen["password"] == "  s3 cret  "

    def test_never_persists_the_password(
        self, manager, monkeypatch, tmp_path, caplog, capsys
    ):
        """No file, keychain, log record or printed line holds the password."""
        import sys
        from unittest.mock import MagicMock

        keyring = MagicMock()
        monkeypatch.setitem(sys.modules, "keyring", keyring)
        monkeypatch.setattr(manager, "connect", lambda *a, **kw: object())
        caplog.set_level(logging.DEBUG)

        manager.create_connection_from_config(
            {
                "host": "omero.example.org",
                "username": "alice",
                "password": "s3cret-pw",
            }
        )

        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert written, "the connection history should have been written"
        for path in written:
            assert b"s3cret-pw" not in path.read_bytes(), path
        assert not keyring.mock_calls
        assert "s3cret-pw" not in caplog.text
        assert "s3cret-pw" not in capsys.readouterr().out

    def test_has_no_password_storage_api(self, manager):
        import inspect

        assert "keyring" not in inspect.getsource(sc)
        assert not [name for name in dir(manager) if "password" in name.lower()]


class TestHistoryMessage:
    def test_history_message_does_not_name_user_or_host(self, manager, caplog):
        manager.save_connection_details("omero.example.org", "alice", verbose=False)

        with caplog.at_level(logging.INFO, logger=LOGGER):
            manager.load_config_files()

        assert "connection history" in caplog.text
        assert "alice" not in caplog.text and "omero.example.org" not in caplog.text


class TestConnectGatewayArgs:
    """What connect() hands to BlitzGateway, and cleanup on failure."""

    @pytest.fixture
    def gateway(self, monkeypatch):
        from unittest.mock import MagicMock

        gw_cls = MagicMock()
        monkeypatch.setattr(sc, "BlitzGateway", gw_cls)
        return gw_cls

    @pytest.mark.parametrize(
        "host", ["wss://omero.example.org/omero-ws", "WS://omero.example.org/omero-ws"]
    )
    def test_url_host_forces_secure_and_drops_port(self, manager, gateway, host):
        manager.connect(host, "alice", "pw", secure=False, port=4064, verbose=False)

        kwargs = gateway.call_args.kwargs
        assert kwargs["secure"] is True
        assert "port" not in kwargs

    def test_plain_host_keeps_secure_and_port(self, manager, gateway):
        manager.connect(
            "omero.example.org", "alice", "pw", secure=False, port=4064, verbose=False
        )

        kwargs = gateway.call_args.kwargs
        assert kwargs["secure"] is False
        assert kwargs["port"] == 4064

    def test_gateway_closed_when_connect_returns_false(self, manager, gateway):
        gateway.return_value.connect.return_value = False

        with pytest.raises(ConnectionFailed, match="log in"):
            manager.connect("omero.example.org", "alice", "pw", verbose=False)
        gateway.return_value.close.assert_called_once()

    def test_gateway_closed_when_connect_raises(self, manager, gateway):
        gateway.return_value.connect.side_effect = RuntimeError("router refused")

        with pytest.raises(ConnectionFailed, match="router refused"):
            manager.connect("omero.example.org", "alice", "pw", verbose=False)
        gateway.return_value.close.assert_called_once()
