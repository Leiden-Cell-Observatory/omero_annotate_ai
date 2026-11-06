"""Tests for SimpleOMEROConnection credential and config handling.

These are unit tests: no OMERO server is contacted. They focus on the paths
where a secret could escape - the keychain, ``.env`` and ``~/.ezomero``.
"""

import json
import os
from pathlib import Path

import pytest

try:
    import omero_annotate_ai.omero.simple_connection as sc

    CONNECTION_AVAILABLE = True
except ImportError:
    CONNECTION_AVAILABLE = False


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not CONNECTION_AVAILABLE, reason="OMERO dependencies not available"
    ),
]


class FakeKeyring:
    """In-memory stand-in for the ``keyring`` module."""

    def __init__(self):
        self.store = {}

    def set_password(self, service, key, value):
        self.store[(service, key)] = value

    def get_password(self, service, key):
        return self.store.get((service, key))

    def delete_password(self, service, key):
        del self.store[(service, key)]

    def get_keyring(self):
        return self.backend


class _GoodBackend:
    priority = 5



@pytest.fixture
def fake_keyring(monkeypatch):
    """Replace the real OS keyring with an in-memory one."""
    keyring = FakeKeyring()
    keyring.backend = _GoodBackend()
    monkeypatch.setattr(sc, "keyring", keyring, raising=False)
    monkeypatch.setattr(sc, "KEYRING_AVAILABLE", True)
    return keyring


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


class TestKeychain:
    """Passwords are stored in the keychain and honour their expiry."""

    def test_save_and_load_round_trip(self, manager, fake_keyring):
        assert manager.save_password("omero.example.org", "alice", "  s3 cret  ")

        loaded = manager.load_password("omero.example.org", "alice")
        assert loaded == "  s3 cret  "

    def test_expired_password_is_dropped_and_deleted(self, manager, fake_keyring):
        manager.save_password("omero.example.org", "alice", "s3cret", expire_hours=-1)

        assert manager.load_password("omero.example.org", "alice") is None
        # The expired secret must be purged, not just hidden
        assert fake_keyring.store == {}

    def test_unexpired_password_survives(self, manager, fake_keyring):
        manager.save_password("omero.example.org", "alice", "s3cret", expire_hours=24)

        assert manager.load_password("omero.example.org", "alice") == "s3cret"

    def test_delete_connection_purges_the_password(self, manager, fake_keyring):
        manager.save_connection_details("omero.example.org", "alice")
        manager.save_password("omero.example.org", "alice", "s3cret")

        assert manager.delete_connection("omero.example.org", "alice")
        assert fake_keyring.store == {}

    def test_save_reports_failure_without_a_keyring(self, manager, monkeypatch):
        monkeypatch.setattr(sc, "KEYRING_AVAILABLE", False)

        assert manager.save_password("omero.example.org", "alice", "s3cret") is False


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

    def test_history_file_holds_no_secret(self, manager, fake_keyring):
        manager.save_connection_details("omero.example.org", "alice", "lab", 6064)
        manager.save_password("omero.example.org", "alice", "s3cret")

        raw = (Path.home() / ".omero-annotate-ai" / "connections.json").read_text()

        assert "s3cret" not in raw
        entry = json.loads(raw)[0]
        assert entry["host"] == "omero.example.org"
        assert entry["port"] == 6064

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
            return None

        monkeypatch.setattr(manager, "connect", fake_connect)

        manager.create_connection_from_config(
            {"host": "omero.example.org", "username": "alice", "password": "  s3 cret  "}
        )

        assert seen["password"] == "  s3 cret  "

    def test_does_not_write_to_the_keychain(self, manager, fake_keyring, monkeypatch):
        """Persisting the password is the caller's job, so it can report failures."""
        monkeypatch.setattr(manager, "connect", lambda *a, **kw: object())

        manager.create_connection_from_config(
            {
                "host": "omero.example.org",
                "username": "alice",
                "password": "s3cret",
                "save_password": True,
                "expire_hours": 24,
            }
        )

        assert fake_keyring.store == {}


def _backend(module, name="Keyring", priority=1):
    cls = type(name, (), {"priority": priority})
    cls.__module__ = module
    return cls()


class TestKeyringBackendChecks:
    """Saving refuses backends that keep nothing, and warns for plaintext ones."""

    @pytest.mark.parametrize("backend", [
        _backend("keyring.backends.fail", priority=0),
        _backend("keyring.backends.null", priority=-1),
        _backend("some.thirdparty", priority=0.5),
    ])
    def test_unusable_backend_refuses_to_save(self, manager, fake_keyring, backend, capsys):
        fake_keyring.backend = backend

        assert manager.save_password("omero.example.org", "alice", "s3cret") is False
        assert fake_keyring.store == {}
        assert "no usable keyring backend" in capsys.readouterr().out

    def test_plaintext_backend_saves_with_warning(self, manager, fake_keyring, capsys):
        fake_keyring.backend = _backend("keyrings.alt.file", "PlaintextKeyring")

        assert manager.save_password("omero.example.org", "alice", "s3cret") is True
        out = capsys.readouterr().out
        assert "unencrypted" in out
        assert "keyrings.alt.file.PlaintextKeyring" in out

    def test_read_back_mismatch_is_not_reported_as_saved(
        self, manager, fake_keyring, monkeypatch, capsys
    ):
        monkeypatch.setattr(fake_keyring, "set_password", lambda *a: None)

        assert manager.save_password("omero.example.org", "alice", "s3cret") is False
        assert "did not keep it" in capsys.readouterr().out

    def test_success_names_the_backend(self, manager, fake_keyring, capsys):
        assert manager.save_password("omero.example.org", "alice", "s3cret") is True
        assert "_GoodBackend" in capsys.readouterr().out


class TestForgetPassword:
    def test_delete_without_history_row_still_removes_password(self, manager, fake_keyring):
        manager.save_password("omero.example.org", "alice", "s3cret")

        assert manager.delete_connection("omero.example.org", "alice") is True
        assert fake_keyring.store == {}


class TestHistoryMessage:
    def test_history_message_does_not_name_user_or_host(self, manager, capsys):
        manager.save_connection_details("omero.example.org", "alice", verbose=False)

        manager.load_config_files()

        out = capsys.readouterr().out
        assert "connection history" in out
        assert "alice" not in out and "omero.example.org" not in out


class TestConnectGatewayArgs:
    """What connect() hands to BlitzGateway, and cleanup on failure."""

    @pytest.fixture
    def gateway(self, monkeypatch):
        from unittest.mock import MagicMock
        gw_cls = MagicMock()
        monkeypatch.setattr(sc, "BlitzGateway", gw_cls)
        return gw_cls

    @pytest.mark.parametrize("host", ["wss://omero.example.org/omero-ws",
                                      "WS://omero.example.org/omero-ws"])
    def test_url_host_forces_secure_and_drops_port(self, manager, gateway, host):
        manager.connect(host, "alice", "pw", secure=False, port=4064, verbose=False)

        kwargs = gateway.call_args.kwargs
        assert kwargs["secure"] is True
        assert "port" not in kwargs

    def test_plain_host_keeps_secure_and_port(self, manager, gateway):
        manager.connect("omero.example.org", "alice", "pw", secure=False,
                        port=4064, verbose=False)

        kwargs = gateway.call_args.kwargs
        assert kwargs["secure"] is False
        assert kwargs["port"] == 4064

    def test_gateway_closed_when_connect_returns_false(self, manager, gateway):
        gateway.return_value.connect.return_value = False

        assert manager.connect("omero.example.org", "alice", "pw", verbose=False) is None
        gateway.return_value.close.assert_called_once()

    def test_gateway_closed_when_connect_raises(self, manager, gateway):
        gateway.return_value.connect.side_effect = RuntimeError("router refused")

        assert manager.connect("omero.example.org", "alice", "pw", verbose=False) is None
        gateway.return_value.close.assert_called_once()
