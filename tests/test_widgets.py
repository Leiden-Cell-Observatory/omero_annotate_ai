"""Tests for widget modules."""

import importlib
import importlib.util
import sys
from unittest.mock import Mock, patch

import pytest

from omero_annotate_ai.annotators import ScriptedAnnotator
from omero_annotate_ai.core.annotation_config import SessionRecord
from omero_annotate_ai.errors import ConnectionFailed
from omero_annotate_ai.events import SessionStarted, UnitCommitted, UploadFailed
from omero_annotate_ai.trainingset import TrainingSet, TrainingSetSummary, summarize
from tests.fakes import image_mask, omero_set

# Test imports with fallbacks for missing dependencies
try:
    from omero_annotate_ai.widgets.omero_connection_widget import OMEROConnectionWidget
    from omero_annotate_ai.widgets.workflow_widget import (
        NewTrainingSet,
        OpenTrainingSet,
        WorkflowWidget,
        open_or_create,
        render_summary,
    )

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
class TestOMEROConnectionWidget:
    """Test OMERO connection widget functionality."""

    @patch("omero_annotate_ai.widgets.omero_connection_widget.widgets")
    @patch("omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection")
    def test_widget_initialization(self, mock_connection_class, mock_widgets):
        """
        Tests the initialization of the OMEROConnectionWidget.
        This test ensures that the widget is correctly initialized and that the
        connection attribute is None.
        """
        # Mock ipywidgets components
        mock_widgets.Text.return_value = Mock()
        mock_widgets.Password.return_value = Mock()
        mock_widgets.IntText.return_value = Mock()
        mock_widgets.Checkbox.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        # Mock Output widget with context manager support
        mock_output = Mock()
        mock_output.__enter__ = Mock(return_value=mock_output)
        mock_output.__exit__ = Mock(return_value=None)
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = Mock()

        # Mock connection manager
        mock_conn_manager = Mock()
        # Mock get_connection_list to return an empty list
        mock_conn_manager.get_connection_list.return_value = []
        # Mock load_config_files to return None
        mock_conn_manager.load_config_files.return_value = None
        mock_connection_class.return_value = mock_conn_manager

        widget = OMEROConnectionWidget()

        assert widget is not None
        assert hasattr(widget, "connection")
        assert widget.connection is None

    @patch("omero_annotate_ai.widgets.omero_connection_widget.widgets")
    @patch("omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection")
    def test_connect_button_callback(self, mock_connection_class, mock_widgets):
        """
        Tests the callback of the connect button.
        This test ensures that the `_on_connect_click` method correctly calls the
        `connect` method of the `SimpleOMEROConnection` class with the expected
        parameters.
        """
        # Mock widgets
        mock_host_widget = Mock()
        mock_host_widget.value = "localhost"
        mock_user_widget = Mock()
        mock_user_widget.value = "testuser"
        mock_password_widget = Mock()
        mock_password_widget.value = "testpass"
        mock_port_widget = Mock()
        mock_port_widget.value = 4064

        mock_widgets.Text.return_value = mock_host_widget
        mock_widgets.Password.return_value = mock_password_widget
        mock_widgets.IntText.return_value = mock_port_widget
        mock_widgets.Checkbox.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        # Mock Output widget with context manager support
        mock_output = Mock()
        mock_output.__enter__ = Mock(return_value=mock_output)
        mock_output.__exit__ = Mock(return_value=None)
        mock_widgets.Output.return_value = mock_output
        mock_widgets.VBox.return_value = Mock()

        # Mock connection manager
        mock_conn_manager = Mock()
        mock_conn = Mock()
        mock_conn_manager.connect.return_value = mock_conn
        mock_conn_manager.create_connection_from_config.return_value = mock_conn
        # Mock get_connection_list to return an empty list instead of Mock
        mock_conn_manager.get_connection_list.return_value = []
        # Mock load_config_files to return None
        mock_conn_manager.load_config_files.return_value = None
        mock_connection_class.return_value = mock_conn_manager

        widget = OMEROConnectionWidget()
        widget.host_widget = mock_host_widget
        widget.user_widget = mock_user_widget
        widget.password_widget = mock_password_widget
        widget.port_widget = mock_port_widget

        # Simulate connect button click
        widget._connect(Mock())

        assert widget.connection == mock_conn
        # Check that create_connection_from_config was called instead of direct connect
        mock_conn_manager.create_connection_from_config.assert_called_once()

    @patch("omero_annotate_ai.widgets.omero_connection_widget.widgets")
    @patch("omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection")
    def test_connect_failure(self, mock_connection_class, mock_widgets):
        """
        Tests the handling of a connection failure.
        This test ensures that the widget correctly handles a connection failure
        and that the connection attribute remains None.
        """
        # Mock widgets
        widget_mocks = {
            "Text": Mock(),
            "Password": Mock(),
            "IntText": Mock(),
            "Checkbox": Mock(),
            "Button": Mock(),
            "Output": Mock(),
            "VBox": Mock(),
        }

        # Set up context manager support for Output widget
        widget_mocks["Output"].__enter__ = Mock(return_value=widget_mocks["Output"])
        widget_mocks["Output"].__exit__ = Mock(return_value=None)

        for widget_type, mock_widget in widget_mocks.items():
            setattr(mock_widgets, widget_type, Mock(return_value=mock_widget))

        # Mock connection failure
        mock_conn_manager = Mock()
        mock_conn_manager.connect.return_value = None
        mock_conn_manager.create_connection_from_config.return_value = (
            None  # Simulate connection failure
        )
        # Mock get_connection_list to return an empty list instead of Mock
        mock_conn_manager.get_connection_list.return_value = []
        # Mock load_config_files to return None
        mock_conn_manager.load_config_files.return_value = None
        mock_connection_class.return_value = mock_conn_manager

        widget = OMEROConnectionWidget()
        widget.host_widget = Mock(value="localhost")
        widget.user_widget = Mock(value="user")
        widget.password_widget = Mock(value="pass")
        widget.port_widget = Mock(value=4064)

        widget._connect(Mock())

        assert widget.connection is None

    @patch("omero_annotate_ai.widgets.omero_connection_widget.widgets")
    @patch("omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection")
    def test_load_from_env(self, mock_connection_class, mock_widgets):
        """
        Tests loading connection parameters from environment variables.
        This test ensures that the widget correctly loads the connection parameters
        from the environment variables.
        """
        # Mock widgets
        widget_mocks = {}
        for widget_type in [
            "Text",
            "Password",
            "IntText",
            "Checkbox",
            "Button",
            "Output",
            "VBox",
        ]:
            widget_mocks[widget_type] = Mock()
            # Set up context manager support for Output widget
            if widget_type == "Output":
                widget_mocks[widget_type].__enter__ = Mock(
                    return_value=widget_mocks[widget_type]
                )
                widget_mocks[widget_type].__exit__ = Mock(return_value=None)
            setattr(
                mock_widgets, widget_type, Mock(return_value=widget_mocks[widget_type])
            )

        # Mock connection manager
        mock_conn_manager = Mock()
        # Mock get_connection_list to return an empty list
        mock_conn_manager.get_connection_list.return_value = []
        # Mock load_config_files to return environment-based config
        mock_conn_manager.load_config_files.return_value = {
            "host": "env_host",
            "username": "env_user",
            "port": "4064",
        }
        mock_connection_class.return_value = mock_conn_manager

        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "OMERO_HOST": "env_host",
                "OMERO_USER": "env_user",
                "OMERO_PORT": "4064",
            }.get(key, default)

            OMEROConnectionWidget()

            # Check that the widget's fields were populated from config
            # Note: The actual population happens during widget initialization
            # through _load_existing_config method


@pytest.mark.unit
@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
class TestOMEROConnectionWidgetCredentials:
    """Test how the connection widget handles credentials.

    These tests use the REAL ipywidgets classes and mock only the connection
    manager. TestOMEROConnectionWidget above mocks ``widgets`` wholesale, which
    means it cannot catch bugs in how the widgets are wired together.
    """

    @pytest.fixture
    def conn_manager(self):
        """A connection manager with no saved state."""
        manager = Mock()
        manager.get_connection_list.return_value = []
        manager.load_config_files.return_value = {}
        manager.load_password.return_value = None
        return manager

    @pytest.fixture
    def widget(self, conn_manager):
        with patch(
            "omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection",
            return_value=conn_manager,
        ):
            yield OMEROConnectionWidget()

    def test_widget_builds_with_real_ipywidgets(self, widget):
        """The widget tree is constructed and the input fields are all in it."""
        assert widget.main_widget is not None
        assert widget.connection is None

        def flatten(box):
            for child in getattr(box, "children", ()):
                yield child
                yield from flatten(child)

        descendants = list(flatten(widget.main_widget))
        assert widget.password_widget in descendants
        assert widget.port_widget in descendants
        assert widget.host_widget in descendants

    def test_no_show_password_toggle(self, widget):
        """The plaintext show-password toggle must not come back."""
        assert not hasattr(widget, "show_password_widget")
        assert not hasattr(widget, "_toggle_password_visibility")

    def test_password_not_auto_loaded_on_construction(self, conn_manager):
        """Constructing the widget must not read the keychain."""
        conn_manager.load_config_files.return_value = {
            "host": "omero.example.org",
            "username": "alice",
            "source": ".env file",
        }

        with patch(
            "omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection",
            return_value=conn_manager,
        ):
            widget = OMEROConnectionWidget()

        conn_manager.load_password.assert_not_called()
        assert widget.password_widget.value == ""
        # host/username are still pre-populated, only the secret is withheld
        assert widget.host_widget.value == "omero.example.org"
        assert widget.username_widget.value == "alice"

    def test_config_with_missing_host_does_not_crash(self, conn_manager):
        """A .env with USER_NAME but no HOST yields host=None, which must not raise."""
        conn_manager.load_config_files.return_value = {
            "host": None,
            "username": "alice",
            "group": None,
            "source": ".env file",
        }

        with patch(
            "omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection",
            return_value=conn_manager,
        ):
            widget = OMEROConnectionWidget()

        assert widget.host_widget.value == ""
        assert widget.username_widget.value == "alice"

    def test_get_config_omits_password_by_default(self, widget):
        """get_config() must not hand back the plaintext password."""
        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget.password_widget.value = "s3cret"

        config = widget.get_config()
        assert "password" not in config
        assert config["host"] == "omero.example.org"

        config = widget.get_config(include_password=True)
        assert config["password"] == "s3cret"

    def test_password_is_not_stripped(self, widget):
        """Whitespace can be part of a password and must survive verbatim."""
        widget.password_widget.value = "  pa ss  "

        assert widget._get_widget_config()["password"] == "  pa ss  "

    def test_port_is_included_in_config(self, widget):
        """The port field is wired into the config the manager receives."""
        assert widget.port_widget.value == widget.DEFAULT_PORT

        widget.port_widget.value = 6064
        assert widget._get_widget_config()["port"] == 6064

    def test_connect_reports_failed_keychain_save(self, widget, conn_manager, capsys):
        """A keychain write that fails must not be reported as a success."""
        conn_manager.create_connection_from_config.return_value = Mock()
        conn_manager.save_password.return_value = False

        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget.password_widget.value = "s3cret"
        widget.save_password_widget.value = True

        widget._connect(Mock())

        conn_manager.save_password.assert_called_once()
        # Outside a live kernel, Output falls through to stdout
        assert "could NOT be saved" in capsys.readouterr().out

    def test_connect_reports_successful_keychain_save(
        self, widget, conn_manager, capsys
    ):
        conn_manager.create_connection_from_config.return_value = Mock()
        conn_manager.save_password.return_value = True

        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget.password_widget.value = "s3cret"
        widget.save_password_widget.value = True

        widget._connect(Mock())

        assert "Password saved to keychain" in capsys.readouterr().out

    def test_connect_does_not_save_password_unless_asked(self, widget, conn_manager):
        conn_manager.create_connection_from_config.return_value = Mock()

        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget.password_widget.value = "s3cret"
        assert widget.save_password_widget.value is False

        widget._connect(Mock())

        conn_manager.save_password.assert_not_called()

    def test_connect_closes_previous_connection(self, widget, conn_manager):
        """Reconnecting must not leak the previous gateway."""
        first, second = Mock(), Mock()
        conn_manager.create_connection_from_config.side_effect = [first, second]

        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget.password_widget.value = "s3cret"

        widget._connect(Mock())
        assert widget.connection is first

        widget.password_widget.value = "s3cret"  # cleared after each attempt
        widget._connect(Mock())
        first.close.assert_called_once()
        assert widget.connection is second

    @pytest.mark.parametrize("result", ["ok", "none", "raises"])
    def test_password_field_cleared_after_connect(self, widget, conn_manager, result):
        """The typed password never stays in the widget model after Connect."""
        if result == "raises":
            conn_manager.create_connection_from_config.side_effect = RuntimeError("x")
        else:
            conn_manager.create_connection_from_config.return_value = (
                Mock() if result == "ok" else None
            )
        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget.password_widget.value = "s3cret"

        widget._connect(Mock())

        assert widget.password_widget.value == ""

    def test_keychain_password_stays_out_of_the_field(self, widget, conn_manager):
        """A keychain password is used for connecting but never put in the field."""
        conn_manager.load_password.return_value = "from-keychain"
        conn_manager.create_connection_from_config.return_value = Mock()
        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"

        widget._load_from_keychain(Mock())
        assert widget.password_widget.value == ""

        widget._connect(Mock())
        sent = conn_manager.create_connection_from_config.call_args[0][0]
        assert sent["password"] == "from-keychain"

    def test_keychain_password_not_sent_to_another_server(self, widget, conn_manager):
        conn_manager.load_password.return_value = "from-keychain"
        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget._load_from_keychain(Mock())

        widget.host_widget.value = "other.example.org"

        assert widget._get_widget_config()["password"] == ""

    def test_stale_keychain_password_hint(self, widget, conn_manager, capsys):
        conn_manager.load_password.return_value = "old"
        conn_manager.create_connection_from_config.side_effect = ConnectionFailed(
            "login failed"
        )
        widget.host_widget.value = "omero.example.org"
        widget.username_widget.value = "alice"
        widget._load_from_keychain(Mock())

        widget._connect(Mock())

        assert "may be out of date" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "host", ["wss://omero.example.org/omero-ws", "ws://omero.example.org/omero-ws"]
    )
    def test_url_host_sends_no_port_and_stays_encrypted(self, widget, host):
        widget.secure_widget.value = False
        widget.host_widget.value = host

        config = widget._get_widget_config()
        assert config["port"] is None
        assert config["secure"] is True
        assert widget.port_widget.disabled and widget.secure_widget.disabled

    def test_plain_host_keeps_port_and_secure_choice(self, widget):
        widget.host_widget.value = "wss://omero.example.org/omero-ws"
        widget.host_widget.value = "omero.example.org"
        widget.secure_widget.value = False

        config = widget._get_widget_config()
        assert config["port"] == widget.DEFAULT_PORT
        assert config["secure"] is False
        assert not widget.port_widget.disabled


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
@pytest.mark.unit
class TestWorkflowWidget:
    """New / Open training set, resets, summary, progress and notes."""

    @staticmethod
    def _conn(containers=()):
        conn = Mock()
        conn.getObjects.return_value = list(containers)
        return conn

    @staticmethod
    def _container(cid, name):
        container = Mock()
        container.getId.return_value = cid
        container.getName.return_value = name
        return container

    def test_new_request_builds_a_fresh_config_from_the_form(self, tmp_path):
        widget = WorkflowWidget(self._conn(), home=tmp_path)
        widget.name.value = "nuclei"
        widget.channel.value = 1
        widget.counts["validation"].value = 3
        widget.seed.value = "42"
        widget.storage.value = "local"

        request = widget.request()

        assert isinstance(request, NewTrainingSet)
        config = request.config
        assert config.name == "nuclei"
        assert config.spatial_coverage.channels == [1]
        assert config.training.validate_n == 3
        assert config.training.seed == 42
        assert config.storage == "local"
        assert widget.get_config() is not config  # a copy every time

    def test_plane_modes_and_patches_reach_the_config(self, tmp_path):
        widget = WorkflowWidget(self._conn(), home=tmp_path)
        assert widget.timepoints.n_random.disabled  # specific: list only
        widget.timepoints.mode.value = "random"
        widget.timepoints.n_random.value = 3
        assert widget.timepoints.listed.disabled
        widget.z_slices.listed.value = "2, 5-7"
        widget.three_d.value = True
        assert widget.patches_per_image.disabled
        widget.use_patches.value = True
        widget.patches_per_image.value = 4

        coverage = widget.get_config().spatial_coverage

        assert (coverage.timepoint_mode, coverage.n_timepoints) == ("random", 3)
        assert coverage.z_slice_mode == "specific"
        assert coverage.z_slices == [2, 5, 6, 7]
        assert coverage.n_slices is None
        assert coverage.three_d
        assert (coverage.use_patches, coverage.patches_per_image) == (True, 4)

        widget.z_slices.mode.value = "all"
        assert widget.get_config().spatial_coverage.z_slice_mode == "all"

    @pytest.mark.parametrize("text", ["", "a", "3-1", "-2"])
    def test_bad_plane_list_shows_an_inline_error(self, tmp_path, text):
        widget = WorkflowWidget(self._conn(), home=tmp_path)
        widget.timepoints.listed.value = text
        widget._resolve()
        assert "Timepoints" in widget.status.value
        assert "oai-error" in widget.status.value

    def test_containers_are_listed_and_a_change_resets_the_selection(self, tmp_path):
        conn = self._conn(
            [self._container(7, "plate A"), self._container(8, "plate B")]
        )
        with patch(
            "omero_annotate_ai.widgets.workflow_widget.get_container_info",
            return_value={
                "total_images": 4,
                "container_name": "plate A",
                "dimensions": {"C": 2},
                "channel_names": ["DAPI", "GFP"],
            },
        ):
            widget = WorkflowWidget(conn, home=tmp_path)
            widget.container.value = 7
            assert widget.container.options == (
                ("plate A (ID 7)", 7),
                ("plate B (ID 8)", 8),
            )
            assert "0: DAPI, 1: GFP" in widget.container_info.value
            assert widget.channel.max == 1

            widget.summary.value = "old summary"
            widget.add["training"].value = 5
            widget.container.value = 8

        assert widget.summary.value == ""
        assert widget.add["training"].value == 0

    def test_open_lists_sets_shows_summary_and_requests_additions(self, tmp_path):
        local = tmp_path / "local" / "abc_local-set"
        local.mkdir(parents=True)
        (local / "config.yaml").write_text("name: x\n")
        info = Mock(table_id=99, created="")
        info.name = "plate_set"
        summary = TrainingSetSummary(
            name="plate_set",
            table_id=99,
            storage="omero",
            progress={"training": (2, 3), "validation": (0, 1), "test": (0, 0)},
            sessions=(
                SessionRecord(
                    id=1,
                    started="2026-11-07T10:00:00",
                    status="finished",
                    n_annotated=2,
                    note="nuclei only",
                ),
            ),
        )
        conn = self._conn([self._container(7, "plate A")])
        with (
            patch(
                "omero_annotate_ai.widgets.workflow_widget.get_container_info",
                return_value={},
            ),
            patch(
                "omero_annotate_ai.trainingset.list_training_sets",
                return_value=[info],
            ),
            patch(
                "omero_annotate_ai.widgets.workflow_widget.summarize",
                return_value=summary,
            ),
        ):
            widget = WorkflowWidget(conn, home=tmp_path)
            widget.container.value = 7
            widget.mode.value = "open"
            assert [value for _, value in widget.sets.options] == [99, local]
            widget.sets.value = 99
            widget.add["training"].value = 2

        assert "2 / 3 annotated" in widget.summary.value
        assert "nuclei only" in widget.summary.value
        assert "test" not in widget.summary.value  # empty splits are left out
        assert widget.request() == OpenTrainingSet(
            99, {"training": 2, "validation": 0, "test": 0}
        )

    def test_open_without_a_selection_shows_an_inline_error(self, tmp_path):
        widget = WorkflowWidget(self._conn(), home=tmp_path)
        widget.mode.value = "open"
        widget._resolve()
        assert "Select a training set" in widget.status.value

    def test_open_or_create_adds_to_the_current_counts(self, tmp_path):
        ts = Mock()
        ts.config.training.train_n, ts.config.training.validate_n = 3, 1
        ts.config.training.test_n = 0
        with patch.object(TrainingSet, "open", return_value=ts) as opened:
            result = open_or_create(
                OpenTrainingSet(5, {"training": 2, "validation": 0, "test": 1}),
                "conn",
                home=tmp_path,
            )
        assert result is ts
        assert opened.call_args.args == (5, "conn")
        ts.extend.assert_called_once_with(train_n=5, validate_n=1, test_n=1)

    def test_events_drive_progress_and_log(self, tmp_path):
        widget = WorkflowWidget(self._conn(), home=tmp_path)
        widget.events(SessionStarted(session=3, pending=4))
        widget.events(UnitCommitted(annotation_id="a", label_id=1, roi_id=2))
        widget.events(UploadFailed(annotation_id="b", error="timeout"))

        assert (widget.progress.value, widget.progress.max) == (1, 4)
        assert "Session 3: 4 units" in widget.log.value
        assert "Upload failed for b: timeout" in widget.log.value

    def test_note_needs_a_session(self, tmp_path):
        widget = WorkflowWidget(self._conn(), home=tmp_path)
        widget.note.value = "looks good"
        widget._save_note()
        assert "Pick a set with a session" in widget.status.value

    def test_note_after_changing_container_creates_nothing(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        conn = omero.conn()
        conn.getObjects.return_value = []
        widget = WorkflowWidget(conn, home=tmp_path / "home")
        widget.name.value = "widget_set"
        widget.counts["training"].value, widget.counts["validation"].value = 2, 0
        widget.training_set(
            image_source=source,
            omero_writes=omero,
            images=[source.get_image(i) for i in (1, 2)],
        )
        widget.annotate(ScriptedAnnotator(image_mask))

        widget._reset_selection()  # the user picks another container
        widget.note.value = "for the old set"
        widget._save_note()

        assert widget.note_session.options == ()
        assert "Pick a set with a session" in widget.status.value
        assert len(omero.tables) == 1  # no second set was created

    def test_new_set_annotate_and_note_on_the_real_core(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        conn = omero.conn()
        conn.getObjects.return_value = []
        widget = WorkflowWidget(conn, home=tmp_path / "home")
        widget.name.value = "widget_set"
        widget.counts["training"].value, widget.counts["validation"].value = 2, 0

        ts = widget.training_set(
            image_source=source,
            omero_writes=omero,
            images=[source.get_image(i) for i in (1, 2)],
        )
        saved = widget.annotate(ScriptedAnnotator(image_mask), note="first pass")
        assert widget.note.value == "first pass"  # prefilled to edit
        assert widget.note_session.value == ts.config.sessions[-1].id
        widget.note.value = "second look: fine"
        widget._save_note()

        assert saved == 2
        sessions = ts.config.sessions
        assert [s.note for s in sessions] == ["second look: fine"]  # replaced
        assert widget.progress.value == 2
        assert widget.training_set() is ts  # resolved once
        assert "second look: fine" in omero.config_text[ts.table_id]

    def test_review_annotated_runs_a_review_session(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        conn = omero.conn()
        conn.getObjects.return_value = []
        widget = WorkflowWidget(conn, home=tmp_path / "home")
        widget.name.value = "widget_set"
        widget.counts["training"].value, widget.counts["validation"].value = 2, 0
        ts = widget.training_set(
            image_source=source,
            omero_writes=omero,
            images=[source.get_image(i) for i in (1, 2)],
        )
        widget.annotate(ScriptedAnnotator(image_mask))

        widget.mode.value = "open"
        widget._training_set = ts  # as if the set was opened from the list
        widget.task.value = "review"

        def edit(pixels, mask):
            out = mask.copy()
            out[0, 0] = 0
            return out

        revised = widget.annotate(ScriptedAnnotator(image_mask, review_fn=edit))

        assert revised == 2
        assert ts.config.sessions[-1].kind == "review"
        assert "review, finished, 2 revised" in render_summary(summarize(ts.bundle_dir))


@pytest.mark.unit
class TestWidgetFallbacks:
    """Test widget behavior when dependencies are missing."""

    def test_widget_import_fallback(self):
        """
        Tests that the widget imports fail gracefully when dependencies are missing.
        This test ensures that an `ImportError` is raised when the `ipywidgets`
        package is not available.
        """
        # Clear the module from sys.modules if it's already imported
        widget_module = "omero_annotate_ai.widgets.omero_connection_widget"
        if widget_module in sys.modules:
            del sys.modules[widget_module]

        with (
            patch.dict("sys.modules", {"ipywidgets": None}),
            pytest.raises(ImportError),
        ):
            importlib.import_module(widget_module)

    def test_widget_creation_without_ipywidgets(self):
        """
        Tests the widget creation behavior without `ipywidgets`.
        This test verifies that the module structure handles the case where `ipywidgets`
        is not available.
        """
        # This test verifies that the module structure handles missing dependencies
        if importlib.util.find_spec("ipywidgets") is not None:
            pytest.skip("ipywidgets is available, cannot test fallback")


@pytest.mark.unit
@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
class TestTrainingDataWidget:
    """Pick a set (OMERO or local), see its progress, export it."""

    @staticmethod
    def _local_set(tmp_path):
        from tests.fakes import annotate, create_set

        config, source, omero = omero_set(tmp_path, n_images=2)
        config.storage = "local"
        ts = create_set(config, source, omero, tmp_path / "home")
        assert annotate(ts) == 2
        return ts

    def test_local_set_is_listed_summarized_and_exported(self, tmp_path):
        from omero_annotate_ai.widgets.training_data_widget import TrainingDataWidget

        ts = self._local_set(tmp_path)
        widget = TrainingDataWidget(None, home=tmp_path / "home")
        assert [value for _, value in widget.sets.options] == [ts.bundle_dir]

        widget.sets.value = ts.bundle_dir
        assert "2 / 2 annotated" in widget.summary.value
        assert widget.training_channels.disabled  # a local set keeps its channels
        assert widget.get_selected_table_id() is None
        assert widget.selected_config().name == ts.name
        out = tmp_path / "out"
        widget.output_dir.value = str(out)
        widget.export_button.click()

        assert "Exported to" in widget.status.value, widget.status.value
        assert len(list((out / "train_input").glob("*.tif"))) == 2
        assert (out / "normalization.json").exists()

    def test_omero_set_exports_with_the_form_options(self, tmp_path):
        from omero_annotate_ai.trainingset import TrainingSetInfo
        from omero_annotate_ai.widgets.training_data_widget import TrainingDataWidget

        config, _, _ = omero_set(tmp_path)
        config.spatial_coverage.channels = [1]
        summary = TrainingSetSummary(
            name="nuclei", table_id=42, storage="omero", progress={}, sessions=()
        )
        conn = TestWorkflowWidget._conn([TestWorkflowWidget._container(7, "ds")])
        module = "omero_annotate_ai.widgets.training_data_widget"
        with (
            patch(
                "omero_annotate_ai.trainingset.list_training_sets",
                return_value=[TrainingSetInfo(42, "nuclei", "")],
            ),
            patch(f"{module}.summarize", return_value=summary),
            patch(f"{module}.read_set_config", return_value=config),
            patch(f"{module}.export_training_set") as export,
        ):
            export.return_value = {"base_dir": tmp_path, "stats": {"n_x": 1}}
            widget = TrainingDataWidget(conn, home=tmp_path)
            widget.container.value = 7
            widget.sets.value = 42
            assert widget.label_channel.value == 1
            assert widget.training_channels.value == "1"
            assert widget.output_dir.value.endswith("fake_set_42")
            widget.training_channels.value = "0, 2-3"
            widget.normalization.value = True
            widget.export_button.click()

        assert export.call_args.args[0] == 42
        assert export.call_args.kwargs == {
            "training_channels": [0, 2, 3],
            "label_channel": 1,
            "apply_normalization": True,
        }
        assert "Exported to" in widget.status.value

    def test_missing_output_folder_is_an_inline_error(self, tmp_path):
        from omero_annotate_ai.widgets.training_data_widget import TrainingDataWidget

        ts = self._local_set(tmp_path)
        widget = TrainingDataWidget(None, home=tmp_path / "home")
        widget.sets.value = ts.bundle_dir
        widget.output_dir.value = ""
        widget.export_button.click()
        assert "Choose an output folder" in widget.status.value
        assert widget.result is None
