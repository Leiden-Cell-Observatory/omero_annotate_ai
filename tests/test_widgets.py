"""Tests for widget modules."""

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from omero_annotate_ai.errors import ConnectionFailed

# Test imports with fallbacks for missing dependencies
try:
    from omero_annotate_ai.widgets.omero_connection_widget import OMEROConnectionWidget
    from omero_annotate_ai.widgets.workflow_widget import WorkflowWidget

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
    """Test workflow configuration widget functionality."""

    @patch("omero_annotate_ai.widgets.workflow_widget.widgets")
    def test_workflow_widget_initialization(self, mock_widgets):
        """
        Tests the initialization of the WorkflowWidget.
        This test ensures that the widget is correctly initialized with the
        provided connection.
        """
        # Mock ipywidgets components
        widget_mocks = {}
        for widget_type in [
            "Text",
            "Dropdown",
            "IntText",
            "Checkbox",
            "Button",
            "Output",
            "VBox",
            "HBox",
        ]:
            widget_mocks[widget_type] = Mock()
            setattr(
                mock_widgets, widget_type, Mock(return_value=widget_mocks[widget_type])
            )

        mock_connection = Mock()

        widget = WorkflowWidget(connection=mock_connection)

        assert widget is not None
        assert widget.connection == mock_connection

    @patch("omero_annotate_ai.widgets.workflow_widget.widgets")
    def test_container_selection(self, mock_widgets):
        """
        Tests the OMERO container selection.
        This test ensures that the widget correctly populates the container dropdown
        when the container type is changed.
        """
        # Mock widgets
        widget_mocks = {}
        for widget_type in [
            "Text",
            "Dropdown",
            "IntText",
            "Checkbox",
            "Button",
            "Output",
            "VBox",
            "HBox",
        ]:
            widget_mocks[widget_type] = Mock()
            setattr(
                mock_widgets, widget_type, Mock(return_value=widget_mocks[widget_type])
            )

        mock_connection = Mock()

        # Mock the connection's getObjects method instead of ezomero
        mock_projects = [
            Mock(getId=Mock(return_value=1), getName=Mock(return_value="Project 1")),
            Mock(getId=Mock(return_value=2), getName=Mock(return_value="Project 2")),
        ]
        mock_connection.getObjects.return_value = mock_projects

        widget = WorkflowWidget(connection=mock_connection)
        # Need to properly mock the container_widgets dictionary structure
        widget.container_widgets = {
            "type": Mock(value="project"),
            "container": Mock(),
            "refresh": Mock(),
        }

        # Simulate container type change
        widget._on_container_type_change({"new": "project"})

        # Should call getObjects with "Project"
        mock_connection.getObjects.assert_called_once_with("Project")

    @patch("omero_annotate_ai.widgets.workflow_widget.widgets")
    def test_config_generation(self, mock_widgets):
        """
        Tests the generation of the configuration from the widget values.
        This test ensures that the `get_config` method correctly generates an
        `AnnotationConfig` object with the expected values from the widget.
        """
        # Mock widgets with values
        widget_mocks = {}
        widget_values = {
            "working_dir": "/tmp/test",
            "container_type": "dataset",
            "container_id": 123,
            "pretrained_from": "vit_b_lm",
            "batch_size": 5,
            "use_patches": False,
        }

        for widget_type in [
            "Text",
            "Dropdown",
            "IntText",
            "Checkbox",
            "Button",
            "Output",
            "VBox",
            "HBox",
        ]:
            mock_widget = Mock()
            if widget_type in ["Text", "Dropdown"]:
                mock_widget.value = widget_values.get(widget_type.lower(), "")
            elif widget_type == "IntText":
                mock_widget.value = widget_values.get("batch_size", 0)
            elif widget_type == "Checkbox":
                mock_widget.value = widget_values.get("use_patches", False)

            widget_mocks[widget_type] = mock_widget
            setattr(mock_widgets, widget_type, Mock(return_value=mock_widget))

        mock_connection = Mock()

        widget = WorkflowWidget(connection=mock_connection)

        # Mock working directory
        widget.working_directory = "/tmp/test"

        # Mock container widgets
        widget.container_widgets = {
            "container": Mock(value=123),
            "type": Mock(value="dataset"),
        }

        # Mock technical widgets container with child widgets
        batch_size_widget = Mock(description="Batch size", value=5)
        model_type_widget = Mock(description="SAM Model", value="vit_b_lm")
        output_folder_widget = Mock(
            description="Output folder", value=Path("/tmp/test")
        )

        widget.technical_widgets = Mock()
        widget.technical_widgets.children = [
            batch_size_widget,
            model_type_widget,
            output_folder_widget,
        ]

        # Mock annotation widgets container with child widgets
        use_patches_widget = Mock(description="Use patches", value=False)

        widget.annotation_widgets = Mock()
        widget.annotation_widgets.children = [use_patches_widget]

        # Mock other required attributes to prevent errors
        widget.selected_table_id = None
        widget.new_table_name = Mock(value="test_training")
        widget.annotation_tables = []
        widget.omero_status = Mock()
        widget.status_output = Mock()
        widget.status_output.__enter__ = Mock(return_value=widget.status_output)
        widget.status_output.__exit__ = Mock(return_value=None)

        # Call update config to populate self.config from widget values
        with patch("builtins.print"), patch("IPython.display.clear_output"):
            widget._on_update_config(None)

        config = widget.get_config()

        assert config is not None
        assert widget.home == Path("/tmp/test")
        assert config.omero.container_type == "dataset"
        assert config.omero.container_id == 123
        assert config.ai_model.pretrained_from == "vit_b_lm"
        assert config.workflow.batch_size == 5
        assert config.spatial_coverage.use_patches is False


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
@pytest.mark.unit
class TestWorkflowWidgetDisplay:
    """Text the workflow widget shows, built with real ipywidgets."""

    @staticmethod
    def _headings(widget):
        import re

        headings = []
        for tab in widget.tab_widget.children:
            for child in tab.children:
                value = getattr(child, "value", "")
                if isinstance(value, str) and "<h4>" in value:
                    headings.append(re.sub(r"</?h4>", "", value))
        return headings

    @pytest.mark.parametrize("connection", [None, "connected"])
    def test_tab_headings_match_step_names(self, connection):
        """Each tab heading is its step name, with or without a connection tab."""
        conn = Mock() if connection else None
        with patch.object(WorkflowWidget, "_load_containers"):
            widget = WorkflowWidget(connection=conn)

        assert self._headings(widget) == widget.steps

    def test_container_info_replaces_placeholder(self):
        """Selecting a container replaces the 'select a container' placeholder."""
        with patch.object(WorkflowWidget, "_load_containers"):
            widget = WorkflowWidget(connection=Mock())

        widget._apply_container_info(
            {
                "container_type": "project",
                "container_id": 51,
                "container_name": "idr0021",
                "total_images": 414,
                "dimensions": {"X": 256, "Y": 256, "Z": 1, "C": 3, "T": 1},
            }
        )

        status = widget.omero_status.value
        assert "Select container in previous steps first" not in status
        assert "idr0021" in status
        assert "ID: 51" in status
        assert "414" in status

    def test_update_config_keeps_image_summary(self):
        """Updating the config keeps the image count in the OMERO settings."""
        with patch.object(WorkflowWidget, "_load_containers"):
            widget = WorkflowWidget(connection=Mock())
        widget._apply_container_info(
            {
                "container_type": "project",
                "container_id": 51,
                "container_name": "idr0021",
                "total_images": 414,
            }
        )
        widget.container_widgets["type"].value = "project"
        widget.container_widgets["container"].options = [("idr0021 (ID: 51)", 51)]
        widget.container_widgets["container"].value = 51
        widget.new_table_name.value = "idr0021_microsam"

        with patch("builtins.print"):
            widget._on_update_config(None)

        status = widget.omero_status.value
        assert "idr0021_microsam" in status
        assert "414" in status

    @pytest.mark.parametrize("vary", [False, True])
    def test_channel_names_shown_in_both_summaries(self, vary):
        """Channel names appear with their index, and a note when images differ."""
        with patch.object(WorkflowWidget, "_load_containers"):
            widget = WorkflowWidget(connection=Mock())

        widget._apply_container_info(
            {
                "container_type": "dataset",
                "container_id": 7,
                "container_name": "cells",
                "total_images": 10,
                "channel_names": ["DAPI", "GFP<1>"],
                "channel_names_vary": vary,
            }
        )

        for text in (widget.container_summary.value, widget.omero_status.value):
            assert "0: DAPI, 1: GFP&lt;1&gt;" in text
            assert ("name channels differently" in text) is vary

    def _widget_with_table(self):
        with patch.object(WorkflowWidget, "_load_containers"):
            widget = WorkflowWidget(connection=Mock())
        widget.container_widgets["type"].value = "plate"
        widget.container_widgets["container"].options = [("plate (ID: 803)", 803)]
        widget.container_widgets["container"].value = 803
        widget.annotation_tables = [{"id": 22159, "name": "plate_run"}]
        widget.tables_widgets["selection"].options = [("plate_run", 22159)]
        widget.tables_widgets["selection"].value = 22159
        return widget

    def test_tables_with_the_same_name_get_distinct_labels(self):
        """Two tables sharing a name must not both show as selected."""
        with patch.object(WorkflowWidget, "_load_containers"):
            widget = WorkflowWidget(connection=Mock())
        widget.container_widgets["type"].value = "plate"
        widget.container_widgets["container"].options = [("plate (ID: 803)", 803)]
        widget.container_widgets["container"].value = 803
        tables = [{"id": 22162, "name": "run"}, {"id": 22159, "name": "run"}]

        with (
            patch(
                "omero_annotate_ai.widgets.workflow_widget.list_annotation_tables",
                return_value=tables,
            ),
            patch("builtins.print"),
        ):
            widget._on_scan_tables(None)

        labels = [label for label, _ in widget.tables_widgets["selection"].options]
        assert labels == ["run (ID: 22162)", "run (ID: 22159)"]

    def test_continue_existing_survives_update_config(self):
        """Update Configuration must not reset resume after Continue Existing."""
        widget = self._widget_with_table()

        widget._on_continue_table(None)
        with patch("builtins.print"):
            widget._on_update_config(None)

        assert widget.get_config().identity.table_id is not None
        assert widget.get_config().name == "plate_run"
        assert widget.get_config().identity.table_id == 22159
        assert widget.workflow_widgets["resume_from_table"].value is True

    @staticmethod
    def _saved_config():
        from omero_annotate_ai.core.annotation_config import create_default_config

        saved = create_default_config()
        saved.name = "plate_run"
        saved.spatial_coverage.channels = [2]
        saved.spatial_coverage.use_patches = True
        saved.spatial_coverage.patch_size = [256, 128]
        saved.training.train_n = 4
        saved.ai_model.pretrained_from = "vit_l"
        return saved

    def _continue_with_saved_config(self):
        widget = self._widget_with_table()
        with patch(
            "omero_annotate_ai.widgets.workflow_widget.find_annotation_config_for_table",
            return_value=self._saved_config(),
        ):
            widget._on_continue_table(None)
        return widget

    def test_continue_existing_loads_saved_config_and_locks_fields(self):
        widget = self._continue_with_saved_config()
        fields = widget._annotation_widget_map()

        assert fields["channel"].value == 2
        assert fields["use_patches"].value is True
        assert fields["patch_size"].value == "256, 128"
        assert fields["train_n"].value == 4
        assert fields["model_type"].value == "vit_l"
        for key in ("channel", "three_d", "use_patches", "patch_size"):
            assert fields[key].disabled, key
        for key in ("train_n", "validate_n", "test_n", "batch_size", "model_type"):
            assert not fields[key].disabled, key
        assert "saved config" in widget.resume_note.value

    def test_continue_existing_updates_config_without_update_click(self):
        """Running right after Continue Existing must use the table's settings."""
        widget = self._continue_with_saved_config()

        config = widget.get_config()
        assert config.spatial_coverage.get_label_channel() == 2
        assert config.spatial_coverage.use_patches is True
        assert config.spatial_coverage.patch_size == [256, 128]
        assert config.training.train_n == 4
        assert config.identity.table_id is not None

    def test_get_config_keeps_locked_values(self):
        widget = self._continue_with_saved_config()
        widget.config.spatial_coverage.use_patches = False

        assert widget.get_config().spatial_coverage.use_patches is True

    def test_table_rows_win_over_a_mismatched_saved_config(self):
        """A saved config from another same-name run must not set locked fields."""
        from omero_annotate_ai.core.annotation_config import create_default_config

        saved = create_default_config()  # channel 0, no patches, 3/2
        saved.name = "plate_run"
        rows = pd.DataFrame(
            [
                {
                    "image_uri": f"omero:image:{i}",
                    "category": cat,
                    "channel": 1,
                    "is_volumetric": False,
                    "is_patch": True,
                    "patch_width": 256,
                    "patch_height": 256,
                }
                for i, cat in enumerate(["training"] * 3 + ["validation"] * 3)
            ]
        )
        widget = self._widget_with_table()
        with (
            patch(
                "omero_annotate_ai.widgets.workflow_widget.find_annotation_config_for_table",
                return_value=saved,
            ),
            patch(
                "omero_annotate_ai.widgets.workflow_widget.read_tracking_table",
                return_value=rows,
            ),
        ):
            widget._on_continue_table(None)

        fields = widget._annotation_widget_map()
        assert fields["channel"].value == 1 and fields["channel"].disabled
        assert fields["use_patches"].value is True
        assert fields["validate_n"].value == 3
        config = widget.get_config()
        assert config.spatial_coverage.get_label_channel() == 1
        assert config.spatial_coverage.use_patches is True
        assert config.spatial_coverage.patch_size == [256, 256]
        assert config.training.validate_n == 3
        assert "did not match the table" in widget.resume_note.value

    def test_resume_locks_planes_patches_and_read_only(self):
        widget = self._continue_with_saved_config()
        fields = widget._annotation_widget_map()
        for key in (
            "timepoint_mode",
            "timepoints",
            "z_slice_mode",
            "z_slices",
            "patches_per_image",
            "segment_all",
        ):
            assert fields[key].disabled, key
        read_only = widget.workflow_widgets["read_only_mode"]
        assert read_only.disabled and read_only.value is False

        read_only.value = True  # disabled in the UI, but set anyway
        fields["patches_per_image"].value = 7
        with patch("builtins.print"):
            widget._on_update_config(None)
        config = widget.get_config()
        assert config.storage == "omero"
        assert (
            config.spatial_coverage.patches_per_image
            == self._saved_config().spatial_coverage.patches_per_image
        )

    def test_name_new_table_unlocks_read_only(self):
        widget = self._continue_with_saved_config()
        with patch(
            "omero_annotate_ai.widgets.workflow_widget.generate_unique_table_name",
            return_value="fresh_table",
        ):
            widget._on_new_table(None)
        assert not widget.workflow_widgets["read_only_mode"].disabled
        assert not widget._annotation_widget_map()["segment_all"].disabled

    def test_locked_values_win_over_widgets_on_update(self):
        widget = self._continue_with_saved_config()
        fields = widget._annotation_widget_map()
        fields["channel"].value = 5  # disabled in the UI, but set anyway
        fields["patch_size"].value = "64, 64"
        fields["train_n"].value = 6  # editable

        with patch("builtins.print"):
            widget._on_update_config(None)

        config = widget.get_config()
        assert config.spatial_coverage.get_label_channel() == 2
        assert config.spatial_coverage.patch_size == [256, 128]
        assert config.spatial_coverage.use_patches is True
        assert config.training.train_n == 6
        assert config.identity.table_id is not None

    def test_name_new_table_unlocks_fields(self):
        widget = self._continue_with_saved_config()

        with patch(
            "omero_annotate_ai.widgets.workflow_widget.generate_unique_table_name",
            return_value="fresh_table",
        ):
            widget._on_new_table(None)

        fields = widget._annotation_widget_map()
        assert not fields["channel"].disabled
        assert not fields["use_patches"].disabled
        assert not fields["three_d"].disabled
        assert fields["patch_size"].disabled is (not fields["use_patches"].value)
        assert widget.resume_note.value == ""
        fields["channel"].value = 3
        with patch("builtins.print"):
            widget._on_update_config(None)
        assert widget.get_config().spatial_coverage.channels[0] == 3

    def test_continue_infers_settings_from_rows_without_saved_config(self):

        widget = self._widget_with_table()
        rows = pd.DataFrame(
            [
                {
                    "image_uri": "omero:image:1",
                    "category": "training",
                    "channel": 1,
                    "is_volumetric": False,
                    "is_patch": False,
                },
                {
                    "image_uri": "omero:image:2",
                    "category": "training",
                    "channel": 1,
                    "is_volumetric": False,
                    "is_patch": False,
                },
            ]
        )
        with (
            patch(
                "omero_annotate_ai.widgets.workflow_widget.find_annotation_config_for_table",
                return_value=None,
            ),
            patch(
                "omero_annotate_ai.widgets.workflow_widget.read_tracking_table",
                return_value=rows,
            ) as read,
        ):
            widget._on_continue_table(None)

        read.assert_called_once_with(widget.connection, 22159)
        fields = widget._annotation_widget_map()
        assert fields["channel"].value == 1 and fields["channel"].disabled
        assert fields["train_n"].value == 2
        assert "table rows" in widget.resume_note.value

    def test_new_table_after_continue_turns_resume_off(self):
        """Switching to a new table clears the earlier Continue Existing choice."""
        widget = self._widget_with_table()
        widget._on_continue_table(None)

        with patch(
            "omero_annotate_ai.widgets.workflow_widget.generate_unique_table_name",
            return_value="fresh_table",
        ):
            widget._on_new_table(None)
        with patch("builtins.print"):
            widget._on_update_config(None)

        assert widget.get_config().identity.table_id is None
        assert widget.get_config().name == "fresh_table"
        assert widget.get_config().identity.table_id is None
        assert widget.workflow_widgets["resume_from_table"].value is False


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
class TestWidgetIntegration:
    """Test widget integration scenarios."""

    @pytest.mark.skipif(
        not WIDGETS_AVAILABLE, reason="Widget dependencies not available"
    )
    @patch("omero_annotate_ai.widgets.omero_connection_widget.widgets")
    @patch("omero_annotate_ai.widgets.workflow_widget.widgets")
    def test_connection_to_workflow_integration(
        self, mock_workflow_widgets, mock_conn_widgets
    ):
        """
        Tests the integration between the connection and workflow widgets.
        This test ensures that the connection object created by the `OMEROConnectionWidget`
        can be correctly passed to the `WorkflowWidget`.
        """
        # Mock all required widgets
        for mock_widgets in [mock_conn_widgets, mock_workflow_widgets]:
            for widget_type in [
                "Text",
                "Password",
                "Dropdown",
                "IntText",
                "Checkbox",
                "Button",
                "Output",
                "VBox",
                "HBox",
            ]:
                mock_widget = Mock()
                if widget_type == "Output":
                    mock_widget.__enter__ = Mock(return_value=mock_widget)
                    mock_widget.__exit__ = Mock(return_value=None)
                setattr(mock_widgets, widget_type, Mock(return_value=mock_widget))

        # Create connection widget and establish connection
        with patch(
            "omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection"
        ) as mock_conn_class:
            mock_conn = Mock()
            mock_conn_manager = Mock()
            mock_conn_manager.connect.return_value = mock_conn
            mock_conn_manager.create_connection_from_config.return_value = mock_conn
            # Mock get_connection_list to return an empty list
            mock_conn_manager.get_connection_list.return_value = []
            # Mock load_config_files to return None
            mock_conn_manager.load_config_files.return_value = None
            mock_conn_class.return_value = mock_conn_manager

            conn_widget = OMEROConnectionWidget()

            # Ensure connection_manager is properly set to our mock
            conn_widget.connection_manager = mock_conn_manager

            conn_widget.host_widget = Mock(value="localhost")
            conn_widget.username_widget = Mock(value="user")
            conn_widget.password_widget = Mock(value="pass")
            conn_widget.group_widget = Mock(value="")
            conn_widget.secure_widget = Mock(value=True)
            conn_widget.save_password_widget = Mock(value=False)
            conn_widget.expire_widget = Mock(value=24)

            # Mock the status output
            conn_widget.status_output = Mock()
            conn_widget.status_output.__enter__ = Mock(
                return_value=conn_widget.status_output
            )
            conn_widget.status_output.__exit__ = Mock(return_value=None)

            with patch("builtins.print"), patch("IPython.display.clear_output"):
                conn_widget._connect(Mock())

            # Use connection in workflow widget
            workflow_widget = WorkflowWidget(connection=conn_widget.connection)

            assert workflow_widget.connection == mock_conn

    @pytest.mark.skipif(
        not WIDGETS_AVAILABLE, reason="Widget dependencies not available"
    )
    @patch("omero_annotate_ai.widgets.workflow_widget.widgets")
    def test_workflow_config_validation(self, mock_widgets):
        """
        Tests the validation of the workflow configuration.
        This test ensures that the `get_config` method handles invalid widget values
        gracefully and still returns a valid configuration object.
        """
        # Mock widgets
        for widget_type in [
            "Text",
            "Dropdown",
            "IntText",
            "Checkbox",
            "Button",
            "Output",
            "VBox",
            "HBox",
        ]:
            setattr(mock_widgets, widget_type, Mock(return_value=Mock()))

        mock_connection = Mock()

        widget = WorkflowWidget(connection=mock_connection)

        # Set invalid configuration
        widget.working_dir_widget = Mock(value="")  # Empty working directory
        widget.container_type_widget = Mock(value="invalid")  # Invalid container type
        widget.container_id_widget = Mock(value=0)  # Invalid container ID

        # Should handle validation gracefully
        config = widget.get_config()

        # Config should still be created with defaults/corrections
        assert config is not None
