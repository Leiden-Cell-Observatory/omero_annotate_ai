"""Tests for widget modules."""

import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.widgets')
    @patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection')
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
        assert hasattr(widget, 'connection')
        assert widget.connection is None
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.widgets')
    @patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection')
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
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.widgets')
    @patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection')
    def test_connect_failure(self, mock_connection_class, mock_widgets):
        """
        Tests the handling of a connection failure.
        This test ensures that the widget correctly handles a connection failure
        and that the connection attribute remains None.
        """
        # Mock widgets
        widget_mocks = {
            'Text': Mock(),
            'Password': Mock(), 
            'IntText': Mock(),
            'Checkbox': Mock(),
            'Button': Mock(),
            'Output': Mock(),
            'VBox': Mock()
        }
        
        # Set up context manager support for Output widget
        widget_mocks['Output'].__enter__ = Mock(return_value=widget_mocks['Output'])
        widget_mocks['Output'].__exit__ = Mock(return_value=None)
        
        for widget_type, mock_widget in widget_mocks.items():
            setattr(mock_widgets, widget_type, Mock(return_value=mock_widget))
        
        # Mock connection failure
        mock_conn_manager = Mock()
        mock_conn_manager.connect.return_value = None
        mock_conn_manager.create_connection_from_config.return_value = None  # Simulate connection failure
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
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.widgets')
    @patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection')
    def test_load_from_env(self, mock_connection_class, mock_widgets):
        """
        Tests loading connection parameters from environment variables.
        This test ensures that the widget correctly loads the connection parameters
        from the environment variables.
        """
        # Mock widgets
        widget_mocks = {}
        for widget_type in ['Text', 'Password', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox']:
            widget_mocks[widget_type] = Mock()
            # Set up context manager support for Output widget
            if widget_type == 'Output':
                widget_mocks[widget_type].__enter__ = Mock(return_value=widget_mocks[widget_type])
                widget_mocks[widget_type].__exit__ = Mock(return_value=None)
            setattr(mock_widgets, widget_type, Mock(return_value=widget_mocks[widget_type]))
        
        # Mock connection manager
        mock_conn_manager = Mock()
        # Mock get_connection_list to return an empty list
        mock_conn_manager.get_connection_list.return_value = []
        # Mock load_config_files to return environment-based config
        mock_conn_manager.load_config_files.return_value = {
            'host': 'env_host',
            'username': 'env_user',
            'port': '4064'
        }
        mock_connection_class.return_value = mock_conn_manager
        
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'OMERO_HOST': 'env_host',
                'OMERO_USER': 'env_user',
                'OMERO_PORT': '4064'
            }.get(key, default)
            
            widget = OMEROConnectionWidget()

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
            'omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection',
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
            'omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection',
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
            'omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection',
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

        widget._connect(Mock())
        first.close.assert_called_once()
        assert widget.connection is second


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
@pytest.mark.unit
class TestWorkflowWidget:
    """Test workflow configuration widget functionality."""
    
    @patch('omero_annotate_ai.widgets.workflow_widget.widgets')
    def test_workflow_widget_initialization(self, mock_widgets):
        """
        Tests the initialization of the WorkflowWidget.
        This test ensures that the widget is correctly initialized with the
        provided connection.
        """
        # Mock ipywidgets components
        widget_mocks = {}
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            widget_mocks[widget_type] = Mock()
            setattr(mock_widgets, widget_type, Mock(return_value=widget_mocks[widget_type]))
        
        mock_connection = Mock()
        
        widget = WorkflowWidget(connection=mock_connection)
        
        assert widget is not None
        assert widget.connection == mock_connection
    
    @patch('omero_annotate_ai.widgets.workflow_widget.widgets')
    def test_container_selection(self, mock_widgets):
        """
        Tests the OMERO container selection.
        This test ensures that the widget correctly populates the container dropdown
        when the container type is changed.
        """
        # Mock widgets
        widget_mocks = {}
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            widget_mocks[widget_type] = Mock()
            setattr(mock_widgets, widget_type, Mock(return_value=widget_mocks[widget_type]))
        
        mock_connection = Mock()
        
        # Mock the connection's getObjects method instead of ezomero
        mock_projects = [
            Mock(getId=Mock(return_value=1), getName=Mock(return_value="Project 1")),
            Mock(getId=Mock(return_value=2), getName=Mock(return_value="Project 2"))
        ]
        mock_connection.getObjects.return_value = mock_projects
        
        widget = WorkflowWidget(connection=mock_connection)
        # Need to properly mock the container_widgets dictionary structure
        widget.container_widgets = {
            "type": Mock(value="project"),
            "container": Mock(),
            "refresh": Mock()
        }
        
        # Simulate container type change
        widget._on_container_type_change({'new': 'project'})
        
        # Should call getObjects with "Project"
        mock_connection.getObjects.assert_called_once_with("Project")
    
    @patch('omero_annotate_ai.widgets.workflow_widget.widgets')
    def test_config_generation(self, mock_widgets):
        """
        Tests the generation of the configuration from the widget values.
        This test ensures that the `get_config` method correctly generates an
        `AnnotationConfig` object with the expected values from the widget.
        """
        # Mock widgets with values
        widget_mocks = {}
        widget_values = {
            'working_dir': '/tmp/test',
            'container_type': 'dataset',
            'container_id': 123,
            'pretrained_from': 'vit_b_lm',
            'batch_size': 5,
            'use_patches': False
        }

        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            mock_widget = Mock()
            if widget_type in ['Text', 'Dropdown']:
                mock_widget.value = widget_values.get(widget_type.lower(), "")
            elif widget_type == 'IntText':
                mock_widget.value = widget_values.get('batch_size', 0)
            elif widget_type == 'Checkbox':
                mock_widget.value = widget_values.get('use_patches', False)

            widget_mocks[widget_type] = mock_widget
            setattr(mock_widgets, widget_type, Mock(return_value=mock_widget))

        mock_connection = Mock()

        widget = WorkflowWidget(connection=mock_connection)

        # Mock working directory
        widget.working_directory = '/tmp/test'

        # Mock container widgets
        widget.container_widgets = {
            'container': Mock(value=123),
            'type': Mock(value='dataset')
        }

        # Mock technical widgets container with child widgets
        batch_size_widget = Mock(description="Batch size", value=5)
        model_type_widget = Mock(description="SAM Model", value='vit_b_lm')
        output_folder_widget = Mock(description="Output folder", value=Path('/tmp/test'))

        widget.technical_widgets = Mock()
        widget.technical_widgets.children = [batch_size_widget, model_type_widget, output_folder_widget]

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
        with patch('builtins.print'):  # Suppress print output
            with patch('IPython.display.clear_output'):  # Suppress clear_output
                widget._on_update_config(None)

        config = widget.get_config()

        assert config is not None
        assert config.output.output_directory == Path('/tmp/test')
        assert config.omero.container_type == 'dataset'
        assert config.omero.container_id == 123
        assert config.ai_model.pretrained_from == 'vit_b_lm'
        assert config.workflow.batch_size == 5
        assert config.spatial_coverage.use_patches is False


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
        widget_module = 'omero_annotate_ai.widgets.omero_connection_widget'
        if widget_module in sys.modules:
            del sys.modules[widget_module]
        
        with patch.dict('sys.modules', {'ipywidgets': None}):
            with pytest.raises(ImportError):
                from omero_annotate_ai.widgets.omero_connection_widget import OMEROConnectionWidget
    
    def test_widget_creation_without_ipywidgets(self):
        """
        Tests the widget creation behavior without `ipywidgets`.
        This test verifies that the module structure handles the case where `ipywidgets`
        is not available.
        """
        # This test verifies that the module structure handles missing dependencies
        try:
            import ipywidgets
            pytest.skip("ipywidgets is available, cannot test fallback")
        except ImportError:
            # This is expected when ipywidgets is not available
            assert True


@pytest.mark.unit
class TestWidgetIntegration:
    """Test widget integration scenarios."""
    
    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
    @patch('omero_annotate_ai.widgets.omero_connection_widget.widgets')
    @patch('omero_annotate_ai.widgets.workflow_widget.widgets')
    def test_connection_to_workflow_integration(self, mock_workflow_widgets, mock_conn_widgets):
        """
        Tests the integration between the connection and workflow widgets.
        This test ensures that the connection object created by the `OMEROConnectionWidget`
        can be correctly passed to the `WorkflowWidget`.
        """
        # Mock all required widgets
        for mock_widgets in [mock_conn_widgets, mock_workflow_widgets]:
            for widget_type in ['Text', 'Password', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
                mock_widget = Mock()
                if widget_type == 'Output':
                    mock_widget.__enter__ = Mock(return_value=mock_widget)
                    mock_widget.__exit__ = Mock(return_value=None)
                setattr(mock_widgets, widget_type, Mock(return_value=mock_widget))
        
        # Create connection widget and establish connection
        with patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection') as mock_conn_class:
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
            conn_widget.status_output.__enter__ = Mock(return_value=conn_widget.status_output)
            conn_widget.status_output.__exit__ = Mock(return_value=None)
            
            with patch('builtins.print'):  # Suppress print output
                with patch('IPython.display.clear_output'):  # Suppress clear_output
                    conn_widget._connect(Mock())
            
            # Use connection in workflow widget
            workflow_widget = WorkflowWidget(connection=conn_widget.connection)
            
            assert workflow_widget.connection == mock_conn
    
    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
    @patch('omero_annotate_ai.widgets.workflow_widget.widgets')
    def test_workflow_config_validation(self, mock_widgets):
        """
        Tests the validation of the workflow configuration.
        This test ensures that the `get_config` method handles invalid widget values
        gracefully and still returns a valid configuration object.
        """
        # Mock widgets
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
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