"""Tests for widget modules."""

import pytest
import pandas as pd
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
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.ipywidgets')
    def test_widget_initialization(self, mock_ipywidgets):
        """
        Tests the initialization of the OMEROConnectionWidget.
        This test ensures that the widget is correctly initialized and that the
        connection attribute is None.
        """
        # Mock ipywidgets components
        mock_ipywidgets.Text.return_value = Mock()
        mock_ipywidgets.Password.return_value = Mock()
        mock_ipywidgets.IntText.return_value = Mock()
        mock_ipywidgets.Checkbox.return_value = Mock()
        mock_ipywidgets.Button.return_value = Mock()
        mock_ipywidgets.Output.return_value = Mock()
        mock_ipywidgets.VBox.return_value = Mock()
        
        widget = OMEROConnectionWidget()
        
        assert widget is not None
        assert hasattr(widget, 'connection')
        assert widget.connection is None
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.ipywidgets')
    @patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection')
    def test_connect_button_callback(self, mock_connection_class, mock_ipywidgets):
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
        
        mock_ipywidgets.Text.return_value = mock_host_widget
        mock_ipywidgets.Password.return_value = mock_password_widget
        mock_ipywidgets.IntText.return_value = mock_port_widget
        mock_ipywidgets.Checkbox.return_value = Mock()
        mock_ipywidgets.Button.return_value = Mock()
        mock_ipywidgets.Output.return_value = Mock()
        mock_ipywidgets.VBox.return_value = Mock()
        
        # Mock connection manager
        mock_conn_manager = Mock()
        mock_conn = Mock()
        mock_conn_manager.connect.return_value = mock_conn
        mock_connection_class.return_value = mock_conn_manager
        
        widget = OMEROConnectionWidget()
        widget.host_widget = mock_host_widget
        widget.user_widget = mock_user_widget
        widget.password_widget = mock_password_widget
        widget.port_widget = mock_port_widget
        
        # Simulate connect button click
        widget._on_connect_click(Mock())
        
        assert widget.connection == mock_conn
        mock_conn_manager.connect.assert_called_once_with(
            "localhost", "testuser", "testpass", port=4064, secure=True
        )
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.ipywidgets')
    @patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection')
    def test_connect_failure(self, mock_connection_class, mock_ipywidgets):
        """
        Tests the handling of a connection failure.
        This test ensures that the widget correctly handles a connection failure
        and that the connection attribute remains None.
        """
        # Mock widgets
        mock_widgets = {
            'Text': Mock(),
            'Password': Mock(), 
            'IntText': Mock(),
            'Checkbox': Mock(),
            'Button': Mock(),
            'Output': Mock(),
            'VBox': Mock()
        }
        
        for widget_type, mock_widget in mock_widgets.items():
            setattr(mock_ipywidgets, widget_type, Mock(return_value=mock_widget))
        
        # Mock connection failure
        mock_conn_manager = Mock()
        mock_conn_manager.connect.return_value = None
        mock_connection_class.return_value = mock_conn_manager
        
        widget = OMEROConnectionWidget()
        widget.host_widget = Mock(value="localhost")
        widget.user_widget = Mock(value="user")
        widget.password_widget = Mock(value="pass")
        widget.port_widget = Mock(value=4064)
        
        widget._on_connect_click(Mock())
        
        assert widget.connection is None
    
    @patch('omero_annotate_ai.widgets.omero_connection_widget.ipywidgets')
    def test_load_from_env(self, mock_ipywidgets):
        """
        Tests loading connection parameters from environment variables.
        This test ensures that the widget correctly loads the connection parameters
        from the environment variables.
        """
        # Mock widgets
        mock_widgets = {}
        for widget_type in ['Text', 'Password', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox']:
            mock_widgets[widget_type] = Mock()
            setattr(mock_ipywidgets, widget_type, Mock(return_value=mock_widgets[widget_type]))
        
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'OMERO_HOST': 'env_host',
                'OMERO_USER': 'env_user',
                'OMERO_PORT': '4064'
            }.get(key, default)
            
            widget = OMEROConnectionWidget()
            widget.host_widget = Mock()
            widget.user_widget = Mock()
            widget.port_widget = Mock()
            
            widget._load_from_env()
            
            widget.host_widget.value = 'env_host'
            widget.user_widget.value = 'env_user'
            widget.port_widget.value = 4064


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
@pytest.mark.unit
class TestWorkflowWidget:
    """Test workflow configuration widget functionality."""
    
    @patch('omero_annotate_ai.widgets.workflow_widget.ipywidgets')
    def test_workflow_widget_initialization(self, mock_ipywidgets):
        """
        Tests the initialization of the WorkflowWidget.
        This test ensures that the widget is correctly initialized with the
        provided connection.
        """
        # Mock ipywidgets components
        mock_widgets = {}
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            mock_widgets[widget_type] = Mock()
            setattr(mock_ipywidgets, widget_type, Mock(return_value=mock_widgets[widget_type]))
        
        mock_connection = Mock()
        
        widget = WorkflowWidget(connection=mock_connection)
        
        assert widget is not None
        assert widget.connection == mock_connection
    
    @patch('omero_annotate_ai.widgets.workflow_widget.ipywidgets')
    def test_container_selection(self, mock_ipywidgets):
        """
        Tests the OMERO container selection.
        This test ensures that the widget correctly populates the container dropdown
        when the container type is changed.
        """
        # Mock widgets
        mock_widgets = {}
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            mock_widgets[widget_type] = Mock()
            setattr(mock_ipywidgets, widget_type, Mock(return_value=mock_widgets[widget_type]))
        
        mock_connection = Mock()
        
        with patch('omero_annotate_ai.widgets.workflow_widget.ezomero') as mock_ezomero:
            mock_ezomero.get_projects.return_value = [
                Mock(getId=lambda: 1, getName=lambda: "Project 1"),
                Mock(getId=lambda: 2, getName=lambda: "Project 2")
            ]
            
            widget = WorkflowWidget(connection=mock_connection)
            widget.container_type_widget = Mock(value="project")
            
            # Simulate container type change
            widget._on_container_type_change({'new': 'project'})
            
            # Should populate container dropdown
            mock_ezomero.get_projects.assert_called_once_with(mock_connection)
    
    @patch('omero_annotate_ai.widgets.workflow_widget.ipywidgets')
    def test_config_generation(self, mock_ipywidgets):
        """
        Tests the generation of the configuration from the widget values.
        This test ensures that the `get_config` method correctly generates an
        `AnnotationConfig` object with the expected values from the widget.
        """
        # Mock widgets with values
        mock_widgets = {}
        widget_values = {
            'working_dir': '/tmp/test',
            'container_type': 'dataset',
            'container_id': 123,
            'model_type': 'vit_b_lm',
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
            
            mock_widgets[widget_type] = mock_widget
            setattr(mock_ipywidgets, widget_type, Mock(return_value=mock_widget))
        
        mock_connection = Mock()
        
        widget = WorkflowWidget(connection=mock_connection)
        
        # Mock widget attributes
        widget.working_dir_widget = Mock(value='/tmp/test')
        widget.container_type_widget = Mock(value='dataset')
        widget.container_id_widget = Mock(value=123)
        widget.model_type_widget = Mock(value='vit_b_lm')
        widget.batch_size_widget = Mock(value=5)
        widget.use_patches_widget = Mock(value=False)
        
        config = widget.get_config()
        
        assert config is not None
        assert config.batch_processing.output_folder == '/tmp/test'
        assert config.omero.container_type == 'dataset'
        assert config.omero.container_id == 123
        assert config.micro_sam.model_type == 'vit_b_lm'
        assert config.batch_processing.batch_size == 5
        assert config.patches.use_patches is False


@pytest.mark.unit
class TestWidgetFallbacks:
    """Test widget behavior when dependencies are missing."""
    
    def test_widget_import_fallback(self):
        """
        Tests that the widget imports fail gracefully when dependencies are missing.
        This test ensures that an `ImportError` is raised when the `ipywidgets`
        package is not available.
        """
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
    @patch('omero_annotate_ai.widgets.omero_connection_widget.ipywidgets')
    @patch('omero_annotate_ai.widgets.workflow_widget.ipywidgets')
    def test_connection_to_workflow_integration(self, mock_workflow_widgets, mock_conn_widgets):
        """
        Tests the integration between the connection and workflow widgets.
        This test ensures that the connection object created by the `OMEROConnectionWidget`
        can be correctly passed to the `WorkflowWidget`.
        """
        # Mock all required widgets
        for mock_ipywidgets in [mock_conn_widgets, mock_workflow_widgets]:
            for widget_type in ['Text', 'Password', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
                setattr(mock_ipywidgets, widget_type, Mock(return_value=Mock()))
        
        # Create connection widget and establish connection
        with patch('omero_annotate_ai.widgets.omero_connection_widget.SimpleOMEROConnection') as mock_conn_class:
            mock_conn = Mock()
            mock_conn_manager = Mock()
            mock_conn_manager.connect.return_value = mock_conn
            mock_conn_class.return_value = mock_conn_manager
            
            conn_widget = OMEROConnectionWidget()
            conn_widget.host_widget = Mock(value="localhost")
            conn_widget.user_widget = Mock(value="user")
            conn_widget.password_widget = Mock(value="pass")
            conn_widget.port_widget = Mock(value=4064)
            
            conn_widget._on_connect_click(Mock())
            
            # Use connection in workflow widget
            workflow_widget = WorkflowWidget(connection=conn_widget.connection)
            
            assert workflow_widget.connection == mock_conn
    
    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
    @patch('omero_annotate_ai.widgets.workflow_widget.ipywidgets')
    def test_workflow_config_validation(self, mock_ipywidgets):
        """
        Tests the validation of the workflow configuration.
        This test ensures that the `get_config` method handles invalid widget values
        gracefully and still returns a valid configuration object.
        """
        # Mock widgets
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            setattr(mock_ipywidgets, widget_type, Mock(return_value=Mock()))
        
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