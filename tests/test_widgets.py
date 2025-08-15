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
        """Test widget initialization."""
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
        """Test connect button callback."""
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
        """Test connection failure handling."""
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
        """Test loading connection parameters from environment."""
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
        """Test workflow widget initialization."""
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
        """Test OMERO container selection."""
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
        """Test configuration generation from widget values."""
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
        """Test that widget imports fail gracefully when dependencies are missing."""
        with patch.dict('sys.modules', {'ipywidgets': None}):
            with pytest.raises(ImportError):
                from omero_annotate_ai.widgets.omero_connection_widget import OMEROConnectionWidget
    
    def test_widget_creation_without_ipywidgets(self):
        """Test widget creation behavior without ipywidgets."""
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
        """Test integration between connection and workflow widgets."""
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
        """Test workflow configuration validation."""
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


@pytest.mark.unit
class TestTableUpdates:
    """Test table update functionality."""
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.delete_table')
    def test_update_tracking_table_rows_success(self, mock_delete_table, mock_ezomero):
        """Test successful tracking table rows update."""
        from omero_annotate_ai.omero.omero_functions import update_tracking_table_rows
        
        # Mock connection
        mock_conn = Mock()
        
        # Mock existing table data
        mock_df = pd.DataFrame({
            'image_id': [1, 2, 3],
            'processed': [False, False, False],
            'label_id': ['None', 'None', 'None'],
            'roi_id': ['None', 'None', 'None'],
            'annotation_type': ['segmentation_mask', 'segmentation_mask', 'segmentation_mask'],
            'annotation_creation_time': ['None', 'None', 'None']
        })
        mock_ezomero.get_table.return_value = mock_df
        
        # Mock file annotation for table name
        mock_file_ann = Mock()
        mock_file = Mock()
        mock_file.getName.return_value = "test_table.csv"
        mock_file_ann.getFile.return_value = mock_file
        mock_conn.getObject.return_value = mock_file_ann
        
        # Mock table creation
        mock_ezomero.post_table.return_value = 456
        mock_delete_table.return_value = True
        
        # Test update
        result = update_tracking_table_rows(
            conn=mock_conn,
            table_id=123,
            row_indices=[0, 1],
            status="completed",
            annotation_type="segmentation_mask",
            label_id=789,
            roi_id=101,
            container_type="dataset",
            container_id=555
        )
        
        # Verify result
        assert result == 456
        mock_ezomero.get_table.assert_called_once_with(mock_conn, 123)
        mock_delete_table.assert_called_once_with(mock_conn, 123)
        mock_ezomero.post_table.assert_called_once()
        
        # Verify the posted table has correct data
        call_args = mock_ezomero.post_table.call_args
        updated_df = call_args[1]['table']  # table parameter in post_table call
        
        # Check that rows were updated correctly
        assert updated_df.loc[0, 'processed'] == True
        assert updated_df.loc[1, 'processed'] == True
        assert updated_df.loc[2, 'processed'] == False  # Unchanged
        assert updated_df.loc[0, 'label_id'] == '789'
        assert updated_df.loc[1, 'label_id'] == '789'
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    def test_update_tracking_table_rows_table_not_found(self, mock_ezomero):
        """Test update when table cannot be retrieved."""
        from omero_annotate_ai.omero.omero_functions import update_tracking_table_rows
        
        mock_conn = Mock()
        mock_ezomero.get_table.return_value = None
        
        result = update_tracking_table_rows(
            conn=mock_conn,
            table_id=123,
            row_indices=[0],
            status="completed",
            annotation_type="segmentation_mask",
            container_type="dataset",
            container_id=555
        )
        
        # Should return original table_id
        assert result == 123
        mock_ezomero.get_table.assert_called_once_with(mock_conn, 123)
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    def test_update_tracking_table_rows_no_ezomero(self, mock_ezomero):
        """Test update when ezomero is not available."""
        from omero_annotate_ai.omero.omero_functions import update_tracking_table_rows
        
        # Mock ezomero as None (simulating import failure)
        with patch('omero_annotate_ai.omero.omero_functions.ezomero', None):
            mock_conn = Mock()
            
            with pytest.raises(ImportError, match="ezomero is required"):
                update_tracking_table_rows(
                    conn=mock_conn,
                    table_id=123,
                    row_indices=[0],
                    status="completed",
                    annotation_type="segmentation_mask",
                    container_type="dataset",
                    container_id=555
                )
    
    @patch('omero_annotate_ai.omero.omero_functions.ezomero')
    @patch('omero_annotate_ai.omero.omero_functions.delete_table')
    def test_update_tracking_table_rows_failed_status(self, mock_delete_table, mock_ezomero):
        """Test tracking table update with failed status."""
        from omero_annotate_ai.omero.omero_functions import update_tracking_table_rows
        
        mock_conn = Mock()
        
        # Mock existing table data
        mock_df = pd.DataFrame({
            'image_id': [1, 2],
            'processed': [False, False],
            'label_id': ['None', 'None'],
            'roi_id': ['None', 'None'],
            'annotation_type': ['segmentation_mask', 'segmentation_mask'],
            'annotation_creation_time': ['None', 'None']
        })
        mock_ezomero.get_table.return_value = mock_df
        
        # Mock file annotation
        mock_file_ann = Mock()
        mock_file = Mock()
        mock_file.getName.return_value = "test_table.csv"
        mock_file_ann.getFile.return_value = mock_file
        mock_conn.getObject.return_value = mock_file_ann
        
        mock_ezomero.post_table.return_value = 789
        mock_delete_table.return_value = True
        
        # Test with failed status
        result = update_tracking_table_rows(
            conn=mock_conn,
            table_id=123,
            row_indices=[0],
            status="failed",
            annotation_type="segmentation_mask",
            container_type="dataset",
            container_id=555
        )
        
        # Verify the posted table has correct data (failed status)
        call_args = mock_ezomero.post_table.call_args
        updated_df = call_args[1]['table']
        
        # Check that processed is False for failed status
        assert updated_df.loc[0, 'processed'] == False
        # Label and ROI IDs should not be updated for failed status
        assert updated_df.loc[0, 'label_id'] == 'None'
        assert updated_df.loc[0, 'roi_id'] == 'None'


@pytest.mark.unit
class TestWidgetErrorHandling:
    """Test widget error handling and edge cases."""
    
    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
    @patch('omero_annotate_ai.widgets.omero_connection_widget.ipywidgets')
    def test_connection_widget_with_invalid_inputs(self, mock_ipywidgets):
        """Test connection widget with invalid inputs."""
        # Mock widgets
        for widget_type in ['Text', 'Password', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox']:
            setattr(mock_ipywidgets, widget_type, Mock(return_value=Mock()))
        
        widget = OMEROConnectionWidget()
        
        # Test with empty host
        widget.host_widget = Mock(value="")
        widget.user_widget = Mock(value="user")
        widget.password_widget = Mock(value="pass")
        widget.port_widget = Mock(value=4064)
        
        widget._on_connect_click(Mock())
        
        # Should handle gracefully
        assert widget.connection is None
    
    @pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="Widget dependencies not available")
    @patch('omero_annotate_ai.widgets.workflow_widget.ipywidgets')
    def test_workflow_widget_with_no_connection(self, mock_ipywidgets):
        """Test workflow widget with no OMERO connection."""
        # Mock widgets
        for widget_type in ['Text', 'Dropdown', 'IntText', 'Checkbox', 'Button', 'Output', 'VBox', 'HBox']:
            setattr(mock_ipywidgets, widget_type, Mock(return_value=Mock()))
        
        # Create widget without connection
        widget = WorkflowWidget(connection=None)
        
        assert widget.connection is None
        
        # Should still be able to create basic config
        widget.working_dir_widget = Mock(value='/tmp')
        widget.container_type_widget = Mock(value='dataset')
        widget.container_id_widget = Mock(value=1)
        
        config = widget.get_config()
        assert config is not None