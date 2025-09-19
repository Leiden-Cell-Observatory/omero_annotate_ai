"""Integration tests for OMERO functionality using docker-compose."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
    from omero_annotate_ai.omero.omero_utils import (
        get_table_by_name,
        list_user_tables,
        validate_omero_permissions
    )
    OMERO_AVAILABLE = True
except ImportError:
    OMERO_AVAILABLE = False

# Always import the widget for test discovery, even if OMERO is unavailable
try:
    from omero_annotate_ai.widgets.omero_connection_widget import create_omero_connection_widget
except ImportError:
    create_omero_connection_widget = None


@pytest.mark.omero
@pytest.mark.integration
class TestOMEROIntegration:
    """Integration tests for OMERO functionality."""
    
    def test_simple_connection_manager(self, omero_connection):
        """
        Tests the `SimpleOMEROConnection` class against a real OMERO server.
        This test ensures that the `SimpleOMEROConnection` class can correctly
        connect to an OMERO server and that the connection object is valid.
        """
        conn_manager = SimpleOMEROConnection()

        # Test connection

        params = {
            "host": "localhost",
            "username": "root",
            "password": "omero",
            "secure": True,
            "port": 6064
        }

        print(f"[DEBUG] Attempting OMERO connection with params: {params}")
        conn = conn_manager.connect(**params)
        if conn is None:
            print("[ERROR] OMERO connection failed. Check if the server is running and accessible at the specified host/port.")
        assert conn is not None
        assert conn.isConnected()

        # Test connection retrieval
        last_conn = conn_manager.get_last_connection()
        assert last_conn is not None
        assert last_conn.isConnected()

        conn.close()
    
    def test_omero_utils_functions(self, omero_connection):
        """
        Tests the OMERO utility functions against a real OMERO server.
        This test ensures that the utility functions in `omero_utils.py` can
        correctly interact with a real OMERO server.
        """
        # Test table listing
        tables = list_user_tables(omero_connection)
        assert isinstance(tables, list)
    
    @pytest.mark.skipif(not OMERO_AVAILABLE, reason="OMERO dependencies not available")
    def test_connection_widget_creation(self):
        """
        Tests that the OMERO connection widget can be created.
        This test ensures that the `create_omero_connection_widget` function can
        be called without errors and that it returns a valid widget object.
        """
        # Mock the problematic print statements that cause Unicode encoding issues
        with patch('builtins.print'):
            widget = create_omero_connection_widget()
            assert widget is not None
            
            # Test widget configuration
            config = widget.get_config()
            assert isinstance(config, dict)
            assert "host" in config
            assert "username" in config
    
    def test_connection_from_widget_config(self, docker_omero_server):
        """
        Tests creating a connection from the widget's configuration.
        This test ensures that the `SimpleOMEROConnection` class can correctly
        create a connection from the configuration dictionary returned by the
        connection widget.
        """
        conn_manager = SimpleOMEROConnection()
        
        widget_config = {
            "host": docker_omero_server["host"],
            "username": docker_omero_server["user"], 
            "password": docker_omero_server["password"],
            "secure": docker_omero_server["secure"],
            "port": docker_omero_server.get("port", 6064)
        }
        
        conn = conn_manager.create_connection_from_config(widget_config)
        assert conn is not None
        assert conn.isConnected()
        
        conn.close()


@pytest.mark.omero
@pytest.mark.integration  
class TestOMEROConnectionManager:
    """Test the OMERO connection management features."""
    
    def test_connection_history(self, omero_connection):
        """
        Tests the connection history functionality.
        This test ensures that the `SimpleOMEROConnection` class can correctly
        save and load connection details to and from the connection history.
        """
        conn_manager = SimpleOMEROConnection()
        
        # Test saving connection details
        conn_manager.save_connection_details(
            host="localhost",
            username="root", 
            group=""
        )
        
        # Test loading connection history
        history = conn_manager.load_connection_history()
        assert isinstance(history, list)
        
        # Should have at least one entry now
        if history:
            entry = history[0]
            assert "host" in entry
            assert "username" in entry
            assert "created_at" in entry or "last_used" in entry  # Check for actual timestamp fields
    
    def test_keychain_integration(self):
        """
        Tests the keychain integration for password storage.
        This test ensures that the `SimpleOMEROConnection` class can correctly
        save and load passwords from the system keychain, if it is available.
        """
        conn_manager = SimpleOMEROConnection()
        
        # Test password storage/retrieval
        test_host = "test_omero_server"
        test_username = "test_user"
        test_password = "test_password"
        
        # Save password
        success = conn_manager.save_password(test_host, test_username, test_password)
        
        if success:  # Only test if keychain is available
            # Retrieve password
            retrieved = conn_manager.load_password(test_host, test_username)
            assert retrieved == test_password
            
            # Clean up (using private method since there's no public delete method)
            conn_manager._delete_password(test_host, test_username)


@pytest.mark.unit
class TestOMEROConnectionManagerUnit:
    """Unit tests that don't require OMERO server."""
    
    def test_connection_manager_creation(self):
        """
        Tests that the `SimpleOMEROConnection` manager can be created.
        This is a simple smoke test to ensure that the `SimpleOMEROConnection`
        class can be instantiated without errors.
        """
        conn_manager = SimpleOMEROConnection()
        assert conn_manager is not None
        assert hasattr(conn_manager, 'connect')
        assert hasattr(conn_manager, 'get_last_connection')
    
    def test_config_loading_without_files(self):
        """
        Tests the configuration loading when no config files exist.
        This test ensures that the `load_config_files` method handles the case
        where no environment or configuration files are present and returns a
        dictionary with default empty values.
        """
        conn_manager = SimpleOMEROConnection()
        
        # Environment variables to clear
        env_vars_to_clear = ['HOST', 'USER_NAME', 'GROUP', 'OMERO_USER', 'OMERO_PASSWORD', 'OMERO_HOST', 'OMERO_GROUP', 'OMERO_PORT']
        
        with patch.dict(os.environ, {}, clear=True), \
            patch.object(Path, 'exists', return_value=False), \
            patch.object(conn_manager, 'load_connection_history', return_value=[]), \
            patch('builtins.print'):  # Suppress print statements
            
            config = conn_manager.load_config_files()
            
            assert isinstance(config, dict)
            # Should be completely empty when no config sources exist
            assert config == {}
            # Or if you expect default empty values:
            assert config.get("host", "") == ""
            assert config.get("username", "") == ""
            assert config.get("group", "") == ""