"""Integration tests for OMERO functionality using docker-compose."""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
    from omero_annotate_ai.omero.omero_utils import (
        get_table_by_name,
        list_user_tables,
        validate_omero_permissions
    )
    from omero_annotate_ai.widgets.omero_connection_widget import create_omero_connection_widget
    import ezomero
    OMERO_AVAILABLE = True
except ImportError:
    OMERO_AVAILABLE = False


@pytest.mark.omero
@pytest.mark.integration
@pytest.mark.docker
class TestOMEROIntegration:
    """Integration tests for OMERO functionality."""
    
    def test_simple_connection_manager(self, omero_connection):
        """Test SimpleOMEROConnection against real OMERO."""
        conn_manager = SimpleOMEROConnection()
        
        # Test connection
        params = {
            "host": "localhost",
            "username": "root",
            "password": "omero",
            "secure": False
        }
        
        conn = conn_manager.connect(**params)
        assert conn is not None
        assert conn.isConnected()
        
        # Test connection retrieval
        last_conn = conn_manager.get_last_connection()
        assert last_conn is not None
        assert last_conn.isConnected()
        
        conn.close()
    
    def test_omero_utils_functions(self, omero_connection):
        """Test OMERO utility functions."""
        # Test table listing
        tables = list_user_tables(omero_connection)
        assert isinstance(tables, list)
        
        # Test permissions validation
        has_perms = validate_omero_permissions(omero_connection)
        assert isinstance(has_perms, bool)
        
        # Test table retrieval (should return None for non-existent table)
        table = get_table_by_name(omero_connection, "non_existent_table")
        assert table is None
    
    def test_connection_widget_creation(self):
        """Test that the connection widget can be created."""
        widget = create_omero_connection_widget()
        assert widget is not None
        
        # Test widget configuration
        config = widget.get_config()
        assert isinstance(config, dict)
        assert "host" in config
        assert "username" in config
    
    def test_connection_from_widget_config(self, docker_omero_server):
        """Test creating connection from widget configuration."""
        conn_manager = SimpleOMEROConnection()
        
        widget_config = {
            "host": docker_omero_server["host"],
            "username": docker_omero_server["user"], 
            "password": docker_omero_server["password"],
            "port": docker_omero_server["port"],
            "secure": docker_omero_server["secure"]
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
        """Test connection history functionality."""
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
            assert "timestamp" in entry
    
    def test_keychain_integration(self):
        """Test keychain password storage if available."""
        conn_manager = SimpleOMEROConnection()
        
        # Test password storage/retrieval
        test_key = "test_omero_server"
        test_password = "test_password"
        
        # Save password
        success = conn_manager.save_password(test_key, test_password)
        
        if success:  # Only test if keychain is available
            # Retrieve password
            retrieved = conn_manager.get_password(test_key)
            assert retrieved == test_password
            
            # Clean up
            conn_manager.delete_password(test_key)


@pytest.mark.unit
class TestOMEROConnectionManagerUnit:
    """Unit tests that don't require OMERO server."""
    
    def test_connection_manager_creation(self):
        """Test that connection manager can be created."""
        conn_manager = SimpleOMEROConnection()
        assert conn_manager is not None
        assert hasattr(conn_manager, 'connect')
        assert hasattr(conn_manager, 'get_last_connection')
    
    def test_config_loading_without_files(self):
        """Test config loading when no config files exist."""
        conn_manager = SimpleOMEROConnection()
        config = conn_manager.load_env_config()
        assert isinstance(config, dict)
        # Should have default empty values
        assert config.get("host", "") == ""
        assert config.get("username", "") == ""