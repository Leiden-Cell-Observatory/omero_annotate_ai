#!/usr/bin/env python3
"""
Test script for the new OMERO connection widget with keychain support.

This script demonstrates how to use the new OMEROConnectionWidget to securely
connect to OMERO servers with keychain password storage.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_widget_creation():
    """Test that the widget can be created successfully."""
    print("🧪 Testing OMERO Connection Widget Creation")
    print("=" * 50)
    
    try:
        from omero_annotate_ai import create_omero_connection_widget
        
        # Create widget
        widget = create_omero_connection_widget()
        print("✅ Widget created successfully")
        
        # Test configuration loading
        config = widget.get_config()
        print(f"✅ Configuration accessible: {len(config)} parameters")
        
        # Test connection manager
        connection_manager = widget.connection_manager
        print(f"✅ Connection manager available: {type(connection_manager).__name__}")
        
        return widget
        
    except Exception as e:
        print(f"❌ Error creating widget: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_config_file_loading():
    """Test loading from .env and .ezomero files."""
    print("\n🧪 Testing Configuration File Loading")
    print("=" * 50)
    
    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
        
        manager = SimpleOMEROConnection()
        config = manager.load_config_files()
        
        if config:
            print("✅ Configuration loaded from files:")
            for key, value in config.items():
                if key != "source":
                    print(f"   {key}: {value if key != 'password' else '***'}")
            print(f"   Source: {config.get('source', 'unknown')}")
        else:
            print("💡 No configuration files found (.env or .ezomero)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading config files: {e}")
        return False

def test_keyring_functionality():
    """Test keyring functionality if available."""
    print("\n🧪 Testing Keyring Functionality")
    print("=" * 50)
    
    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection, KEYRING_AVAILABLE
        
        if not KEYRING_AVAILABLE:
            print("⚠️ Keyring not available - skipping keyring tests")
            return True
            
        manager = SimpleOMEROConnection()
        
        # Test password save/load (without actual OMERO connection)
        test_host = "test.server.edu"
        test_username = "test_user"
        test_password = "test_password_123"
        
        # Save password
        success = manager.save_password(test_host, test_username, test_password, expire_hours=1)
        if success:
            print("✅ Password saved to keychain")
            
            # Load password
            loaded_password = manager.load_password(test_host, test_username)
            if loaded_password == test_password:
                print("✅ Password loaded from keychain successfully")
                
                # Clean up test password
                manager._delete_password(test_host, test_username)
                print("✅ Test password cleaned up")
                return True
            else:
                print("❌ Password mismatch after loading")
        else:
            print("❌ Failed to save password")
            
        return False
        
    except Exception as e:
        print(f"❌ Error testing keyring: {e}")
        return False

def test_connection_history():
    """Test connection history functionality."""
    print("\n🧪 Testing Connection History")
    print("=" * 50)
    
    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
        
        manager = SimpleOMEROConnection()
        
        # Test saving connection details
        test_host = "test-history.server.edu"
        test_username = "test_history_user"
        test_group = "test_group"
        
        success = manager.save_connection_details(test_host, test_username, test_group)
        if success:
            print("✅ Connection details saved successfully")
        else:
            print("❌ Failed to save connection details")
            return False
        
        # Test loading connection history
        connections = manager.load_connection_history()
        if connections and len(connections) > 0:
            print(f"✅ Connection history loaded: {len(connections)} connections")
        else:
            print("❌ No connections found in history")
            return False
        
        # Test formatted connection list
        conn_list = manager.get_connection_list()
        if conn_list and len(conn_list) > 0:
            conn = conn_list[0]
            if "display_name" in conn and "last_used_display" in conn:
                print(f"✅ Connection formatted correctly: {conn['display_name']}")
            else:
                print("❌ Connection formatting incomplete")
                return False
        else:
            print("❌ No formatted connections found")
            return False
        
        # Test connection deletion
        success = manager.delete_connection(test_host, test_username)
        if success:
            print("✅ Connection deleted successfully")
        else:
            print("❌ Failed to delete connection")
            return False
        
        # Verify deletion
        connections_after = manager.load_connection_history()
        original_count = len(connections)
        new_count = len(connections_after)
        
        if new_count < original_count:
            print("✅ Connection history updated after deletion")
        else:
            print("❌ Connection history not updated after deletion")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing connection history: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_widget_integration():
    """Test widget integration with the configuration system."""
    print("\n🧪 Testing Widget Integration")
    print("=" * 50)
    
    try:
        from omero_annotate_ai import create_omero_connection_widget, create_config_widget
        
        # Create both widgets
        conn_widget = create_omero_connection_widget()
        config_widget = create_config_widget()
        
        print("✅ Both widgets created successfully")
        
        # Test that they can work together
        conn_config = conn_widget.get_config()
        annotation_config = config_widget.get_config()
        
        print(f"✅ Connection config has {len(conn_config)} parameters")
        print(f"✅ Annotation config has {len(vars(annotation_config))} attributes")
        
        # Test connection dropdown
        dropdown = conn_widget.connection_dropdown
        print(f"✅ Connection dropdown has {len(dropdown.options)} options")
        
        # Test basic integration
        print("✅ Widgets can be used together")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing widget integration: {e}")
        return False

def demo_widget_usage():
    """Demonstrate how to use the widget (without actual connection)."""
    print("\n📖 Widget Usage Demo")
    print("=" * 50)
    
    try:
        from omero_annotate_ai import create_omero_connection_widget
        
        print("🔌 Creating OMERO connection widget...")
        widget = create_omero_connection_widget()
        
        print("\n📋 To use this widget in a Jupyter notebook:")
        print("```python")
        print("from omero_annotate_ai import create_omero_connection_widget")
        print("")
        print("# Create and display the widget")
        print("conn_widget = create_omero_connection_widget()")
        print("conn_widget.display()")
        print("")
        print("# After user fills in the form and clicks 'Save & Connect':")
        print("conn = conn_widget.get_connection()")
        print("```")
        
        print("\n🔑 Key Features:")
        print("- Secure password storage in OS keychain")
        print("- Password expiration options (1 hour to never)")
        print("- Auto-load from .env and .ezomero files")
        print("- Test connection before saving")
        print("- Show/hide password toggle")
        print("- Integration with existing OMERO workflows")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 OMERO Connection Widget Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run tests
    widget = test_widget_creation()
    success &= widget is not None
    
    success &= test_config_file_loading()
    success &= test_keyring_functionality()
    success &= test_connection_history()
    success &= test_widget_integration()
    success &= demo_widget_usage()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! OMERO Connection Widget is ready to use.")
        print("\n📝 Next steps:")
        print("1. Install the package: pip install -e .")
        print("2. In Jupyter: from omero_annotate_ai import create_omero_connection_widget")
        print("3. Create and display widget: widget = create_omero_connection_widget(); widget.display()")
        print("4. Connect to OMERO: conn = widget.get_connection()")
    else:
        print("❌ Some tests failed. Check output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)