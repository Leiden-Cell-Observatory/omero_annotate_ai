#!/usr/bin/env python3
"""
Demo script showing how to use the OMERO Connection Widget.

This script demonstrates the basic usage of the new OMEROConnectionWidget
which provides secure password storage and easy OMERO server connections.
"""

import sys
import os

def demo_basic_usage():
    """Demonstrate basic widget usage."""
    print("🔌 OMERO Connection Widget Demo")
    print("=" * 50)
    
    # Import the widget
    from omero_annotate_ai import create_omero_connection_widget
    
    print("📋 Creating OMERO connection widget...")
    
    # Create the widget
    conn_widget = create_omero_connection_widget()
    
    print("✅ Widget created successfully!")
    print("\n🔧 Widget Features:")
    print("- Connection history dropdown with saved connections")
    print("- Auto-loads from .env and .ezomero files")
    print("- Secure password storage in OS keychain")
    print("- Password expiration options")
    print("- Connection testing")
    print("- Show/hide password toggle")
    print("- Save/delete connection management")
    
    print("\n📊 Current configuration:")
    config = conn_widget.get_config()
    for key, value in config.items():
        if key not in ['password'] and value:
            print(f"  {key}: {value}")
    
    print("\n💡 To use in Jupyter notebook:")
    print("```python")
    print("# Import and create widget")
    print("from omero_annotate_ai import create_omero_connection_widget")
    print("conn_widget = create_omero_connection_widget()")
    print("")
    print("# Display the widget")
    print("conn_widget.display()")
    print("")
    print("# After user interaction, get the connection")
    print("conn = conn_widget.get_connection()")
    print("```")
    
    return conn_widget

def demo_integration_with_pipeline():
    """Demonstrate integration with the annotation pipeline."""
    print("\n🔗 Integration with Annotation Pipeline")
    print("=" * 50)
    
    from omero_annotate_ai import create_omero_connection_widget, create_config_widget, AnnotationPipeline
    
    print("📋 Creating both widgets...")
    
    # Create widgets
    conn_widget = create_omero_connection_widget()
    config_widget = create_config_widget()
    
    print("✅ Both widgets created!")
    
    print("\n💡 Complete workflow in Jupyter:")
    print("```python")
    print("from omero_annotate_ai import create_omero_connection_widget, create_config_widget, AnnotationPipeline")
    print("")
    print("# Step 1: Create OMERO connection")
    print("conn_widget = create_omero_connection_widget()")
    print("conn_widget.display()")
    print("conn = conn_widget.get_connection()")
    print("")
    print("# Step 2: Configure annotation pipeline")
    print("config_widget = create_config_widget()")
    print("config_widget.display()")
    print("config = config_widget.get_config()")
    print("")
    print("# Step 3: Run annotation pipeline")
    print("pipeline = AnnotationPipeline(config, conn)")
    print("table_id, images = pipeline.run_full_workflow()")
    print("```")
    
    return True

def demo_connection_history():
    """Demonstrate connection history features."""
    print("\n📚 Connection History Features")
    print("=" * 50)
    
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
    
    manager = SimpleOMEROConnection()
    
    print("🔧 Connection History Management:")
    print("  - Automatic saving of successful connections")
    print("  - Connection dropdown with recent connections")
    print("  - Easy selection and auto-population")
    print("  - Delete unwanted connection history")
    print("")
    
    # Show current connection history
    connections = manager.load_connection_history()
    if connections:
        print(f"📋 Current saved connections: {len(connections)}")
        for i, conn in enumerate(connections[:3]):  # Show first 3
            conn_list = manager.get_connection_list()
            if conn_list:
                display_name = conn_list[i]["display_name"]
                last_used = conn_list[i]["last_used_display"]
                print(f"  {i+1}. {display_name} (last used: {last_used})")
        if len(connections) > 3:
            print(f"  ... and {len(connections) - 3} more")
    else:
        print("📋 No saved connections found")
    
    print("\n💡 How connection history works:")
    print("  1. When you successfully connect, details are automatically saved")
    print("  2. Host, username, and group are stored (passwords in keychain)")
    print("  3. Dropdown shows recent connections for easy selection")
    print("  4. Use 'Save Connection' to save without connecting")
    print("  5. Use 'Delete Connection' to remove unwanted entries")
    
    print("\n🔄 Priority order for auto-loading:")
    print("  1. .env file (development priority)")
    print("  2. Connection history (most recent)")
    print("  3. .ezomero file")
    print("  4. Manual entry")

def demo_security_features():
    """Demonstrate security features."""
    print("\n🔐 Security Features")
    print("=" * 50)
    
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection, KEYRING_AVAILABLE
    
    print(f"🔑 Keyring available: {KEYRING_AVAILABLE}")
    
    if KEYRING_AVAILABLE:
        print("✅ Secure password storage enabled")
        print("  - Windows: Windows Credential Manager")
        print("  - macOS: Keychain Access")
        print("  - Linux: GNOME Keyring/KDE KWallet")
        print("")
        print("🕒 Password expiration options:")
        print("  - 1 hour (high security)")
        print("  - 8 hours (work session)")
        print("  - 24 hours (daily use)")
        print("  - 1 week (convenience)")
        print("  - Never expires (maximum convenience)")
    else:
        print("⚠️ Keyring backend not available")
        print("  This is common in some environments (like WSL)")
        print("  Widget still works with manual password entry")
        print("  Consider installing: pip install keyrings.alt")
    
    print("\n🛡️ Security Best Practices:")
    print("  - Passwords never stored in config files")
    print("  - Uses OS-native secure storage")
    print("  - Automatic expiration and cleanup")
    print("  - Fallback to .env files for development")

def main():
    """Run the complete demo."""
    print("🚀 OMERO Connection Widget Complete Demo")
    print("=" * 60)
    
    try:
        # Run demos
        widget = demo_basic_usage()
        demo_integration_with_pipeline()
        demo_connection_history()
        demo_security_features()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n📝 Next Steps:")
        print("1. Try the widget in a Jupyter notebook")
        print("2. Test connection to your OMERO server")
        print("3. Use with annotation pipeline for micro-SAM workflows")
        print("4. Explore password storage and expiration options")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)