#!/usr/bin/env python3
"""
Example script demonstrating the new project-based annotation workflow.

This example shows how to:
1. Connect to OMERO
2. Select a project and manage annotation tables
3. Create or continue annotation workflows with unique table names
4. Run the annotation pipeline

Usage:
    python project_annotation_workflow.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Demonstrate the project annotation workflow."""
    
    print("🎯 Project-Based Annotation Workflow Demo")
    print("=" * 50)
    
    # Step 1: Create OMERO connection
    print("\n1️⃣ Creating OMERO Connection")
    try:
        from omero_annotate_ai import create_omero_connection_widget
        
        # Create connection widget
        conn_widget = create_omero_connection_widget()
        print("✅ Connection widget created")
        
        # In a Jupyter notebook, you would display and use the widget:
        # display(conn_widget.display())
        # connection = conn_widget.get_connection()
        
        # For this demo, we'll simulate having a connection
        connection = None  # Replace with actual connection in real use
        
    except Exception as e:
        print(f"❌ Error creating connection widget: {e}")
        return
    
    # Step 2: Create project annotation widget
    print("\n2️⃣ Creating Project Annotation Widget")
    try:
        from omero_annotate_ai import create_project_annotation_widget
        
        # Create project annotation widget
        project_widget = create_project_annotation_widget(connection)
        print("✅ Project annotation widget created")
        
        # In a Jupyter notebook, you would display and use the widget:
        # display(project_widget.display())
        # project_config = project_widget.get_configuration()
        
        # For this demo, we'll simulate project configuration
        project_config = {
            'project_id': 123,
            'action': 'new',  # or 'continue'
            'table_name': 'microsam_demo_project_20250107_120000',
            'roi_namespace': 'omero_annotate_ai.table.microsam_demo_project_20250107_120000'
        }
        print(f"📊 Simulated project config: {project_config}")
        
    except Exception as e:
        print(f"❌ Error creating project widget: {e}")
        return
    
    # Step 3: Create annotation configuration
    print("\n3️⃣ Creating Annotation Configuration")
    try:
        from omero_annotate_ai import create_default_config
        
        # Create default configuration
        config = create_default_config()
        
        # Customize configuration for your needs
        config.omero.container_type = "project"
        config.omero.container_id = project_config['project_id']
        config.microsam.model_type = "vit_b_lm"
        config.training.trainingset_name = "demo_training"
        
        print("✅ Annotation configuration created")
        print(f"   Container: {config.omero.container_type} {config.omero.container_id}")
        print(f"   Model: {config.microsam.model_type}")
        
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return
    
    # Step 4: Create and run pipeline
    print("\n4️⃣ Creating Annotation Pipeline")
    try:
        from omero_annotate_ai import create_pipeline
        
        # Create pipeline with project configuration
        pipeline = create_pipeline(
            config=config,
            conn=connection,
            project_config=project_config
        )
        print("✅ Pipeline created with project configuration")
        
        # Show what the pipeline would do
        print(f"   Table name: {project_config['table_name']}")
        print(f"   ROI namespace: {project_config['roi_namespace']}")
        print(f"   Action: {project_config['action']}")
        
        # In real usage, you would run the pipeline:
        # if connection:
        #     table_id, results = pipeline.run_full_workflow()
        #     print(f"✅ Pipeline completed. Table ID: {table_id}")
        
        print("✅ Pipeline ready to run (connection required)")
        
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return
    
    # Step 5: Demonstrate table management functions
    print("\n5️⃣ Demonstrating Table Management Functions")
    try:
        from omero_annotate_ai.omero.omero_functions import (
            generate_unique_table_name,
            create_roi_namespace_for_table
        )
        
        # Generate unique table names
        if connection:
            unique_name = generate_unique_table_name(connection, 123, "my_annotation")
            print(f"   Unique table name: {unique_name}")
        else:
            print("   (Connection required for unique name generation)")
        
        # Create ROI namespace
        roi_namespace = create_roi_namespace_for_table("demo_table")
        print(f"   ROI namespace: {roi_namespace}")
        
        print("✅ Table management functions demonstrated")
        
    except Exception as e:
        print(f"❌ Error demonstrating functions: {e}")
    
    print("\n🎉 Demo Complete!")
    print("\nNext Steps:")
    print("1. Run this in a Jupyter notebook with actual OMERO connection")
    print("2. Use the widgets interactively to select projects and manage tables")
    print("3. Run the pipeline to perform actual annotations")
    print("\nKey Benefits:")
    print("✅ Unique table names prevent conflicts")
    print("✅ Project-based organization")
    print("✅ Can continue interrupted workflows")
    print("✅ Consistent ROI naming")


if __name__ == "__main__":
    main()