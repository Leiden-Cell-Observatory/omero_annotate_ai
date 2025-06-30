#!/usr/bin/env python3
"""
Test script for the omero-annotate-ai package annotation pipeline.

This script replicates the functionality of the Jupyter notebook 
omero-annotate-ai-batch.ipynb in a standalone Python script format.

Usage:
    python test_annotation_pipeline.py [config.yaml]
    
If no config file is provided, it will use test_config.yaml
"""

import sys
import os
import tempfile
import shutil
import argparse
from pathlib import Path
from getpass import getpass
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description='Test omero-annotate-ai annotation pipeline')
    parser.add_argument('config', nargs='?', default='test_config.yaml',
                       help='YAML configuration file (default: test_config.yaml)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running the pipeline')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory')
    args = parser.parse_args()

    print("🚀 OMERO Micro-SAM Annotation Pipeline Test")
    print("=" * 50)

    # 1. Import the package
    print("\n📦 Importing omero-annotate-ai package...")
    try:
        import omero_annotate_ai
        from omero_annotate_ai.core.config import AnnotationConfig
        from omero_annotate_ai.core.pipeline import create_pipeline
        print(f"✅ Package version: {omero_annotate_ai.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import omero-annotate-ai: {e}")
        print("💡 Install the package with: pip install -e .[omero]")
        sys.exit(1)

    # 2. Check OMERO dependencies
    print("\n🔗 Checking OMERO dependencies...")
    try:
        import omero
        from omero.gateway import BlitzGateway
        import ezomero
        print("✅ OMERO functionality available")
        OMERO_AVAILABLE = True
    except ImportError as e:
        print(f"❌ OMERO functionality not available: {e}")
        print("💡 Install with: pip install -e .[omero]")
        if not args.dry_run:
            sys.exit(1)
        OMERO_AVAILABLE = False

    # 3. Load configuration
    print(f"\n📄 Loading configuration from {args.config}...")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        config = AnnotationConfig.from_yaml(str(config_path))
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # 4. Validate configuration
    print("\n🔍 Validating configuration...")
    try:
        config.validate()
        print("✅ Configuration is valid")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

    # 5. Display configuration summary
    print("\n📋 Configuration Summary:")
    print(f"   🔬 Model: {config.microsam.model_type}")
    print(f"   📦 Container: {config.omero.container_type} (ID: {config.omero.container_id})")
    print(f"   📺 Channel: {config.omero.channel}")
    print(f"   🎯 Training Set: {config.training.trainingset_name}")
    print(f"   📊 Batch Size: {config.batch_processing.batch_size}")
    print(f"   🧩 Use Patches: {config.patches.use_patches}")
    if config.patches.use_patches:
        print(f"   📐 Patch Size: {config.patches.patch_size}")
    print(f"   🔄 Resume from Table: {config.workflow.resume_from_table}")
    print(f"   📖 Read-only Mode: {config.workflow.read_only_mode}")
    print(f"   🧠 3D Processing: {config.microsam.three_d}")

    # Override output directory if specified
    if args.output_dir:
        config.batch_processing.output_folder = args.output_dir
        print(f"📁 Output directory overridden: {args.output_dir}")
    else:
        # Create temporary output directory
        output_directory = tempfile.mkdtemp()
        config.batch_processing.output_folder = output_directory
        print(f"📁 Created temporary output directory: {output_directory}")

    if args.dry_run:
        print("\n✅ Dry run completed successfully!")
        print("🔍 Configuration is valid and ready for use")
        return

    if not OMERO_AVAILABLE:
        print("❌ Cannot run pipeline without OMERO functionality")
        sys.exit(1)

    # 6. Setup OMERO connection
    print("\n🔐 Setting up OMERO connection...")
    load_dotenv(override=True)
    
    # Get credentials
    host = os.environ.get("HOST")
    username = os.environ.get("USER_NAME") 
    group = os.environ.get("GROUP")
    password = os.environ.get("PASSWORD")
    
    if not host or not username:
        print("❌ Missing OMERO credentials in environment")
        print("💡 Create a .env file with HOST, USER_NAME, and GROUP")
        sys.exit(1)
    
    if not password:
        password = getpass("Enter OMERO server password: ")
    
    # Connect to OMERO
    conn = BlitzGateway(
        host=host,
        username=username,
        passwd=password,
        group=group,
        secure=True
    )
    
    connection_status = conn.connect()
    if connection_status:
        print("✅ Connected to OMERO Server")
        print(f"👤 User: {conn.getUser().getName()}")
        print(f"🏢 Group: {conn.getGroupFromContext().getName()}")
    else:
        print("❌ Connection to OMERO Server Failed")
        sys.exit(1)
    
    conn.c.enableKeepAlive(60)

    try:
        # 7. Create pipeline and validate container
        print("\n🏗️ Creating annotation pipeline...")
        pipeline = create_pipeline(config, conn)
        
        # 8. Get container details and preview images
        print("\n📂 Validating container and getting image list...")
        container_type = config.omero.container_type
        container_id = config.omero.container_id
        
        # Validate container exists
        if container_type == "dataset":
            container = conn.getObject("Dataset", container_id)
            if container is None:
                raise ValueError(f"Dataset with ID {container_id} not found")
            print(f"📁 Dataset: {container.getName()} (ID: {container_id})")
        elif container_type == "project":
            container = conn.getObject("Project", container_id)
            if container is None:
                raise ValueError(f"Project with ID {container_id} not found")
            print(f"📂 Project: {container.getName()} (ID: {container_id})")
        elif container_type == "plate":
            container = conn.getObject("Plate", container_id)
            if container is None:
                raise ValueError(f"Plate with ID {container_id} not found")
            print(f"🧪 Plate: {container.getName()} (ID: {container_id})")
        elif container_type == "image":
            container = conn.getObject("Image", container_id)
            if container is None:
                raise ValueError(f"Image with ID {container_id} not found")
            print(f"🖼️ Image: {container.getName()} (ID: {container_id})")
        else:
            raise ValueError(f"Unsupported container type: {container_type}")
        
        # Get images list
        images_list = pipeline.get_images_from_container()
        print(f"📊 Found {len(images_list)} images to process")
        
        # Show first few images
        print("\n🖼️ Sample images:")
        for i, img in enumerate(images_list[:3]):
            print(f"   {i+1}. {img.getName()} (ID: {img.getId()})")
        if len(images_list) > 3:
            print(f"   ... and {len(images_list) - 3} more images")

        # 9. Run the annotation pipeline
        print(f"\n🚀 Starting annotation pipeline...")
        print(f"   Processing {len(images_list)} images using micro-SAM")
        print(f"   Model: {config.microsam.model_type}")
        print(f"   Napari will open for interactive annotation")
        
        # Run the complete workflow
        table_id, processed_images = pipeline.run_full_workflow()
        
        print(f"\n✅ Annotation pipeline completed successfully!")
        print(f"📊 Processed {len(processed_images)} images")
        print(f"📋 Tracking table ID: {table_id}")
        
        if config.workflow.read_only_mode:
            print(f"💾 Annotations saved locally to: {config.workflow.local_output_dir}")
        else:
            print(f"☁️ Annotations uploaded to OMERO")

        # 10. Display results
        print("\n📊 Checking results...")
        if table_id is not None:
            try:
                tracking_df = ezomero.get_table(conn, table_id)
                print(f"📋 Tracking table contains {len(tracking_df)} rows")
                print(f"✅ Processed: {tracking_df['processed'].sum()} units")
                print(f"⏳ Pending: {(~tracking_df['processed']).sum()} units")
                
                if not config.training.segment_all:
                    train_count = tracking_df['train'].sum()
                    validate_count = tracking_df['validate'].sum()
                    print(f"🎓 Training samples: {train_count}")
                    print(f"✅ Validation samples: {validate_count}")
                    
            except Exception as e:
                print(f"⚠️ Could not retrieve tracking table details: {e}")

        # 11. Save configuration with results
        results_config_path = f"results_config_{config.training.trainingset_name}.yaml"
        config.save_yaml(results_config_path)
        print(f"💾 Configuration with results saved to: {results_config_path}")

    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        raise
    finally:
        # 12. Clean up
        print("\n🧹 Cleaning up...")
        
        # Clean up temporary directory if we created one
        if not args.output_dir and 'output_directory' in locals():
            try:
                shutil.rmtree(output_directory)
                print(f"🗑️ Removed temporary directory: {output_directory}")
            except Exception as e:
                print(f"⚠️ Error removing temporary directory: {e}")
        
        # Close OMERO connection
        if 'conn' in locals() and conn is not None:
            conn.close()
            print("🔌 OMERO connection closed")

    print("\n✨ Test completed successfully!")
    print(f"📄 Configuration used: {args.config}")
    if table_id:
        print(f"📋 OMERO tracking table ID: {table_id}")


if __name__ == "__main__":
    main()