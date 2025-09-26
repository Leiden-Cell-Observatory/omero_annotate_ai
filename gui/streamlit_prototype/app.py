"""
OMERO Image Annotation Workflow - Multi-Step Wizard

Properly integrated with the actual WorkflowWidget and AnnotationPipeline.
This replicates the exact 5-step WorkflowWidget process in a Streamlit interface.
"""

import streamlit as st
import sys
import tempfile
from pathlib import Path
import os
from datetime import datetime
import traceback

# Add src to path for development
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from omero_annotate_ai import create_omero_connection_widget, create_workflow_widget, create_pipeline
from omero_annotate_ai.core.annotation_config import create_default_config
from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
from omero_annotate_ai.omero.omero_utils import get_container_info
from omero_annotate_ai.omero.omero_functions import (
    generate_unique_table_name,
    list_annotation_tables,
)

# Page configuration
st.set_page_config(
    page_title="OMERO Annotation Wizard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 OMERO Image Annotation Workflow")
st.markdown("*Multi-step wizard that mirrors the complete WorkflowWidget functionality*")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "connection" not in st.session_state:
    st.session_state.connection = None
if "config" not in st.session_state:
    st.session_state.config = create_default_config()
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "working_directory" not in st.session_state:
    st.session_state.working_directory = None
if "containers" not in st.session_state:
    st.session_state.containers = []
if "annotation_tables" not in st.session_state:
    st.session_state.annotation_tables = []
if "selected_table_id" not in st.session_state:
    st.session_state.selected_table_id = None
if "container_info" not in st.session_state:
    st.session_state.container_info = None

# Step indicator
steps = [
    "🔌 OMERO Connection",
    "📁 Working Directory",
    "🔬 Choose Container",
    "📋 Check Tables",
    "⚙️ Configure Parameters",
    "💾 Save & Review",
    "🚀 Run Pipeline"
]

# Progress bar
progress = (st.session_state.step - 1) / (len(steps) - 1)
st.progress(progress)

# Step indicator
cols = st.columns(len(steps))
for i, (col, step_name) in enumerate(zip(cols, steps)):
    step_num = i + 1
    if step_num == st.session_state.step:
        col.markdown(f"**{step_num}. {step_name}** ✨")
    elif step_num < st.session_state.step:
        col.markdown(f"~~{step_num}. {step_name}~~ ✅")
    else:
        col.markdown(f"{step_num}. {step_name}")

st.markdown("---")

# STEP 1: OMERO Connection
if st.session_state.step == 1:
    st.header("🔌 OMERO Server Connection")

    with st.form("omero_connection_form"):
        st.write("Enter your OMERO server connection details:")

        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", value="localhost")
            username = st.text_input("Username")
        with col2:
            port = st.number_input("Port", value=4064, min_value=1, max_value=65535)
            password = st.text_input("Password", type="password")

        submit = st.form_submit_button("Connect to OMERO")

        if submit:
            try:
                with st.spinner("Connecting to OMERO..."):
                    # Create connection using SimpleOMEROConnection
                    conn_obj = SimpleOMEROConnection()
                    conn = conn_obj.connect(host, username, password, port)

                    if conn and conn.isConnected():
                        st.session_state.connection = conn
                        st.success(f"✅ Connected to OMERO as {username}")

                        # Automatically advance to next step
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("❌ Failed to connect to OMERO")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

    # Show current connection status
    if st.session_state.connection:
        st.info("Already connected to OMERO. You can proceed to the next step.")
        if st.button("Continue to Working Directory →"):
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    st.header("📁 Select Working Directory")

    if not st.session_state.connection:
        st.warning("Please establish OMERO connection first.")
        if st.button("← Back to Connection"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.write("Select or create a local working directory for your annotation project:")

        # Default directory
        default_dir = str(Path.home() / "omero_annotate_ai" / "omero_annotations")

        with st.form("directory_form"):
            working_dir = st.text_input(
                "Working Directory",
                value=st.session_state.working_directory or default_dir
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                submit = st.form_submit_button("Set Directory")
            with col2:
                use_temp = st.form_submit_button("Use Temp Directory")
            with col3:
                create_dir = st.form_submit_button("Create Directory")

            if submit or create_dir:
                try:
                    dir_path = Path(working_dir).expanduser()
                    dir_path.mkdir(parents=True, exist_ok=True)
                    st.session_state.working_directory = str(dir_path)
                    st.session_state.config.output.output_directory = dir_path
                    st.success(f"✅ Directory ready: {dir_path}")

                    if submit:
                        st.session_state.step = 3
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating directory: {e}")

            if use_temp:
                temp_dir = tempfile.mkdtemp(prefix="omero_annotations_")
                st.session_state.working_directory = temp_dir
                st.session_state.config.output.output_directory = Path(temp_dir)
                st.success(f"✅ Temporary directory created: {temp_dir}")
                st.session_state.step = 3
                st.rerun()

        # Show current directory status
        if st.session_state.working_directory:
            st.info(f"Current working directory: {st.session_state.working_directory}")
            if st.button("Continue to Container Selection →"):
                st.session_state.step = 3
                st.rerun()

        # Navigation
        if st.button("← Back to Connection"):
            st.session_state.step = 1
            st.rerun()

elif st.session_state.step == 3:
    st.header("🔬 Choose Container")

    if not st.session_state.working_directory:
        st.warning("Please set working directory first.")
        if st.button("← Back to Directory"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.write("Choose the OMERO container (project, dataset, plate, or screen) for annotation:")

        with st.form("container_form"):
            col1, col2 = st.columns(2)

            with col1:
                container_type = st.selectbox(
                    "Container Type",
                    ["project", "dataset", "plate", "screen"]
                )

            with col2:
                load_containers = st.form_submit_button("Load Containers")

            if load_containers:
                try:
                    with st.spinner(f"Loading {container_type}s..."):
                        conn = st.session_state.connection

                        if container_type == "project":
                            containers = list(conn.getObjects("Project"))
                        elif container_type == "dataset":
                            containers = list(conn.getObjects("Dataset"))
                        elif container_type == "plate":
                            containers = list(conn.getObjects("Plate"))
                        elif container_type == "screen":
                            containers = list(conn.getObjects("Screen"))

                        st.session_state.containers = [(c.getName(), c.getId()) for c in containers]
                        st.success(f"Found {len(containers)} {container_type}s")

                except Exception as e:
                    st.error(f"Error loading containers: {e}")

        # Container selection
        if st.session_state.containers:
            with st.form("select_container_form"):
                container_options = [f"{name} (ID: {id})" for name, id in st.session_state.containers]
                selected_idx = st.selectbox(
                    f"Select {container_type}",
                    range(len(container_options)),
                    format_func=lambda i: container_options[i]
                )

                submit = st.form_submit_button("Select Container")

                if submit:
                    _, container_id = st.session_state.containers[selected_idx]
                    st.session_state.config.omero.container_type = container_type
                    st.session_state.config.omero.container_id = container_id

                    # Get container info
                    try:
                        container_info = get_container_info(
                            st.session_state.connection,
                            container_type,
                            container_id
                        )
                        st.session_state.container_info = container_info
                        st.success(f"✅ Selected {container_type} (ID: {container_id})")

                        if container_info:
                            st.write(f"**Total images:** {container_info.get('total_images', 'Unknown')}")

                        st.session_state.step = 4
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error getting container info: {e}")

        # Navigation
        if st.button("← Back to Directory"):
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 4:
    st.header("📋 Check Existing Tables")

    if not st.session_state.config.omero.container_id:
        st.warning("Please select container first.")
        if st.button("← Back to Container"):
            st.session_state.step = 3
            st.rerun()
    else:
        st.write("Check for existing annotation tables and decide how to proceed:")

        # Scan for tables button
        if st.button("🔍 Scan for Tables"):
            try:
                with st.spinner("Scanning for annotation tables..."):
                    tables = list_annotation_tables(
                        st.session_state.connection,
                        st.session_state.config.omero.container_type,
                        st.session_state.config.omero.container_id
                    )
                    st.session_state.annotation_tables = tables

                    if tables:
                        st.success(f"Found {len(tables)} annotation tables")
                        for i, table in enumerate(tables):
                            st.write(f"{i+1}. {table.get('name', 'Unknown')} (ID: {table.get('id')})")
                    else:
                        st.info("No annotation tables found.")

            except Exception as e:
                st.error(f"Error scanning tables: {e}")

        # Table selection options
        if st.session_state.annotation_tables:
            st.subheader("Existing Tables Found")

            # Radio button for table selection
            table_options = [f"{table.get('name', 'Unknown')} (ID: {table.get('id')})"
                           for table in st.session_state.annotation_tables]

            selected_table_idx = st.radio(
                "Select existing table to continue:",
                range(len(table_options)),
                format_func=lambda i: table_options[i]
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Continue Existing Table"):
                    selected_table = st.session_state.annotation_tables[selected_table_idx]
                    st.session_state.selected_table_id = selected_table.get('id')
                    st.session_state.config.name = selected_table.get('name', 'unknown')
                    st.session_state.config.workflow.resume_from_table = True
                    st.success(f"Will continue with table: {selected_table.get('name')}")
                    st.session_state.step = 5
                    st.rerun()

        # Create new table option
        st.subheader("Create New Table")

        with st.form("new_table_form"):
            custom_name = st.text_input(
                "New Table Name (optional)",
                placeholder="Leave empty for auto-generation"
            )

            submit = st.form_submit_button("➕ Create New Table")

            if submit:
                try:
                    if custom_name.strip():
                        unique_name = generate_unique_table_name(
                            st.session_state.connection,
                            st.session_state.config.omero.container_type,
                            st.session_state.config.omero.container_id,
                            custom_name.strip()
                        )
                    else:
                        unique_name = generate_unique_table_name(
                            st.session_state.connection,
                            st.session_state.config.omero.container_type,
                            st.session_state.config.omero.container_id
                        )

                    st.session_state.config.name = unique_name
                    st.session_state.config.workflow.resume_from_table = False
                    st.success(f"✅ Generated table name: {unique_name}")
                    st.session_state.step = 5
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating table name: {e}")

        # Navigation
        if st.button("← Back to Container"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 5:
    st.header("⚙️ Configure Parameters")

    if not st.session_state.config.name:
        st.warning("Please complete table selection first.")
        if st.button("← Back to Tables"):
            st.session_state.step = 4
            st.rerun()
    else:
        # Show OMERO status
        st.subheader("📡 OMERO Settings")
        st.write(f"**Container:** {st.session_state.config.omero.container_type} (ID: {st.session_state.config.omero.container_id})")
        st.write(f"**Training Set:** {st.session_state.config.name}")
        st.write(f"**Resume from table:** {st.session_state.config.workflow.resume_from_table}")

        if st.session_state.container_info:
            st.write(f"**Total images:** {st.session_state.container_info.get('total_images', 'Unknown')}")

        # Configuration tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Annotation Settings", "⚙️ Technical Settings", "🔄 Workflow Settings"])

        with tab1:
            st.subheader("Annotation Settings")

            col1, col2 = st.columns(2)
            with col1:
                segment_all = st.checkbox(
                    "Annotate all images",
                    value=st.session_state.config.training.segment_all
                )
                st.session_state.config.training.segment_all = segment_all

                if not segment_all:
                    max_images = st.session_state.container_info.get('total_images', 50) if st.session_state.container_info else 50
                    train_n = st.slider(
                        "Training images",
                        1, max_images,
                        min(st.session_state.config.training.train_n, max_images)
                    )
                    validate_n = st.slider(
                        "Validation images",
                        1, max_images,
                        min(st.session_state.config.training.validate_n, max_images)
                    )
                    st.session_state.config.training.train_n = train_n
                    st.session_state.config.training.validate_n = validate_n

                channel = st.number_input(
                    "Channel",
                    min_value=0,
                    value=st.session_state.config.spatial_coverage.channels[0]
                )
                st.session_state.config.spatial_coverage.channels = [channel]

            with col2:
                timepoint_mode = st.selectbox(
                    "Timepoints",
                    ["specific", "all", "random"],
                    index=["specific", "all", "random"].index(st.session_state.config.spatial_coverage.timepoint_mode)
                )
                st.session_state.config.spatial_coverage.timepoint_mode = timepoint_mode

                if timepoint_mode == "specific":
                    timepoints_str = st.text_input(
                        "Timepoint list",
                        value=", ".join(map(str, st.session_state.config.spatial_coverage.timepoints))
                    )
                    try:
                        timepoints = [int(x.strip()) for x in timepoints_str.split(",") if x.strip()]
                        st.session_state.config.spatial_coverage.timepoints = timepoints
                    except:
                        pass

                z_slice_mode = st.selectbox(
                    "Z-slices",
                    ["specific", "all", "random"],
                    index=["specific", "all", "random"].index(st.session_state.config.spatial_coverage.z_slice_mode)
                )
                st.session_state.config.spatial_coverage.z_slice_mode = z_slice_mode

                if z_slice_mode == "specific":
                    z_slices_str = st.text_input(
                        "Z-slice list",
                        value=", ".join(map(str, st.session_state.config.spatial_coverage.z_slices))
                    )
                    try:
                        z_slices = [int(x.strip()) for x in z_slices_str.split(",") if x.strip()]
                        st.session_state.config.spatial_coverage.z_slices = z_slices
                    except:
                        pass

            # Patches settings
            st.subheader("Patches")
            use_patches = st.checkbox(
                "Use patches",
                value=st.session_state.config.processing.use_patches
            )
            st.session_state.config.processing.use_patches = use_patches

            if use_patches:
                col1, col2 = st.columns(2)
                with col1:
                    patches_per_image = st.slider(
                        "Patches per image",
                        1, 20,
                        st.session_state.config.processing.patches_per_image
                    )
                    st.session_state.config.processing.patches_per_image = patches_per_image

                with col2:
                    patch_h, patch_w = st.session_state.config.processing.patch_size
                    patch_size_str = st.text_input(
                        "Patch size (height, width)",
                        value=f"{patch_h}, {patch_w}"
                    )
                    try:
                        sizes = [int(x.strip()) for x in patch_size_str.split(",") if x.strip()]
                        if len(sizes) == 2:
                            st.session_state.config.processing.patch_size = sizes
                    except:
                        pass

        with tab2:
            st.subheader("Technical Settings")

            batch_size = st.slider(
                "Batch size (0 = process all together)",
                0, 10,
                st.session_state.config.processing.batch_size
            )
            st.session_state.config.processing.batch_size = batch_size

            model_type = st.selectbox(
                "SAM Model",
                ["vit_b_lm", "vit_b", "vit_l", "vit_h"],
                index=["vit_b_lm", "vit_b", "vit_l", "vit_h"].index(st.session_state.config.ai_model.model_type)
            )
            st.session_state.config.ai_model.model_type = model_type

        with tab3:
            st.subheader("Workflow Settings")

            read_only_mode = st.checkbox(
                "Read-only mode (save locally only)",
                value=st.session_state.config.workflow.read_only_mode
            )
            st.session_state.config.workflow.read_only_mode = read_only_mode

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Tables"):
                st.session_state.step = 4
                st.rerun()

        with col2:
            if st.button("Continue to Save →"):
                st.session_state.step = 6
                st.rerun()

elif st.session_state.step == 6:
    st.header("💾 Save & Review Configuration")

    if not st.session_state.config.name:
        st.warning("Please complete configuration first.")
        if st.button("← Back to Configuration"):
            st.session_state.step = 5
            st.rerun()
    else:
        st.subheader("Configuration Preview")

        # Update working directory in config
        if st.session_state.working_directory:
            st.session_state.config.output.output_directory = Path(st.session_state.working_directory)

        # Show configuration as YAML
        config_yaml = st.session_state.config.to_yaml()
        st.code(config_yaml, language="yaml")

        # Save configuration
        if st.button("💾 Save Configuration"):
            try:
                if st.session_state.working_directory:
                    config_path = Path(st.session_state.working_directory) / "annotation_config.yaml"
                    st.session_state.config.save_yaml(config_path)
                    st.success(f"✅ Configuration saved to: {config_path}")
                else:
                    st.error("No working directory set")
            except Exception as e:
                st.error(f"❌ Error saving configuration: {e}")

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Configuration"):
                st.session_state.step = 5
                st.rerun()

        with col2:
            if st.button("Create Pipeline →"):
                try:
                    # Create pipeline with config file path
                    config_path = Path(st.session_state.working_directory) / "annotation_config.yaml"
                    pipeline = create_pipeline(
                        st.session_state.config,
                        st.session_state.connection,
                        config_path
                    )
                    st.session_state.pipeline = pipeline
                    st.success("✅ Pipeline created successfully!")
                    st.session_state.step = 7
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creating pipeline: {str(e)}")

elif st.session_state.step == 7:
    st.header("🚀 Run Annotation Pipeline")

    if not st.session_state.pipeline:
        st.warning("Please create pipeline first.")
        if st.button("← Back to Review"):
            st.session_state.step = 6
            st.rerun()
    else:
        st.write("Ready to start the annotation workflow. This will launch napari for interactive annotation.")

        # Show pipeline info
        config = st.session_state.config
        st.info(
            f"**Pipeline Details:**\n"
            f"- Training Set: {config.name}\n"
            f"- Container: {config.omero.container_type} (ID: {config.omero.container_id})\n"
            f"- Model: {config.ai_model.model_type}\n"
            f"- Output: {config.output.output_directory}\n"
            f"- Read-only Mode: {'Yes' if config.workflow.read_only_mode else 'No'}"
        )

        # Warning about napari
        st.warning(
            "⚠️ **Important:** This will launch napari on your local machine for interactive annotation. "
            "Make sure you have napari and micro-sam installed."
        )

        # Launch button
        if st.button("🚀 Launch Pipeline", type="primary"):
            try:
                with st.spinner("Launching annotation pipeline..."):
                    # Run the full micro-SAM workflow
                    pipeline = st.session_state.pipeline
                    table_id, updated_config = pipeline.run_full_micro_sam_workflow()

                    st.success(f"✅ Pipeline completed successfully!")
                    st.write(f"**Results:**")
                    st.write(f"- Table ID: {table_id}")
                    st.write(f"- Processed annotations: {len([a for a in updated_config.annotations if a.processed])}")
                    st.write(f"- Total annotations: {len(updated_config.annotations)}")

                    # Show download links if in read-only mode
                    if config.workflow.read_only_mode:
                        st.write("**Local Files Created:**")
                        output_path = Path(config.output.output_directory)
                        if output_path.exists():
                            for folder in output_path.iterdir():
                                if folder.is_dir():
                                    st.write(f"- {folder.name}/: {len(list(folder.glob('*')))} files")

            except Exception as e:
                st.error(f"❌ Pipeline execution failed: {str(e)}")
                st.write("**Error details:**")
                st.code(traceback.format_exc())

        # Navigation
        col1, col2 = st.columns([1, 1])

        with col2:
            if st.button("← Back to Review"):
                st.session_state.step = 6
                st.rerun()

        with col1:
            if st.button("🔄 Reset Workflow"):
                # Reset session state
                for key in list(st.session_state.keys()):
                    if key.startswith(('step', 'connection', 'config', 'pipeline', 'working_directory', 'containers', 'annotation_tables', 'selected_table_id', 'container_info')):
                        del st.session_state[key]
                st.session_state.step = 1
                st.session_state.config = create_default_config()
                st.rerun()

# Footer
st.markdown("---")
st.markdown("*OMERO Annotation Workflow - Complete 7-step wizard with full WorkflowWidget integration*")