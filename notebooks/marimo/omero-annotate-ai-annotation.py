# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "omero-annotate-ai",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # OMERO Image Annotation Workflow with micro-SAM

    This notebook guides you through annotating images in OMERO using the micro-SAM annotation plugin.

    **Workflow Steps:**
    1. Connect to your OMERO server
    2. Select a container (project, dataset, or plate)
    3. Configure annotation parameters
    4. Run the annotation pipeline
    5. Save your configuration for future use
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. OMERO Connection
    """)
    return


@app.cell
def _(mo):
    omero_host = mo.ui.text(value="", label="OMERO Host", placeholder="omero.server.edu")
    omero_username = mo.ui.text(value="", label="Username", placeholder="your_username")
    omero_password = mo.ui.text(value="", label="Password", placeholder="password", kind="password")
    omero_group = mo.ui.text(value="", label="Group (optional)", placeholder="Leave empty for default")
    omero_secure = mo.ui.checkbox(value=True, label="Secure connection")
    connect_button = mo.ui.run_button(label="Connect to OMERO")

    mo.vstack([
        omero_host,
        omero_username,
        omero_password,
        omero_group,
        omero_secure,
        connect_button,
    ])
    return omero_host, omero_username, omero_password, omero_group, omero_secure, connect_button


@app.cell
def _(connect_button, omero_host, omero_username, omero_password, omero_group, omero_secure, establish_connection, mo):
    conn = None
    connection_status = "Not connected"

    if connect_button.value:
        host = omero_host.value
        username = omero_username.value
        password = omero_password.value
        group = omero_group.value if omero_group.value else None
        secure = omero_secure.value

        if host and username and password:
            try:
                conn = establish_connection(host, username, password, group, secure)
                if conn and conn.isConnected():
                    user = conn.getUser().getName()
                    group_name = conn.getGroupFromContext().getName()
                    connection_status = f"Connected as **{user}** to group **{group_name}**"
                else:
                    connection_status = "Connection failed"
            except Exception as e:
                connection_status = f"Connection error: {str(e)}"
        else:
            connection_status = "Please fill in host, username, and password"

    mo.md(f"**Status:** {connection_status}")
    return (conn,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Container Selection
    """)
    return


@app.cell
def _(mo):
    container_type = mo.ui.dropdown(
        options=["project", "dataset", "plate", "screen"],
        value="dataset",
        label="Container Type",
    )
    container_id = mo.ui.number(
        value=0,
        label="Container ID",
        start=0,
    )
    mo.hstack([container_type, container_id], justify="start", gap=2)
    return container_id, container_type


@app.cell
def _(conn, container_id, container_type, mo):
    container_info = None
    if conn and conn.isConnected() and container_id.value > 0:
        try:
            container = conn.getObject(container_type.value.capitalize(), container_id.value)
            if container:
                container_info = f"Found: **{container.getName()}**"
            else:
                container_info = f"Container not found with ID {container_id.value}"
        except Exception as e:
            container_info = f"Error: {str(e)}"
    elif container_id.value > 0:
        container_info = "Connect to OMERO first to validate container"

    mo.md(container_info) if container_info else mo.md("")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Annotation Settings
    """)
    return


@app.cell
def _(mo):
    segment_all = mo.ui.checkbox(value=False, label="Annotate all images")
    train_n = mo.ui.slider(start=1, stop=50, value=3, label="Training images")
    validate_n = mo.ui.slider(start=0, stop=50, value=2, label="Validation images")
    test_n = mo.ui.slider(start=0, stop=50, value=0, label="Test images")

    mo.vstack([
        segment_all,
        mo.md("**Image split** (ignored if 'Annotate all' is checked):") if not segment_all.value else mo.md(""),
        mo.hstack([train_n, validate_n, test_n], justify="start", gap=2) if not segment_all.value else mo.md(""),
    ])
    return segment_all, test_n, train_n, validate_n


@app.cell
def _(mo):
    channel = mo.ui.number(value=0, label="Channel", start=0)
    timepoint_mode = mo.ui.dropdown(
        options=["specific", "all", "random"],
        value="specific",
        label="Timepoint Mode",
    )
    timepoints = mo.ui.text(value="0", label="Timepoints (comma-separated)")
    z_slice_mode = mo.ui.dropdown(
        options=["specific", "all", "random"],
        value="specific",
        label="Z-slice Mode",
    )
    z_slices = mo.ui.text(value="0", label="Z-slices (comma-separated)")
    three_d = mo.ui.checkbox(value=False, label="3D volumetric processing")

    mo.vstack([
        mo.hstack([channel], justify="start"),
        mo.hstack([timepoint_mode, timepoints], justify="start", gap=2),
        mo.hstack([z_slice_mode, z_slices], justify="start", gap=2),
        three_d,
    ])
    return channel, three_d, timepoint_mode, timepoints, z_slice_mode, z_slices


@app.cell
def _(mo):
    mo.md("""
    ## 4. Processing Settings
    """)
    return


@app.cell
def _(mo):
    use_patches = mo.ui.checkbox(value=False, label="Use patches")
    patches_per_image = mo.ui.slider(start=1, stop=20, value=1, label="Patches per image")
    patch_size = mo.ui.text(value="512, 512", label="Patch size (width, height)")

    mo.vstack([
        use_patches,
        mo.hstack([patches_per_image, patch_size], justify="start", gap=2) if use_patches.value else mo.md(""),
    ])
    return patch_size, patches_per_image, use_patches


@app.cell
def _(mo):
    batch_size = mo.ui.slider(start=0, stop=10, value=0, label="Batch size (0 = all at once)")
    model_type = mo.ui.dropdown(
        options=["vit_b_lm", "vit_b", "vit_l", "vit_h"],
        value="vit_b_lm",
        label="SAM Model",
    )
    read_only_mode = mo.ui.checkbox(value=False, label="Read-only mode (save locally only)")

    mo.vstack([
        mo.hstack([batch_size, model_type], justify="start", gap=2),
        read_only_mode,
    ])
    return batch_size, model_type, read_only_mode


@app.cell
def _(mo):
    mo.md("""
    ## 5. Output Settings
    """)
    return


@app.cell
def _(Path, datetime, mo):
    output_directory = mo.ui.text(
        value=str(Path.home() / "omero_annotate_ai" / "omero_annotations"),
        label="Output Directory",
        full_width=True,
    )
    workflow_name = mo.ui.text(
        value=f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        label="Workflow Name",
    )
    mo.vstack([output_directory, workflow_name])
    return output_directory, workflow_name


@app.cell
def _(mo):
    mo.md("""
    ## 6. Configuration Summary
    """)
    return


@app.cell
def _(
    batch_size,
    build_config,
    channel,
    container_id,
    container_type,
    mo,
    model_type,
    output_directory,
    patch_size,
    patches_per_image,
    read_only_mode,
    segment_all,
    test_n,
    three_d,
    timepoint_mode,
    timepoints,
    train_n,
    use_patches,
    validate_n,
    workflow_name,
    z_slice_mode,
    z_slices,
):
    config = build_config(
        name=workflow_name.value,
        container_type=container_type.value,
        container_id=int(container_id.value),
        output_dir=output_directory.value,
        segment_all=segment_all.value,
        train_n=train_n.value,
        validate_n=validate_n.value,
        test_n=test_n.value,
        channel=int(channel.value),
        timepoint_mode=timepoint_mode.value,
        timepoints=timepoints.value,
        z_slice_mode=z_slice_mode.value,
        z_slices=z_slices.value,
        three_d=three_d.value,
        use_patches=use_patches.value,
        patches_per_image=patches_per_image.value,
        patch_size=patch_size.value,
        batch_size=batch_size.value,
        model_type=model_type.value,
        read_only_mode=read_only_mode.value,
    )

    mo.md(f"""
    **Configuration Summary:**

    | Setting | Value |
    |---------|-------|
    | Workflow Name | {config.name} |
    | Container | {config.omero.container_type} (ID: {config.omero.container_id}) |
    | Output Directory | {config.output.output_directory} |
    | Model | {config.ai_model.model_type} |
    | Channel | {config.spatial_coverage.channels} |
    | Segment All | {config.training.segment_all} |
    | Train/Val/Test | {config.training.train_n}/{config.training.validate_n}/{config.training.test_n} |
    | Use Patches | {config.processing.use_patches} |
    | Read-only Mode | {config.workflow.read_only_mode} |
    """)
    return (config,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Run Annotation Pipeline
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Annotation Pipeline", kind="success")
    run_button
    return (run_button,)


@app.cell
def _(config, conn, container_id, create_pipeline, mo, run_button):
    mo.stop(not run_button.value, mo.md("Click 'Run Annotation Pipeline' to start"))

    if not conn or not conn.isConnected():
        mo.output.replace(mo.md("**Error:** Please connect to OMERO first"))
    elif container_id.value <= 0:
        mo.output.replace(mo.md("**Error:** Please select a valid container ID"))
    else:
        try:
            pipeline = create_pipeline(config, conn)
            mo.output.replace(mo.md("**Starting annotation pipeline...** Napari will open for interactive annotation."))

            table_id, updated_config = pipeline.run_full_micro_sam_workflow()

            result_md = f"""
            **Annotation pipeline completed successfully!**

            - Tracking table ID: {table_id}
            - Total images processed: {len(updated_config.get_processed())}
            - Output directory: {updated_config.output.output_directory}
            """
            mo.output.replace(mo.md(result_md))
        except Exception as e:
            mo.output.replace(mo.md(f"**Error during pipeline execution:** {str(e)}"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Save Configuration
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save Configuration to YAML")
    save_button
    return (save_button,)


@app.cell
def _(Path, config, mo, save_button):
    mo.stop(not save_button.value, mo.md("Click 'Save Configuration' to export your settings"))

    _result = None
    try:
        config_path = Path(config.output.output_directory) / f"annotation_config_{config.name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.save_yaml(config_path)
        _result = mo.md(f"""
        **Configuration saved!**

        File: `{config_path}`

        To reuse this configuration:
        ```python
        from omero_annotate_ai import load_config
        config = load_config('{config_path}')
        ```
        """)
    except Exception as e:
        _result = mo.md(f"**Error saving configuration:** {str(e)}")
    _result
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    from datetime import datetime
    return Path, datetime


@app.cell
def _():
    from omero_annotate_ai import create_default_config, create_pipeline

    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
        OMERO_AVAILABLE = True
    except ImportError:
        OMERO_AVAILABLE = False
    return (
        OMERO_AVAILABLE,
        SimpleOMEROConnection,
        create_default_config,
        create_pipeline,
    )


@app.cell
def _(OMERO_AVAILABLE, SimpleOMEROConnection):
    def establish_connection(host, username, password, group=None, secure=True):
        if not OMERO_AVAILABLE:
            raise ImportError("OMERO functionality not available. Install with: pip install -e .[omero]")

        connection = SimpleOMEROConnection()
        config = {
            "host": host,
            "username": username,
            "password": password,
            "secure": secure,
        }
        if group:
            config["group"] = group

        conn = connection.create_connection_from_config(config)
        if conn:
            conn.c.enableKeepAlive(60)
        return conn
    return (establish_connection,)


@app.cell
def _():
    def parse_int_list(text):
        if not text or not text.strip():
            return [0]
        try:
            return [int(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            return [0]

    def parse_patch_size(text):
        try:
            parts = [int(x.strip()) for x in text.split(",")]
            if len(parts) == 2:
                return parts
            return [512, 512]
        except ValueError:
            return [512, 512]
    return parse_int_list, parse_patch_size


@app.cell
def _(Path, create_default_config, parse_int_list, parse_patch_size):
    def build_config(
        name,
        container_type,
        container_id,
        output_dir,
        segment_all,
        train_n,
        validate_n,
        test_n,
        channel,
        timepoint_mode,
        timepoints,
        z_slice_mode,
        z_slices,
        three_d,
        use_patches,
        patches_per_image,
        patch_size,
        batch_size,
        model_type,
        read_only_mode,
    ):
        config = create_default_config()

        config.name = name
        config.omero.container_type = container_type
        config.omero.container_id = container_id
        config.output.output_directory = Path(output_dir)

        config.training.segment_all = segment_all
        config.training.train_n = train_n
        config.training.validate_n = validate_n
        config.training.test_n = test_n

        config.spatial_coverage.channels = [channel]
        config.spatial_coverage.timepoint_mode = timepoint_mode
        config.spatial_coverage.timepoints = parse_int_list(timepoints)
        config.spatial_coverage.z_slice_mode = z_slice_mode
        config.spatial_coverage.z_slices = parse_int_list(z_slices)
        config.spatial_coverage.three_d = three_d

        config.spatial_coverage.use_patches = use_patches
        config.spatial_coverage.patches_per_image = patches_per_image
        config.spatial_coverage.patch_size = parse_patch_size(patch_size)
        config.workflow.batch_size = batch_size

        config.ai_model.pretrained_from = model_type

        config.workflow.read_only_mode = read_only_mode

        return config
    return (build_config,)


if __name__ == "__main__":
    app.run()
