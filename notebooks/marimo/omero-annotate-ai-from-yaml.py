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
    # OMERO Annotation Workflow from YAML Configuration

    Load an existing YAML configuration and run the annotation pipeline without using the configuration widgets.

    **Workflow Steps:**
    1. Connect to your OMERO server
    2. Load a YAML configuration file
    3. Review the configuration
    4. Run the annotation pipeline
    5. Save the updated configuration
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
    ## 2. Load YAML Configuration

    Provide the path to your YAML configuration file.
    """)
    return


@app.cell
def _(mo):
    yaml_path = mo.ui.text(
        value="",
        label="YAML Configuration Path",
        placeholder="/path/to/annotation_config.yaml",
        full_width=True,
    )
    load_button = mo.ui.run_button(label="Load Configuration")
    mo.vstack([yaml_path, load_button])
    return yaml_path, load_button


@app.cell
def _(load_button, load_config_from_yaml, mo, yaml_path):
    config = None
    mo.stop(not load_button.value, mo.md("Enter a YAML path and click 'Load Configuration'"))

    if not yaml_path.value:
        mo.output.replace(mo.md("**Error:** Please provide a YAML file path"))
    else:
        try:
            config = load_config_from_yaml(yaml_path.value)
            mo.output.replace(mo.md(f"""
            **Configuration loaded!**

            | Setting | Value |
            |---------|-------|
            | Workflow Name | {config.name} |
            | Container | {config.omero.container_type} (ID: {config.omero.container_id}) |
            | Output Directory | {config.output.output_directory} |
            | Model | {config.ai_model.model_type} |
            | Channel | {config.spatial_coverage.channels} |
            | Read-only Mode | {config.workflow.read_only_mode} |
            | Resume from Table | {config.workflow.resume_from_table} |
            """))
        except FileNotFoundError:
            mo.output.replace(mo.md(f"**Error:** File not found: `{yaml_path.value}`"))
        except Exception as e:
            mo.output.replace(mo.md(f"**Error loading configuration:** {str(e)}"))
    return (config,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Run Annotation Pipeline

    Choose which workflow to run with the loaded configuration.
    """)
    return


@app.cell
def _(mo):
    workflow_type = mo.ui.dropdown(
        options=["micro-SAM (interactive)", "CellPose (export for annotation)"],
        value="micro-SAM (interactive)",
        label="Workflow Type",
    )
    run_button = mo.ui.run_button(label="Run Pipeline", kind="success")
    mo.hstack([workflow_type, run_button], justify="start", gap=2)
    return workflow_type, run_button


@app.cell
def _(config, conn, create_pipeline, mo, run_button, workflow_type):
    mo.stop(not run_button.value, mo.md("Click 'Run Pipeline' to start"))

    if config is None:
        mo.output.replace(mo.md("**Error:** Load a configuration first"))
    elif not conn or not conn.isConnected():
        mo.output.replace(mo.md("**Error:** Please connect to OMERO first"))
    else:
        try:
            pipeline = create_pipeline(config, conn)

            if "micro-SAM" in workflow_type.value:
                mo.output.replace(mo.md("**Starting micro-SAM pipeline...** Napari will open for interactive annotation."))
                table_id, updated_config = pipeline.run_full_micro_sam_workflow()
            else:
                mo.output.replace(mo.md("**Starting CellPose export pipeline...**"))
                table_id, updated_config = pipeline.run_cp_workflow()

            result_md = f"""
            **Pipeline completed!**

            - Tracking table ID: {table_id}
            - Total images processed: {len(updated_config.get_processed())}
            - Output directory: {updated_config.output.output_directory}
            """
            mo.output.replace(mo.md(result_md))
        except Exception as e:
            mo.output.replace(mo.md(f"**Error:** {str(e)}"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Save Updated Configuration
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
    if config is None:
        _result = mo.md("**Error:** No configuration loaded")
    else:
        try:
            config_path = Path(config.output.output_directory) / f"annotation_config_{config.name}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config.save_yaml(config_path)
            _result = mo.md(f"""
            **Configuration saved!**

            File: `{config_path}`

            To reuse: `config = load_config_from_yaml('{config_path}')`
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
    return (Path,)


@app.cell
def _():
    from omero_annotate_ai import load_config_from_yaml, create_pipeline

    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
        OMERO_AVAILABLE = True
    except ImportError:
        OMERO_AVAILABLE = False
    return (
        OMERO_AVAILABLE,
        SimpleOMEROConnection,
        load_config_from_yaml,
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


if __name__ == "__main__":
    app.run()
