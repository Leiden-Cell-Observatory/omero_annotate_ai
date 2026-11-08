# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "omero-annotate-ai",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Micro-SAM Training from OMERO Data

    Train micro-SAM models using annotation tables from OMERO with automated data preparation.

    **Workflow Steps:**
    1. Connect to your OMERO server
    2. Select an annotation table with training data
    3. Download and organize training data
    4. Configure and run micro-SAM training
    5. Optionally export to BioImage.IO format
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
    omero_host = mo.ui.text(
        value="", label="OMERO Host", placeholder="omero.server.edu"
    )
    omero_username = mo.ui.text(value="", label="Username", placeholder="your_username")
    omero_password = mo.ui.text(
        value="", label="Password", placeholder="password", kind="password"
    )
    omero_group = mo.ui.text(
        value="", label="Group (optional)", placeholder="Leave empty for default"
    )
    omero_secure = mo.ui.checkbox(value=True, label="Encrypt data transfer")
    connect_button = mo.ui.run_button(label="Connect to OMERO")

    mo.vstack(
        [
            omero_host,
            omero_username,
            omero_password,
            omero_group,
            omero_secure,
            connect_button,
        ]
    )
    return (
        connect_button,
        omero_group,
        omero_host,
        omero_password,
        omero_secure,
        omero_username,
    )


@app.cell
def _(
    connect_button,
    establish_connection,
    mo,
    omero_group,
    omero_host,
    omero_password,
    omero_secure,
    omero_username,
):
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
                    connection_status = (
                        f"Connected as **{user}** to group **{group_name}**"
                    )
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
    ## 2. Training Data Selection

    Select the OMERO container and annotation table to use for training.
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
def _(conn, container_id, container_type, list_training_sets, mo, summarize):
    tables_info = ""

    if conn and conn.isConnected() and container_id.value > 0:
        try:
            _sets = list_training_sets(conn, container_type.value, container_id.value)
            if _sets:
                _lines = []
                for _info in _sets:
                    _progress = summarize(_info.table_id, conn).progress
                    _split = " · ".join(
                        f"{name} {done}/{total}"
                        for name, (done, total) in _progress.items()
                        if total
                    )
                    _lines.append(f"    | {_info.table_id} | {_info.name} | {_split} |")
                rows = "\n".join(_lines)
                tables_info = f"""
    Found **{len(_sets)}** training sets:

    | Table ID | Name | Annotated per split |
    |----------|------|---------------------|
    {rows}
                """
            else:
                tables_info = "No training sets found in this container."
        except Exception as e:
            tables_info = f"Error: {str(e)}"

    mo.md(tables_info) if tables_info else mo.md("")
    return


@app.cell
def _(mo):
    table_id = mo.ui.number(value=0, label="Table ID to use for training", start=0)
    table_id
    return (table_id,)


@app.cell
def _(
    conn,
    container_id,
    container_type,
    find_annotation_config_for_table,
    mo,
    table_id,
):
    annotation_config = None
    _config_info = ""
    if conn and conn.isConnected() and table_id.value > 0 and container_id.value > 0:
        try:
            annotation_config = find_annotation_config_for_table(
                conn, int(table_id.value)
            )
            if annotation_config:
                _config_info = f"Loaded config **{annotation_config.name}** (model: `{annotation_config.ai_model.pretrained_from}`)"
            else:
                _config_info = (
                    "No annotation config found for this table — using defaults."
                )
        except Exception as e:
            _config_info = f"Could not load config: {e}"
    mo.md(_config_info) if _config_info else mo.md("")
    return (annotation_config,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Training Configuration
    """)
    return


@app.cell
def _(Path, annotation_config, datetime, mo):
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    _default_name = (
        annotation_config.name
        if annotation_config
        else f"micro_sam_training_{_timestamp}"
    )
    _default_model = (
        annotation_config.ai_model.pretrained_from if annotation_config else "vit_b_lm"
    ) or "vit_b_lm"
    _default_patch = (
        f"{annotation_config.spatial_coverage.patch_size[0]}, {annotation_config.spatial_coverage.patch_size[1]}"
        if annotation_config and annotation_config.spatial_coverage.patch_size
        else "512, 512"
    )

    output_directory = mo.ui.text(
        value=str(
            Path.home()
            / "omero-annotate-ai"
            / "micro-sam_models"
            / f"micro-sam-{_timestamp}"
        ),
        label="Output Directory",
        full_width=True,
    )
    model_name = mo.ui.text(value=_default_name, label="Model Name")
    model_type = mo.ui.dropdown(
        options=["vit_b_lm", "vit_b", "vit_l", "vit_h"],
        value=_default_model,
        label="SAM Model",
    )
    epochs = mo.ui.slider(
        start=1, stop=200, value=10, label="Epochs (use 50+ for real training)"
    )
    batch_size = mo.ui.slider(start=1, stop=8, value=1, label="Batch Size")
    patch_shape = mo.ui.text(value=_default_patch, label="Patch Shape (width, height)")
    n_iterations = mo.ui.number(
        value=0, start=0, label="Iterations (0 = derive from epochs)"
    )

    mo.vstack(
        [
            output_directory,
            mo.hstack([model_name, model_type], justify="start", gap=2),
            mo.hstack([epochs, batch_size], justify="start", gap=2),
            mo.hstack([patch_shape, n_iterations], justify="start", gap=2),
        ]
    )
    return (
        batch_size,
        epochs,
        model_name,
        model_type,
        n_iterations,
        output_directory,
        patch_shape,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 4. Prepare Training Data

    Download and organize training data from OMERO.
    """)
    return


@app.cell
def _(mo):
    prepare_button = mo.ui.run_button(label="Prepare Training Data", kind="success")
    prepare_button
    return (prepare_button,)


@app.cell
def _(
    Path,
    conn,
    mo,
    model_name,
    output_directory,
    prepare_button,
    prepare_training_data_from_table,
    table_id,
):
    training_result = None
    mo.stop(
        not prepare_button.value,
        mo.md("Click 'Prepare Training Data' to download from OMERO"),
    )

    if not conn or not conn.isConnected():
        mo.output.replace(mo.md("**Error:** Please connect to OMERO first"))
    elif table_id.value <= 0:
        mo.output.replace(mo.md("**Error:** Please enter a valid table ID"))
    else:
        try:
            _out_dir = Path(output_directory.value)
            _out_dir.mkdir(parents=True, exist_ok=True)

            training_result = prepare_training_data_from_table(
                conn=conn,
                table_id=int(table_id.value),
                training_name=model_name.value,
                output_dir=_out_dir,
                clean_existing=True,
            )

            stats_rows = "\n".join(
                f"    | {k} | {v} |" for k, v in training_result["stats"].items()
            )
            mo.output.replace(
                mo.md(f"""
            **Training data prepared!**

            | Statistic | Value |
            |-----------|-------|
    {stats_rows}

            - Training images: `{training_result["train_input"]}`
            - Training labels: `{training_result["train_label"]}`
            - Validation images: `{training_result["val_input"]}`
            - Validation labels: `{training_result["val_label"]}`
            - Test images: `{training_result.get("test_input", "none")}`

            The splits come from the training set (each unit's category).
            """)
            )
        except Exception as e:
            mo.output.replace(mo.md(f"**Error:** {str(e)}"))
    return (training_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Run Training
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    batch_size,
    epochs,
    mo,
    model_name,
    model_type,
    n_iterations,
    patch_shape,
    run_training,
    setup_training,
    train_button,
    training_result,
):
    mo.stop(
        not train_button.value,
        mo.md("Click 'Start Training' to begin micro-SAM fine-tuning"),
    )

    training_config = None
    training_results = None
    if training_result is None:
        mo.output.replace(mo.md("**Error:** Prepare training data first"))
    else:
        try:
            _patch = parse_patch_size(patch_shape.value)
            training_config = setup_training(
                training_result,
                model_name=model_name.value,
                epochs=epochs.value,
                n_iterations=int(n_iterations.value) or None,
                batch_size=batch_size.value,
                learning_rate=1e-5,
                patch_shape=tuple(_patch),
                model_type=model_type.value,
                n_objects_per_batch=25,
            )

            mo.output.replace(
                mo.md(
                    f"**Training started...** This may take a while.\n\n- Model: {model_name.value}\n- Epochs: {epochs.value}\n- Iterations: {training_config['n_iterations']}"
                )
            )

            training_results = run_training(training_config, framework="microsam")

            mo.output.replace(
                mo.md(f"""
            **Training completed!**

            - Model name: {training_results["model_name"]}
            - Final model: `{training_results.get("final_model_path", "N/A")}`
            - Checkpoints: {len(training_results.get("checkpoints", []))}
            - Output: `{training_results["output_dir"]}`
            """)
            )
        except Exception as e:
            mo.output.replace(mo.md(f"**Error during training:** {str(e)}"))
    return training_config, training_results


@app.cell
def _(mo):
    mo.md("""
    ## 6. Export to the BioImage Model Zoo (optional)

    Packages the best checkpoint in [BioImage.IO](https://bioimage.io) format, with one
    validation image as the test input. Fill in authors and a description first.
    """)
    return


@app.cell
def _(mo):
    bioimageio_authors = mo.ui.text(
        value="Your Name; Your Institution", label="Author; affiliation"
    )
    bioimageio_description = mo.ui.text(
        value="micro-SAM model fine-tuned on OMERO annotations",
        label="Description",
        full_width=True,
    )
    export_button = mo.ui.run_button(label="Export BioImage.IO model")
    mo.vstack([bioimageio_authors, bioimageio_description, export_button])
    return bioimageio_authors, bioimageio_description, export_button


@app.cell
def _(
    Path,
    bioimageio_authors,
    bioimageio_description,
    export_button,
    mo,
    training_config,
    training_result,
    training_results,
):
    mo.stop(not export_button.value)
    mo.stop(training_results is None, mo.md("**Train a model first.**"))
    import tifffile
    from micro_sam.bioimageio.model_export import export_sam_model

    _images = sorted(Path(training_result["val_input"]).glob("*.tif"))
    _labels = sorted(Path(training_result["val_label"]).glob("*.tif"))
    _checkpoint = training_results["best_checkpoint"] or training_results["latest_checkpoint"]
    mo.stop(not (_images and _labels), mo.md("**No validation data to use as test.**"))
    mo.stop(_checkpoint is None, mo.md("**No checkpoint found from training.**"))
    _name, _, _affiliation = bioimageio_authors.value.partition(";")
    _out = Path(training_results["output_dir"]) / "bioimage_io_model"
    export_sam_model(
        image=tifffile.imread(_images[0]),
        label_image=tifffile.imread(_labels[0]),
        model_type=training_config["model_type"],
        name=training_config["model_name"],
        output_path=str(_out),
        checkpoint_path=str(_checkpoint),
        authors=[{"name": _name.strip(), "affiliation": _affiliation.strip()}],
        description=bioimageio_description.value,
    )
    mo.md(f"BioImage.IO model written to `{_out}`.")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from datetime import datetime
    from pathlib import Path

    return Path, datetime


@app.cell
def _():
    from omero_annotate_ai import (
        prepare_training_data_from_table,
        run_training,
        setup_training,
    )
    from omero_annotate_ai.omero.omero_functions import (
        find_annotation_config_for_table,
    )
    from omero_annotate_ai.trainingset import list_training_sets, summarize

    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

        OMERO_AVAILABLE = True
    except ImportError:
        OMERO_AVAILABLE = False
    return (
        OMERO_AVAILABLE,
        SimpleOMEROConnection,
        find_annotation_config_for_table,
        list_training_sets,
        prepare_training_data_from_table,
        run_training,
        setup_training,
        summarize,
    )


@app.cell
def _(OMERO_AVAILABLE, SimpleOMEROConnection):
    def establish_connection(host, username, password, group=None, secure=True):
        if not OMERO_AVAILABLE:
            raise ImportError(
                "OMERO functionality not available. Install with: pip install -e .[omero]"
            )

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


@app.function
def parse_patch_size(text):
    try:
        parts = [int(x.strip()) for x in text.split(",")]
        if len(parts) == 2:
            return parts
        return [512, 512]
    except ValueError:
        return [512, 512]


if __name__ == "__main__":
    app.run()
