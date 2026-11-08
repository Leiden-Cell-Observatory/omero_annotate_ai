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
    2. Pick a training set (OMERO or local)
    3. Export it into the training layout
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
    ## 2. Training Set

    Pick the container, then the training set (local sets are listed too). Its
    progress per split is shown below.
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
def _(conn, container_id, container_type, mo, training_set_choices):
    _connected = conn is not None and conn.isConnected()
    _note = ""
    try:
        _choices = training_set_choices(
            conn if _connected else None,
            container_type.value,
            int(container_id.value) if container_id.value else None,
        )
    except Exception as _error:  # an OMERO error still lists the local sets
        _note = f"Could not list the sets on OMERO: {_error}"
        _choices = training_set_choices()
    training_set = mo.ui.dropdown(
        options=dict(_choices), label="Training set", full_width=True
    )
    mo.vstack(
        [
            training_set if _choices else mo.md("No training sets found yet."),
            mo.md(_note),
        ]
    )
    return (training_set,)


@app.cell
def _(conn, mo, read_set_config, summarize, training_set):
    mo.stop(training_set.value is None, mo.md("Pick a training set above."))
    annotation_config = read_set_config(training_set.value, conn)
    _summary = summarize(training_set.value, conn)
    _where = f"table {_summary.table_id}" if _summary.table_id else "local"
    _rows = "\n".join(
        f"| {_split} | {_done} / {_total} |"
        for _split, (_done, _total) in _summary.progress.items()
        if _total
    )
    mo.md(f"""
    **{_summary.name}** ({_where})

    | Split | Annotated |
    |---|---|
    {_rows}
    """)
    return (annotation_config,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Training Configuration
    """)
    return


@app.cell
def _(annotation_config, mo):
    _coverage = annotation_config.spatial_coverage
    _default_model = annotation_config.ai_model.pretrained_from or "vit_b_lm"
    _default_patch = (
        f"{_coverage.patch_size[0]}, {_coverage.patch_size[1]}"
        if _coverage.patch_size
        else "512, 512"
    )
    _default_name = annotation_config.name

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
        patch_shape,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 4. Prepare Training Data

    Export the training set's annotated units into `train_*`, `val_*` and `test_*`
    folders: from OMERO for an OMERO set, from its folder for a local set.
    """)
    return


@app.cell
def _(annotation_config, default_export_dir, mo, training_set):
    _coverage = annotation_config.spatial_coverage
    _local = not isinstance(training_set.value, int)  # a local set keeps its channels
    export_directory = mo.ui.text(
        value=str(default_export_dir(training_set.value, annotation_config.name)),
        label="Export folder",
        full_width=True,
    )
    training_channels = mo.ui.text(
        value=", ".join(map(str, _coverage.get_training_channels())),
        label="Training channels",
        disabled=_local,
    )
    label_channel = mo.ui.number(
        value=_coverage.get_label_channel(),
        start=0,
        label="Label channel",
        disabled=_local,
    )
    apply_normalization = mo.ui.checkbox(
        value=bool(annotation_config.normalization.apply_at_export),
        label="Apply the normalisation at export (float32; default keeps native bit depth)",
    )
    prepare_button = mo.ui.run_button(label="Prepare Training Data", kind="success")
    mo.vstack(
        [
            export_directory,
            mo.hstack([training_channels, label_channel], justify="start", gap=2),
            apply_normalization,
            prepare_button,
        ]
    )
    return (
        apply_normalization,
        export_directory,
        label_channel,
        prepare_button,
        training_channels,
    )


@app.cell
def _(
    Path,
    apply_normalization,
    conn,
    export_directory,
    export_training_set,
    label_channel,
    mo,
    parse_indices,
    prepare_button,
    training_channels,
    training_set,
):
    training_result = None
    mo.stop(
        not prepare_button.value,
        mo.md("Click 'Prepare Training Data' to export the training set"),
    )
    _local = not isinstance(training_set.value, int)
    try:
        training_result = export_training_set(
            training_set.value,
            Path(export_directory.value).expanduser(),
            conn,
            training_channels=None
            if _local
            else parse_indices(training_channels.value, "Training channels"),
            label_channel=None if _local else int(label_channel.value),
            apply_normalization=apply_normalization.value,
        )
        _stats = "\n".join(
            f"    | {_key} | {_value} |"
            for _key, _value in training_result["stats"].items()
        )
        mo.output.replace(
            mo.md(f"""
        **Training data prepared** in `{training_result["base_dir"]}`

        | Statistic | Value |
        |-----------|-------|
    {_stats}

        The splits come from the training set (each unit's category).
        """)
        )
    except Exception as _error:
        mo.output.replace(mo.md(f"**Error:** {_error}"))
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
    _checkpoint = (
        training_results["best_checkpoint"] or training_results["latest_checkpoint"]
    )
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
    from pathlib import Path

    return (Path,)


@app.cell
def _():
    from omero_annotate_ai import (
        export_training_set,
        run_training,
        setup_training,
    )
    from omero_annotate_ai.planning import parse_indices
    from omero_annotate_ai.processing.training_functions import default_export_dir
    from omero_annotate_ai.trainingset import (
        read_set_config,
        summarize,
        training_set_choices,
    )

    try:
        from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

        OMERO_AVAILABLE = True
    except ImportError:
        OMERO_AVAILABLE = False
    return (
        OMERO_AVAILABLE,
        SimpleOMEROConnection,
        default_export_dir,
        export_training_set,
        parse_indices,
        read_set_config,
        run_training,
        setup_training,
        summarize,
        training_set_choices,
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
