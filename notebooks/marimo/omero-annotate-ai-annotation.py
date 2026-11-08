# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "omero-annotate-ai[microsam]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from omero_annotate_ai import TrainingSet, create_default_config
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection
    from omero_annotate_ai.trainingset import list_training_sets, summarize

    return (
        SimpleOMEROConnection,
        TrainingSet,
        create_default_config,
        list_training_sets,
        mo,
        summarize,
    )


@app.cell
def _(mo):
    mo.md("""
    # Annotate OMERO images with micro-SAM or Cellpose

    Create a **training set** (images picked from an OMERO container and split into
    training/validation/test) or open an earlier one, then annotate it: interactively in
    napari with micro-SAM, or in the Cellpose GUI (images are exported, you annotate, the
    notebook collects the masks). Masks are uploaded to OMERO as ROIs and label files and tracked in the set's
    table; a local copy lives in the set's folder under `~/omero_annotate_ai/`. You can stop
    at any time: opening the set again continues where you left off and repairs anything a
    crash left half-done.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Connect to OMERO
    """)
    return


@app.cell
def _(mo):
    omero_host = mo.ui.text(label="Host", placeholder="omero.example.org")
    omero_username = mo.ui.text(label="Username")
    omero_password = mo.ui.text(label="Password", kind="password")
    omero_group = mo.ui.text(label="Group (optional)")
    omero_secure = mo.ui.checkbox(value=True, label="Encrypt data transfer")
    connect_button = mo.ui.run_button(label="Connect")
    mo.vstack(
        [
            mo.hstack([omero_host, omero_username, omero_password], justify="start"),
            mo.hstack([omero_group, omero_secure, connect_button], justify="start"),
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
    SimpleOMEROConnection,
    connect_button,
    mo,
    omero_group,
    omero_host,
    omero_password,
    omero_secure,
    omero_username,
):
    mo.stop(
        not connect_button.value, mo.md("Fill in the fields and click **Connect**.")
    )
    _settings = {
        "host": omero_host.value,
        "username": omero_username.value,
        "password": omero_password.value,
        "secure": omero_secure.value,
    }
    if omero_group.value:
        _settings["group"] = omero_group.value
    try:
        conn = SimpleOMEROConnection().create_connection_from_config(_settings)
    except Exception as _error:  # ConnectionFailed: show it, don't crash the notebook
        conn = None
        _status = f"**Not connected:** {_error}"
    else:
        conn.c.enableKeepAlive(60)
        _status = f"Connected to group **{conn.getGroupFromContext().getName()}**."
    mo.md(_status)
    return (conn,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Container and training set
    """)
    return


@app.cell
def _(mo):
    container_type = mo.ui.dropdown(
        options=["dataset", "project", "plate", "screen"],
        value="dataset",
        label="Container type",
    )
    container_id = mo.ui.number(value=0, start=0, label="Container ID")
    mode = mo.ui.radio(
        options=["New training set", "Open training set"],
        value="New training set",
    )
    mo.vstack([mo.hstack([container_type, container_id], justify="start"), mode])
    return container_id, container_type, mode


@app.cell
def _(conn, container_id, container_type, list_training_sets, mo, mode):
    existing_sets = []
    if conn is not None and container_id.value > 0:
        existing_sets = list_training_sets(
            conn, container_type.value, int(container_id.value)
        )
    set_choice = mo.ui.dropdown(
        options={f"{s.name} (table {s.table_id})": s.table_id for s in existing_sets},
        label="Training set",
    )
    _view = (
        set_choice if existing_sets else mo.md("No training sets on this container.")
    )
    _view if mode.value == "Open training set" else mo.md("")
    return (set_choice,)


@app.cell
def _(conn, mo, mode, set_choice, summarize):
    mo.stop(mode.value != "Open training set" or set_choice.value is None)
    _summary = summarize(set_choice.value, conn)
    _progress = "<br>".join(
        f"{name}: {done}/{total} annotated"
        for name, (done, total) in _summary.progress.items()
        if total
    )
    _sessions = "<br>".join(
        f"#{s.id} {s.started or ''}: {s.n_annotated} saved"
        + (f", *{s.note}*" if s.note else "")
        for s in _summary.sessions[-5:]
    )
    mo.md(f"""
    | {_summary.name} | |
    |---|---|
    | Progress | {_progress or "-"} |
    | Last sessions | {_sessions or "none yet"} |
    """)
    return


@app.cell
def _(mo):
    set_name = mo.ui.text(value="", label="Name", placeholder="e.g. hela_nuclei_v1")
    train_n = mo.ui.number(value=3, start=0, label="Training images")
    validate_n = mo.ui.number(value=2, start=0, label="Validation images")
    test_n = mo.ui.number(value=0, start=0, label="Test images")
    seed = mo.ui.text(value="", label="Seed (empty = random)")
    channel = mo.ui.number(value=0, start=0, label="Channel")
    timepoints = mo.ui.text(value="0", label="Timepoints")
    z_slices = mo.ui.text(value="0", label="Z-slices")
    three_d = mo.ui.checkbox(value=False, label="3D (volumes)")
    use_patches = mo.ui.checkbox(value=False, label="Patches")
    patch_size = mo.ui.text(value="512, 512", label="Patch size (w, h)")
    patches_per_image = mo.ui.number(value=1, start=1, label="Patches per image")
    storage = mo.ui.dropdown(
        options={
            "OMERO (upload masks, keep a table)": "omero",
            "Local folder only": "local",
        },
        value="OMERO (upload masks, keep a table)",
        label="Storage",
    )
    return (
        channel,
        patch_size,
        patches_per_image,
        seed,
        set_name,
        storage,
        test_n,
        three_d,
        timepoints,
        train_n,
        use_patches,
        validate_n,
        z_slices,
    )


@app.cell
def _(mo):
    add_train = mo.ui.number(value=0, start=0, label="Training images (0 = keep)")
    add_validate = mo.ui.number(value=0, start=0, label="Validation images (0 = keep)")
    add_test = mo.ui.number(value=0, start=0, label="Test images (0 = keep)")
    return add_test, add_train, add_validate


@app.cell
def _(
    add_test,
    add_train,
    add_validate,
    channel,
    mo,
    mode,
    patch_size,
    patches_per_image,
    seed,
    set_name,
    storage,
    test_n,
    three_d,
    timepoints,
    train_n,
    use_patches,
    validate_n,
    z_slices,
):
    if mode.value == "New training set":
        _form = mo.vstack(
            [
                set_name,
                mo.hstack([train_n, validate_n, test_n, seed], justify="start"),
                mo.hstack([channel, timepoints, z_slices, three_d], justify="start"),
                mo.hstack(
                    [use_patches, patch_size, patches_per_image], justify="start"
                ),
                storage,
            ]
        )
    else:
        _form = mo.vstack(
            [
                mo.md("Raise a count to add images to the set (counts only go up):"),
                mo.hstack([add_train, add_validate, add_test], justify="start"),
            ]
        )
    _form
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Annotate
    """)
    return


@app.cell
def _(mo):
    tool = mo.ui.dropdown(
        options=["micro-SAM (napari)", "Cellpose GUI"],
        value="micro-SAM (napari)",
        label="Annotation tool",
    )
    model_type = mo.ui.dropdown(
        options=["vit_b_lm", "vit_l_lm", "vit_b", "vit_l", "vit_h"],
        value="vit_b_lm",
        label="micro-SAM model",
    )
    batch_size = mo.ui.number(
        value=0, start=0, label="Images per napari session (0 = all)"
    )
    note = mo.ui.text_area(
        label="Note for this session (optional)",
        placeholder="e.g. nuclei only, small debris skipped",
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Start", kind="success")
    mo.vstack(
        [mo.hstack([tool, model_type, batch_size], justify="start"), note, run_button]
    )
    return batch_size, model_type, note, run_button, tool


@app.cell
def _(
    TrainingSet,
    add_test,
    add_train,
    add_validate,
    batch_size,
    channel,
    conn,
    container_id,
    container_type,
    create_default_config,
    mo,
    mode,
    model_type,
    note,
    patch_size,
    patches_per_image,
    run_button,
    seed,
    set_choice,
    set_name,
    storage,
    test_n,
    three_d,
    timepoints,
    tool,
    train_n,
    use_patches,
    validate_n,
    z_slices,
):
    mo.stop(not run_button.value, mo.md("Click **Start**."))
    mo.stop(conn is None, mo.md("**Connect to OMERO first.**"))

    def _ints(text):
        return [int(v) for v in text.replace(" ", "").split(",") if v]

    _added = 0
    if mode.value == "Open training set":
        mo.stop(set_choice.value is None, mo.md("**Choose a training set.**"))
        ts = TrainingSet.open(set_choice.value, conn)
        _added = ts.extend(
            train_n=add_train.value or None,
            validate_n=add_validate.value or None,
            test_n=add_test.value or None,
        )
    else:
        mo.stop(
            not set_name.value or container_id.value <= 0,
            mo.md("**Give the set a name and choose a container.**"),
        )
        _config = create_default_config()
        _config.name = set_name.value
        _config.storage = storage.value
        _config.omero.container_type = container_type.value
        _config.omero.container_id = int(container_id.value)
        _config.training.train_n = int(train_n.value)
        _config.training.validate_n = int(validate_n.value)
        _config.training.test_n = int(test_n.value)
        _config.training.seed = int(seed.value) if seed.value.strip() else None
        _config.spatial_coverage.channels = [int(channel.value)]
        _config.spatial_coverage.timepoints = _ints(timepoints.value)
        _config.spatial_coverage.z_slices = _ints(z_slices.value)
        _config.spatial_coverage.three_d = three_d.value
        _config.spatial_coverage.use_patches = use_patches.value
        _config.spatial_coverage.patch_size = _ints(patch_size.value)
        _config.spatial_coverage.patches_per_image = int(patches_per_image.value)
        _config.ai_model.pretrained_from = model_type.value
        if tool.value == "Cellpose GUI":
            _config.ai_model.framework = "cellpose"
        ts = TrainingSet.create(_config, conn)

    _table = ts.table_id if ts.table_id is not None else "local"
    if tool.value == "Cellpose GUI":
        _exported = ts.prepare_external()
        _result = f"""
    **{_exported} images exported for Cellpose** (training set **{ts.name}**, table {_table}).

    1. Open the Cellpose GUI (`cellpose`; install with `pixi add cellpose` or
       `pip install cellpose`).
    2. Annotate the images in `{ts.bundle_dir / "annotation_input"}`.
    3. Save each mask in `{ts.bundle_dir / "external_output"}` as
       `<image name>_mask.tif` (e.g. `12345_0_0.tif` → `12345_0_0_mask.tif`; Cellpose's
       own `_seg.npy` / `_cp_masks.png` files are not picked up).
    4. Come back and click **Collect masks** below.
    """
    else:
        _saved = ts.annotate(
            model=model_type.value,
            batch_size=int(batch_size.value) or None,
            note=note.value or None,
        )
        _done = sum(r.processed for r in ts.units())
        _result = f"""
    **Session finished: {_saved} units saved.**

    - Training set: **{ts.name}** (table {_table})
    - Annotated: {_done} of {len(ts.units())}
    - Images added: {_added}
    - Local folder: `{ts.bundle_dir}`
    """
    mo.md(_result)
    return (ts,)


@app.cell
def _(mo, tool, ts):
    collect_button = mo.ui.run_button(label="Collect masks")
    collect_button if tool.value == "Cellpose GUI" else mo.md("")
    return (collect_button,)


@app.cell
def _(collect_button, mo, ts):
    mo.stop(not collect_button.value)
    _collected = ts.collect_masks()
    _done = sum(r.processed for r in ts.units())
    mo.md(
        f"**{_collected} masks saved** (uploaded for OMERO sets). "
        f"Annotated: {_done} of {len(ts.units())}."
    )
    return


@app.cell
def _(mo, ts):
    later_note = mo.ui.text_area(label="Add or change the note of the last session")
    save_note = mo.ui.run_button(label="Save note")
    mo.vstack([later_note, save_note]) if ts.config.sessions else mo.md("")
    return later_note, save_note


@app.cell
def _(later_note, mo, save_note, ts):
    mo.stop(not save_note.value)
    _session = ts.config.sessions[-1]
    ts.set_session_note(_session.id, later_note.value)
    mo.md(f"Note saved to session #{_session.id}.")
    return


if __name__ == "__main__":
    app.run()
