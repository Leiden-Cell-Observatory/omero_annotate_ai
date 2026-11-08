# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "omero-annotate-ai[microsam,zarr]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from omero_annotate_ai import TrainingSet, load_config_from_yaml
    from omero_annotate_ai.omero.simple_connection import SimpleOMEROConnection

    return Path, SimpleOMEROConnection, TrainingSet, load_config_from_yaml, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Annotate from a YAML configuration (OMERO or OME-Zarr)

    Load a YAML configuration and create a training set from it, or open an existing set
    from its folder, then annotate with micro-SAM or export images for Cellpose.

    - `source: omero` reads images from an OMERO container (connect below).
    - `source: zarr` reads OME-Zarr images, local or remote (e.g. the BioImage Archive),
      with no OMERO server. Such sets use `storage: local`. Try
      `notebooks/marimo/configs/zarr-s-biad855-example.yaml`.
    """)
    return


@app.cell
def _(mo):
    yaml_path = mo.ui.text(
        value="notebooks/marimo/configs/zarr-s-biad855-example.yaml",
        label="YAML configuration, or a training set folder to open",
        full_width=True,
    )
    load_button = mo.ui.run_button(label="Load")
    mo.vstack([yaml_path, load_button])
    return load_button, yaml_path


@app.cell
def _(Path, load_button, load_config_from_yaml, mo, yaml_path):
    mo.stop(not load_button.value, mo.md("Enter a path and click **Load**."))
    _path = Path(yaml_path.value).expanduser()
    set_folder = _path if _path.is_dir() else None
    config = load_config_from_yaml(_path / "config.yaml" if set_folder else _path)
    _what = config.zarr.uris if config.source == "zarr" else (
        f"{config.omero.container_type} {config.omero.container_id}"
    )
    mo.md(f"""
    | {"Open" if set_folder else "New"} training set | **{config.name}** |
    |---|---|
    | Source | {config.source}: {_what} |
    | Storage | {config.storage} |
    | Splits | {config.training.train_n} / {config.training.validate_n} / {config.training.test_n} |
    | Channels | {config.spatial_coverage.channels} |
    | Model | {config.ai_model.pretrained_from} |
    """)
    return config, set_folder


@app.cell
def _(config, mo):
    mo.stop(config.source != "omero" and config.storage != "omero")
    omero_host = mo.ui.text(label="OMERO host")
    omero_username = mo.ui.text(label="Username")
    omero_password = mo.ui.text(label="Password", kind="password")
    connect_button = mo.ui.run_button(label="Connect")
    mo.hstack([omero_host, omero_username, omero_password, connect_button])
    return connect_button, omero_host, omero_password, omero_username


@app.cell
def _(
    SimpleOMEROConnection,
    config,
    connect_button,
    mo,
    omero_host,
    omero_password,
    omero_username,
):
    conn = None
    if config.source == "omero" or config.storage == "omero":
        mo.stop(not connect_button.value, mo.md("This set needs OMERO: connect."))
        try:
            conn = SimpleOMEROConnection().create_connection_from_config(
                {
                    "host": omero_host.value,
                    "username": omero_username.value,
                    "password": omero_password.value,
                }
            )
        except Exception as _error:  # ConnectionFailed: report and stop
            mo.stop(True, mo.md(f"**Not connected:** {_error}"))
    return (conn,)


@app.cell
def _(TrainingSet, config, conn, mo, pd, set_folder):
    ts = (
        TrainingSet.open(set_folder, conn)
        if set_folder
        else TrainingSet.create(config, conn)
    )
    _rows = [
        {
            "split": r.category,
            "image": r.image_name,
            "well": r.well,
            "field": r.field,
            "annotated": r.processed,
        }
        for r in ts.units()
    ]
    mo.vstack(
        [
            mo.md(f"**{len(_rows)} units** in `{ts.bundle_dir}`"),
            mo.ui.table(pd.DataFrame(_rows)),
        ]
    )
    return (ts,)


@app.cell
def _(mo):
    tool = mo.ui.dropdown(
        options=["micro-SAM (napari)", "Cellpose GUI"],
        value="micro-SAM (napari)",
        label="Annotation tool",
    )
    note = mo.ui.text(label="Session note (optional)", full_width=True)
    run_button = mo.ui.run_button(label="Start", kind="success")
    mo.vstack([mo.hstack([tool, run_button], justify="start"), note])
    return note, run_button, tool


@app.cell
def _(mo, note, run_button, tool, ts):
    mo.stop(not run_button.value, mo.md("Click **Start**."))
    if tool.value == "Cellpose GUI":
        _n = ts.prepare_external()
        _result = (
            f"**{_n} images exported** to `{ts.bundle_dir / 'annotation_input'}`. "
            "Annotate them in the Cellpose GUI, save each mask as "
            f"`{ts.bundle_dir / 'annotation_output'}/<image name>_mask.tif`, then "
            "click **Collect masks**."
        )
    else:
        _saved = ts.annotate(note=note.value or None)
        _result = f"**{_saved} units saved** in `{ts.bundle_dir}`."
    mo.md(_result)
    return


@app.cell
def _(mo, tool):
    collect_button = mo.ui.run_button(label="Collect masks")
    collect_button if tool.value == "Cellpose GUI" else mo.md("")
    return (collect_button,)


@app.cell
def _(collect_button, mo, ts):
    mo.stop(not collect_button.value)
    _collected = ts.collect_masks()
    mo.md(f"**{_collected} masks saved.**")
    return


if __name__ == "__main__":
    app.run()
