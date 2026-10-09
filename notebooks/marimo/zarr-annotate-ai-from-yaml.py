# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "omero-annotate-ai[zarr]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from omero_annotate_ai import create_pipeline, load_config_from_yaml

    return create_pipeline, load_config_from_yaml, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Annotate OME-Zarr images from a YAML configuration

    Runs the annotation pipeline on (remote) OME-Zarr data, no OMERO server needed.
    Outputs (masks, images, tracking table, well matching) are written to the output
    directory of the configuration.

    1. Load a YAML configuration with `source: zarr`
    2. Discover the images it selects
    3. Run micro-SAM or CellPose preparation
    """)
    return


@app.cell
def _(mo):
    yaml_path = mo.ui.text(
        value="notebooks/marimo/configs/zarr-s-biad855-example.yaml",
        label="YAML configuration path",
        full_width=True,
    )
    load_button = mo.ui.run_button(label="Load configuration")
    mo.vstack([yaml_path, load_button])
    return load_button, yaml_path


@app.cell
def _(load_button, load_config_from_yaml, mo, yaml_path):
    mo.stop(not load_button.value, mo.md("Enter a YAML path and click 'Load configuration'"))
    config = load_config_from_yaml(yaml_path.value)
    # Keep the source YAML untouched; progress and the updated config go to the output directory.
    config.config_file_path = None
    mo.stop(
        config.source != "zarr",
        mo.md("This configuration reads from OMERO; use `omero-annotate-ai-from-yaml.py`."),
    )
    mo.md(f"""
    | Setting | Value |
    |---|---|
    | Workflow name | {config.name} |
    | URIs | {"<br>".join(config.zarr.uris)} |
    | Level | {config.zarr.level} |
    | Fields | {config.zarr.fields} |
    | Wells | {config.zarr.wells or "all"} |
    | Well metadata | {config.zarr.well_metadata or "-"} |
    | Well filters | {config.zarr.well_filters or "-"} ({config.zarr.well_filter_mode}) |
    | Channels | {config.spatial_coverage.channels} |
    | Model | {config.ai_model.pretrained_from} |
    | Output directory | {config.output.output_directory} |
    """)
    return (config,)


@app.cell
def _(mo):
    discover_button = mo.ui.run_button(label="Discover images")
    discover_button
    return (discover_button,)


@app.cell
def _(config, create_pipeline, discover_button, mo, pd):
    mo.stop(not discover_button.value, mo.md("Click 'Discover images' to list what the config selects"))
    pipeline = create_pipeline(config)
    _uris = pipeline.get_image_uris()
    _rows = [
        {"plate": r.plate, "well": r.well, "field": r.field, "uri": r.uri, **r.well_metadata}
        for r in (pipeline._zarr_refs[u] for u in _uris)
    ]
    mo.vstack([
        mo.md(f"**{len(_uris)} images found.** Well matching: "
              f"`{config.output.output_directory}/well_matching.csv`"),
        mo.ui.table(pd.DataFrame(_rows)) if _rows else mo.md("No images selected."),
    ])
    return (pipeline,)


@app.cell
def _(mo):
    workflow_type = mo.ui.dropdown(
        options=["micro-SAM (interactive)", "CellPose (export for annotation)"],
        value="micro-SAM (interactive)",
        label="Workflow type",
    )
    run_button = mo.ui.run_button(label="Run pipeline", kind="success")
    mo.hstack([workflow_type, run_button], justify="start", gap=2)
    return run_button, workflow_type


@app.cell
def _(mo, pipeline, run_button, workflow_type):
    mo.stop(not run_button.value, mo.md("Click 'Run pipeline' to start"))
    if "micro-SAM" in workflow_type.value:
        _, updated_config = pipeline.run_full_micro_sam_workflow()
    else:
        _, updated_config = pipeline.run_cp_workflow()
    _summary = updated_config.get_progress_summary()
    mo.md(f"""
    **Pipeline finished.**

    - Processed: {_summary["completed_units"]} / {_summary["total_units"]}
    - Output directory: `{updated_config.output.output_directory}`
    - Tracking table: `{updated_config.output.output_directory}/tracking_table.csv`
    """)
    return


if __name__ == "__main__":
    app.run()
