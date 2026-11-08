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
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import requests

    from omero_annotate_ai import (
        TrainingSet,
        load_config_from_yaml,
        reorganize_local_data_for_training,
        run_training,
        setup_training,
    )

    return (
        Path,
        TrainingSet,
        datetime,
        load_config_from_yaml,
        mo,
        pd,
        reorganize_local_data_for_training,
        requests,
        run_training,
        setup_training,
    )


@app.cell
def _(mo):
    mo.md("""
    # Annotate public IDR data (OME-Zarr)

    The [Image Data Resource (IDR)](https://idr.openmicroscopy.org) publishes many
    studies as OME-Zarr on public S3 storage. This notebook annotates such a plate with
    micro-SAM and trains a model on the result. No OMERO login is needed and nothing is
    uploaded: the training set is kept in a local folder (`storage: local`).

    The example is HeLa nuclei (H2B-GFP) from the MitoCheck screen (study idr0013A).
    Only studies converted to OME-Zarr can be used; step 1 checks a screen for you.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Find a plate (optional)

    Enter an IDR screen ID and the study's OME-Zarr folder (study name plus `A`, `B`, …
    as on the IDR, e.g. `idr0013A`). The IDR web API lists the screen's plates and each
    plate is checked for an OME-Zarr copy.
    """)
    return


@app.cell
def _(mo):
    screen_id = mo.ui.number(value=1101, start=1, label="IDR screen ID")
    zarr_study = mo.ui.text(value="idr0013A", label="OME-Zarr study folder")
    n_plates = mo.ui.number(value=5, start=1, stop=50, label="Plates to check")
    find_button = mo.ui.run_button(label="Find plates")
    mo.hstack([screen_id, zarr_study, n_plates, find_button], justify="start")
    return find_button, n_plates, screen_id, zarr_study


@app.cell
def _(find_button, mo, n_plates, pd, requests, screen_id, zarr_study):
    mo.stop(not find_button.value, mo.md("Click **Find plates** to check a screen."))
    _plates = requests.get(
        "https://idr.openmicroscopy.org/webclient/api/plates/",
        params={"id": int(screen_id.value)},
        timeout=30,
    ).json()["plates"][: int(n_plates.value)]
    _rows = []
    for _plate in _plates:
        _url = (
            "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/"
            f"{zarr_study.value.strip()}/{_plate['id']}.zarr"
        )
        _ok = requests.head(_url + "/.zattrs", timeout=15).status_code == 200
        _rows.append(
            {
                "plate": _plate["id"],
                "name": _plate["name"],
                "OME-Zarr": _url if _ok else "-",
            }
        )
    mo.ui.table(pd.DataFrame(_rows)) if _rows else mo.md("No plates in this screen.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Choose the images

    The defaults come from `notebooks/marimo/configs/idr0013-mitocheck-zarr.yaml`.
    """)
    return


@app.cell
def _(Path, load_config_from_yaml):
    example = load_config_from_yaml(
        str(Path("notebooks/marimo/configs/idr0013-mitocheck-zarr.yaml"))
    )
    return (example,)


@app.cell
def _(example, mo):
    plate_url = mo.ui.text(
        value=example.zarr.uris[0], label="OME-Zarr plate", full_width=True
    )
    wells = mo.ui.text(
        value=", ".join(example.zarr.wells or []), label="Wells (empty = all)"
    )
    channel = mo.ui.number(
        value=example.spatial_coverage.channels[0], start=0, label="Channel"
    )
    level = mo.ui.number(value=example.zarr.level, start=0, label="Resolution level")
    train_n = mo.ui.number(
        value=example.training.train_n, start=1, label="Training images"
    )
    validate_n = mo.ui.number(
        value=example.training.validate_n, start=0, label="Validation images"
    )
    seed = mo.ui.number(value=example.training.seed or 42, label="Seed")
    mo.vstack(
        [
            plate_url,
            mo.hstack([wells, channel, level], justify="start"),
            mo.hstack([train_n, validate_n, seed], justify="start"),
        ]
    )
    return channel, level, plate_url, seed, train_n, validate_n, wells


@app.cell
def _(
    channel,
    datetime,
    example,
    level,
    plate_url,
    seed,
    train_n,
    validate_n,
    wells,
):
    config = example.model_copy(deep=True)
    config.name = f"idr_zarr_{datetime.now():%Y%m%d_%H%M%S}"
    config.zarr.uris = [plate_url.value.strip()]
    config.zarr.wells = [w.strip() for w in wells.value.split(",") if w.strip()] or None
    config.zarr.level = int(level.value)
    config.spatial_coverage.channels = [int(channel.value)]
    config.training.train_n = int(train_n.value)
    config.training.validate_n = int(validate_n.value)
    config.training.seed = int(seed.value)
    return (config,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Create the training set

    Lists the plate's wells and fields and picks the images (one random frame of each
    time-lapse in the example). Only metadata is read here.
    """)
    return


@app.cell
def _(mo):
    create_button = mo.ui.run_button(label="Create training set")
    create_button
    return (create_button,)


@app.cell
def _(TrainingSet, config, create_button, mo, pd):
    mo.stop(not create_button.value, mo.md("Click **Create training set**."))
    ts = TrainingSet.create(config)
    _rows = [
        {"split": u.category, "well": u.well, "field": u.field, "frame": u.timepoint}
        for u in ts.units()
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
    mo.md("""
    ## 4. Annotate

    napari opens with micro-SAM. Segment the nuclei, then close the window; each image
    you segmented is saved. Open the set again later with `TrainingSet.open(folder)`.
    """)
    return


@app.cell
def _(mo):
    note = mo.ui.text(label="Session note (optional)", full_width=True)
    annotate_button = mo.ui.run_button(label="Annotate in napari", kind="success")
    mo.vstack([note, annotate_button])
    return annotate_button, note


@app.cell
def _(annotate_button, mo, note, ts):
    mo.stop(not annotate_button.value, mo.md("Click **Annotate in napari**."))
    _saved = ts.annotate(note=note.value or None)
    mo.md(f"**{_saved} units saved** in `{ts.bundle_dir}`.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Train

    The annotated images are copied into a training layout next to the set's folder
    and a micro-SAM model is fine-tuned on them. A few epochs only check that it runs;
    use 50 or more for a useful model.
    """)
    return


@app.cell
def _(mo):
    epochs = mo.ui.number(value=5, start=1, stop=500, label="Epochs")
    train_button = mo.ui.run_button(label="Prepare data and train", kind="success")
    mo.hstack([epochs, train_button], justify="start")
    return epochs, train_button


@app.cell
def _(
    datetime,
    epochs,
    mo,
    reorganize_local_data_for_training,
    run_training,
    setup_training,
    train_button,
    ts,
):
    mo.stop(
        not train_button.value,
        mo.md("Annotate first, then click **Prepare data and train**."),
    )
    mo.stop(
        not any(u.processed for u in ts.units()),
        mo.md("**No annotations yet.** Run step 4 first."),
    )
    _data = reorganize_local_data_for_training(ts.config, annotation_dir=ts.bundle_dir)
    _training = setup_training(
        _data,
        model_name=f"idr_microsam_{datetime.now():%Y%m%d_%H%M%S}",
        model_type="vit_b_lm",
        epochs=int(epochs.value),
        batch_size=1,
        patch_shape=(512, 512),
    )
    _result = run_training(_training, framework="microsam")
    mo.md(f"""
    **Training finished.** Model: `{_result.get("final_model_path", "-")}`
    (output in `{_result["output_dir"]}`)
    """)
    return


if __name__ == "__main__":
    app.run()
