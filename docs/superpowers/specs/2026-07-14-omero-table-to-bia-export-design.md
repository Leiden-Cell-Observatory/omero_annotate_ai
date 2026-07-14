# OMERO table → BioImage Archive export

## Goal

Add a notebook that exports a BioImage Archive (BIA) submission bundle starting from an
**annotation table on OMERO**, mirroring the training notebooks: select a table, fetch the
metadata, download the data, package for BIA.

The existing `prepare_bioimage_archive.ipynb` is a self-contained demo that assumes the
annotation run's `output_directory` is already on the local disk. That only works on the
machine that did the annotating. This adds the OMERO-driven path.

## Blocking problem: `annotation_id` is not persisted

`AnnotationConfig.to_dataframe()` writes the OMERO table columns (`image_id`, `label_id`,
`patch_*`, `processed`, …) but **not** `annotation_id`. `from_dataframe()` therefore leaves
`annotation_id=""` on every `ImageAnnotation`.

Every local BIA path is derived from that id, in `mifa_export._file_ids()`:

```python
return (f"output/{img.annotation_id}_mask.tif", f"{input_dir}/{img.annotation_id}.tif")
```

So a config loaded from a table collapses every row to `output/_mask.tif` — colliding paths
and a broken bundle. The pipeline generates these ids (`{image_id}_{t}_{z}`,
`{image_id}_{t}_{z}_{patch_i}`, `{image_id}_{t}_3d`) but never stores them. Fixing this is a
prerequisite for any table-driven export, and is the reason this feature is not just notebook
glue.

## Design

### 1. Persist `annotation_id` (`core/annotation_config.py`)

- `to_dataframe()`: add an `annotation_id` column.
- `from_dataframe()`: read it back when present.
- Legacy tables (written before this change) have no such column. Regenerate the id
  deterministically from `image_id` / `timepoint` / `z_slice` / `is_volumetric` / patch
  coordinates, following the pipeline's scheme. Patches get a stable index by sorting rows
  within an `(image_id, timepoint, z_slice)` group on `(patch_x, patch_y)`.

A regenerated id may differ from the name the original annotation run used on its own disk.
That is acceptable: the bundle is self-consistent because the fetch step (below) writes the
image and mask files under the same ids the file lists reference.

### 2. `prepare_bia_data_from_table()` (new `processing/bia_data.py`)

```python
prepare_bia_data_from_table(conn, table_id, output_dir, config=None, clean_existing=False) -> dict
```

Downloads the data referenced by the table into the on-disk layout `save_bia_package()`
already expects:

- image plane / patch → `input/{annotation_id}.tif`, or `label_input/{annotation_id}.tif`
  when `config.spatial_coverage.uses_separate_channels()`
- mask `FileAnnotation` (`label_id`) → `output/{annotation_id}_mask.tif`

Keeps rows that are `processed` **and** carry a `label_id`. Unprocessed or maskless rows are
counted and skipped with a warning, never fatal. Returns a stats dict (`output_dir`,
`n_images`, `n_masks`, `skipped_unprocessed`, `skipped_no_mask`).

**Images are written at native bit depth.** The training path
(`training_functions._prepare_dataset_from_table`) normalizes to 8-bit (`img_8bit`, carrying
its own `TODO make this optional`), which is lossy and wrong for an archive submission. This
is the reason the fetch is a separate function rather than a reuse of the training one, and
it justifies the ~40 lines of ezomero plane/patch extraction the two now have in common. The
alternative — refactoring a shared fetch out of the ~450-line `_prepare_dataset_from_table` —
risks a working training path for modest gain, and is rejected.

### 3. `TrainingDataWidget.get_selected_container()` (`widgets/training_data_widget.py`)

Returns `{"type": <container_type>, "id": <container_id>}`. The widget already tracks both in
its dropdowns but exposes only `get_selected_table_id()` / `get_selected_table_info()`. The
notebook needs the container to locate the config YAML attached to it.

### 4. Notebook `notebooks/jupyter/export_annotations/omero-table-to-bioimage-archive.ipynb`

1. Imports.
2. `create_omero_connection_widget()` → `conn`.
3. `create_training_data_widget(conn)` → `table_id` + selected container.
4. `download_annotation_config_from_omero(conn, container_type, container_id)` → config.
   Falls back to `create_default_config()` with a warning when no YAML is attached.
5. `sync_omero_table_to_config(conn, table_id, config)` → `config.annotations`.
6. Editable study-metadata cell: title, description, keywords, license, authors, funding,
   accession. Prefilled from the config that came off OMERO.
7. `prepare_bia_data_from_table(conn, table_id, output_dir, config)`; set
   `config.output.output_directory`.
8. `config.save_bia_package(bundle_dir, accession=...)`.
9. Print the file lists and the MIFA `Annotations` YAML for inspection.
10. Markdown: how to transfer the bundle to the BioImage Archive.
11. Close the connection.

The notebook is **read-only against OMERO** — it writes the bundle to local disk and stops.

### 5. Tests (`tests/test_bia_data.py`, mocked `conn` / `ezomero`)

- `annotation_id` survives a `to_dataframe()` → `from_dataframe()` round-trip.
- Legacy tables (no `annotation_id` column) get deterministic, non-colliding regenerated ids,
  including the patch and volumetric cases.
- Row filtering: unprocessed rows and rows without a `label_id` are skipped and counted.
- Layout naming: `input/{id}.tif` + `output/{id}_mask.tif`; separate-channel configs route the
  image to `label_input/`.
- Images are written at native dtype (a uint16 plane does not come back as uint8).
- The stats dict reports the right counts.

## Out of scope

- Uploading MIFA metadata back to OMERO as file annotations.
- A metadata-entry widget (study fields are edited in a notebook cell).
- Zipping the bundle.
