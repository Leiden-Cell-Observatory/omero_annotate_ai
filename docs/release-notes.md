# Release notes

Newest first. The full list of pull requests per version is on
[GitHub releases](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/releases).

## Unreleased

- **Croissant and bioimage.io descriptions.** `ts.write_croissant()` writes
  `croissant.json` (MLCommons Croissant: masks, image regions, splits, classes,
  sessions) and `ts.write_bioimageio_dataset()` writes `rdf.yaml` (a bioimage.io
  dataset description) into the set's folder, from the schema library. They need the
  set's `description`, `authors` and `url` (bioimage.io also `keywords` and a paper with
  a DOI) and name what is missing.
- **Agent skill** in `skills/omero-annotate-ai`: helps coding agents (Claude Code, Codex,
  Cursor, ...) set up, run and debug omero-annotate-ai. See the README.
- **Finding data in the widgets.** The workflow and training-data widgets pick
  containers by group (switching the session's group), owner (*Me* by default, all
  members or one colleague) and a search box that filters by name or ID; *Go to ID*
  finds a container in any of your groups. The workflow widget picks several containers
  of one type for a multi-container set.
- **More of the config in the workflow widget**, in folded sections: the channels to
  train on, extra napari context channels with colours, the normalisation, and well
  filters on plate metadata (MIHCSME assay conditions first, with a live count of the
  wells kept).
- **Well filters apply to screens** too (to the wells of each plate of the screen).
- **Stay logged in.** A password login can save a token (on by default in the
  Jupyter connection widget, the marimo notebooks and the napari plugin): next time,
  choose it under *Saved login* and log in without a password. One token per server and
  user, so logins for several OMERO servers live side by side and the form still logs
  in to another one. Tokens end after 7 days unused; *Forget* ends one on the server.
  Kept in `~/.omero-annotate-ai/tokens.json` (user-only permissions); passwords are
  still never stored. `SimpleOMEROConnection().connect_saved(host=...)` does the same
  in scripts, and a token from `omero-user-token set` is picked up too.
- **The napari plugin is a wizard now: from your OMERO data to annotated images
  without a notebook.** Step 1 logs in (or uses a saved login); step 2 finds the data
  (group, owner, search by name or ID, *Go to ID*, several containers); step 3 opens
  a set (and can add images to it) or makes a **new** one, in its own wide window, with the same choices as the
  Jupyter widget: channels to annotate, train on and show, planes, 3D, patches,
  counts, seed, storage, model, batch size and plate well filters. *Show sample image
  in viewer* shows each channel as a layer. Context channels chosen for a set are now
  shown next to the image while annotating in the dock. The form logic lives in
  `omero_annotate_ai.forms`, shared by both front ends.
- **Two more Jupyter notebooks**, next to their marimo versions: the IDR OME-Zarr demo
  (`annotation/idr-zarr-demo.ipynb`) and run from a YAML file
  (`annotation/annotate-from-yaml.ipynb`).
- **IDR OME-Zarr moved** to `livingobjects.ebi.ac.uk`; the old host
  (`uk1s3.embassy.ebi.ac.uk`) refuses connections. Example config, notebooks and the
  tutorial use the new address.
- **Config version 4: sets are described with the Bioimage Annotation Schema.** The
  config's description sections are now the models of the
  [Bioimage Annotation Schema](https://github.com/Leiden-Cell-Observatory/bioimage-annotation-schema)
  (`bioimage_annotation_schema.datamodel`): `name`, `description`, `license`,
  `version`, `changes`, `previous_version`, `ai_models_trained`, `authors`,
  `keywords`, `links`, `study`, `source_dataset`, `methodology`, `classes`,
  `sampling`, `split`, `notes` and `extensions`. The tool sections (`omero`, `zarr`,
  `storage`, `training`, `normalization`, `ai_model`, `workflow`, `identity`,
  `sessions`) stay, plus a new `display` section for context channels. Configs start
  with `config_version: 4`; version 3 configs are not read, and every moved key raises
  `RemovedConfigKey` naming its new place (`spatial_coverage` → `sampling`,
  `annotation_methodology` → `methodology`, `dataset` → `source_dataset` and
  `license`, `feature_types` → `classes`, `training.seed` / `split_unit` /
  `stratify_by` → `split`, `custom_metadata` → `extensions`, ...). Annotation
  types use MIFA's names (`segmentation_mask`, `point_annotations`, ...); the licence
  is `CC-BY-4.0` or `CC0-1.0`. `config.label_channel()`, `config.training_channels()`
  and `config.uses_separate_channels()` replace the `spatial_coverage` methods.
- **`TrainingSet.describe(**fields)`** changes the description of an existing set
  (version, keywords, study, links, ...), locally and on OMERO. `sampling` and `split`
  are fixed once the set exists. `ts.config` is a copy; changing it does not change
  the set.
- **Split report with every export.** `ts.write_ro_crate()` and `save_bia_package`
  write `split-report.yaml`: the images, wells and conditions per split, those found
  in more than one split and the conditions that miss a split.
- **Exports come from the schema library.** `ts.write_ro_crate()` now also writes
  `annotation-set.yaml` and `annotation-records.csv` (checked against the schema) into
  the set's folder. `save_bia_package` moved to `omero_annotate_ai.schema_export`; it
  no longer takes `funding_statement=` and does not fill in placeholders: it raises
  `MifaNotReady` listing the MIFA fields the config lacks (study description and
  funding statement, keywords, a link, version). The modules
  `omero_annotate_ai.crate` and `omero_annotate_ai.core.mifa_export` are removed; the
  MIFA functions (`save_mifa`, `check_mifa`, `save_bia_package`) live in
  `bioimage_annotation_schema.mifa`.
- **ISCC codes say how they were made.** When units carry ISCC codes, the exported
  `annotation-set.yaml` has a `content_codes.region` block: code type (Image-Code,
  64 bits), which pixels, how they were prepared, and the package and `iscc-lib`
  versions. ISCC codes are only comparable when made the same way, so other tools
  (for example iscc-bio, whose ISCC-SUMs cover the whole source image) can tell
  whether a code is theirs. The export checks that the set and its records agree
  (`bioimage_annotation_schema.check_documents`).
- **`url`: where the set is published** (DOI or archive page). The exported
  `annotation-set.yaml` carries it; the schema's Croissant and bioimage.io dataset
  descriptions need it. Set it with `ts.describe(url=...)` once the set is deposited.
- **The OMERO server is recorded.** Records name images as `omero:image:<id>`,
  which is not traceable without the server. The config's new `omero_server` (host,
  database UUID, optionally `web_url`) is filled in from the connection, the exported
  `annotation-set.yaml` carries it, and the RO-Crate links each source image to it.
  Exporting OMERO images without it fails; open an older set once with a connection
  to fill it in.
- **Install:** `bioimage-annotation-schema` is a new dependency. Until it is on PyPI,
  install it from GitHub (see [Installation](installation.md)).

## 0.5.3

- **Passwords are no longer stored anywhere.** The Jupyter connection widget lost its
  "Save password to keychain" option and its **Load from Keychain** button, and
  `SimpleOMEROConnection.save_password` / `load_password` are gone; `keyring` is no
  longer a dependency. Enter the password each time you connect; the connection history
  still remembers server, port, username and group. Passwords saved by earlier
  versions stay in the OS keychain under the service name `omero-annotate-ai`, with
  the key `<host>:<username>`. Remove them with your system's keychain tool (Seahorse
  on Linux, Keychain Access on macOS, Credential Manager on Windows) or with
  `pipx run keyring del omero-annotate-ai <host>:<username>` (`keyring` itself is no
  longer installed with the package).
- **The connection widget has no Test Connection button any more.** It cleared the
  password, which then had to be typed again for **Connect**; Connect reports the same
  errors. `SimpleOMEROConnection.test_connection` is removed with it.
- **The BioImage Archive notebook no longer invents study metadata.**
  `omero-table-to-bioimage-archive.ipynb` filled an empty title, author, funding
  statement, organism and accession with example values; it now keeps what the config
  holds and stops until the required fields are filled in.
- **Notebooks:** the marimo notebooks take a session key too, and
  `annotate-from-yaml.py` a group and the encryption switch. The patch
  shape of the training notebook is labelled (height, width), which is how it is read. The inference
  notebook no longer carries an image ID and checkpoint name from our own server.
- **A dropped connection is restored.** Before remote work (opening a set, each batch,
  each save, listing sets) the connection is checked; if the network dropped it
  rejoins its OMERO session, and if the session itself ended (the server's idle
  timeout) it logs in again with the credentials held in memory for this run (never
  on disk). Only when that fails are you asked to connect again.
- **Log in with a session key** instead of username and password: a *Session key*
  field in the Jupyter connection widget and the napari plugin (and `session_key` in
  `create_connection_from_config`), e.g. for a session started in OMERO.web or
  BIOMERO. Closing such a connection only detaches; the session stays usable where it
  came from. A session-key connection cannot log in again once its session has ended.
- **`import omero_annotate_ai` works without the `notebooks` extra.** `python-dotenv`
  is now a base dependency, and the widget factories are imported on first use; without
  `ipywidgets` they raise an error naming `omero-annotate-ai[notebooks]` (#110).
- **A local set opens from its folder without an OMERO connection**:
  `TrainingSet.open(folder)` no longer needs one for `storage: local` sets with OMERO
  images; the connection is asked for only when an image is read (#112).
- `spatial_coverage.patch_size` is documented as `[height, width]`, which is how it has
  always been read (#111).

## 0.5.2

- **Session numbers start at 1.** A new training set numbered its first session 0
  (unannotated rows carry session -1). Existing sets keep their numbers; the next
  session follows the highest one.
- **napari plugin:** while the next image's embedding is computed, that image is shown
  dimmed, the previous image's objects and prompts are hidden and micro-SAM's dock is
  disabled, so nothing is drawn against the wrong embedding. Each image gets its own
  contrast limits; the opacity you set on the image layer is kept.
- **The napari plugin no longer offers to store the OMERO password** in the system
  keychain; it is asked for at each connect and the field is cleared straight away.
- **napari plugin:** the dock can be made as short as the annotate page, so micro-SAM's
  dock below it gets the room (it kept the height of the start page).
- **napari plugin: Prev** goes back through every image of the session, also saved
  ones: they open with their objects, and Accept replaces the saved mask and upload
  (`AnnotationSession.revise`). The next image's embedding is computed while you
  annotate the current one.
- **One pixel reader.** The training export and the BioImage Archive export read
  planes, patches and volumes through the same image source as annotation
  (`omero_annotate_ai.sources.read_unit`) instead of two separate ezomero copies.
  This fixes the training export of 3D units, which wrote one plane instead of the
  annotated z range.
- **napari plugin: Exclude** leaves an image out for good (out of focus, empty well,
  artefact): it counts as done, is not shown again and is not exported. Prev back to
  it and Accept saves it after all. Skip still means "later".
- **Exclude:** `AnnotationSession.exclude(batch, ids)` marks units as excluded (bad,
  out of focus). An excluded unit is done: it is recorded as `annotation_type:
  excluded` with `processed` true and no mask, label file or ROI, is never handed out
  or reviewed again, and is left out of the training and BioImage Archive exports.
  Progress counts it as done and reports it separately (`summarize(...).excluded`,
  `progress_summary(...)["excluded_units"]`); a session's `n_annotated` does not count
  it. Committing a mask for it later in the same session makes it a normal annotation.
  Excluding a unit whose earlier upload failed halfway deletes that upload. Tables and
  configs stay readable by 0.5.0 and 0.5.1: no column or config key is added (those
  versions see an excluded unit as annotated without a mask). Resuming now compares
  against the local copy as the last run left it (it was rewritten when leftover
  claims were released first).

## 0.5.1

- **napari plugin** (experimental, `pip install "omero-annotate-ai[napari]"`): a dock in
  your own napari viewer to open a training set, annotate with micro-SAM image by image
  (Accept / Skip / Prev), review earlier annotations with their saved masks, and finish
  with a note. Closing the window or the dock ends the session and frees its claims; see
  [napari plugin](napari-plugin.md).
- **Review** counts a mask with the same labels in a wider data type (micro-SAM writes
  uint32) as unchanged, so it is not uploaded again. A batch that fails to load (or is
  cancelled while loading) releases its claims at once.
- **ISCC content codes** (optional `provenance` extra, `iscc-lib`): each unit records an
  ISCC Image-Code of its image and mask next to the SHA-256 digests. Resume and export
  warnings say whether a changed file is a near-duplicate or different content, and the
  BioImage Archive file lists carry `pixel_sha256` and `iscc` per file. Tables made by
  0.5.0 keep working without the new columns.
- **RO-Crate**: `ts.write_ro_crate()` describes a training set's folder as an RO-Crate
  1.1 (`ro-crate-metadata.json`): files with digests and codes, source images, and the
  sessions with their notes.
- **Training-data widget** rebuilt on the 0.5 API: pick an OMERO or local training set,
  see its progress, choose output folder, channels and normalisation, and export.
  The marimo training notebook offers the same choices; both call the new
  `export_training_set` (`get_selected_table_info` is gone; use `selected_config()`).
- **BioImage Archive export** from an OMERO training set works again (the rows had no
  image IDs), and a bundle holds only annotated units.
- **Public IDR data via OME-Zarr.** The IDR's OMERO API is no longer open for
  logins, so the [public-data tutorial](tutorials/public-data-idr.md) and the new
  `idr-zarr-demo.py` notebook (replacing `omero-idr-demo.py`) read IDR studies as
  OME-Zarr from its public S3 storage, with an example config
  (`notebooks/marimo/configs/idr0013-mitocheck-zarr.yaml`).

## 0.5.0

0.5.0 makes annotation runs reliable and traceable, adds OME-Zarr as an image source, and
reorganises the package around **training sets**. It is a breaking release.

### Highlights

- **Training sets.** `TrainingSet.create(config, conn)` selects the images and starts
  tracking; `TrainingSet.open(table_id)` (or a folder) continues it. The tracking table
  is updated in place, so its ID identifies the set for good.
- **Crash-safe and resumable.** Each mask is saved locally, uploaded with a tag
  (`omero-annotate-ai:<table id>:<unit id>`) and then recorded. Opening a set repairs
  anything a crash left half-done, without duplicate uploads. A failed upload leaves the
  unit unannotated instead of marking it done.
- **Integrity checks.** Pixel checksums of every image and mask are stored in the table;
  a resume or an export reports a mask that changed after upload.
- **Several computers, one set.** Sessions claim the units they show, so two people can
  annotate the same OMERO set at the same time without doing an image twice. Your own
  claims (e.g. left by a dropped connection) never block you; other computers' claims
  expire after `workflow.claim_hours`, and a unit saved elsewhere is never uploaded again.
- **Review and correct.** `ts.review()` (widget: *Review annotated*) opens annotated
  units with their saved masks in napari; changed masks replace the uploads on OMERO.
- **Sessions and notes.** Every annotation session is recorded (who, when, model, how
  many units) with an optional note, e.g. "nuclei only, small debris skipped".
- **One folder per set** under `~/omero_annotate_ai/` (or `$OMERO_ANNOTATE_AI_HOME`), with
  the same layout whether masks go to OMERO or stay local.
- **OME-Zarr images** (local or remote, e.g. the BioImage Archive) with `source: zarr`,
  no OMERO needed. Images are identified by URI (`omero:image:<id>` or a Zarr URL).
- **Reproducible selection.** A stored `training.seed` drives every random choice;
  splits can keep wells together (`split_unit: well`) and be stratified by well metadata.
- **Training export** keeps native bit depth, includes the test split, records the
  normalisation in `normalization.json`, and micro-SAM training normalises at load.
- **New workflow widget**: *New training set* (timepoints and z-slices: specific, all or
  random) or *Open training set* with a summary of progress and earlier sessions, a
  progress bar and a note editor for any session.
- **Safer delete**: `delete_training_set` finds everything by table ID and tag, runs as a
  dry run first, waits for the server, and refuses objects owned by other users.
- **Cellpose route**: `ts.prepare_external()` exports images; masks saved in
  `external_output/` are collected with `ts.collect_masks()`.

## 0.4.0

- MIFA / BioImage Archive export (#49).
- One folder layout for annotation and training (`annotation_input/`, `model_input/`,
  `annotation_output/`) and typed OMERO table columns (#50).
- numpy 2 support (#52); `__version__` matches the installed package (#57).

## 0.3.3 – 0.3.1

- Training folders cleaned correctly (#46).
- Workflow widget states what it creates; safer credential handling in the connection
  widget (#45, #44).
- Masks and ROIs uploaded to the right OMERO images (file-ordering bug, #42).

Older versions: see [GitHub releases](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/releases).
