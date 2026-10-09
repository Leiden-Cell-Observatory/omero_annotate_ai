# OME-Zarr image source + URI identity (Phase 2, sub-projects A+B)

**Date:** 2026-10-09
**Status:** approved 2026-10-09 (user: "implement it in a minimal reproducible way")
**Builds on:** Phase 1 `ImageSource` seam (branch `feat/image-source-seam`, plan
`docs/superpowers/plans/2026-10-08-image-source-seam.md`)
**Assessment:** `~/.claude/plans/the-package-now-is-glowing-spark.md`

## Minimal-scope rule

The user asked for a minimal, reproducible implementation. So: small tasks, each with
its own tests; no feature beyond what the success criteria need; existing method names
kept where a new one would only be a rename. Cuts made for this reason are marked
*(minimal)*.

## Goal

Run the existing annotation workflow (micro-SAM and CellPose preparation) on
(remote) OME-Zarr images, driven by a YAML config, without an OMERO server. Images
are identified by URI everywhere, for both backends.

## What the user decided

- No backwards compatibility in the code paths we write. Images are identified by URI:
  `omero:image:<id>` for OMERO, the image group's URL for Zarr.
- `annotation_id = {hash8(image_uri)}_{t}_{z}[_{patch}]` for every source, so files and
  table rows stay linked to the image whatever the backend.
- The multiscale `level` is configurable and recorded per annotation; segmenting at a
  lower resolution is a real use case.
- Well metadata for Zarr plates comes from a CSV referenced in the config, plus a plain
  `wells:` list. Because the CSV-to-well link is something *we* make (OME-Zarr stores no
  such link), both sides are recorded: the Zarr well that was read and the CSV values it
  was matched to.
- Read Zarr through the official `ome-zarr-py` library, behind our `ImageSource`.
- Outputs for Zarr are local only in A+B (masks, images, CSV table, YAML config).
  Writing labels into the OME-Zarr `labels/` group is sub-project C.
- No Jupyter widget changes. A marimo notebook that runs from YAML, like
  `notebooks/marimo/omero-annotate-ai-from-yaml.py`, is the user-facing entry point.

## Out of scope (later sub-projects)

- C: labels into OME-Zarr (sparse), `LocalSink`/`OmeroSink` split.
- D: training from a Zarr table, BIA annotations-only export with remote `source_image`.
- `s3://` URIs and `storage_options` (users pass the HTTPS endpoint, e.g. EBI's
  `https://livingobjects.ebi.ac.uk/...`).
- Resuming a Zarr run from a previous local table.
- Fetching well metadata from IDR/BIA APIs (could later *generate* the CSV).
- Python 3.12 / newest ome-zarr (see Dependencies).

## Verified facts (probe, 2026-10-09)

ome-zarr 0.16.0 with zarr 3.1.5 on Python 3.11, against S-BIAD855 plate `20585`
(`https://livingobjects.ebi.ac.uk/bioimaging-integrator-data/S-BIAD855/781ac3d7-673f-47be-a4d2-3fdf3f477047/781ac3d7-673f-47be-a4d2-3fdf3f477047.zarr`):

- `parse_url(plate).root_attrs["plate"]`: name `20585`, 384 wells, `field_count` 9,
  version 0.4. The wells list is **not** in row/column order (first entry `H/21`).
- `parse_url(plate + "/B/2").root_attrs["well"]["images"]`: paths `0`..`8`.
- `Reader(parse_url(plate + "/B/2/0"))()` gives one `Multiscales` node in 1.1 s:
  `node.data` = 4 dask levels `(5,520,696)` … `(5,65,87)` uint16;
  `node.metadata["axes"]` = `c,y,x`; `node.metadata["channel_names"]` = the 5 stains.
- A 64×64 slice of level 1 computes in 0.2 s.
- ome-zarr prints `version mismatch: detected: FormatV04, requested: FormatV05` on every
  open of a 0.4 store. Harmless, but must be silenced (it goes through `logging`).

## Design

### 1. Config

`AnnotationConfig` gets a source selector and a `zarr` section. `omero` stays as is.

```yaml
source: zarr                 # "omero" (default) | "zarr"
zarr:
  uris:                      # each: plate, well, or single image group (http(s) URL or local path)
    - https://livingobjects.ebi.ac.uk/.../781ac3d7-....zarr
  level: 0                   # multiscale level to read; all pixel coordinates are in this level's grid
  fields: first              # plates/wells only: "first" | "all"
  wells: [B2, C5]            # optional: only these wells (any plate)
  well_metadata: layout.csv  # optional: local path or URL of a CSV with well metadata
  well_filters:              # optional: same semantics as omero.well_filters, on CSV values
    compound: [DMSO]
  well_filter_mode: include  # include | exclude
```

Pydantic model `ZarrConfig` (in `annotation_config.py`):

| field | type | default |
|---|---|---|
| `uris` | `List[str]` | `[]` |
| `level` | `int`, `ge=0` | `0` |
| `fields` | `Literal["first","all"]` | `"first"` |
| `wells` | `Optional[List[str]]` | `None` |
| `well_metadata` | `Optional[str]` | `None` |
| `well_filters` | `Optional[Dict[str, List[str]]]` | `None` |
| `well_filter_mode` | `Literal["include","exclude"]` | `"include"` |

Validation (model validator on `AnnotationConfig`):

- `source == "zarr"` requires a non-empty `zarr.uris`.
- `zarr.well_filters` requires `zarr.well_metadata`.

A Zarr run never writes to OMERO, which is exactly what `workflow.read_only_mode`
already means in the pipeline (masks to `annotation_output/`, table as local CSV, no
uploads). So the same model validator sets `workflow.read_only_mode = True` when
`source == "zarr"` *(minimal: replaces a new `writes_to_omero` helper that would have
touched ~8 call sites)*. The saved YAML then shows `read_only_mode: true`, which is
truthful.

### 2. Image identity

`ImageAnnotation`:

- `image_uri: str = ""` — the identity. A model validator fills it with
  `omero_uri(image_id)` when it is empty and `image_id` is set, so OMERO code and tests
  that build `ImageAnnotation(image_id=…)` keep working *(minimal)*.
- `image_id: Optional[int] = None` — the OMERO image id, set only for OMERO URIs. It is
  kept because the OMERO table's typed `ImageColumn` (PR #50) and ROI/attachment
  uploads need an integer id.
- New: `level: int = 0`, `well: str = ""`, `field: str = ""`,
  `well_metadata: Dict[str, str] = {}`.

New module `src/omero_annotate_ai/sources/uri.py` (pure functions, no I/O):

- `omero_uri(image_id: int) -> str` → `"omero:image:<id>"`.
- `parse_omero_uri(uri: str) -> Optional[int]` → id, or `None` for non-OMERO URIs.
- `normalize_zarr_uri(uri: str) -> str` → canonical form: `http(s)` URLs with any query
  string and trailing `/` removed; local paths made absolute (no `file://`). A local
  path's hash therefore differs between machines; that is accepted.
- `uri_key(uri: str) -> str` → first 8 hex chars of `sha1(uri)`.
- `annotation_id(uri, t, z, *, volumetric=False, patch=None) -> str` →
  `{key}_{t}_{z}` / `{key}_{t}_3d`, plus `_{patch}` when `patch` is given. Replaces the
  four f-strings in `_prepare_processing_units`. `_annotation_id_base` in
  `annotation_config.py` is **not** changed: it regenerates ids for legacy tables that
  never stored one, so it must keep the old `{image_id}_…` form.

Table columns (`to_dataframe` / `from_dataframe`, local CSV and OMERO table):

- Add `image_uri` (str), `level` (int), `well` (str), `field` (str),
  `well_metadata` (str: JSON object, `"{}"` when empty).
- `image_id` stays an int column; Zarr rows write `0` (the existing "not set" sentinel,
  see `_optional_int_to_omero`).
- `from_dataframe` on a table without `image_uri` (written before this change) derives it
  as `omero_uri(image_id)` and keeps the stored `annotation_id`. This is a one-line read
  of an existing column, not a migration; tables on users' servers keep resuming.

`ImageMeta.image_id` is renamed `ImageMeta.image_uri: str`.

### 3. `ZarrImageSource` (`src/omero_annotate_ai/sources/zarr.py`)

Implements the Phase 1 `ImageSource` protocol. Handles are a small frozen dataclass:

```python
@dataclass(frozen=True)
class ZarrImage:
    uri: str              # normalized image-group URI
    name: str
    level: int
    data: Any             # dask array of the chosen level, in the file's own axis order
    axes: Tuple[str, ...] # e.g. ("c", "y", "x")
```

- `ZarrImageSource(level: int = 0)`.
- `get_image(uri)` → opens with `Reader(parse_url(uri))`, takes the first node with a
  `Multiscales` spec, picks `node.data[level]`, reads axis names from
  `node.metadata["axes"]` (NGFF ≥0.4 list of dicts; plain strings also accepted).
  Raises `LookupError` when the URI does not exist or has no multiscale image, and
  `ValueError` when `level` is beyond the available levels (message lists how many).
  Handles are cached per URI inside the source (opening costs ~1 s over HTTP).
- `metadata(image)` → sizes from the level's shape mapped onto t/c/z/y/x; axes absent from
  the file count as size 1. dtype = `str(data.dtype)`.
- `read_plane(image, *, t, c, z, region=None)` → builds an index tuple in the file's axis
  order (`t`/`c`/`z` → the given index, `y`/`x` → slices; region clipped with Phase 1
  `clamp_region`), then `np.asarray(data[idx].compute())`. Indices for axes the file does
  not have must be 0, otherwise `IndexError`.
- `read_volume(...)` → one dask slice over `z_start..z_end` (inclusive), computed once.
  Requires a `z` axis when `z_end > z_start`.
- Image name: `"{plate}_{well}_{field}"` when it came from a plate (set by discovery),
  otherwise `node.metadata["name"]` stripped of slashes, falling back to the last
  path component.
- ome-zarr's version-mismatch log lines are silenced by raising the level of the
  `ome_zarr` logger to `WARNING` around opens.

### 4. Discovery (`src/omero_annotate_ai/sources/zarr_discovery.py`)

Turns `ZarrConfig` into the list of images to annotate. No pixel reads.

```python
@dataclass(frozen=True)
class ZarrImageRef:
    uri: str
    plate: str = ""           # NGFF plate name
    well: str = ""            # canonical "B2"
    field: str = ""           # field path, e.g. "0"
    well_metadata: Dict[str, str] = field(default_factory=dict)

def discover_zarr_images(cfg: ZarrConfig) -> List[ZarrImageRef]
```

Per URI, read `root_attrs` (0.4 top-level keys, 0.5 under `"ome"`):

- `plate` → wells from `plate.wells[].path` sorted by (rowIndex, columnIndex); keep the
  selected wells (below); for each, read the well's `images` and pick fields
  (`first` = first listed, or `all`). Random field sampling is left out *(minimal)*; the
  pipeline's existing random image selection already samples across the list.
- `well` → its fields, same field rule; plate/well names from the parent path when the
  parent is a plate, otherwise the URI's last two path parts.
- `multiscales` → a single image ref.
- anything else → `ValueError` naming the URI and the keys found.

Well selection, applied **before** any well's `images` are read (one request per
selected well):

1. `wells` list, if given (well names normalized: `B2`, `B02`, `B/2` all → `B2`).
2. `well_metadata` CSV, if given: read with pandas (local path or URL). Required column
   `well`; optional column `plate` (matched against the NGFF plate name). Match key is
   `plate+well` when the CSV has a `plate` column, else `well`. All other columns are
   metadata, stored as strings.
3. `well_filters` / `well_filter_mode` on the matched metadata, with the same AND logic
   as OMERO. The logic of the existing `AnnotationPipeline._check_well_filter` moves to a
   pure function `match_well_filter(kv: Dict[str, str], filters) -> bool` in
   `sources/wells.py`; `_check_well_filter` stays as a one-line call to it, so the OMERO
   path and its tests are unchanged.

**Recording the link (user requirement).** Discovery writes
`well_matching.csv` into the output directory: one row per discovered well with
`plate, well, well_uri, matched (bool), included (bool), <csv columns…>`. Wells with no
CSV row and CSV rows with no well each produce one warning summarizing the count and
the first few names. Unmatched wells are excluded when `well_filters` is set in
`include` mode, kept otherwise. Each selected image's `ZarrImageRef.well_metadata` ends
up in the annotation's `well_metadata` column. The config keeps `well_metadata` and the
filters, so the YAML plus `well_matching.csv` show exactly which decision was made.

### 5. Pipeline changes (`annotation_pipeline.py`)

- Source selection: new `make_image_source(config, conn) -> ImageSource` in
  `sources/__init__.py`: `OmeroImageSource(conn)` for `omero`,
  `ZarrImageSource(level=config.zarr.level)` for `zarr`. `AnnotationPipeline.__init__`
  uses it when no `image_source` is passed. `create_pipeline(config, conn=None, …)`
  works unchanged for Zarr.
- `OmeroImageSource.get_image` accepts `omero:image:<id>` URIs (and still ints, for the
  OMERO helpers that hold ids).
- New `get_image_uris()` returns URIs and is what `initialize_workflow` calls:
  OMERO → `[omero_uri(i) for i in self.get_image_ids_from_container()]` (existing method
  and its tests untouched) *(minimal)*;
  Zarr → `discover_zarr_images`, keeping the refs on the pipeline
  (`self._zarr_refs: Dict[str, ZarrImageRef]`) so `_prepare_processing_units` can fill
  `well`, `field`, `well_metadata` and the name.
  `get_images_by_ids` keeps its name and passes whatever it gets to
  `image_source.get_image` *(minimal)*.
- `_prepare_processing_units` fills `image_uri`, `image_id` (OMERO only), `level`
  (`config.zarr.level` for Zarr, `0` for OMERO), and uses `uri.annotation_id(...)`.
- Every place that reads `meta["image_id"]` for an image lookup uses `image_uri`; OMERO
  uploads keep using `annotation.image_id` and are only reached when
  `config.workflow.read_only_mode` is false (never for Zarr).
- `_validate_setup` needs no change: Zarr configs are read-only (section 1) and the
  default source for them is not an `OmeroImageSource`.
- The "No images found in {container_type} {container_id}" error names the Zarr URIs for
  Zarr runs.

### 6. Notebook (`notebooks/marimo/zarr-annotate-ai-from-yaml.py`)

Same cell flow as `omero-annotate-ai-from-yaml.py`, minus the OMERO connection:

1. imports (`marimo as mo`, `load_config_from_yaml`, `create_pipeline`, polars).
2. YAML path text field + run button → `load_config_from_yaml`; `mo.stop` if
   `config.source != "zarr"` with a pointer to the OMERO notebook.
3. Config summary table (name, URIs, level, fields, wells, well_metadata, filters,
   output directory, model type).
4. "Discover" button → `pipeline.get_image_uris()`; show a polars table of discovered
   images (plate, well, field, metadata) and the `well_matching.csv` path.
5. Workflow dropdown: micro-SAM (`run_full_micro_sam_workflow()`) or CellPose
   (`run_cp_workflow()`), run button.
6. Results: output folder, progress summary, save-config cell.

Plus an example config `notebooks/marimo/configs/zarr-s-biad855-example.yaml` pointing at
the S-BIAD855 plate (2 training images, level 1, `fields: first`, wells `[B2, C5, D10]`).
`marimo check --fix` must pass.

### 7. Dependencies

- pyproject `[project.optional-dependencies]`: `zarr = ["ome-zarr>=0.16"]`, and add it to
  `all`. ome-zarr brings `aiohttp`, `requests` and `ome-zarr-models`.
- pixi: `ome-zarr = ">=0.16"` in a new `zarr` feature; environments become
  `default = ["zarr"]` (the notebook runs in the default env) and
  `dev = ["dev","all","mifa","zarr"]`. Solves to ome-zarr 0.16.0 and moves dask
  2026.1.2 → 2026.8.0 (ome-zarr excludes dask 2026.1/2026.2); verified with a trial solve.
- `sources/zarr.py` imports `ome_zarr` lazily and raises `ImportError` with
  `pip install omero-annotate-ai[zarr]` when missing, so an OMERO-only install works.
- ome-zarr ≥0.18 needs Python 3.12; we pin 3.11. Upgrading Python is a separate change.

### 8. Errors

| situation | behaviour |
|---|---|
| URI unreachable / not NGFF | `LookupError` from `get_image`; discovery raises `ValueError` naming the URI |
| `level` too high | `ValueError` with available level count |
| channel/t/z index beyond the image | `IndexError` with the axis and size |
| `wells` names not in plate | warning listing them; if nothing is left, the existing "No images found" error |
| CSV missing `well` column | `ValueError` at discovery |
| CSV / well mismatch | one summarized warning each way; recorded in `well_matching.csv` |
| ome-zarr not installed | `ImportError` with install hint |

### 9. Testing

Fixtures are written with `ome_zarr.writer` into `tmp_path` (no network in the suite):
a single image in NGFF 0.4 (`c,y,x`, 2 levels), a single image in 0.5 (`t,c,z,y,x`),
and a small 0.4 plate (2×3 wells, 2 fields).

- `tests/test_image_sources.py`: `TestZarrImageSource` — protocol, metadata per axis
  layout, plane/region/volume reads match the written array, level selection, too-high
  level, missing URI → `LookupError`, handle caching. `TestUri` — omero round trip,
  normalization, stable `uri_key`, `annotation_id` forms.
- `tests/test_well_filtering.py`: `TestZarrDiscovery` — plate/well/image detection,
  sorted wells, `fields` first/all, `wells` normalization, CSV matching with
  and without `plate`, filters include/exclude, `well_matching.csv` contents,
  mismatch warnings. `match_well_filter` unit tests (moved).
- `tests/test_config.py`: `ZarrConfig` validation, Zarr forces `read_only_mode`, YAML round trip,
  new table columns, `from_dataframe` without `image_uri`.
- `tests/test_pipeline.py`: an end-to-end Zarr run on the plate fixture with a fake
  annotation function (`run_custom_annotation`): annotation ids are hash-based, masks and
  CSV land locally, rows carry `image_uri`, `level`, `well`, `field`, `well_metadata`;
  no connection needed. Existing OMERO tests updated to URIs.
- One `@pytest.mark.slow` network test against S-BIAD855 (skipped without network):
  discover 1 well, read one 64×64 patch at level 1.

## Success criteria

1. `pixi run -e dev pytest tests/ -v` is green, including the new tests.
2. The example YAML runs in the marimo notebook on S-BIAD855 without OMERO: discovery
   lists the configured wells, micro-SAM opens patches at the chosen level, and
   `annotation_output/`, the CSV table and `well_matching.csv` appear in the output
   folder with linked `annotation_id`s.
3. An OMERO run (live check by the user, as for Phase 1) behaves as before apart from
   the new id format and columns.
