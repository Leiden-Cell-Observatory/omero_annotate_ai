# OME-Zarr Source + URI Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the annotation workflow from a YAML config on (remote) OME-Zarr images without OMERO, with every image identified by a URI.

**Architecture:** Images get a string URI (`omero:image:<id>` or the Zarr image URL) and hash-based annotation ids. A `ZarrImageSource` (ome-zarr-py behind the Phase 1 `ImageSource` protocol) reads pixels; a discovery module expands plate/well URIs into image URIs and applies well selection from a list or a CSV, recording the match in `well_matching.csv`. A Zarr config is forced read-only, so the existing local-output path is reused.

**Tech Stack:** Python 3.11, pydantic 2, ome-zarr 0.16 (zarr 3.1.5, dask), pandas, pytest, marimo.

**Spec:** `docs/superpowers/specs/2026-10-09-zarr-source-and-uri-identity-design.md`

## Global Constraints

- Minimal scope: implement only what the spec says; keep existing method names where a new one would only be a rename.
- `annotation_id = {uri_key(image_uri)}_{t}_{z}` / `{uri_key}_{t}_3d`, plus `_{patch}`; `uri_key` = first 8 hex of `sha1(uri)`.
- `_annotation_id_base` / `_regenerate_annotation_ids` in `annotation_config.py` are NOT changed (legacy tables).
- OMERO URI form is exactly `omero:image:<id>`.
- `ome-zarr>=0.16` (resolves to 0.16.0 on Python 3.11); imported lazily, missing → `ImportError` mentioning `pip install 'omero-annotate-ai[zarr]'`.
- `source == "zarr"` forces `workflow.read_only_mode = True`.
- Zarr rows write `image_id` = `0` (OMERO_ID_UNSET) in tables.
- Tests run with `pixi run -e dev pytest tests/ -v`; tests go in the existing test files per CLAUDE.md mapping (sources → `tests/test_image_sources.py`).
- Work on branch `feat/image-source-seam` (Phase 2 builds on Phase 1); do not touch the user's unstaged notebook / `.vscode` changes.

## Review Focus

1. Windows local paths: discovery joins with `/` and splits well URIs on `/`; a Windows path with `\` separators may mis-derive the plate name for a direct well URI (plate URIs still work).
2. A CSV `plate` column holding a different plate name than the NGFF `plate.name` silently matches nothing — the mismatch warning must fire (pinned in Task 5).
3. A `wells:` entry that is not in the plate must not crash; it just selects nothing (pinned in Task 5), and with nothing left the "No images found" error names the Zarr URIs (pinned in Task 6).
4. Channel index from `spatial_coverage.channels` beyond the Zarr `c` size must raise a clear `IndexError`, not return garbage (pinned in Task 4).
5. Tables read back from CSV have NaN in empty string columns (`image_uri`, `well`, `well_metadata`); `from_dataframe` must not produce `"nan"` (pinned in Task 2).

---

### Task 1: URI helpers

**Files:**
- Create: `src/omero_annotate_ai/sources/uri.py`
- Test: `tests/test_image_sources.py` (add `TestUri`)

**Interfaces:**
- Produces: `omero_uri(image_id: int) -> str`, `parse_omero_uri(uri: Any) -> Optional[int]`, `normalize_zarr_uri(uri: str) -> str`, `uri_key(uri: str) -> str`, `annotation_id(uri: str, t: int, z: int, *, volumetric: bool = False, patch: Optional[int] = None) -> str` — all in `omero_annotate_ai.sources.uri`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_sources.py`:

```python
from omero_annotate_ai.sources.uri import (
    annotation_id,
    normalize_zarr_uri,
    omero_uri,
    parse_omero_uri,
    uri_key,
)


@pytest.mark.unit
class TestUri:
    def test_omero_round_trip(self):
        assert omero_uri(42) == "omero:image:42"
        assert parse_omero_uri("omero:image:42") == 42

    @pytest.mark.parametrize("value", ["https://x/y.zarr", "omero:image:abc", 42, None])
    def test_parse_non_omero_returns_none(self, value):
        assert parse_omero_uri(value) is None

    def test_normalize_http_drops_query_and_trailing_slash(self):
        assert (
            normalize_zarr_uri("https://h/p.zarr/B/2/0/?anonymous=true")
            == "https://h/p.zarr/B/2/0"
        )

    def test_normalize_local_path_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert normalize_zarr_uri("img.zarr/") == str(tmp_path / "img.zarr")

    def test_uri_key_is_stable_sha1_prefix(self):
        assert uri_key("omero:image:1") == "4ba674d4"
        assert uri_key("omero:image:2") != uri_key("omero:image:1")

    def test_annotation_id_forms(self):
        key = uri_key("omero:image:1")
        assert annotation_id("omero:image:1", 0, 3) == f"{key}_0_3"
        assert annotation_id("omero:image:1", 0, 3, patch=2) == f"{key}_0_3_2"
        assert annotation_id("omero:image:1", 1, 0, volumetric=True) == f"{key}_1_3d"
        assert (
            annotation_id("omero:image:1", 1, 0, volumetric=True, patch=0)
            == f"{key}_1_3d_0"
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run -e dev pytest tests/test_image_sources.py::TestUri -v`
Expected: collection error `ModuleNotFoundError: No module named 'omero_annotate_ai.sources.uri'`

- [ ] **Step 3: Implement**

Create `src/omero_annotate_ai/sources/uri.py`:

```python
"""Image URIs: one string identity for images from any source.

OMERO images are ``omero:image:<id>``; OME-Zarr images are the URL (or local path) of
their image group. Annotation ids are derived from a short hash of the URI so files and
table rows stay linked to the image whatever the backend.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Optional

_OMERO_PREFIX = "omero:image:"


def omero_uri(image_id: int) -> str:
    """URI of an OMERO image."""
    return f"{_OMERO_PREFIX}{int(image_id)}"


def parse_omero_uri(uri: Any) -> Optional[int]:
    """OMERO image id of ``uri``, or ``None`` when it is not an OMERO URI."""
    if isinstance(uri, str) and uri.startswith(_OMERO_PREFIX):
        rest = uri[len(_OMERO_PREFIX):]
        if rest.isdigit():
            return int(rest)
    return None


def normalize_zarr_uri(uri: str) -> str:
    """Canonical form of a Zarr image URI: no query string or trailing slash for
    http(s) URLs, an absolute path for local stores."""
    if re.match(r"^https?://", uri):
        return uri.split("?", 1)[0].rstrip("/")
    return os.path.abspath(os.path.expanduser(uri))


def uri_key(uri: str) -> str:
    """Short stable key for ``uri``: the first 8 hex characters of its SHA-1."""
    return hashlib.sha1(uri.encode("utf-8")).hexdigest()[:8]


def annotation_id(
    uri: str,
    t: int,
    z: int,
    *,
    volumetric: bool = False,
    patch: Optional[int] = None,
) -> str:
    """Annotation id ``{key}_{t}_{z}`` (or ``{key}_{t}_3d``), plus ``_{patch}``."""
    base = f"{uri_key(uri)}_{t}_3d" if volumetric else f"{uri_key(uri)}_{t}_{z}"
    return base if patch is None else f"{base}_{patch}"
```

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run -e dev pytest tests/test_image_sources.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/sources/uri.py tests/test_image_sources.py
git commit -m "feat: add image URI and hash-based annotation id helpers"
```

---

### Task 2: URI identity in the config and tables

**Files:**
- Modify: `src/omero_annotate_ai/core/annotation_config.py` (`ImageAnnotation` ~L209-297, `to_dataframe` ~L999-1097, `from_dataframe` ~L1099-1170)
- Test: `tests/test_config.py` (add `TestImageUriIdentity`)

**Interfaces:**
- Consumes: `omero_uri`, `parse_omero_uri` from Task 1.
- Produces: `ImageAnnotation.image_uri: str` (auto-filled from `image_id`), `ImageAnnotation.image_id: Optional[int]`, `ImageAnnotation.level: int`, `.well: str`, `.field: str`, `.well_metadata: Dict[str, str]`; table columns `image_uri`, `level`, `well`, `field`, `well_metadata` (JSON string).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` (it already imports `pytest`, `pd`, `ImageAnnotation`, `create_default_config`; add any missing import):

```python
@pytest.mark.unit
class TestImageUriIdentity:
    def test_uri_is_derived_from_omero_id(self):
        ann = ImageAnnotation(image_id=7, image_name="a")
        assert ann.image_uri == "omero:image:7"

    def test_zarr_annotation_has_no_omero_id(self):
        ann = ImageAnnotation(image_uri="https://h/p.zarr/B/2/0", image_name="p_B2_0")
        assert ann.image_id is None

    def test_new_columns_round_trip(self):
        config = create_default_config()
        config.annotations.append(
            ImageAnnotation(
                image_uri="https://h/p.zarr/B/2/0", image_name="p_B2_0",
                annotation_id="abc_0_0", level=1, well="B2", field="0",
                well_metadata={"compound": "DMSO"},
            )
        )
        df = config.to_dataframe()
        assert df.loc[0, "image_uri"] == "https://h/p.zarr/B/2/0"
        assert df.loc[0, "image_id"] == 0
        assert df.loc[0, "level"] == 1
        assert df.loc[0, "well_metadata"] == '{"compound": "DMSO"}'

        restored = create_default_config()
        restored.from_dataframe(df)
        ann = restored.annotations[0]
        assert (ann.image_uri, ann.image_id, ann.level, ann.well, ann.field) == (
            "https://h/p.zarr/B/2/0", None, 1, "B2", "0"
        )
        assert ann.well_metadata == {"compound": "DMSO"}

    def test_csv_round_trip_has_no_nan_strings(self, tmp_path):
        config = create_default_config()
        config.annotations.append(
            ImageAnnotation(image_id=5, image_name="a", annotation_id="x_0_0")
        )
        path = tmp_path / "t.csv"
        config.to_dataframe().to_csv(path, index=False)
        restored = create_default_config()
        restored.from_dataframe(pd.read_csv(path))
        ann = restored.annotations[0]
        assert ann.image_uri == "omero:image:5"
        assert (ann.well, ann.field, ann.well_metadata) == ("", "", {})

    def test_legacy_table_without_uri_column(self):
        df = pd.DataFrame([{"image_id": 101, "image_name": "img",
                            "annotation_id": "101_0_0", "timepoint": 0, "z_slice": 0}])
        config = create_default_config()
        config.from_dataframe(df)
        ann = config.annotations[0]
        assert ann.image_uri == "omero:image:101"
        assert ann.annotation_id == "101_0_0"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run -e dev pytest tests/test_config.py::TestImageUriIdentity -v`
Expected: FAIL (`image_uri` unknown / missing attribute)

- [ ] **Step 3: Implement**

In `annotation_config.py`, add the import after the pydantic import:

```python
from ..sources.uri import omero_uri
```

Add a helper next to `_row_int`:

```python
def _row_str(row, key: str, default: str = "") -> str:
    """Read a string field from a table row; NaN/blank (e.g. from CSV) gives ``default``."""
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)
```

In `ImageAnnotation`, replace the `image_id` field and add the new fields + validator:

```python
    # Image identification
    image_uri: str = Field(
        default="", description="Image URI: omero:image:<id> or the OME-Zarr image URL"
    )
    image_id: Optional[int] = Field(
        default=None, description="OMERO image ID (OMERO images only)"
    )
    image_name: str = Field(description="Image name")
    annotation_id: str = Field(default="", description="Unique annotation identifier")

    # Where the image came from (OME-Zarr)
    level: int = Field(default=0, description="Multiscale level the pixel coordinates refer to")
    well: str = Field(default="", description="Plate well, e.g. B2")
    field: str = Field(default="", description="Field (well sample) path, e.g. 0")
    well_metadata: Dict[str, str] = Field(
        default_factory=dict, description="Well metadata the image was selected with"
    )
```

(keep the existing `image_name`/`annotation_id` lines only once), and add inside the class:

```python
    @model_validator(mode="after")
    def _fill_image_uri(self):
        if not self.image_uri and self.image_id is not None:
            self.image_uri = omero_uri(self.image_id)
        return self
```

In `to_dataframe`, in the row dict replace `"image_id": annotation.image_id,` with:

```python
                "image_id": _optional_int_to_omero(annotation.image_id),
                "image_uri": annotation.image_uri,
```

and append to the row dict:

```python
                "level": annotation.level,
                "well": annotation.well,
                "field": annotation.field,
                "well_metadata": json.dumps(annotation.well_metadata, sort_keys=True),
```

In the `columns` list insert `"image_uri",` after `"image_id",` and append `"level", "well", "field", "well_metadata",` at the end. Add `"level",` to `numeric_columns`.

In `from_dataframe`, in `annotation_data` replace `"image_id": int(row.get("image_id", -1)),` with:

```python
                "image_id": _omero_id_to_optional_int(row.get("image_id")),
                "image_uri": _row_str(row, "image_uri"),
                "level": _row_int(row, "level", 0),
                "well": _row_str(row, "well"),
                "field": _row_str(row, "field"),
                "well_metadata": json.loads(_row_str(row, "well_metadata") or "{}"),
```

- [ ] **Step 4: Run the new tests, then the whole config/pipeline suite**

Run: `pixi run -e dev pytest tests/test_config.py::TestImageUriIdentity -v`
Expected: PASS
Run: `pixi run -e dev pytest tests/ -q -p no:cacheprovider > .superpowers/t2.log 2>&1; tail -5 .superpowers/t2.log`
Expected: all pass. If a test compares a full `to_dataframe()` column list or asserts `image_id == -1` for a missing id, update that expectation to the new columns / `None` (the spec changes both) and note it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/core/annotation_config.py tests/test_config.py
git commit -m "feat: identify annotated images by URI in config and tables"
```

---

### Task 3: Pipeline uses URIs and hash-based ids

**Files:**
- Modify: `src/omero_annotate_ai/sources/base.py` (`ImageMeta.image_id` → `image_uri: str`)
- Modify: `src/omero_annotate_ai/sources/omero.py` (`get_image` accepts URIs, `metadata` returns URI)
- Modify: `src/omero_annotate_ai/core/annotation_pipeline.py` (`_prepare_processing_units` L255-420, `_run_micro_sam_annotation` L508-512, `_load_image_data` L584, `_save_original_image_for_annotation` L821-826, `_save_local_progress` L1142-1147, `_convert_annotations_to_processing_units` L1356-1381)
- Test: `tests/test_image_sources.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1 helpers; Task 2 `ImageAnnotation.image_uri`.
- Produces: `ImageMeta(image_uri: str, name, size_t, size_c, size_z, size_y, size_x, dtype)`; `OmeroImageSource.get_image(image: int | str)`; `AnnotationPipeline._identity_fields(image_meta: ImageMeta) -> dict` (keys `image_uri`, `image_id`, `image_name`); processing-unit tuples `(image_uri, annotation_id, meta, row_index)` with `meta["image_uri"]` and `meta["image_id"]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_image_sources.py`, change `test_metadata`'s expected value to `ImageMeta(image_uri="omero:image:7", name="img.tif", size_t=2, size_c=3, size_z=4, size_y=10, size_x=12, dtype="uint16")` and add to `TestOmeroImageSource`:

```python
    def test_get_image_accepts_omero_uri(self):
        assert self.source.get_image("omero:image:7") is self.image
        self.conn.getObject.assert_called_once_with("Image", 7)

    def test_get_image_rejects_non_omero_uri(self):
        with pytest.raises(LookupError, match="not an OMERO image"):
            self.source.get_image("https://h/p.zarr/B/2/0")
```

In `tests/test_pipeline.py`, update the helpers so the fake speaks URIs (add `from omero_annotate_ai.sources.uri import annotation_id as make_annotation_id, omero_uri, parse_omero_uri`):

```python
class FakeImageSource:
    """In-memory ImageSource over (t, c, z, y, x) arrays; records read calls."""

    def __init__(self, arrays):
        self.arrays = arrays  # {image_id: ndarray (t, c, z, y, x)}
        self.reads = []

    def get_image(self, image_id):
        if isinstance(image_id, str):
            image_id = parse_omero_uri(image_id)
        if image_id not in self.arrays:
            raise LookupError(image_id)
        return FakeImage(image_id, f"img{image_id}")

    def metadata(self, image):
        t, c, z, y, x = self.arrays[image.image_id].shape
        return ImageMeta(omero_uri(image.image_id), image.name, t, c, z, y, x,
                         str(self.arrays[image.image_id].dtype))
```

(`read_plane` / `read_volume` unchanged.) Add to `TestImageSourceSeam`:

```python
    def test_annotations_carry_uri_and_hash_ids(self, tmp_path):
        source = FakeImageSource({1: np.zeros((1, 1, 2, 16, 16), np.uint16)})
        config = _read_only_config()
        config.output.output_directory = tmp_path
        config.spatial_coverage.z_slice_mode = "all"
        config.training.segment_all = True
        pipeline = AnnotationPipeline(config, conn=None, image_source=source)

        pipeline._prepare_processing_units([source.get_image(1)])

        anns = pipeline.config.annotations
        assert {a.image_uri for a in anns} == {"omero:image:1"}
        assert {a.image_id for a in anns} == {1}
        assert sorted(a.annotation_id for a in anns) == sorted(
            make_annotation_id("omero:image:1", 0, z) for z in (0, 1)
        )

    def test_processing_units_are_keyed_by_uri(self, tmp_path):
        config = _read_only_config()
        config.output.output_directory = tmp_path
        config.annotations.append(
            ImageAnnotation(image_id=3, image_name="a", annotation_id="k_0_0")
        )
        pipeline = AnnotationPipeline(config, conn=None, image_source=FakeImageSource({}))
        (key, ann_id, meta, row), = pipeline._convert_annotations_to_processing_units(
            config.annotations
        )
        assert key == "omero:image:3"
        assert (meta["image_uri"], meta["image_id"]) == ("omero:image:3", 3)
```

(import `ImageAnnotation` in `tests/test_pipeline.py` if it is not imported yet.)

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run -e dev pytest tests/test_image_sources.py tests/test_pipeline.py::TestImageSourceSeam -v`
Expected: FAIL (`ImageMeta` has no `image_uri`; non-URI ids)

- [ ] **Step 3: Implement**

`sources/base.py` — in `ImageMeta` replace `image_id: Any` with `image_uri: str  # omero:image:<id> or the OME-Zarr image URL`.

`sources/omero.py` — import `from .uri import omero_uri, parse_omero_uri`; replace `get_image` and the first `metadata` field:

```python
    def get_image(self, image: Any) -> Any:
        """Look up an image by OMERO id or ``omero:image:<id>`` URI."""
        if self.conn is None:
            raise ValueError("OMERO connection is not set.")
        image_id = image if isinstance(image, int) else parse_omero_uri(image)
        if image_id is None:
            raise LookupError(f"{image!r} is not an OMERO image")
        found = self.conn.getObject("Image", image_id)
        if found is None:
            raise LookupError(f"OMERO image {image_id} not found or not accessible")
        return found
```

```python
            image_uri=omero_uri(image.getId()),
```

`annotation_pipeline.py`:

1. Imports: `from ..sources import ImageMeta, ImageSource, OmeroImageSource` and `from ..sources.uri import annotation_id as make_annotation_id, parse_omero_uri`.
2. Add method above `_prepare_processing_units`:

```python
    def _identity_fields(self, image_meta: ImageMeta) -> dict:
        """ImageAnnotation fields that say which image a unit belongs to."""
        uri = image_meta.image_uri
        return {
            "image_uri": uri,
            "image_id": parse_omero_uri(uri),
            "image_name": image_meta.name,
        }
```

3. In `_prepare_processing_units`, replace

```python
            image_meta = self.image_source.metadata(image)
            image_id = image_meta.image_id
            image_name = image_meta.name
```

with

```python
            image_meta = self.image_source.metadata(image)
            identity = self._identity_fields(image_meta)
            image_uri = identity["image_uri"]
```

and in the four `ImageAnnotation(...)` calls replace the `image_id=image_id, image_name=image_name, annotation_id=f"..."` lines with, respectively:

```python
                                **identity,
                                annotation_id=make_annotation_id(image_uri, t, z_start, volumetric=True, patch=i),
```
```python
                            **identity,
                            annotation_id=make_annotation_id(image_uri, t, z_start, volumetric=True),
```
```python
                                    **identity,
                                    annotation_id=make_annotation_id(image_uri, t, z, patch=i),
```
```python
                                **identity,
                                annotation_id=make_annotation_id(image_uri, t, z),
```

4. `_run_micro_sam_annotation`: replace the three lines building `meta_with_image_id` with `metadata.append((annotation_id, meta.copy(), row_idx))` (meta already carries `image_uri`/`image_id`, step 6).
5. `_load_image_data`: `image_id = self.image_source.metadata(image_obj).image_id` → `image_uri = self.image_source.metadata(image_obj).image_uri` and use `{image_uri}` in the warning.
6. `_convert_annotations_to_processing_units`: add `"image_uri": annotation.image_uri,` to `metadata` and append `(annotation.image_uri, annotation.annotation_id, metadata, i)`.
7. `_save_original_image_for_annotation`: replace the lookup block with

```python
        image_key = meta.get("image_uri") or meta.get("image_id")
        if not image_key:
            self._debug_print(f"No image found for annotation {annotation_id}")
            return

        image_obj = self.image_source.get_image(image_key)
```

8. `_save_local_progress`: add `"image_uri": meta.get("image_uri", ""),` after the `"image_id"` entry.

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/ -q -p no:cacheprovider > .superpowers/t3.log 2>&1; tail -5 .superpowers/t3.log`
Expected: all pass. A failing test that builds `ImageMeta(` positionally or reads `.image_id` off an `ImageMeta` gets the URI form; a test asserting an annotation id of the old `{image_id}_{t}_{z}` form *for ids made by the pipeline* gets `make_annotation_id(...)` (tests on legacy-table regeneration such as `tests/test_bia_data.py` keep `101_0_0`).

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/sources tests/test_image_sources.py tests/test_pipeline.py src/omero_annotate_ai/core/annotation_pipeline.py
git commit -m "feat: pipeline keys images by URI and uses hash-based annotation ids"
```

---

### Task 4: Zarr config + ZarrImageSource

**Files:**
- Modify: `pyproject.toml` (extra + pixi feature), `pixi.lock`
- Modify: `src/omero_annotate_ai/core/annotation_config.py` (add `ZarrConfig`, `AnnotationConfig.source`, `.zarr`, validator)
- Create: `src/omero_annotate_ai/sources/zarr.py`
- Modify: `src/omero_annotate_ai/sources/__init__.py`
- Modify: `tests/conftest.py` (NGFF fixtures)
- Test: `tests/test_config.py` (`TestZarrConfig`), `tests/test_image_sources.py` (`TestZarrImageSource`)

**Interfaces:**
- Consumes: `normalize_zarr_uri` (Task 1), `ImageMeta(image_uri=…)` (Task 3), `clamp_region`.
- Produces: `ZarrConfig(uris, level, fields, wells, well_metadata, well_filters, well_filter_mode)`; `AnnotationConfig.source: Literal["omero","zarr"]`, `AnnotationConfig.zarr: ZarrConfig`; `ZarrImage(uri, name, level, data, axes)`; `ZarrImageSource(level: int = 0)` with the `ImageSource` methods; `load_ome_zarr() -> (parse_url, Reader, Multiscales)` in `sources/zarr.py`; conftest fixture `ngff_stores` → dict with keys `img04`, `img05`, `plate` (paths as `str`), `arr04` `(2,64,80)` uint16, `arr05` `(2,2,3,32,40)` uint16.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`:
- under `[project.optional-dependencies]` add `zarr = ["ome-zarr>=0.16"]` and change `all` to `"omero-annotate-ai[notebooks,microsam,zarr]"`;
- add after the mifa feature block:

```toml
# OME-Zarr reading (optional) - ome-zarr-py, imported lazily by sources/zarr.py.
[tool.pixi.feature.zarr.dependencies]
ome-zarr = ">=0.16"
```

- in `[tool.pixi.environments]`: `default = ["zarr"]` and `dev = ["dev","all","mifa","zarr"]`.

Run: `pixi install && pixi install -e dev && pixi list -e dev | grep -E "^(ome-zarr|dask-core|zarr) "`
Expected: `ome-zarr 0.16.0`, `dask-core 2026.8.0` (or newer), `zarr 3.1.5`.
Run: `pixi run -e dev pytest tests/ -q -p no:cacheprovider > .superpowers/t4dep.log 2>&1; tail -3 .superpowers/t4dep.log`
Expected: same pass count as after Task 3 (the dask bump breaks nothing). Commit:

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add optional ome-zarr dependency"
```

- [ ] **Step 2: Write the failing config tests**

Append to `tests/test_config.py`:

```python
@pytest.mark.unit
class TestZarrConfig:
    def test_default_source_is_omero(self):
        config = create_default_config()
        assert config.source == "omero"
        assert config.zarr.uris == []

    def test_zarr_requires_uris(self):
        with pytest.raises(ValueError, match="zarr.uris"):
            AnnotationConfig(name="z", source="zarr")

    def test_filters_require_well_metadata(self):
        with pytest.raises(ValueError, match="well_metadata"):
            AnnotationConfig(name="z", source="zarr",
                             zarr={"uris": ["a.zarr"], "well_filters": {"c": ["x"]}})

    def test_zarr_forces_read_only(self):
        config = AnnotationConfig(name="z", source="zarr", zarr={"uris": ["a.zarr"]})
        assert config.workflow.read_only_mode is True

    def test_yaml_round_trip(self):
        config = AnnotationConfig(name="z", source="zarr",
                                  zarr={"uris": ["a.zarr"], "level": 1, "fields": "all",
                                        "wells": ["B2"]})
        again = AnnotationConfig.from_yaml(config.to_yaml())
        assert again.source == "zarr"
        assert (again.zarr.level, again.zarr.fields, again.zarr.wells) == (1, "all", ["B2"])

    def test_bad_fields_value_rejected(self):
        with pytest.raises(ValueError):
            AnnotationConfig(name="z", source="zarr", zarr={"uris": ["a"], "fields": 3})
```

Run: `pixi run -e dev pytest tests/test_config.py::TestZarrConfig -v`
Expected: FAIL (`source` unknown)

- [ ] **Step 3: Implement the config**

In `annotation_config.py`, after `OMEROConfig`:

```python
class ZarrConfig(BaseModel):
    """OME-Zarr image selection (used when ``source == "zarr"``)."""

    uris: List[str] = Field(
        default_factory=list,
        description="OME-Zarr plates, wells or images (http(s) URL or local path)",
    )
    level: int = Field(default=0, ge=0, description="Multiscale level to read")
    fields: Literal["first", "all"] = Field(
        default="first", description="Fields per well: the first one, or all"
    )
    wells: Optional[List[str]] = Field(default=None, description="Only these wells, e.g. ['B2']")
    well_metadata: Optional[str] = Field(
        default=None, description="CSV with a 'well' column (optional 'plate') and metadata columns"
    )
    well_filters: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Filter wells on well_metadata values (AND logic)"
    )
    well_filter_mode: Literal["include", "exclude"] = Field(default="include")
```

In `AnnotationConfig`, next to `omero:` add:

```python
    source: Literal["omero", "zarr"] = Field(default="omero", description="Where images come from")
    zarr: ZarrConfig = Field(default_factory=ZarrConfig)
```

and the validator (inside `AnnotationConfig`):

```python
    @model_validator(mode="after")
    def _check_zarr_source(self):
        if self.source == "zarr":
            if not self.zarr.uris:
                raise ValueError("source 'zarr' needs at least one entry in zarr.uris")
            # A Zarr run never writes to OMERO; read_only_mode is what makes the
            # pipeline write masks and the table locally instead.
            self.workflow.read_only_mode = True
        if self.zarr.well_filters and not self.zarr.well_metadata:
            raise ValueError("zarr.well_filters needs zarr.well_metadata (a CSV) to filter on")
        return self
```

Run: `pixi run -e dev pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 4: Add NGFF fixtures**

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def ngff_stores(tmp_path_factory):
    """Small OME-Zarr stores: NGFF 0.4 image (c,y,x), 0.5 image (t,c,z,y,x), 0.4 plate.

    Each image has 2 levels. The plate is named "testplate" with wells A/1, A/2, B/3,
    two fields ("0", "1") each, every field holding the 0.4 image data.
    """
    pytest.importorskip("ome_zarr")
    import zarr
    from ome_zarr.format import FormatV04, FormatV05
    from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

    root = tmp_path_factory.mktemp("ngff")
    arr04 = np.arange(2 * 64 * 80, dtype=np.uint16).reshape(2, 64, 80)
    arr05 = np.arange(2 * 2 * 3 * 32 * 40, dtype=np.uint16).reshape(2, 2, 3, 32, 40)

    g04 = zarr.open_group(str(root / "img04.zarr"), mode="w", zarr_format=2)
    write_image(arr04, g04, axes="cyx", fmt=FormatV04(), scale_factors=(2,))
    g05 = zarr.open_group(str(root / "img05.zarr"), mode="w", zarr_format=3)
    write_image(arr05, g05, axes="tczyx", fmt=FormatV05(), scale_factors=(2,))

    plate = zarr.open_group(str(root / "plate.zarr"), mode="w", zarr_format=2)
    well_paths = ["A/1", "A/2", "B/3"]
    write_plate_metadata(plate, ["A", "B"], ["1", "2", "3"], well_paths,
                         name="testplate", fmt=FormatV04())
    for well_path in well_paths:
        row, col = well_path.split("/")
        well = plate.require_group(row).require_group(col)
        write_well_metadata(well, ["0", "1"], fmt=FormatV04())
        for field_path in ("0", "1"):
            write_image(arr04, well.require_group(field_path), axes="cyx",
                        fmt=FormatV04(), scale_factors=(2,))

    return {
        "img04": str(root / "img04.zarr"),
        "img05": str(root / "img05.zarr"),
        "plate": str(root / "plate.zarr"),
        "arr04": arr04,
        "arr05": arr05,
    }
```

(add `import numpy as np` to conftest's imports if missing.)

- [ ] **Step 5: Write the failing source tests**

Append to `tests/test_image_sources.py`:

```python
@pytest.mark.unit
class TestZarrImageSource:
    @pytest.fixture(autouse=True)
    def _stores(self, ngff_stores):
        from omero_annotate_ai.sources.zarr import ZarrImageSource

        self.s = ngff_stores
        self.source = ZarrImageSource()
        self.Source = ZarrImageSource

    def test_satisfies_protocol(self):
        assert isinstance(self.source, ImageSource)

    def test_metadata_cyx_fills_missing_axes_with_one(self):
        meta = self.source.metadata(self.source.get_image(self.s["img04"]))
        assert meta == ImageMeta(image_uri=self.s["img04"], name="img04.zarr", size_t=1,
                                 size_c=2, size_z=1, size_y=64, size_x=80, dtype="uint16")

    def test_metadata_tczyx_v05(self):
        meta = self.source.metadata(self.source.get_image(self.s["img05"]))
        assert (meta.size_t, meta.size_c, meta.size_z, meta.size_y, meta.size_x) == (2, 2, 3, 32, 40)

    def test_read_plane_cyx(self):
        image = self.source.get_image(self.s["img04"])
        np.testing.assert_array_equal(self.source.read_plane(image, t=0, c=1, z=0),
                                      self.s["arr04"][1])

    def test_read_plane_region_and_clip(self):
        image = self.source.get_image(self.s["img04"])
        np.testing.assert_array_equal(
            self.source.read_plane(image, t=0, c=0, z=0, region=(10, 5, 8, 4)),
            self.s["arr04"][0, 5:9, 10:18])
        assert self.source.read_plane(image, t=0, c=0, z=0, region=(70, 60, 20, 20)).shape == (4, 10)

    def test_read_plane_tczyx(self):
        image = self.source.get_image(self.s["img05"])
        np.testing.assert_array_equal(self.source.read_plane(image, t=1, c=1, z=2),
                                      self.s["arr05"][1, 1, 2])

    def test_read_volume_tczyx(self):
        image = self.source.get_image(self.s["img05"])
        vol = self.source.read_volume(image, t=1, c=0, z_start=0, z_end=2)
        np.testing.assert_array_equal(vol, self.s["arr05"][1, 0, 0:3])

    def test_read_volume_without_z_axis(self):
        image = self.source.get_image(self.s["img04"])
        assert self.source.read_volume(image, t=0, c=0, z_start=0, z_end=0).shape == (1, 64, 80)

    def test_level_one(self):
        meta = self.Source(level=1).metadata(self.Source(level=1).get_image(self.s["img04"]))
        assert (meta.size_y, meta.size_x) == (32, 40)

    def test_level_too_high(self):
        with pytest.raises(ValueError, match="2 levels"):
            self.Source(level=5).get_image(self.s["img04"])

    def test_channel_out_of_range(self):
        image = self.source.get_image(self.s["img04"])
        with pytest.raises(IndexError, match="c=5"):
            self.source.read_plane(image, t=0, c=5, z=0)

    def test_missing_store(self, tmp_path):
        with pytest.raises(LookupError):
            self.source.get_image(str(tmp_path / "nope.zarr"))

    def test_plate_is_not_an_image(self):
        with pytest.raises(LookupError, match="not an image"):
            self.source.get_image(self.s["plate"])

    def test_handles_are_cached(self):
        assert self.source.get_image(self.s["img04"]) is self.source.get_image(self.s["img04"] + "/")
```

Run: `pixi run -e dev pytest tests/test_image_sources.py::TestZarrImageSource -v`
Expected: FAIL (`No module named 'omero_annotate_ai.sources.zarr'`)

- [ ] **Step 6: Implement `sources/zarr.py`**

```python
"""ImageSource for OME-Zarr images (NGFF 0.4 and 0.5), read through ome-zarr-py.

Handles are :class:`ZarrImage`: the dask array of one multiscale level in the file's
own axis order, plus the axis names. Pixel coordinates are in that level's grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ImageMeta, Region, clamp_region
from .uri import normalize_zarr_uri


def load_ome_zarr():
    """Import ome-zarr lazily so an OMERO-only install works without it."""
    try:
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Multiscales, Reader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading OME-Zarr needs ome-zarr: pip install 'omero-annotate-ai[zarr]'"
        ) from exc
    # parse_url logs "version mismatch" for every NGFF 0.4 store; it is harmless.
    logging.getLogger("ome_zarr.io").setLevel(logging.ERROR)
    return parse_url, Reader, Multiscales


@dataclass(frozen=True)
class ZarrImage:
    uri: str
    name: str
    level: int
    data: Any  # dask array of the chosen level
    axes: Tuple[str, ...]


def _default_name(uri: str, metadata_name: Optional[str]) -> str:
    name = (metadata_name or "").strip("/")
    if name:
        return name.replace("/", "_")
    return uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _check_index(axis: str, index: int, size: int) -> None:
    if not 0 <= index < size:
        raise IndexError(f"{axis}={index} is outside the image ({axis} size {size})")


class ZarrImageSource:
    """Read OME-Zarr images at one multiscale ``level``."""

    def __init__(self, level: int = 0):
        self.level = level
        self._cache: Dict[str, ZarrImage] = {}

    def get_image(self, uri: str) -> ZarrImage:
        uri = normalize_zarr_uri(uri)
        if uri not in self._cache:
            self._cache[uri] = self._open(uri)
        return self._cache[uri]

    def _open(self, uri: str) -> ZarrImage:
        parse_url, Reader, Multiscales = load_ome_zarr()
        location = parse_url(uri)
        if location is None:
            raise LookupError(f"No OME-Zarr found at {uri}")
        if "multiscales" not in location.root_attrs:
            raise LookupError(
                f"{uri} is not an image (keys: {sorted(location.root_attrs)})"
            )
        node = next(
            n for n in Reader(location)() if any(isinstance(s, Multiscales) for s in n.specs)
        )
        levels = node.data
        if self.level >= len(levels):
            raise ValueError(
                f"Level {self.level} requested but {uri} has {len(levels)} levels"
            )
        axes = tuple(
            a["name"] if isinstance(a, dict) else str(a) for a in node.metadata.get("axes", [])
        )
        data = levels[self.level]
        if len(axes) != data.ndim:
            raise ValueError(f"{uri} has no usable axes metadata (NGFF >= 0.4 needed)")
        return ZarrImage(uri, _default_name(uri, node.metadata.get("name")), self.level, data, axes)

    def metadata(self, image: ZarrImage) -> ImageMeta:
        sizes = dict(zip(image.axes, image.data.shape))
        return ImageMeta(
            image_uri=image.uri,
            name=image.name,
            size_t=sizes.get("t", 1),
            size_c=sizes.get("c", 1),
            size_z=sizes.get("z", 1),
            size_y=sizes["y"],
            size_x=sizes["x"],
            dtype=str(image.data.dtype),
        )

    def _index(self, image: ZarrImage, t: int, c: int, z: Any, region: Optional[Region]):
        sizes = dict(zip(image.axes, image.data.shape))
        _check_index("t", t, sizes.get("t", 1))
        _check_index("c", c, sizes.get("c", 1))
        for z_value in ((z.start, z.stop - 1) if isinstance(z, slice) else (z,)):
            _check_index("z", z_value, sizes.get("z", 1))
        if region is None:
            ys, xs = slice(None), slice(None)
        else:
            x, y, width, height = clamp_region(region, sizes["y"], sizes["x"])
            ys, xs = slice(y, y + height), slice(x, x + width)
        picks = {"t": t, "c": c, "z": z, "y": ys, "x": xs}
        return tuple(picks[axis] for axis in image.axes)

    def read_plane(
        self, image: ZarrImage, *, t: int, c: int, z: int, region: Optional[Region] = None
    ) -> np.ndarray:
        return np.asarray(image.data[self._index(image, t, c, z, region)].compute())

    def read_volume(
        self,
        image: ZarrImage,
        *,
        t: int,
        c: int,
        z_start: int,
        z_end: int,
        region: Optional[Region] = None,
    ) -> np.ndarray:
        volume = np.asarray(
            image.data[self._index(image, t, c, slice(z_start, z_end + 1), region)].compute()
        )
        return volume if "z" in image.axes else volume[np.newaxis]
```

In `sources/__init__.py` add `from .zarr import ZarrImage, ZarrImageSource` and both names to `__all__`.

- [ ] **Step 7: Run tests**

Run: `pixi run -e dev pytest tests/test_image_sources.py tests/test_config.py -v -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/omero_annotate_ai/core/annotation_config.py src/omero_annotate_ai/sources tests/conftest.py tests/test_config.py tests/test_image_sources.py
git commit -m "feat: add zarr config section and ZarrImageSource on ome-zarr"
```

---

### Task 5: Zarr discovery and well selection

**Files:**
- Create: `src/omero_annotate_ai/sources/wells.py`, `src/omero_annotate_ai/sources/zarr_discovery.py`
- Modify: `src/omero_annotate_ai/core/annotation_pipeline.py` (`_check_well_filter` body → one-line call)
- Test: `tests/test_well_filtering.py` (`TestWellHelpers`, `TestZarrDiscovery`)

**Interfaces:**
- Consumes: `ZarrConfig` (Task 4), `load_ome_zarr` (Task 4), `normalize_zarr_uri` (Task 1), fixture `ngff_stores`.
- Produces: `normalize_well(name: str) -> str`, `match_well_filter(kv: Dict[str, str], filters: Dict[str, List[str]]) -> bool` in `sources/wells.py`; `ZarrImageRef(uri, plate="", well="", field="", well_metadata={})` with property `name -> str`; `discover_zarr_images(cfg: ZarrConfig, output_dir: Optional[Path] = None) -> List[ZarrImageRef]` in `sources/zarr_discovery.py` (writes `output_dir/well_matching.csv` when any well was considered).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_well_filtering.py`:

```python
import warnings
from pathlib import Path

import pandas as pd

from omero_annotate_ai.core.annotation_config import ZarrConfig
from omero_annotate_ai.sources.wells import match_well_filter, normalize_well
from omero_annotate_ai.sources.zarr_discovery import discover_zarr_images


@pytest.mark.unit
class TestWellHelpers:
    @pytest.mark.parametrize("raw", ["B2", "b02", "B/2", "B_02", " B 2 "])
    def test_normalize_well(self, raw):
        assert normalize_well(raw) == "B2"

    def test_normalize_rejects_garbage(self):
        with pytest.raises(ValueError, match="well name"):
            normalize_well("field0")

    def test_match_well_filter_and_logic(self):
        kv = {"compound": "DMSO", "dose": "0"}
        assert match_well_filter(kv, {"compound": ["DMSO", "X"]})
        assert not match_well_filter(kv, {"compound": ["DMSO"], "dose": ["1"]})
        assert not match_well_filter(kv, {"cellline": ["U2OS"]})


@pytest.mark.unit
class TestZarrDiscovery:
    @pytest.fixture(autouse=True)
    def _stores(self, ngff_stores, tmp_path):
        self.s = ngff_stores
        self.out = tmp_path / "out"

    def _csv(self, tmp_path, rows):
        path = tmp_path / "layout.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_single_image(self):
        refs = discover_zarr_images(ZarrConfig(uris=[self.s["img04"]]))
        assert [(r.uri, r.well, r.name) for r in refs] == [(self.s["img04"], "", "")]

    def test_plate_first_field_in_row_column_order(self):
        refs = discover_zarr_images(ZarrConfig(uris=[self.s["plate"]]))
        plate = self.s["plate"]
        assert [r.uri for r in refs] == [f"{plate}/A/1/0", f"{plate}/A/2/0", f"{plate}/B/3/0"]
        assert refs[0].name == "testplate_A1_0"

    def test_plate_all_fields(self):
        assert len(discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], fields="all"))) == 6

    def test_well_uri(self):
        (ref,) = discover_zarr_images(ZarrConfig(uris=[self.s["plate"] + "/A/2"]))
        assert (ref.plate, ref.well, ref.field) == ("testplate", "A2", "0")

    def test_wells_list_normalized_and_unknown_ignored(self):
        refs = discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], wells=["a02", "H12"]))
        assert [r.well for r in refs] == ["A2"]

    def test_csv_filter_include_records_the_match(self, tmp_path):
        csv = self._csv(tmp_path, [{"well": "A1", "compound": "DMSO"},
                                   {"well": "A02", "compound": "X"},
                                   {"well": "B3", "compound": "DMSO"}])
        cfg = ZarrConfig(uris=[self.s["plate"]], well_metadata=csv,
                         well_filters={"compound": ["DMSO"]})
        refs = discover_zarr_images(cfg, output_dir=self.out)
        assert [r.well for r in refs] == ["A1", "B3"]
        assert refs[0].well_metadata == {"compound": "DMSO"}
        table = pd.read_csv(self.out / "well_matching.csv")
        assert list(table["well"]) == ["A1", "A2", "B3"]
        assert list(table["matched"]) == [True, True, True]
        assert list(table["included"]) == [True, False, True]
        assert list(table["compound"]) == ["DMSO", "X", "DMSO"]

    def test_csv_filter_exclude(self, tmp_path):
        csv = self._csv(tmp_path, [{"well": w, "compound": c}
                                   for w, c in [("A1", "DMSO"), ("A2", "X"), ("B3", "DMSO")]])
        cfg = ZarrConfig(uris=[self.s["plate"]], well_metadata=csv,
                         well_filters={"compound": ["DMSO"]}, well_filter_mode="exclude")
        assert [r.well for r in discover_zarr_images(cfg)] == ["A2"]

    def test_csv_with_plate_column(self, tmp_path):
        csv = self._csv(tmp_path, [{"plate": "testplate", "well": "A1", "compound": "DMSO"},
                                   {"plate": "other", "well": "A2", "compound": "DMSO"}])
        cfg = ZarrConfig(uris=[self.s["plate"]], well_metadata=csv,
                         well_filters={"compound": ["DMSO"]})
        with pytest.warns(UserWarning):
            refs = discover_zarr_images(cfg)
        assert [r.well for r in refs] == ["A1"]

    def test_mismatch_warnings(self, tmp_path):
        csv = self._csv(tmp_path, [{"well": "A1", "c": "1"}, {"well": "A2", "c": "1"},
                                   {"well": "H12", "c": "1"}])
        with pytest.warns(UserWarning) as record:
            discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], well_metadata=csv))
        messages = " ".join(str(w.message) for w in record)
        assert "B3" in messages and "H12" in messages

    def test_csv_without_well_column(self, tmp_path):
        csv = self._csv(tmp_path, [{"position": "A1"}])
        with pytest.raises(ValueError, match="'well' column"):
            discover_zarr_images(ZarrConfig(uris=[self.s["plate"]], well_metadata=csv))

    def test_not_ngff(self, tmp_path):
        (tmp_path / "plain").mkdir()
        with pytest.raises(ValueError, match="OME-Zarr"):
            discover_zarr_images(ZarrConfig(uris=[str(tmp_path / "plain")]))
```

Run: `pixi run -e dev pytest tests/test_well_filtering.py -k "WellHelpers or ZarrDiscovery" -v`
Expected: collection error (`No module named 'omero_annotate_ai.sources.wells'`)

- [ ] **Step 2: Implement `sources/wells.py`**

```python
"""Well names and well filters, shared by the OMERO and OME-Zarr paths."""

from __future__ import annotations

import re
from typing import Dict, List

_WELL = re.compile(r"^\s*([A-Za-z]+)\s*[/_\- ]?\s*0*(\d+)\s*$")


def normalize_well(name: str) -> str:
    """Canonical well name: ``B2`` for ``B2``, ``b02``, ``B/2`` or ``B_02``."""
    match = _WELL.match(str(name))
    if not match:
        raise ValueError(f"Not a well name: {name!r} (expected e.g. B2, B02 or B/2)")
    return f"{match.group(1).upper()}{int(match.group(2))}"


def match_well_filter(kv: Dict[str, str], filters: Dict[str, List[str]]) -> bool:
    """True when ``kv`` meets ALL filter conditions (key present, value accepted)."""
    for key, accepted in filters.items():
        if key not in kv or kv[key] not in accepted:
            return False
    return True
```

In `annotation_pipeline.py`, import `from ..sources.wells import match_well_filter` and replace the body of `_check_well_filter` (keep its docstring) with `return match_well_filter(well_kv_pairs, filters)`.

- [ ] **Step 3: Implement `sources/zarr_discovery.py`**

```python
"""Expand OME-Zarr plate / well / image URIs into the images to annotate.

Reads only NGFF metadata, never pixels. Well selection (a ``wells`` list, then a
metadata CSV with optional filters) happens before a well's fields are listed. OME-Zarr
stores no link between such a CSV and its wells, so the match we make is written to
``well_matching.csv``.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from .uri import normalize_zarr_uri
from .wells import match_well_filter, normalize_well
from .zarr import load_ome_zarr

WellKey = Union[str, Tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class ZarrImageRef:
    uri: str
    plate: str = ""
    well: str = ""
    field: str = ""
    well_metadata: Dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        """``{plate}_{well}_{field}`` for plate images, else empty."""
        if not self.well:
            return ""
        return "_".join(part for part in (self.plate, self.well, self.field) if part)


def _attrs(uri: str) -> Optional[dict]:
    parse_url, _, _ = load_ome_zarr()
    location = parse_url(uri)
    return None if location is None else location.root_attrs


def _read_well_metadata(path: str) -> Tuple[Dict[WellKey, Dict[str, str]], bool]:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "well" not in df.columns:
        raise ValueError(f"{path} needs a 'well' column (found: {list(df.columns)})")
    by_plate = "plate" in df.columns
    rows: Dict[WellKey, Dict[str, str]] = {}
    for record in df.to_dict("records"):
        well = normalize_well(record.pop("well"))
        key = (record.pop("plate"), well) if by_plate else well
        rows[key] = {k: str(v) for k, v in record.items()}
    return rows, by_plate


class _Selector:
    """Decides per well whether it is used, and records every decision."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.wanted: Optional[Set[str]] = (
            {normalize_well(w) for w in cfg.wells} if cfg.wells else None
        )
        self.rows, self.by_plate = (
            _read_well_metadata(cfg.well_metadata) if cfg.well_metadata else ({}, False)
        )
        self.seen: Set[WellKey] = set()
        self.records: List[dict] = []

    def select(self, plate: str, well: str, well_uri: str) -> Optional[Dict[str, str]]:
        """The well's metadata if it is selected, else None."""
        key: WellKey = (plate, well) if self.by_plate else well
        self.seen.add(key)
        if self.wanted is not None and well not in self.wanted:
            return None
        metadata = self.rows.get(key)
        matched = metadata is not None
        kv = metadata or {}
        if self.cfg.well_filters:
            hit = matched and match_well_filter(kv, self.cfg.well_filters)
            included = hit if self.cfg.well_filter_mode == "include" else not hit
        else:
            included = True
        self.records.append({"plate": plate, "well": well, "well_uri": well_uri,
                             "matched": matched, "included": included, **kv})
        return kv if included else None

    def report(self, output_dir: Optional[Path]) -> None:
        if self.cfg.well_metadata:
            no_row = [r["well"] for r in self.records if not r["matched"]]
            if no_row:
                warnings.warn(f"{len(no_row)} wells have no row in {self.cfg.well_metadata}: "
                              f"{', '.join(no_row[:5])}")
            no_well = [str(k) for k in self.rows if k not in self.seen]
            if no_well:
                warnings.warn(f"{len(no_well)} rows in {self.cfg.well_metadata} match no "
                              f"well: {', '.join(no_well[:5])}")
        if output_dir is not None and self.records:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.records).to_csv(output_dir / "well_matching.csv", index=False)


def _field_refs(selector: _Selector, well_uri: str, plate: str, well: str,
                well_attrs: Optional[dict] = None) -> List[ZarrImageRef]:
    metadata = selector.select(plate, well, well_uri)
    if metadata is None:
        return []
    if well_attrs is None:
        well_attrs = _attrs(well_uri) or {}
    fields = [image["path"] for image in well_attrs.get("well", {}).get("images", [])]
    if selector.cfg.fields == "first":
        fields = fields[:1]
    return [ZarrImageRef(f"{well_uri}/{f}", plate, well, f, dict(metadata)) for f in fields]


def discover_zarr_images(cfg, output_dir: Optional[Path] = None) -> List[ZarrImageRef]:
    """Images to annotate for a :class:`ZarrConfig`, in plate row/column order."""
    selector = _Selector(cfg)
    refs: List[ZarrImageRef] = []
    for raw in cfg.uris:
        uri = normalize_zarr_uri(raw)
        attrs = _attrs(uri)
        if attrs and "plate" in attrs:
            plate = attrs["plate"]
            name = str(plate.get("name", ""))
            wells = sorted(plate.get("wells", []),
                           key=lambda w: (w.get("rowIndex", 0), w.get("columnIndex", 0)))
            for well in wells:
                refs += _field_refs(selector, f"{uri}/{well['path']}", name,
                                    normalize_well(well["path"]))
        elif attrs and "well" in attrs:
            parent, row, col = uri.rsplit("/", 2)
            parent_attrs = _attrs(parent) or {}
            name = str(parent_attrs.get("plate", {}).get("name", ""))
            refs += _field_refs(selector, uri, name, normalize_well(f"{row}/{col}"), attrs)
        elif attrs and "multiscales" in attrs:
            refs.append(ZarrImageRef(uri))
        else:
            raise ValueError(
                f"{uri} is not an OME-Zarr plate, well or image "
                f"(found keys: {sorted(attrs or {})})"
            )
    selector.report(output_dir)
    return refs
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_well_filtering.py -v -q`
Expected: PASS (including the existing OMERO well-filter tests)

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/sources/wells.py src/omero_annotate_ai/sources/zarr_discovery.py src/omero_annotate_ai/core/annotation_pipeline.py tests/test_well_filtering.py
git commit -m "feat: discover OME-Zarr images with well selection from a list or CSV"
```

---

### Task 6: Pipeline runs on a Zarr config

**Files:**
- Modify: `src/omero_annotate_ai/sources/__init__.py` (`make_image_source`)
- Modify: `src/omero_annotate_ai/core/annotation_pipeline.py` (`__init__`, `_identity_fields`, new `get_image_uris`, `initialize_workflow` L1251-1257, `_run_annotation_batches` L1429-1432, `run_cellpose_preparation` L1526-1527)
- Test: `tests/test_pipeline.py` (`TestZarrWorkflow`)

**Interfaces:**
- Consumes: `ZarrImageSource` (Task 4), `discover_zarr_images`, `ZarrImageRef` (Task 5), `omero_uri` (Task 1), `_identity_fields` (Task 3).
- Produces: `make_image_source(config, conn=None) -> ImageSource`; `AnnotationPipeline.get_image_uris() -> List[str]`; read-only runs write `output/tracking_table.csv`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py` (add `import tifffile` and `from omero_annotate_ai.core.annotation_config import AnnotationConfig` if missing):

```python
def _zarr_config(tmp_path, uris, **zarr):
    config = AnnotationConfig(name="zarr_run", source="zarr", zarr={"uris": uris, **zarr})
    config.output.output_directory = tmp_path / "out"
    config.training.segment_all = True
    config.spatial_coverage.use_patches = False
    return config


@pytest.mark.integration
class TestZarrWorkflow:
    def test_default_source_for_zarr_config(self, tmp_path, ngff_stores):
        from omero_annotate_ai.sources import ZarrImageSource

        pipeline = AnnotationPipeline(_zarr_config(tmp_path, [ngff_stores["plate"]], level=1))
        assert isinstance(pipeline.image_source, ZarrImageSource)
        assert pipeline.image_source.level == 1

    def test_no_images_error_names_the_uris(self, tmp_path, ngff_stores):
        config = _zarr_config(tmp_path, [ngff_stores["plate"]], wells=["H12"])
        with pytest.raises(ValueError, match="plate.zarr"):
            AnnotationPipeline(config).initialize_workflow()

    def test_end_to_end_custom_annotation(self, tmp_path, ngff_stores):
        config = _zarr_config(tmp_path, [ngff_stores["plate"]], level=1, wells=["A1", "B3"])
        pipeline = AnnotationPipeline(config)
        seg_dir = tmp_path / "seg"

        def fake_annotate(batch):
            seg_dir.mkdir(exist_ok=True)
            metadata = []
            for i, (image_obj, ann_id, meta, row_idx) in enumerate(batch):
                plane = pipeline._load_image_data(image_obj, meta)
                tifffile.imwrite(seg_dir / f"seg_{i:05d}.tif",
                                 (plane > plane.mean()).astype(np.uint32))
                metadata.append((ann_id, meta, row_idx))
            return {"metadata": metadata, "annotations_path": seg_dir}

        pipeline.run_custom_annotation(fake_annotate)

        anns = pipeline.config.annotations
        assert [(a.well, a.field, a.level, a.image_id) for a in anns] == [
            ("A1", "0", 1, None), ("B3", "0", 1, None)]
        assert anns[0].image_name == "testplate_A1_0"
        assert anns[0].annotation_id == make_annotation_id(anns[0].image_uri, 0, 0)
        out = tmp_path / "out"
        for ann in anns:
            mask = tifffile.imread(out / "annotation_output" / f"{ann.annotation_id}_mask.tif")
            assert mask.shape == (32, 40)  # level 1 grid
            assert (out / "annotation_input" / f"{ann.annotation_id}.tif").exists()
        table = pd.read_csv(out / "tracking_table.csv")
        assert list(table["image_uri"]) == [a.image_uri for a in anns]
        assert table["processed"].all()
        assert (out / "well_matching.csv").exists()
```

(If `pd` is not imported in `tests/test_pipeline.py`, add `import pandas as pd`.)

Run: `pixi run -e dev pytest tests/test_pipeline.py::TestZarrWorkflow -v`
Expected: FAIL (default source is OMERO / `ValueError: OMERO connection is required to read images`)

- [ ] **Step 2: Implement**

`sources/__init__.py` — add:

```python
def make_image_source(config, conn=None) -> ImageSource:
    """The ImageSource a config asks for: OME-Zarr for ``source: zarr``, else OMERO."""
    if getattr(config, "source", "omero") == "zarr":
        return ZarrImageSource(level=config.zarr.level)
    return OmeroImageSource(conn)
```

and `"make_image_source"` to `__all__`.

`annotation_pipeline.py`:

1. Imports: `from ..sources import ImageMeta, ImageSource, OmeroImageSource, make_image_source`, `from ..sources.uri import annotation_id as make_annotation_id, omero_uri, parse_omero_uri`, `from ..sources.zarr_discovery import ZarrImageRef, discover_zarr_images`.
2. `__init__`: `self.image_source = image_source if image_source is not None else make_image_source(config, conn)` and after it `self._zarr_refs: Dict[str, ZarrImageRef] = {}`. Update the docstring line to "Defaults to the source the config asks for (``make_image_source``)."
3. Replace `_identity_fields` with:

```python
    def _identity_fields(self, image_meta: ImageMeta) -> dict:
        """ImageAnnotation fields that say which image a unit belongs to."""
        uri = image_meta.image_uri
        fields = {
            "image_uri": uri,
            "image_id": parse_omero_uri(uri),
            "image_name": image_meta.name,
            "level": self.config.zarr.level if self.config.source == "zarr" else 0,
        }
        ref = self._zarr_refs.get(uri)
        if ref is not None:
            fields.update(
                image_name=ref.name or image_meta.name,
                well=ref.well,
                field=ref.field,
                well_metadata=dict(ref.well_metadata),
            )
        return fields
```

4. Add next to `get_image_ids_from_container`:

```python
    def get_image_uris(self) -> List[str]:
        """URIs of the images the config selects (OMERO container or OME-Zarr URIs)."""
        if self.config.source == "zarr":
            refs = discover_zarr_images(
                self.config.zarr, output_dir=Path(self.config.output.output_directory)
            )
            self._zarr_refs = {ref.uri: ref for ref in refs}
            print(f"Found {len(refs)} OME-Zarr images")
            return list(self._zarr_refs)
        return [omero_uri(i) for i in self.get_image_ids_from_container()]
```

5. `initialize_workflow`: replace `image_ids = self.get_image_ids_from_container()` with `image_ids = self.get_image_uris()`, and the "No images found" `raise` with:

```python
                where = (
                    f"OME-Zarr URIs {self.config.zarr.uris}"
                    if self.config.source == "zarr"
                    else f"{self.config.omero.container_type} {self.config.omero.container_id}"
                )
                raise ValueError(f"No images found in {where}")
```

6. `_run_annotation_batches`: change

```python
                if not self.config.workflow.read_only_mode:
                    self._debug_print("   Updating OMERO table...")
                    self._replace_omero_table_from_config()
                    self._debug_print(f"   OMERO table updated (ID: {self.table_id})")
```

to add an `else:` branch:

```python
                else:
                    self._save_local_progress_from_annotations(self.config.get_processed())
```

7. `run_cellpose_preparation`: same — after `if not self.config.workflow.read_only_mode: self._replace_omero_table_from_config()` add `else: self._save_local_progress_from_annotations(self.config.get_processed())`.

- [ ] **Step 3: Run tests**

Run: `pixi run -e dev pytest tests/ -q -p no:cacheprovider > .superpowers/t6.log 2>&1; tail -5 .superpowers/t6.log`
Expected: all pass. If an existing read-only test now finds an unexpected `tracking_table.csv`, that is the intended new behaviour; adjust the assertion.

- [ ] **Step 4: Commit**

```bash
git add src/omero_annotate_ai/sources/__init__.py src/omero_annotate_ai/core/annotation_pipeline.py tests/test_pipeline.py
git commit -m "feat: run the annotation pipeline on an OME-Zarr config"
```

---

### Task 7: Zarr YAML notebook, example config, remote check

**Files:**
- Create: `notebooks/marimo/configs/zarr-s-biad855-example.yaml`
- Create: `notebooks/marimo/zarr-annotate-ai-from-yaml.py`
- Modify: `CLAUDE.md` (notebook list: one line)
- Test: `tests/test_config.py` (`TestZarrConfig.test_example_yaml_loads`), `tests/test_image_sources.py` (`TestZarrRemote`, slow)

**Interfaces:**
- Consumes: `load_config_from_yaml`, `create_pipeline`, `AnnotationPipeline.get_image_uris`, `pipeline._zarr_refs`, `discover_zarr_images`, `ZarrImageSource`.
- Produces: user-facing notebook + example config.

- [ ] **Step 1: Write the failing tests**

Add to `TestZarrConfig` in `tests/test_config.py`:

```python
    def test_example_yaml_loads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config_from_yaml(
            str(root / "notebooks/marimo/configs/zarr-s-biad855-example.yaml"))
        assert config.source == "zarr"
        assert config.zarr.level == 1
        assert config.workflow.read_only_mode is True
```

(import `load_config_from_yaml` and `Path` in `tests/test_config.py` if missing.)

Append to `tests/test_image_sources.py`:

```python
S_BIAD855 = ("https://livingobjects.ebi.ac.uk/bioimaging-integrator-data/S-BIAD855/"
             "781ac3d7-673f-47be-a4d2-3fdf3f477047/781ac3d7-673f-47be-a4d2-3fdf3f477047.zarr")


@pytest.mark.slow
class TestZarrRemote:
    def test_discover_and_read_patch(self, tmp_path):
        pytest.importorskip("ome_zarr")
        requests = pytest.importorskip("requests")
        try:
            requests.head(S_BIAD855 + "/.zattrs", timeout=10).raise_for_status()
        except Exception as exc:
            pytest.skip(f"S-BIAD855 not reachable: {exc}")
        from omero_annotate_ai.core.annotation_config import ZarrConfig
        from omero_annotate_ai.sources.zarr import ZarrImageSource
        from omero_annotate_ai.sources.zarr_discovery import discover_zarr_images

        (ref,) = discover_zarr_images(ZarrConfig(uris=[S_BIAD855], wells=["B2"]))
        assert (ref.plate, ref.well, ref.field) == ("20585", "B2", "0")
        source = ZarrImageSource(level=1)
        image = source.get_image(ref.uri)
        patch = source.read_plane(image, t=0, c=2, z=0, region=(0, 0, 64, 64))
        assert patch.shape == (64, 64) and patch.dtype == np.uint16
```

Run: `pixi run -e dev pytest tests/test_config.py::TestZarrConfig::test_example_yaml_loads tests/test_image_sources.py::TestZarrRemote -v`
Expected: `test_example_yaml_loads` FAIL (file not found); `TestZarrRemote` PASS or SKIP (no network) — it exercises code from Tasks 4-5.

- [ ] **Step 2: Create the example config**

`notebooks/marimo/configs/zarr-s-biad855-example.yaml`:

```yaml
# Annotate a public Cell Painting plate (BioImage Archive S-BIAD855) straight from OME-Zarr.
# Load it in notebooks/marimo/zarr-annotate-ai-from-yaml.py.
name: s-biad855-zarr-demo
source: zarr
zarr:
  uris:
    - https://livingobjects.ebi.ac.uk/bioimaging-integrator-data/S-BIAD855/781ac3d7-673f-47be-a4d2-3fdf3f477047/781ac3d7-673f-47be-a4d2-3fdf3f477047.zarr
  level: 1          # half resolution (260 x 348); masks and coordinates use this grid
  fields: first
  wells: [B2, C5, D10]
spatial_coverage:
  channels: [2]     # Hoechst
  timepoints: [0]
  z_slices: [0]
training:
  segment_all: false
  train_n: 2
  validate_n: 1
  test_n: 0
ai_model:
  framework: micro_sam
  pretrained_from: vit_b_lm
output:
  output_directory: ./zarr_annotations
```

Run: `pixi run -e dev pytest tests/test_config.py::TestZarrConfig -v`
Expected: PASS. (If a field name is rejected, fix the YAML to the model's field name — the test is the check.)

- [ ] **Step 3: Create the notebook**

`notebooks/marimo/zarr-annotate-ai-from-yaml.py`:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "omero-annotate-ai[zarr]",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from omero_annotate_ai import create_pipeline, load_config_from_yaml
    return create_pipeline, load_config_from_yaml, mo, pl


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
def _(config, create_pipeline, discover_button, mo, pl):
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
        mo.ui.table(pl.DataFrame(_rows)) if _rows else mo.md("No images selected."),
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
```

Run: `pixi run -e dev marimo check --fix notebooks/marimo/zarr-annotate-ai-from-yaml.py && pixi run -e dev python -c "import ast,sys; ast.parse(open('notebooks/marimo/zarr-annotate-ai-from-yaml.py').read())"`
Expected: no errors reported (polars must be importable: if `import polars` fails in the dev env, run `pixi add --feature zarr polars` and re-run).

- [ ] **Step 4: Note the notebook in CLAUDE.md**

In `CLAUDE.md` under the `marimo/` notebook list add:

```markdown
  - `zarr-annotate-ai-from-yaml.py` — Run from a YAML config on OME-Zarr (no OMERO; `source: zarr`)
```

- [ ] **Step 5: Run the full suite**

Run: `pixi run -e dev pytest tests/ -q -p no:cacheprovider > .superpowers/t7.log 2>&1; tail -5 .superpowers/t7.log`
Expected: all pass (slow remote test passes or skips).

- [ ] **Step 6: Commit**

```bash
git add notebooks/marimo/configs/zarr-s-biad855-example.yaml notebooks/marimo/zarr-annotate-ai-from-yaml.py CLAUDE.md tests/test_config.py tests/test_image_sources.py pyproject.toml pixi.lock
git commit -m "feat: marimo notebook and example config for OME-Zarr runs"
```

---

## Manual check (user, after the plan)

Open `notebooks/marimo/zarr-annotate-ai-from-yaml.py` in the default env (`pixi run marimo edit notebooks/marimo/zarr-annotate-ai-from-yaml.py`), load the example config, discover (3 images: B2, C5, D10), run micro-SAM, annotate, close napari, and check `zarr_annotations/annotation_output/`, `tracking_table.csv` and `well_matching.csv`.
