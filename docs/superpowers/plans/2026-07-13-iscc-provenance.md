# ISCC Content Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record ISO 24138 pixel-content codes for the source images and annotation masks of an annotation dataset in `config.yaml`, so a recipient can verify offline which images the annotations came from and that the masks are unmodified.

**Architecture:** All ISCC logic lives in one new unit, `processing/provenance.py`, which wraps `iscc-bio` behind guarded imports. Two new `Optional[str]` fields on `ImageAnnotation` (`source_iscc`, `label_iscc`) carry the codes, so they serialize into `config.yaml` and mirror into the OMERO tracking table for free. A `stamp_config()` fills them (during a run, or retroactively on an old config), and `verify_config()` re-checks them either against OMERO or fully offline against a directory of files.

**Tech Stack:** Python ≥3.10, pydantic v2, pandas, ezomero, `iscc-bio` (optional extra), pytest.

## Global Constraints

- `iscc-bio` is an **optional** dependency, pinned `>=0.1,<0.2` — it is PoC software that declares breaking changes may land in any release.
- `iscc-bio` is imported **lazily, inside functions**, never at module top level. The package must import and run with it absent.
- Provenance is **best-effort and must never fail an annotation run.** Any compute error yields `None` for that code plus a warning; the run continues.
- Default is **off**: `iscc_mode: Literal["off", "on"] = "off"`. No compute, no import attempt, zero behaviour change for existing users.
- Codes are computed on **raw source pixels only** — never on the exported 8-bit training tiles. Export rescales intensities (`img * 255/max`) and restacks channels, so tile codes would not match the source.
- Optional `Optional[str]` fields round-trip through the OMERO table via the `"None"` sentinel-string pattern already used for timestamps (`annotation_config.py:996-1003`).
- Tests run with `pixi run -e dev pytest tests/ -v`.
- `pixi.lock` must be refreshed (`pixi install`) and committed before any PR, or CI fails.

**iscc-bio public API** (verified against the source; use exactly this):

```python
from iscc_bio.api import biocode

biocode(source="image.ome.tiff")        # local file
biocode(conn=blitz_gateway, iid=123)    # OMERO image

# returns: List[Dict[str, Any]], one entry per scene
# [{"iscc_code": "ISCC:...", "units": ["ISCC:...", "ISCC:..."]}]
```

---

### Task 1: Provenance unit — code computation

**Files:**
- Create: `src/omero_annotate_ai/processing/provenance.py`
- Create: `tests/test_provenance.py`
- Modify: `pyproject.toml:49-67` (add the `provenance` optional-dependency group)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `iscc_available() -> bool`
  - `compute_file_iscc(path: str | Path) -> Optional[str]`
  - `compute_image_iscc(conn, image_id: int) -> Optional[str]`
  - `compute_label_iscc(conn, label_id: int) -> Optional[str]`

The `provenance` extra is added here because this task's module is the thing that needs the dependency declared.

- [ ] **Step 1: Add the optional-dependency group**

In `pyproject.toml`, inside `[project.optional-dependencies]`, after the `microsam` group (ends line 60), add:

```toml
provenance = [
    # ISO 24138 pixel-content codes for dataset provenance.
    # Pinned tightly: iscc-bio is a proof of concept and declares that breaking
    # changes may be released at any time.
    "iscc-bio>=0.1,<0.2",
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_provenance.py`. `iscc-bio` is an optional dep that will not be installed in CI, so every test fakes it by injecting a stub module into `sys.modules`.

```python
"""Tests for ISCC content provenance (processing/provenance.py)."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omero_annotate_ai.processing import provenance


@pytest.fixture
def fake_iscc(monkeypatch):
    """Inject a stub iscc_bio module and hand back its biocode mock.

    provenance.py imports iscc_bio lazily inside functions, so patching
    sys.modules is enough - no real install needed.
    """
    biocode = MagicMock(
        return_value=[{"iscc_code": "ISCC:AAA", "units": ["ISCC:D", "ISCC:I"]}]
    )
    api_mod = types.ModuleType("iscc_bio.api")
    api_mod.biocode = biocode
    root_mod = types.ModuleType("iscc_bio")
    root_mod.api = api_mod
    monkeypatch.setitem(sys.modules, "iscc_bio", root_mod)
    monkeypatch.setitem(sys.modules, "iscc_bio.api", api_mod)
    return biocode


@pytest.fixture
def no_iscc(monkeypatch):
    """Make `import iscc_bio` raise ImportError, simulating it not installed."""
    monkeypatch.setitem(sys.modules, "iscc_bio", None)


@pytest.mark.unit
class TestIsccAvailability:
    """Guarded-import behaviour."""

    def test_available_when_installed(self, fake_iscc):
        assert provenance.iscc_available() is True

    def test_unavailable_when_missing(self, no_iscc):
        assert provenance.iscc_available() is False


@pytest.mark.unit
class TestComputeFileIscc:
    """compute_file_iscc()."""

    def test_returns_code_for_file(self, fake_iscc, tmp_path):
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) == "ISCC:AAA"
        fake_iscc.assert_called_once_with(source=str(img))

    def test_returns_none_when_iscc_missing(self, no_iscc, tmp_path):
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) is None

    def test_returns_none_on_compute_error(self, fake_iscc, tmp_path):
        fake_iscc.side_effect = RuntimeError("unreadable")
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) is None

    def test_returns_none_on_empty_result(self, fake_iscc, tmp_path):
        fake_iscc.return_value = []
        img = tmp_path / "image.tif"
        img.write_bytes(b"fake")

        assert provenance.compute_file_iscc(img) is None


@pytest.mark.unit
class TestComputeImageIscc:
    """compute_image_iscc() - the canonical source-side code, from raw OMERO pixels."""

    def test_returns_code_for_omero_image(self, fake_iscc):
        conn = MagicMock()

        assert provenance.compute_image_iscc(conn, 123) == "ISCC:AAA"
        fake_iscc.assert_called_once_with(conn=conn, iid=123)

    def test_returns_none_when_iscc_missing(self, no_iscc):
        assert provenance.compute_image_iscc(MagicMock(), 123) is None

    def test_returns_none_on_omero_error(self, fake_iscc):
        fake_iscc.side_effect = RuntimeError("connection lost")

        assert provenance.compute_image_iscc(MagicMock(), 123) is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'omero_annotate_ai.processing.provenance'`

- [ ] **Step 4: Write the implementation**

Create `src/omero_annotate_ai/processing/provenance.py`:

```python
"""ISCC content provenance for published annotation datasets.

Computes ISO 24138:2024 content codes via iscc-bio's IMAGEWALK traversal.

IMAGEWALK is an *exact* code: it packs raw dtype bytes with no bit-depth or
intensity canonicalization, aiming for bit-for-bit identity with OMERO's own
pixel representation. It therefore survives a change of container format
(OME-TIFF / OME-Zarr / OMERO) but NOT a change of pixel values.

Consequence: codes are computed on raw source pixels only. The exported 8-bit
training tiles are rescaled (img * 255/max) and channel-restacked, so their
codes would not match the source, and they are never coded here.

iscc-bio is an optional dependency and is imported lazily. Everything in this
module degrades to None with a warning when it is absent.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_MISSING_MSG = (
    "iscc-bio is not installed; skipping ISCC provenance. "
    "Install it with: pip install 'omero-annotate-ai[provenance]'"
)


def _get_biocode():
    """Return iscc_bio.api.biocode, or None if iscc-bio is not installed."""
    try:
        from iscc_bio.api import biocode
    except ImportError:
        return None
    return biocode


def iscc_available() -> bool:
    """Whether iscc-bio is importable and provenance can be computed."""
    return _get_biocode() is not None


def _first_code(results: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Take the ISCC of the first scene.

    biocode() returns one entry per scene. OMERO images and the exported masks
    are single-scene, so the first entry is the one we want.
    """
    if not results:
        return None
    return results[0].get("iscc_code")


def compute_file_iscc(path: Union[str, Path]) -> Optional[str]:
    """Content code for a local image file. Returns None on any failure."""
    biocode = _get_biocode()
    if biocode is None:
        logger.warning(_MISSING_MSG)
        return None

    try:
        return _first_code(biocode(source=str(path)))
    except Exception as exc:  # provenance is best-effort, never fatal
        logger.warning(f"Could not compute ISCC for file {path}: {exc}")
        return None


def compute_image_iscc(conn, image_id: int) -> Optional[str]:
    """Content code for an OMERO image, from its raw pixels.

    This is the canonical source-side code: iscc-bio reads OMERO's pixel data
    directly, so it is directly comparable to any faithful copy of the image.
    """
    biocode = _get_biocode()
    if biocode is None:
        logger.warning(_MISSING_MSG)
        return None

    try:
        return _first_code(biocode(conn=conn, iid=int(image_id)))
    except Exception as exc:
        logger.warning(f"Could not compute ISCC for OMERO image {image_id}: {exc}")
        return None


def compute_label_iscc(conn, label_id: int) -> Optional[str]:
    """Content code for an annotation mask.

    Masks are stored as OMERO file annotations, so this downloads the mask to a
    temporary directory and codes it as a file.
    """
    biocode = _get_biocode()
    if biocode is None:
        logger.warning(_MISSING_MSG)
        return None

    import ezomero

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = ezomero.get_file_annotation(
                conn, int(label_id), folder_path=tmp_dir
            )
            if mask_path is None:
                logger.warning(f"Label file annotation {label_id} could not be fetched")
                return None
            return compute_file_iscc(mask_path)
    except Exception as exc:
        logger.warning(f"Could not compute ISCC for label {label_id}: {exc}")
        return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_provenance.py -v`
Expected: PASS — 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/omero_annotate_ai/processing/provenance.py tests/test_provenance.py pyproject.toml
git commit -m "feat(provenance): add ISCC code computation via iscc-bio

Lazily-imported wrapper around iscc-bio's biocode(). Degrades to None with a
warning when the optional [provenance] extra is not installed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Schema fields and table round-trip

**Files:**
- Modify: `src/omero_annotate_ai/core/annotation_config.py` — `ImageAnnotation` (`:58-117`), `AnnotationConfig` (`:683`), `to_dataframe()` (`:842-943`), `from_dataframe()` (`:945-1005`), `get_config_template()` (`:1323`)
- Modify: `src/omero_annotate_ai/omero/omero_functions.py:334-337` (`_prepare_dataframe_for_omero`)
- Test: `tests/test_config.py`, `tests/test_omero_functions.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent).
- Produces:
  - `ImageAnnotation.source_iscc: Optional[str]`
  - `ImageAnnotation.label_iscc: Optional[str]`
  - `AnnotationConfig.iscc_mode: Literal["off", "on"]`

Both codes are `Optional[str]`, so they use the **sentinel-string** convention the timestamps already use: `"None"` in the DataFrame, `None` on the model.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
@pytest.mark.unit
class TestIsccProvenanceFields:
    """source_iscc / label_iscc round-trip and iscc_mode toggle."""

    def test_iscc_fields_default_to_none(self):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        ann = ImageAnnotation(image_id=1, image_name="a.tif")

        assert ann.source_iscc is None
        assert ann.label_iscc is None

    def test_iscc_mode_defaults_to_off(self):
        from omero_annotate_ai.core.annotation_config import create_default_config

        assert create_default_config().iscc_mode == "off"

    def test_schema_version_bumped(self):
        from omero_annotate_ai.core.annotation_config import create_default_config

        assert create_default_config().schema_version == "2.1.0"

    def test_old_schema_config_still_loads(self):
        """2.0.0 configs predate the ISCC fields; they must still load, unstamped."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        ann = ImageAnnotation(image_id=1, image_name="a.tif")

        assert ann.source_iscc is None

    def test_iscc_fields_survive_yaml_round_trip(self, tmp_path):
        from omero_annotate_ai.core.annotation_config import (
            AnnotationConfig,
            ImageAnnotation,
            create_default_config,
        )

        config = create_default_config()
        config.iscc_mode = "on"
        config.add_annotation(
            ImageAnnotation(
                image_id=1,
                image_name="a.tif",
                source_iscc="ISCC:SOURCE",
                label_iscc="ISCC:LABEL",
            )
        )

        path = tmp_path / "config.yaml"
        config.save_yaml(path)
        loaded = AnnotationConfig.from_yaml(path)

        assert loaded.iscc_mode == "on"
        assert loaded.annotations[0].source_iscc == "ISCC:SOURCE"
        assert loaded.annotations[0].label_iscc == "ISCC:LABEL"

    def test_iscc_fields_survive_dataframe_round_trip(self):
        from omero_annotate_ai.core.annotation_config import (
            ImageAnnotation,
            create_default_config,
        )

        config = create_default_config()
        config.add_annotation(
            ImageAnnotation(
                image_id=1,
                image_name="a.tif",
                source_iscc="ISCC:SOURCE",
                label_iscc="ISCC:LABEL",
            )
        )

        df = config.to_dataframe()
        assert df.loc[0, "source_iscc"] == "ISCC:SOURCE"
        assert df.loc[0, "label_iscc"] == "ISCC:LABEL"

        reloaded = create_default_config()
        reloaded.from_dataframe(df)

        assert reloaded.annotations[0].source_iscc == "ISCC:SOURCE"
        assert reloaded.annotations[0].label_iscc == "ISCC:LABEL"

    def test_absent_codes_round_trip_as_none(self):
        """An unstamped annotation must come back as None, not the string 'None'."""
        from omero_annotate_ai.core.annotation_config import (
            ImageAnnotation,
            create_default_config,
        )

        config = create_default_config()
        config.add_annotation(ImageAnnotation(image_id=1, image_name="a.tif"))

        df = config.to_dataframe()
        assert df.loc[0, "source_iscc"] == "None"

        reloaded = create_default_config()
        reloaded.from_dataframe(df)

        assert reloaded.annotations[0].source_iscc is None
        assert reloaded.annotations[0].label_iscc is None
```

Add to `tests/test_omero_functions.py`:

```python
@pytest.mark.unit
class TestIsccColumnTyping:
    """_prepare_dataframe_for_omero() types the ISCC columns as strings."""

    def test_iscc_columns_typed_as_string(self):
        import pandas as pd

        from omero_annotate_ai.omero.omero_functions import _prepare_dataframe_for_omero

        df = pd.DataFrame(
            {
                "image_id": [1, 2],
                "source_iscc": ["ISCC:SOURCE", None],
                "label_iscc": [None, "ISCC:LABEL"],
            }
        )

        result = _prepare_dataframe_for_omero(df)

        assert result["source_iscc"].tolist() == ["ISCC:SOURCE", "None"]
        assert result["label_iscc"].tolist() == ["None", "ISCC:LABEL"]
        assert result["source_iscc"].dtype == object
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_config.py::TestIsccProvenanceFields tests/test_omero_functions.py::TestIsccColumnTyping -v`
Expected: FAIL — pydantic rejects the unknown `source_iscc` field / `KeyError: 'source_iscc'`

- [ ] **Step 3: Add the model fields**

In `annotation_config.py`, in `ImageAnnotation`, immediately after the `schema_attachment_id` field (ends line 111) and before the `channel_presentation` field:

```python
    # ISCC content provenance (None until stamped; see processing/provenance.py)
    source_iscc: Optional[str] = Field(
        default=None,
        description="ISO 24138 content code of the source image's raw OMERO pixels",
    )
    label_iscc: Optional[str] = Field(
        default=None,
        description="ISO 24138 content code of the annotation mask",
    )
```

In `AnnotationConfig` (`:683`), alongside the other top-level settings, add:

```python
    iscc_mode: Literal["off", "on"] = Field(
        default="off",
        description="Compute ISCC content provenance during a run. Requires the "
        "'provenance' extra. Off by default: adds per-image compute cost.",
    )
```

- [ ] **Step 4: Add the codes to `to_dataframe()`**

In `to_dataframe()`, in the `row` dict (after `"z_length": annotation.z_length,`, line 881), add — reusing the `"None"` sentinel the timestamps use:

```python
                "source_iscc": annotation.source_iscc or "None",
                "label_iscc": annotation.label_iscc or "None",
```

And append to the `columns` list (after `"z_length",`, line 910):

```python
            "source_iscc",
            "label_iscc",
```

- [ ] **Step 5: Add the codes to `from_dataframe()`**

In `from_dataframe()`, after the timestamp handling (after line 1003, before `self.add_annotation(...)`), add:

```python
            # ISCC provenance codes - "None" sentinel means not yet stamped
            source_iscc_str = str(row.get("source_iscc", "None"))
            if source_iscc_str != "None":
                annotation_data["source_iscc"] = source_iscc_str

            label_iscc_str = str(row.get("label_iscc", "None"))
            if label_iscc_str != "None":
                annotation_data["label_iscc"] = label_iscc_str
```

- [ ] **Step 6: Type the columns for OMERO**

In `omero_functions.py`, in `_prepare_dataframe_for_omero`, after the `datetime_columns` block (ends line 337), add:

```python
    iscc_columns = ["source_iscc", "label_iscc"]
    for col in iscc_columns:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)
```

- [ ] **Step 7: Bump the schema version**

Two new fields change the config schema, so bump the minor version. In `annotation_config.py:687`, change the `schema_version` default from `"2.0.0"` to `"2.1.0"`, and change the matching line in the template at `:1327` from `schema_version: "2.0.0"` to `schema_version: "2.1.0"`.

The bump is minor, not major: both fields default to `None`, so a 2.0.0 config still loads unchanged.

- [ ] **Step 8: Document in the config template**

In `get_config_template()` (`:1323`), add to the commented template:

```yaml
# ISCC content provenance (ISO 24138). Requires: pip install 'omero-annotate-ai[provenance]'
# When "on", each annotation records the content code of its source image
# (source_iscc) and mask (label_iscc), so a published dataset can be verified
# offline. Off by default.
iscc_mode: off
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_config.py tests/test_omero_functions.py -v`
Expected: PASS — all tests, including the new `TestIsccProvenanceFields` (7 tests) and `TestIsccColumnTyping` (1 test)

- [ ] **Step 10: Run the full suite for regressions**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS — the two new columns must not break any existing table test. If an existing test asserts an exact column count or an exact `schema_version` string, update it to the new values.

- [ ] **Step 11: Commit**

```bash
git add src/omero_annotate_ai/core/annotation_config.py src/omero_annotate_ai/omero/omero_functions.py tests/test_config.py tests/test_omero_functions.py
git commit -m "feat(provenance): add source_iscc/label_iscc fields and iscc_mode

Codes serialize into config.yaml and mirror into the OMERO tracking table.
Absent codes use the 'None' sentinel-string convention already used by the
timestamp columns.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `stamp_config()` — fill the codes

**Files:**
- Modify: `src/omero_annotate_ai/processing/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: `compute_image_iscc`, `compute_label_iscc`, `iscc_available` (Task 1); `ImageAnnotation.source_iscc`, `.label_iscc` (Task 2).
- Produces: `stamp_config(config: AnnotationConfig, conn) -> AnnotationConfig` — mutates and returns the config.

Caching is the point of this task: a source image is shared by many annotation rows (every patch, z-slice and timepoint of one image), so it must be coded **once**, not once per row.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_provenance.py`:

```python
def _config_with(annotations):
    """Build a default config carrying the given annotations."""
    from omero_annotate_ai.core.annotation_config import create_default_config

    config = create_default_config()
    for ann in annotations:
        config.add_annotation(ann)
    return config


@pytest.mark.unit
class TestStampConfig:
    """stamp_config() fills codes, and codes each unique id exactly once."""

    def test_fills_source_and_label_codes(self, fake_iscc, monkeypatch):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        monkeypatch.setattr(provenance, "compute_label_iscc", lambda conn, lid: "ISCC:LBL")
        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", label_id=99)]
        )

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc == "ISCC:AAA"
        assert config.annotations[0].label_iscc == "ISCC:LBL"

    def test_codes_each_source_image_only_once(self, fake_iscc):
        """Three patches of one image must trigger exactly one OMERO compute."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with(
            [
                ImageAnnotation(image_id=7, image_name="a.tif", is_patch=True, patch_x=0),
                ImageAnnotation(image_id=7, image_name="a.tif", is_patch=True, patch_x=1),
                ImageAnnotation(image_id=7, image_name="a.tif", is_patch=True, patch_x=2),
            ]
        )

        provenance.stamp_config(config, MagicMock())

        assert fake_iscc.call_count == 1
        assert all(a.source_iscc == "ISCC:AAA" for a in config.annotations)

    def test_skips_annotations_without_a_label(self, fake_iscc):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with([ImageAnnotation(image_id=7, image_name="a.tif")])

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc == "ISCC:AAA"
        assert config.annotations[0].label_iscc is None

    def test_does_not_recompute_existing_codes(self, fake_iscc):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", source_iscc="ISCC:OLD")]
        )

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc == "ISCC:OLD"
        assert fake_iscc.call_count == 0

    def test_no_op_and_warns_when_iscc_missing(self, no_iscc, caplog):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with([ImageAnnotation(image_id=7, image_name="a.tif")])

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc is None
        assert "iscc-bio is not installed" in caplog.text

    def test_one_failing_image_does_not_abort_the_pass(self, fake_iscc):
        """A broken image must not cost us the codes of the healthy ones."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        def flaky(conn=None, iid=None, **kwargs):
            if iid == 7:
                raise RuntimeError("corrupt pixels")
            return [{"iscc_code": "ISCC:OK"}]

        fake_iscc.side_effect = flaky
        config = _config_with(
            [
                ImageAnnotation(image_id=7, image_name="bad.tif"),
                ImageAnnotation(image_id=8, image_name="good.tif"),
            ]
        )

        provenance.stamp_config(config, MagicMock())

        assert config.annotations[0].source_iscc is None
        assert config.annotations[1].source_iscc == "ISCC:OK"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_provenance.py::TestStampConfig -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'stamp_config'`

- [ ] **Step 3: Implement `stamp_config`**

Append to `src/omero_annotate_ai/processing/provenance.py`:

```python
def stamp_config(config, conn):
    """Fill in source_iscc and label_iscc for every annotation in the config.

    Safe to re-run: codes that are already present are left alone, so this can
    be pointed at an existing config.yaml from a previous run to add provenance
    retroactively.

    Each unique image_id and label_id is coded once and cached - a source image
    is typically shared by many annotation rows (patches, z-slices, timepoints),
    and coding it per row would be wasteful.

    Args:
        config: AnnotationConfig to stamp, mutated in place.
        conn: OMERO BlitzGateway connection.

    Returns:
        The same config, for chaining.
    """
    if not iscc_available():
        logger.warning(_MISSING_MSG)
        return config

    image_cache: Dict[int, Optional[str]] = {}
    label_cache: Dict[int, Optional[str]] = {}

    for annotation in config.annotations:
        if annotation.source_iscc is None:
            image_id = annotation.image_id
            if image_id not in image_cache:
                image_cache[image_id] = compute_image_iscc(conn, image_id)
            annotation.source_iscc = image_cache[image_id]

        if annotation.label_iscc is None and annotation.label_id is not None:
            label_id = annotation.label_id
            if label_id not in label_cache:
                label_cache[label_id] = compute_label_iscc(conn, label_id)
            annotation.label_iscc = label_cache[label_id]

    coded = sum(1 for a in config.annotations if a.source_iscc is not None)
    logger.info(
        f"Stamped ISCC provenance: {coded}/{len(config.annotations)} annotations, "
        f"{len(image_cache)} unique source image(s)"
    )
    return config
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_provenance.py -v`
Expected: PASS — 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/processing/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): add stamp_config to fill ISCC codes

Codes each unique image_id/label_id once. Idempotent, so it can retroactively
stamp a config.yaml from a run that predates this feature.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `verify_config()` — check the codes, including fully offline

**Files:**
- Modify: `src/omero_annotate_ai/processing/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: `compute_file_iscc`, `compute_image_iscc`, `compute_label_iscc` (Task 1); the schema fields (Task 2).
- Produces: `verify_config(config, conn=None, data_dir=None) -> ValidationResult`

Two modes, exactly one of which must be supplied:

- `conn` — re-fetch from OMERO and recompute. The author's own check.
- `data_dir` — **offline**, no OMERO at all. This is the mode a recipient of the published dataset runs, and it is the reason the whole feature exists.

The offline mode is **content-addressed, not filename-addressed**: it codes every image file under `data_dir` and asks whether each stored code is present in that set. Renaming a published file therefore does not break verification, which is precisely the property ISCC gives us.

Three outcomes per annotation, mapped onto the existing `ValidationResult` (`annotation_config.py:1145`):

| Outcome | Meaning | Recorded as |
|---|---|---|
| match | recomputed code equals the stored one | nothing |
| mismatch | stored code is absent from the data / differs | **error** (`is_valid` → False) |
| missing | no code was ever stored | **warning** |

A missing code is an absence of evidence, not evidence of tampering, and must never be reported as a failure.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_provenance.py`:

```python
@pytest.mark.unit
class TestVerifyConfigArguments:
    """verify_config() demands exactly one source of truth."""

    def test_rejects_neither_conn_nor_data_dir(self, fake_iscc):
        config = _config_with([])

        with pytest.raises(ValueError, match="exactly one"):
            provenance.verify_config(config)

    def test_rejects_both_conn_and_data_dir(self, fake_iscc, tmp_path):
        config = _config_with([])

        with pytest.raises(ValueError, match="exactly one"):
            provenance.verify_config(config, conn=MagicMock(), data_dir=tmp_path)


@pytest.mark.unit
class TestVerifyConfigOffline:
    """data_dir mode - the recipient's story. No OMERO connection at all."""

    def test_match_when_published_file_carries_the_stored_code(
        self, fake_iscc, tmp_path
    ):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        (tmp_path / "published.tif").write_bytes(b"fake")
        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", source_iscc="ISCC:AAA")]
        )

        result = provenance.verify_config(config, data_dir=tmp_path)

        assert result.is_valid is True
        assert result.errors == []

    def test_mismatch_when_stored_code_absent_from_data(self, fake_iscc, tmp_path):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        (tmp_path / "published.tif").write_bytes(b"fake")
        config = _config_with(
            [
                ImageAnnotation(
                    image_id=7, image_name="a.tif", source_iscc="ISCC:DIFFERENT"
                )
            ]
        )

        result = provenance.verify_config(config, data_dir=tmp_path)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "source_iscc" in result.errors[0].field

    def test_missing_is_a_warning_not_an_error(self, fake_iscc, tmp_path):
        """No stored code is absence of evidence, not evidence of tampering."""
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        (tmp_path / "published.tif").write_bytes(b"fake")
        config = _config_with([ImageAnnotation(image_id=7, image_name="a.tif")])

        result = provenance.verify_config(config, data_dir=tmp_path)

        assert result.is_valid is True
        assert result.errors == []
        assert len(result.warnings) == 1

    def test_label_code_also_verified(self, fake_iscc, tmp_path):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        (tmp_path / "mask.tif").write_bytes(b"fake")
        config = _config_with(
            [
                ImageAnnotation(
                    image_id=7,
                    image_name="a.tif",
                    source_iscc="ISCC:AAA",
                    label_iscc="ISCC:NOTPRESENT",
                )
            ]
        )

        result = provenance.verify_config(config, data_dir=tmp_path)

        assert result.is_valid is False
        assert any("label_iscc" in e.field for e in result.errors)


@pytest.mark.unit
class TestVerifyConfigOmero:
    """conn mode - the author's own check against the live server."""

    def test_match_when_omero_still_has_the_same_pixels(self, fake_iscc):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", source_iscc="ISCC:AAA")]
        )

        result = provenance.verify_config(config, conn=MagicMock())

        assert result.is_valid is True

    def test_mismatch_when_omero_pixels_changed(self, fake_iscc):
        from omero_annotate_ai.core.annotation_config import ImageAnnotation

        fake_iscc.return_value = [{"iscc_code": "ISCC:CHANGED"}]
        config = _config_with(
            [ImageAnnotation(image_id=7, image_name="a.tif", source_iscc="ISCC:AAA")]
        )

        result = provenance.verify_config(config, conn=MagicMock())

        assert result.is_valid is False
        assert len(result.errors) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_provenance.py::TestVerifyConfigOffline -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'verify_config'`

- [ ] **Step 3: Implement `verify_config`**

Append to `src/omero_annotate_ai/processing/provenance.py`:

```python
# Image formats iscc-bio can read, used when scanning a published data directory.
_IMAGE_SUFFIXES = {".tif", ".tiff", ".czi", ".nd2", ".lif", ".dv", ".zarr"}


def _scan_directory_codes(data_dir: Union[str, Path]) -> set:
    """Content-code every image file under data_dir.

    Returns a set of codes, deliberately discarding filenames: verification is
    content-addressed, so renaming a published file must not break it.
    """
    codes = set()
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            code = compute_file_iscc(path)
            if code is not None:
                codes.add(code)
    return codes


def verify_config(config, conn=None, data_dir=None):
    """Verify the ISCC codes stored in a config against actual data.

    Exactly one source must be given:
      conn:     recompute from OMERO. The author's check against the live server.
      data_dir: recompute from local files, fully OFFLINE, no OMERO needed. This
                is what a recipient of a published dataset runs, and it is the
                reason this feature exists.

    In data_dir mode the check is content-addressed: every image file under the
    directory is coded, and each stored code must appear somewhere in that set.
    Filenames are irrelevant.

    Args:
        config: AnnotationConfig carrying stamped codes.
        conn: OMERO BlitzGateway, for the online check.
        data_dir: Directory of published image files, for the offline check.

    Returns:
        ValidationResult. A mismatch is an error; a code that was never stamped
        is a warning, because absence of evidence is not evidence of tampering.
    """
    from ..core.annotation_config import ValidationIssue, ValidationResult

    if (conn is None) == (data_dir is None):
        raise ValueError(
            "verify_config requires exactly one of 'conn' (verify against OMERO) "
            "or 'data_dir' (verify offline against published files)"
        )

    errors = []
    warnings = []

    available_codes = _scan_directory_codes(data_dir) if data_dir is not None else None
    image_cache: Dict[int, Optional[str]] = {}
    label_cache: Dict[int, Optional[str]] = {}

    for annotation in config.annotations:
        label = f"image {annotation.image_id} ({annotation.image_name})"

        # --- source image ---
        if annotation.source_iscc is None:
            warnings.append(
                ValidationIssue(
                    field="source_iscc",
                    message=f"No source ISCC stored for {label}; not verifiable. "
                    "Run stamp_config() to add provenance.",
                )
            )
        elif available_codes is not None:
            if annotation.source_iscc not in available_codes:
                errors.append(
                    ValidationIssue(
                        field="source_iscc",
                        message=f"No file in the data directory matches the recorded "
                        f"source ISCC for {label}. The data does not match the config.",
                    )
                )
        else:
            image_id = annotation.image_id
            if image_id not in image_cache:
                image_cache[image_id] = compute_image_iscc(conn, image_id)
            actual = image_cache[image_id]
            if actual is not None and actual != annotation.source_iscc:
                errors.append(
                    ValidationIssue(
                        field="source_iscc",
                        message=f"Source ISCC mismatch for {label}: config records "
                        f"{annotation.source_iscc}, OMERO now yields {actual}.",
                    )
                )

        # --- annotation mask ---
        if annotation.label_id is None:
            continue

        if annotation.label_iscc is None:
            warnings.append(
                ValidationIssue(
                    field="label_iscc",
                    message=f"No label ISCC stored for {label}; mask not verifiable.",
                )
            )
        elif available_codes is not None:
            if annotation.label_iscc not in available_codes:
                errors.append(
                    ValidationIssue(
                        field="label_iscc",
                        message=f"No file in the data directory matches the recorded "
                        f"label ISCC for {label}. The mask does not match the config.",
                    )
                )
        else:
            label_id = annotation.label_id
            if label_id not in label_cache:
                label_cache[label_id] = compute_label_iscc(conn, label_id)
            actual = label_cache[label_id]
            if actual is not None and actual != annotation.label_iscc:
                errors.append(
                    ValidationIssue(
                        field="label_iscc",
                        message=f"Label ISCC mismatch for {label}: config records "
                        f"{annotation.label_iscc}, OMERO now yields {actual}.",
                    )
                )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        annotation_count=len(config.annotations),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_provenance.py -v`
Expected: PASS — 23 passed

- [ ] **Step 5: Commit**

```bash
git add src/omero_annotate_ai/processing/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): add verify_config with offline data_dir mode

Verifies stored ISCC codes against OMERO (conn) or against a directory of
published files with no OMERO at all (data_dir) - the mode a recipient runs.

Offline verification is content-addressed: every image under data_dir is coded
and each stored code must appear in that set, so renaming published files does
not break it.

Mismatch is an error; a never-stamped code is a warning, not a failure.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Pipeline hook, docs, and lockfile

**Files:**
- Modify: `src/omero_annotate_ai/core/annotation_pipeline.py` (after the annotation run completes, before the config is persisted)
- Modify: `docs/superpowers/specs/2026-07-13-iscc-provenance-design.md` (correct the offline-verify description)
- Modify: `CLAUDE.md`
- Modify: `pixi.lock` (regenerated)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `stamp_config` (Task 3); `AnnotationConfig.iscc_mode` (Task 2).
- Produces: nothing new — this wires the unit into the pipeline.

The hook is deliberately tiny: all the logic lives in `provenance.py`, so the pipeline only decides *whether* to call it. Stamping runs **after** the annotation work is done and `label_id`s exist, so both codes can be filled in one pass.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
@pytest.mark.unit
class TestIsccPipelineHook:
    """The pipeline stamps provenance only when iscc_mode is on."""

    def test_stamp_not_called_when_mode_off(self, monkeypatch):
        from omero_annotate_ai.core import annotation_pipeline
        from omero_annotate_ai.core.annotation_config import create_default_config

        called = []
        monkeypatch.setattr(
            annotation_pipeline, "stamp_config", lambda c, conn: called.append(c)
        )

        config = create_default_config()
        config.iscc_mode = "off"
        pipeline = annotation_pipeline.AnnotationPipeline(config, MagicMock())
        pipeline._stamp_provenance()

        assert called == []

    def test_stamp_called_when_mode_on(self, monkeypatch):
        from omero_annotate_ai.core import annotation_pipeline
        from omero_annotate_ai.core.annotation_config import create_default_config

        called = []
        monkeypatch.setattr(
            annotation_pipeline, "stamp_config", lambda c, conn: called.append(c)
        )

        config = create_default_config()
        config.iscc_mode = "on"
        pipeline = annotation_pipeline.AnnotationPipeline(config, MagicMock())
        pipeline._stamp_provenance()

        assert len(called) == 1

    def test_stamp_failure_never_breaks_the_run(self, monkeypatch):
        """Provenance is best-effort. A failure here must not lose annotations."""
        from omero_annotate_ai.core import annotation_pipeline
        from omero_annotate_ai.core.annotation_config import create_default_config

        def boom(config, conn):
            raise RuntimeError("iscc exploded")

        monkeypatch.setattr(annotation_pipeline, "stamp_config", boom)

        config = create_default_config()
        config.iscc_mode = "on"
        pipeline = annotation_pipeline.AnnotationPipeline(config, MagicMock())

        pipeline._stamp_provenance()  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pipeline.py::TestIsccPipelineHook -v`
Expected: FAIL — `AttributeError: 'AnnotationPipeline' object has no attribute '_stamp_provenance'`

- [ ] **Step 3: Add the import and the hook method**

In `annotation_pipeline.py`, add to the imports at the top of the file:

```python
from ..processing.provenance import stamp_config
```

Add this method to `AnnotationPipeline`:

```python
    def _stamp_provenance(self) -> None:
        """Record ISCC content provenance for the annotated images and masks.

        No-op unless config.iscc_mode == "on". Best-effort: provenance is a
        publication concern, and losing it must never cost us the annotations,
        so any failure is logged and swallowed.
        """
        if self.config.iscc_mode != "on":
            return

        try:
            stamp_config(self.config, self.conn)
        except Exception as exc:
            print(f"⚠️ Could not stamp ISCC provenance: {exc}")
```

- [ ] **Step 4: Call the hook from `_finalize_workflow`**

`_finalize_workflow` (`annotation_pipeline.py:1482`) is the single convergence point — both `run_microsam_annotation` (`:1524`) and `run_custom_annotation` (`:1949`) end there, and it is where the config is persisted: `_auto_save_config()` at `:1489`, then `_upload_annotation_config_to_omero()` at `:1493`.

Insert the call at the **top** of `_finalize_workflow`, before the `_auto_save_config()` line:

```python
    def _finalize_workflow(self, processed_count: int) -> None:
        """Finalize the workflow with cleanup and uploads.

        Args:
            processed_count: Number of units that were processed
        """
        # Stamp ISCC provenance before the config is persisted, so the codes
        # travel into both config.yaml and the OMERO tracking table.
        self._stamp_provenance()

        # Final config save
        self._auto_save_config()
```

Placement matters and this point satisfies both constraints: it runs **after** the annotation batches have completed and masks are uploaded (so `label_id` is populated), and **before** the config is written out (so the codes are actually saved).

Note: `run_cellpose_preparation` does not route through `_finalize_workflow`, so it is not stamped. That is correct — it only prepares local tiles and produces no masks. Cellpose datasets get provenance through the retroactive `stamp_config()` path instead.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pipeline.py -v`
Expected: PASS — including the 3 new `TestIsccPipelineHook` tests

- [ ] **Step 6: Run the full suite**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS — no regressions.

- [ ] **Step 7: Correct the spec's offline-verify description**

The spec described offline verification as mapping each annotation to a file. The implementation is content-addressed instead — better, because it survives renaming. In `docs/superpowers/specs/2026-07-13-iscc-provenance-design.md`, replace:

> The `data_dir` mode is what a third party actually runs: they hold the published images and the
> `config.yaml`, and nothing else. It maps each annotation's file to `compute_file_iscc` and compares
> against the stored `source_iscc` / `label_iscc`.

with:

> The `data_dir` mode is what a third party actually runs: they hold the published images and the
> `config.yaml`, and nothing else. It is **content-addressed**: every image file under the directory is
> coded, and each stored `source_iscc` / `label_iscc` must appear somewhere in that set. Filenames are
> never consulted, so renaming a published file does not break verification — which is exactly the
> property a content code is supposed to give us.

- [ ] **Step 8: Document the feature in CLAUDE.md**

Append a section to `CLAUDE.md`:

```markdown
---

# ISCC Content Provenance

**Date**: 2026-07-13
**Spec**: `docs/superpowers/specs/2026-07-13-iscc-provenance-design.md`

Records ISO 24138 pixel-content codes so a published annotation dataset can be verified.

- `source_iscc` — content code of the source image's **raw OMERO pixels**
- `label_iscc` — content code of the annotation mask

Both are `Optional[str]` fields on `ImageAnnotation`, so they land in `config.yaml`
and mirror into the OMERO tracking table automatically.

## The one thing to remember

IMAGEWALK (iscc-bio) is an **exact** code, not a transform-invariant one. It packs raw
dtype bytes with no bit-depth or intensity canonicalization, matching OMERO's own pixel
representation bit-for-bit. So it survives a change of **container format** (OME-TIFF ↔
Zarr ↔ OMERO) but **not** a change of **pixel values**.

The export path rescales intensities (`img * 255/max`) and restacks channels, so the
8-bit training tiles have codes that do **not** match their source images. Never code the
tiles — they are ephemeral micro-SAM/CellPose scratch and are never published. Codes come
from raw source pixels only.

## Usage

```python
from omero_annotate_ai.processing.provenance import stamp_config, verify_config

# Stamp: fills codes; safe to re-run on an old config.yaml to add provenance
# retroactively. Codes each unique image once, not once per annotation row.
stamp_config(config, conn)

# Verify as the author, against the live server:
result = verify_config(config, conn=conn)

# Verify as a recipient, fully offline, no OMERO:
result = verify_config(config, data_dir="published_data/")
print(result.summary)
```

Offline verification is content-addressed: every image under `data_dir` is coded and each
stored code must appear in that set. Renaming published files does not break it.

Mismatch is an **error**; a code that was never stamped is a **warning** — absence of
evidence is not evidence of tampering.

## Install

`iscc-bio` is optional and pinned `>=0.1,<0.2` (it is PoC software that declares breaking
changes may ship at any time):

```bash
pip install 'omero-annotate-ai[provenance]'
```

Without it the pipeline runs unchanged, codes stay `None`, and one warning is logged.
Set `iscc_mode: "on"` in the config to enable — it is **off** by default.
```

- [ ] **Step 9: Refresh the lockfile**

Run: `pixi install`
Expected: `pixi.lock` is updated. CI fails if this is skipped.

- [ ] **Step 10: Commit**

```bash
git add src/omero_annotate_ai/core/annotation_pipeline.py tests/test_pipeline.py CLAUDE.md docs/superpowers/specs/2026-07-13-iscc-provenance-design.md pixi.lock
git commit -m "feat(provenance): stamp ISCC codes from the pipeline when enabled

Hook runs after masks are uploaded and before the config is persisted, so both
codes travel into config.yaml and the tracking table. No-op when iscc_mode is
off (the default); failures are swallowed so provenance can never cost us the
annotations.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual Verification

The unit tests all mock `iscc-bio`, so they prove the wiring but not that the codes are *real*. These steps need a live OMERO and a real install (`pip install 'omero-annotate-ai[provenance]'`).

- [ ] **The core claim — format independence.** Export a **lossless** copy of a source image in a different container format (OME-TIFF → OME-Zarr). Confirm `compute_file_iscc(copy) == compute_image_iscc(conn, image_id)`. The entire design rests on this; if it fails, nothing else matters.

- [ ] **Stamping.** Run an annotation with `iscc_mode: "on"`. Confirm `source_iscc` and `label_iscc` appear in the saved `config.yaml` and in the OMERO tracking table, and that each unique `image_id` was coded once (check the "N unique source image(s)" log line against the number of annotation rows).

- [ ] **The recipient's story.** With **no OMERO connection**, run `verify_config(config, data_dir=...)` against the published images plus `config.yaml` alone. Every entry must report match. This is the story the feature exists for — if it does not work offline, the feature has failed.

- [ ] **Tamper detection.** Alter the pixels of one published file; confirm `verify_config` reports a **mismatch** error. Clear one stored code; confirm it reports a **missing** warning and `is_valid` stays `True`.

- [ ] **Retrofit.** Take a `config.yaml` from a run predating this feature, run `stamp_config(config, conn)`, and confirm both fields populate.

- [ ] **Graceful degradation.** `pip uninstall iscc-bio`. Confirm the pipeline still runs end to end, codes are `None`, and a single clear warning is logged rather than an exception.
