# Model Zoo Metadata Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the metadata in an `AnnotationConfig` to populate BioImage Model Zoo (BMZ) exports in the micro-sam (retrofit) and BiaPy (new) training notebooks.

**Architecture:** A single pure helper module (`core/modelzoo_metadata.py`) maps an `AnnotationConfig` to a common BMZ-metadata dict, plus two thin adapters that shape it for micro-sam's `export_sam_model(**kwargs)` and BiaPy's `export_model_to_bmz(bmz_cfg=...)`. The notebooks import the helper; the actual export calls stay in the notebooks.

**Tech Stack:** Python 3.11, pydantic v2, pytest, pixi.

## Global Constraints

- The helper module MUST NOT import `biapy`, `micro_sam`, or `bioimageio` — it stays pure and unit-testable everywhere.
- No new project dependencies. BiaPy is a guarded (`try/except ImportError`) import in the notebook only.
- Tests live in `tests/test_config.py`, marked `@pytest.mark.unit`. Run with `pixi run -e dev pytest`.
- BMZ licenses are SPDX strings (e.g. `CC-BY-4.0`), not the MIFA enum codes.
- micro-sam's `export_sam_model` has **no `training_data` kwarg** — the BIA-dataset link rides on `cite`. BiaPy carries it in `bmz_cfg["data"]`.
- New Python is black-formatted (line length 88).
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 0: Environment setup

**Files:** none (worktree already on branch `feat/modelzoo-glue`).

- [ ] **Step 1: Install the dev environment**

Run: `pixi install -e dev`
Expected: "The dev environment has been installed."

- [ ] **Step 2: Confirm a clean baseline**

Run: `pixi run -e dev pytest tests/ -q`
Expected: all tests pass (e.g. `281 passed, 5 skipped`).

---

### Task 1: `config_to_bmz_metadata` + builders (pure helper)

**Files:**
- Create: `src/omero_annotate_ai/core/modelzoo_metadata.py`
- Test: `tests/test_config.py` (append `TestModelZooMetadata`)

**Interfaces:**
- Consumes: `AnnotationConfig` (fields `name`, `authors[].{name,affiliation}`, `tags`, `study.{title,description,keywords,organism,imaging_method}`, `dataset.{license,source_dataset_id,source_dataset_url,source_description}`, `ai_model.{model_name,pretrained_from,framework,model_doi}`).
- Produces: `config_to_bmz_metadata(config) -> dict` with keys `model_name, model_type, description, authors, license, tags, cite, data`; plus `_to_spdx_license(str|None) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
@pytest.mark.unit
class TestModelZooMetadata:
    """Map AnnotationConfig -> BioImage Model Zoo metadata (pure helper)."""

    def test_config_to_bmz_metadata_keys_and_fallbacks(self):
        from omero_annotate_ai.core.modelzoo_metadata import config_to_bmz_metadata
        config = create_default_config()
        config.name = "demo_model"
        meta = config_to_bmz_metadata(config)
        assert set(meta) == {
            "model_name", "model_type", "description",
            "authors", "license", "tags", "cite", "data",
        }
        assert meta["model_name"] == "demo_model"   # ai_model.model_name blank -> name
        assert meta["description"] == "demo_model"  # study blank -> name
        assert meta["model_type"] == "vit_b_lm"     # ai_model.pretrained_from default
        assert meta["license"] == "CC-BY-4.0"

    def test_spdx_license_map(self):
        from omero_annotate_ai.core.modelzoo_metadata import _to_spdx_license
        assert _to_spdx_license("CC-BY-4.0") == "CC-BY-4.0"
        assert _to_spdx_license("CC_BY") == "CC-BY-4.0"
        assert _to_spdx_license("CC0") == "CC0-1.0"
        assert _to_spdx_license("weird") == "CC-BY-4.0"

    def test_authors_drop_nameless(self):
        from omero_annotate_ai.core.annotation_config import AuthorInfo
        from omero_annotate_ai.core.modelzoo_metadata import config_to_bmz_metadata
        config = create_default_config()
        config.authors = [
            AuthorInfo(name="Jane Doe", affiliation="EMBL"),
            AuthorInfo(name=""),
        ]
        assert config_to_bmz_metadata(config)["authors"] == [
            {"name": "Jane Doe", "affiliation": "EMBL"}
        ]

    def test_tags_merge_and_dedupe(self):
        from omero_annotate_ai.core.modelzoo_metadata import config_to_bmz_metadata
        config = create_default_config()
        config.tags = ["seg", "micro_sam"]
        config.study.keywords = ["nuclei", "seg"]
        config.study.organism = "Homo sapiens"
        assert config_to_bmz_metadata(config)["tags"] == [
            "seg", "micro_sam", "nuclei", "Homo sapiens"
        ]

    def test_cite_from_doi(self):
        from omero_annotate_ai.core.modelzoo_metadata import config_to_bmz_metadata
        config = create_default_config()
        assert config_to_bmz_metadata(config)["cite"] == []
        config.ai_model.model_doi = "10.1234/abcd"
        assert config_to_bmz_metadata(config)["cite"] == [
            {"text": "model", "doi": "10.1234/abcd"}
        ]

    def test_data_from_dataset(self):
        from omero_annotate_ai.core.modelzoo_metadata import config_to_bmz_metadata
        config = create_default_config()
        assert config_to_bmz_metadata(config)["data"] == {}
        config.dataset.source_description = "U2OS nuclei"
        config.dataset.source_dataset_id = "S-BIAD123"
        config.study.imaging_method = "confocal"
        assert config_to_bmz_metadata(config)["data"] == {
            "name": "U2OS nuclei", "doi": "S-BIAD123", "image_modality": "confocal"
        }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_config.py -q -k ModelZoo`
Expected: FAIL — `ModuleNotFoundError: No module named 'omero_annotate_ai.core.modelzoo_metadata'`.

- [ ] **Step 3: Implement the module**

Create `src/omero_annotate_ai/core/modelzoo_metadata.py`:

```python
"""Map an AnnotationConfig onto BioImage Model Zoo (BMZ) export metadata.

Two export targets consume the result:
- micro-sam: ``export_sam_model(..., **to_micro_sam_kwargs(meta))``
- BiaPy:     ``BiaPy(...).export_model_to_bmz(folder, bmz_cfg=to_biapy_bmz_cfg(meta))``

This module is pure: it imports neither micro_sam nor biapy, so it stays importable and
unit-testable without either. It only reads an AnnotationConfig.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Dataset license string -> SPDX identifier (BMZ wants SPDX, not the MIFA enum codes).
_SPDX_LICENSE_MAP = {
    "CC-BY-4.0": "CC-BY-4.0",
    "CC-BY": "CC-BY-4.0",
    "CC_BY": "CC-BY-4.0",
    "CC0": "CC0-1.0",
    "CC0-1.0": "CC0-1.0",
    "CC-0": "CC0-1.0",
}
_DEFAULT_SPDX_LICENSE = "CC-BY-4.0"


def _to_spdx_license(license_str: Optional[str]) -> str:
    """Map a dataset license string to an SPDX identifier (default CC-BY-4.0)."""
    return _SPDX_LICENSE_MAP.get((license_str or "").strip(), _DEFAULT_SPDX_LICENSE)


def _build_authors(config) -> List[Dict[str, str]]:
    """Map config authors to BMZ author dicts; drop authors without a name."""
    authors: List[Dict[str, str]] = []
    for author in config.authors:
        if not author.name or not author.name.strip():
            continue
        entry = {"name": author.name.strip()}
        if author.affiliation:
            entry["affiliation"] = author.affiliation
        authors.append(entry)
    return authors


def _build_tags(config) -> List[str]:
    """Merge tags + study keywords + framework + organism, de-duplicated."""
    tags: List[str] = []
    candidates = [
        *config.tags,
        *config.study.keywords,
        config.ai_model.framework,
        config.study.organism,
    ]
    for tag in candidates:
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _build_cite(config) -> List[Dict[str, str]]:
    """Build the BMZ citation list from the model DOI, if present."""
    if config.ai_model.model_doi:
        return [{"text": "model", "doi": config.ai_model.model_doi}]
    return []


def _build_data(config) -> Dict[str, str]:
    """Build the BMZ training-data reference (name / doi / image_modality)."""
    data: Dict[str, str] = {}
    name = config.dataset.source_description or config.study.title
    if name:
        data["name"] = name
    doi = config.dataset.source_dataset_id or (
        str(config.dataset.source_dataset_url)
        if config.dataset.source_dataset_url
        else None
    )
    if doi:
        data["doi"] = doi
    if config.study.imaging_method:
        data["image_modality"] = config.study.imaging_method
    return data


def config_to_bmz_metadata(config) -> Dict[str, Any]:
    """Build the common BMZ metadata dict from an AnnotationConfig.

    Scalar fields fall back to ``config.name`` when blank; list/dict fields may be
    empty so each export tool's own defaults apply.
    """
    return {
        "model_name": config.ai_model.model_name or config.name,
        "model_type": config.ai_model.pretrained_from,
        "description": config.study.description or config.study.title or config.name,
        "authors": _build_authors(config),
        "license": _to_spdx_license(config.dataset.license),
        "tags": _build_tags(config),
        "cite": _build_cite(config),
        "data": _build_data(config),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_config.py -q -k ModelZoo`
Expected: PASS (6 passed).

- [ ] **Step 5: Format and commit**

```bash
pixi run -e dev black src/omero_annotate_ai/core/modelzoo_metadata.py
git add src/omero_annotate_ai/core/modelzoo_metadata.py tests/test_config.py
git commit -m "feat: map AnnotationConfig to BMZ metadata

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: micro-sam / BiaPy adapters

**Files:**
- Modify: `src/omero_annotate_ai/core/modelzoo_metadata.py` (append two functions)
- Test: `tests/test_config.py` (append to `TestModelZooMetadata`)

**Interfaces:**
- Consumes: `config_to_bmz_metadata(config) -> dict` (Task 1).
- Produces:
  - `to_micro_sam_kwargs(meta) -> dict` — keys subset of `{description, license, authors, tags, cite}` (no `data`, no `model_name`, no `model_type`).
  - `to_biapy_bmz_cfg(meta) -> dict` — keys `{model_name, description, license, authors, tags, cite}` plus `data` when non-empty.

- [ ] **Step 1: Write the failing tests**

Append to `TestModelZooMetadata` in `tests/test_config.py`:

```python
    def test_to_micro_sam_kwargs_shape(self):
        from omero_annotate_ai.core.modelzoo_metadata import (
            config_to_bmz_metadata, to_micro_sam_kwargs,
        )
        config = create_default_config()
        config.ai_model.model_doi = "10.1/x"
        kwargs = to_micro_sam_kwargs(config_to_bmz_metadata(config))
        assert kwargs["description"] and kwargs["license"] == "CC-BY-4.0"
        assert kwargs["cite"] == [{"text": "model", "doi": "10.1/x"}]
        assert "data" not in kwargs
        assert "model_name" not in kwargs
        assert "model_type" not in kwargs

    def test_to_micro_sam_kwargs_omits_empty_lists(self):
        from omero_annotate_ai.core.modelzoo_metadata import (
            config_to_bmz_metadata, to_micro_sam_kwargs,
        )
        kwargs = to_micro_sam_kwargs(config_to_bmz_metadata(create_default_config()))
        assert "authors" not in kwargs  # no authors on a default config
        assert "cite" not in kwargs     # no DOI on a default config

    def test_to_biapy_bmz_cfg_shape(self):
        from omero_annotate_ai.core.modelzoo_metadata import (
            config_to_bmz_metadata, to_biapy_bmz_cfg,
        )
        config = create_default_config()
        config.dataset.source_dataset_id = "S-BIAD123"
        cfg = to_biapy_bmz_cfg(config_to_bmz_metadata(config))
        assert {"model_name", "description", "license", "authors", "tags", "cite"} <= set(cfg)
        assert cfg["data"]["doi"] == "S-BIAD123"

    def test_to_biapy_bmz_cfg_omits_empty_data(self):
        from omero_annotate_ai.core.modelzoo_metadata import (
            config_to_bmz_metadata, to_biapy_bmz_cfg,
        )
        cfg = to_biapy_bmz_cfg(config_to_bmz_metadata(create_default_config()))
        assert "data" not in cfg  # blank dataset -> empty data dropped
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_config.py -q -k "to_micro_sam or to_biapy"`
Expected: FAIL — `ImportError: cannot import name 'to_micro_sam_kwargs'`.

- [ ] **Step 3: Implement the adapters**

Append to `src/omero_annotate_ai/core/modelzoo_metadata.py`:

```python
def to_micro_sam_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Metadata kwargs for ``micro_sam ... export_sam_model(**kwargs)``.

    ``name``/``model_type`` are passed positionally by the caller; the dataset link
    rides on ``cite`` (export_sam_model has no training_data kwarg). Empty list fields
    are omitted so micro-sam's defaults apply.
    """
    kwargs: Dict[str, Any] = {
        "description": meta["description"],
        "license": meta["license"],
    }
    if meta["authors"]:
        kwargs["authors"] = meta["authors"]
    if meta["tags"]:
        kwargs["tags"] = meta["tags"]
    if meta["cite"]:
        kwargs["cite"] = meta["cite"]
    return kwargs


def to_biapy_bmz_cfg(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The ``bmz_cfg`` dict for ``BiaPy(...).export_model_to_bmz(folder, bmz_cfg=...)``."""
    cfg: Dict[str, Any] = {
        "model_name": meta["model_name"],
        "description": meta["description"],
        "license": meta["license"],
        "authors": meta["authors"],
        "tags": meta["tags"],
        "cite": meta["cite"],
    }
    if meta["data"]:
        cfg["data"] = meta["data"]
    return cfg
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_config.py -q -k ModelZoo`
Expected: PASS (10 passed).

- [ ] **Step 5: Format and commit**

```bash
pixi run -e dev black src/omero_annotate_ai/core/modelzoo_metadata.py
git add src/omero_annotate_ai/core/modelzoo_metadata.py tests/test_config.py
git commit -m "feat: add micro-sam and BiaPy BMZ metadata adapters

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Retrofit the micro-sam notebook export

**Files:**
- Modify: `notebooks/jupyter/training/omero-training-microsam.ipynb`

**Interfaces:**
- Consumes: `config_to_bmz_metadata`, `to_micro_sam_kwargs` (Task 1-2); `load_config_from_yaml` from `omero_annotate_ai.core.annotation_config`.

Note: this is a training notebook that requires a live OMERO connection and a trained
checkpoint, so it is not executed headlessly. Verification compiles the edited cells.

- [ ] **Step 1: Add a config-load cell before the export cell (cell 18)**

Insert a new code cell immediately before the export cell with this source:

```python
# Load the AnnotationConfig used for this dataset so its metadata (authors, license,
# description, tags, DOI) populates the Model Zoo export. Adjust the path, or use
# download_annotation_config_from_omero(conn, "Dataset", dataset_id) to pull it from OMERO.
from omero_annotate_ai.core.annotation_config import load_config_from_yaml
from omero_annotate_ai.core.modelzoo_metadata import (
    config_to_bmz_metadata,
    to_micro_sam_kwargs,
)

config = load_config_from_yaml(str(output_directory / "annotation_config.yaml"))
```

- [ ] **Step 2: Replace the hardcoded metadata in the `export_sam_model(...)` call**

In the export cell, replace these two lines:

```python
            authors=[{"name": "Your Name", "affiliation": "Your Institution"}],
            description="Micro-SAM model trained on microscopy images",
```

with:

```python
            **to_micro_sam_kwargs(config_to_bmz_metadata(config)),
```

- [ ] **Step 3: Syntax-check the edited cells**

Run:
```bash
pixi run -e dev python -c "
import json
nb = json.load(open('notebooks/jupyter/training/omero-training-microsam.ipynb'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), '<cell>', 'exec')
print('all code cells compile')
"
```
Expected: `all code cells compile`.

- [ ] **Step 4: Confirm the helper call is present and the placeholder is gone**

Run:
```bash
grep -c "to_micro_sam_kwargs(config_to_bmz_metadata(config))" notebooks/jupyter/training/omero-training-microsam.ipynb
grep -c "Your Name" notebooks/jupyter/training/omero-training-microsam.ipynb
```
Expected: first `1`, second `0`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/jupyter/training/omero-training-microsam.ipynb
git commit -m "feat: populate micro-sam Model Zoo export from AnnotationConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Add the BiaPy notebook Model Zoo export section

**Files:**
- Modify: `notebooks/jupyter/training/omero-training_biapy.ipynb`

**Interfaces:**
- Consumes: `config_to_bmz_metadata`, `to_biapy_bmz_cfg` (Task 1-2); `load_config_from_yaml`.

- [ ] **Step 1: Append a markdown section header cell**

Add a markdown cell at the end:

```markdown
## 6. Export the trained model to the BioImage Model Zoo

After training with BiaPy (outside this notebook), package the model for the BioImage
Model Zoo, reusing the metadata captured in your `AnnotationConfig`. Run this section in
the environment where you trained with BiaPy.
```

- [ ] **Step 2: Append the metadata-building code cell**

Add a code cell:

```python
from omero_annotate_ai.core.annotation_config import load_config_from_yaml
from omero_annotate_ai.core.modelzoo_metadata import (
    config_to_bmz_metadata,
    to_biapy_bmz_cfg,
)

# Load the AnnotationConfig for this dataset (or download_annotation_config_from_omero(...)).
config = load_config_from_yaml(str(output_directory / "annotation_config.yaml"))
bmz_cfg = to_biapy_bmz_cfg(config_to_bmz_metadata(config))
bmz_cfg
```

- [ ] **Step 3: Append the guarded export code cell**

Add a code cell:

```python
bmz_folder = str(output_directory / "bmz_model")
try:
    from biapy import BiaPy  # noqa: F401
except ImportError:
    print(
        "BiaPy is not installed in this environment. Run this section where you trained "
        "with BiaPy (pip install biapy); the bmz_cfg above is ready to use."
    )
else:
    # `biapy` is your trained BiaPy job: biapy = BiaPy(...); biapy.run_job()
    biapy.export_model_to_bmz(bmz_folder, bmz_cfg=bmz_cfg)
    print(f"Exported BMZ model to: {bmz_folder}")
```

- [ ] **Step 4: Syntax-check the edited cells**

Run:
```bash
pixi run -e dev python -c "
import json
nb = json.load(open('notebooks/jupyter/training/omero-training_biapy.ipynb'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), '<cell>', 'exec')
print('all code cells compile')
"
```
Expected: `all code cells compile`.

- [ ] **Step 5: Confirm the export glue is present**

Run:
```bash
grep -c "to_biapy_bmz_cfg(config_to_bmz_metadata(config))" notebooks/jupyter/training/omero-training_biapy.ipynb
grep -c "export_model_to_bmz(bmz_folder, bmz_cfg=bmz_cfg)" notebooks/jupyter/training/omero-training_biapy.ipynb
```
Expected: both `1`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/jupyter/training/omero-training_biapy.ipynb
git commit -m "feat: add BiaPy Model Zoo export reusing AnnotationConfig metadata

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Final verification

**Files:** none.

- [ ] **Step 1: Run the full test suite**

Run: `pixi run -e dev pytest tests/ -q`
Expected: previous baseline + 10 new tests pass, 0 failures.

- [ ] **Step 2: Confirm the helper stays pure (no heavy imports)**

Run:
```bash
pixi run -e dev python -c "
import sys
import omero_annotate_ai.core.modelzoo_metadata  # noqa: F401
assert 'biapy' not in sys.modules, 'helper must not import biapy'
assert 'micro_sam' not in sys.modules, 'helper must not import micro_sam'
print('helper is pure')
"
```
Expected: `helper is pure`.

## Self-Review

- **Spec coverage:** module `modelzoo_metadata.py` (Tasks 1-2), micro-sam retrofit (Task 3), BiaPy new export (Task 4), tests (Tasks 1-2), no new deps / pure helper (Global Constraints + Task 5) — all spec sections covered.
- **Placeholder scan:** every code step contains complete code; notebook cells are given verbatim; no TBD/TODO.
- **Type consistency:** `config_to_bmz_metadata` returns the 8-key dict consumed identically by both adapters; `to_micro_sam_kwargs`/`to_biapy_bmz_cfg` names match between definition (Task 2) and notebook use (Tasks 3-4).
