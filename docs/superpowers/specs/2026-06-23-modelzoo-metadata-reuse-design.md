# Design: Reuse AnnotationConfig metadata for BioImage Model Zoo export

**Date:** 2026-06-23
**Branch:** `feat/modelzoo-glue` (off `main`)
**Status:** Approved design — pending spec review → implementation plan

## Context

Trained models from this project are published to the **BioImage Model Zoo (BMZ)**.
Today the metadata for those exports is entered by hand:

- The **micro-sam** training notebook (`notebooks/jupyter/training/omero-training-microsam.ipynb`,
  cell 18) already calls `export_sam_model(...)`, but with **hardcoded** metadata
  (`authors=[{"name": "Your Name", ...}]`, a generic `description`).
- The **BiaPy** training notebook (`omero-training_biapy.ipynb`) stops at data preparation
  and has **no Model Zoo export at all**, even though BiaPy ships `export_model_to_bmz`.

Meanwhile, the `AnnotationConfig` already captures exactly the metadata a BMZ export wants
(authors, license, study description, keywords, dataset DOI). This work reuses that config
metadata to populate both exports, replacing the hand-entered values.

## Decisions (locked during brainstorming)

1. Reuse `AnnotationConfig` metadata for **both** notebooks — add the BiaPy export *and*
   retrofit the existing micro-sam cell so both are consistent.
2. The mapping lives in a **shipped, unit-tested helper module**, not duplicated inline in
   the two notebooks.
3. The BiaPy export cell is **guarded/illustrative** (`try: import biapy`), meant to run in
   the user's own BiaPy environment. BiaPy is *not* added as a project dependency. The
   shipped helper has no BiaPy/micro-sam import, so it stays importable and testable
   everywhere.

## Verified upstream APIs

- **micro-sam:** `export_sam_model(image, label_image, model_type, name, output_path,
  checkpoint_path=None, **kwargs)`. The source forwards these metadata kwargs:
  `authors` (list of dicts, e.g. `{"name", "affiliation"}`), `description`, `cite`
  (list of `{"text", "doi"}`), `license` (SPDX string), `tags`, `git_repo`,
  `documentation`, `covers`. **No `training_data` kwarg** → the BIA-dataset link rides on
  `cite` (the dataset DOI).
- **BiaPy:** `export_model_to_bmz` is an **instance method**:
  `biapy = BiaPy(...); biapy.run_job(); biapy.export_model_to_bmz(bmz_folder, bmz_cfg=...)`.
  `bmz_cfg` shape:
  ```python
  {"description", "model_name",
   "authors": [{"name", "github_user"}],
   "license": "CC-BY-4.0", "tags": [...],
   "cite": [{"text", "doi"}],
   "data": {"name", "doi", "image_modality"}}
  ```
  The BIA-dataset link rides on `data` (name + DOI).

Sources: <https://biapy.readthedocs.io/en/latest/get_started/bmz.html>,
<https://github.com/BiaPyX/BiaPy/blob/master/biapy/utils/scripts/export_bmz_test.py>.

## Architecture

### Component 1 — `src/omero_annotate_ai/core/modelzoo_metadata.py` (new, pure)

No `biapy`/`micro_sam`/`bioimageio` imports — only reads an `AnnotationConfig`. One clear
purpose: translate config metadata into the shapes each export tool wants.

- `config_to_bmz_metadata(config) -> dict` — the common intermediate:
  | key | source | fallback |
  |---|---|---|
  | `model_name` | `ai_model.model_name` → `name` | non-empty |
  | `model_type` | `ai_model.pretrained_from` (e.g. `vit_b_lm`) | passed through (micro-sam validates) |
  | `description` | `study.description` → `study.title` → `name` | non-empty |
  | `authors` | `config.authors` → `{name, affiliation}` | drop entries without a name; ORCID dropped (no BMZ field); `github_user` omitted (`AuthorInfo` has none) |
  | `license` | `dataset.license` → SPDX | `CC-BY-4.0` passthrough; `CC_BY`→`CC-BY-4.0`; `CC0`/`CC0-1.0`→`CC0-1.0`; default `CC-BY-4.0` |
  | `tags` | `config.tags` + `study.keywords` (+ `ai_model.framework`, `study.organism`) | deduped; may be empty |
  | `cite` | `ai_model.model_doi` (+ dataset DOI) | `[{"text", "doi"}]` when a DOI exists; else empty |
  | `data` | `{name: dataset.source_description or study.title, doi: dataset.source_dataset_id/url, image_modality: study.imaging_method}` | keys omitted when blank |

- `to_micro_sam_kwargs(meta) -> dict` — `{authors, description, license, tags, cite}` for
  `export_sam_model(**kwargs)`. (`name`/`model_type` are passed positionally by the
  notebook; the dataset link is in `cite`.)
- `to_biapy_bmz_cfg(meta) -> dict` — the full `bmz_cfg` dict (description, model_name,
  authors, license, tags, cite, data).
- Private helpers: `_to_spdx_license(s)`, `_build_authors(config)`, `_build_cite(config)`,
  `_build_tags(config)`, `_build_data(config)`.

Guiding principle (same as the MIFA exporter): emit only what we have; omit blanks so each
tool's own defaults apply. Blank/default configs still produce a usable dict.

### Component 2 — micro-sam notebook retrofit

In `omero-training-microsam.ipynb` cell 18, replace the hardcoded
`authors=[...]`/`description=...` with
`**to_micro_sam_kwargs(config_to_bmz_metadata(config))`. Add a cell to obtain the
`AnnotationConfig` if it is not already loaded in the notebook (e.g.
`download_annotation_config_from_omero(...)` from the selected table, or
`load_config_from_yaml(...)`).

### Component 3 — BiaPy notebook new export section

Append a Model Zoo export section to `omero-training_biapy.ipynb`:
1. Obtain the config (`download_annotation_config_from_omero(...)`).
2. `bmz_cfg = to_biapy_bmz_cfg(config_to_bmz_metadata(config))`; display it.
3. A guarded, illustrative cell:
   ```python
   try:
       from biapy import BiaPy
   except ImportError:
       print("Run this cell in your BiaPy environment (pip install biapy).")
   else:
       biapy = BiaPy(...)        # your BiaPy training/config
       biapy.run_job()
       biapy.export_model_to_bmz(bmz_folder, bmz_cfg=bmz_cfg)
   ```

## Data flow

```
AnnotationConfig
      │  config_to_bmz_metadata()
      ▼
  common bmz metadata dict
      ├── to_micro_sam_kwargs() ──► export_sam_model(**kwargs)      (BIA link via cite)
      └── to_biapy_bmz_cfg()    ──► BiaPy.export_model_to_bmz(bmz_cfg)  (BIA link via data)
```

## Testing

`TestModelZooMetadata` in `tests/test_config.py` (`@pytest.mark.unit`; **no** biapy/micro-sam
needed — the helper is pure):
- `config_to_bmz_metadata` returns the expected keys; `model_name`/`description` fall back
  to `config.name` on a blank config.
- SPDX license map (`CC-BY-4.0`→passthrough, `CC_BY`→`CC-BY-4.0`, `CC0`→`CC0-1.0`, unknown→default).
- author mapping (name/affiliation; nameless dropped; empty authors → `[]`).
- `cite` built from `ai_model.model_doi`; empty when no DOI.
- `tags` merge + dedupe.
- `data` populated from `dataset`/`study.imaging_method`; keys omitted when blank.
- `to_micro_sam_kwargs` shape (no `data`/`model_name`); `to_biapy_bmz_cfg` shape (full dict).

Run: `pixi run -e dev pytest tests/test_config.py -v -k ModelZoo`.

## Dependencies

None added. The helper is pure Python; `micro_sam` is already a core dependency; BiaPy stays
out of the project env (guarded import in the notebook).

## Non-goals / out of scope

- No package code that *runs* the export (no wrapper around `export_sam_model` /
  `export_model_to_bmz`) — the export calls stay in the notebooks.
- No BiaPy dependency; no running BiaPy training in-project.
- No changes to the MIFA/BIA export work (separate PR/branch).
- No `github_user` on `AuthorInfo` (could be a later addition if BMZ author linking matters).

## To verify at implementation time

- Exact variable names available in each notebook (`config`, `model_name`,
  `training_config`, `output_directory`) so the retrofit/new cells reference real values.
- Whether the micro-sam notebook already loads an `AnnotationConfig` or needs the new
  download cell.
