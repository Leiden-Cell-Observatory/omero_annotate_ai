# Architecture

How the package is put together, for contributors. It describes the code as it is now.

```mermaid
flowchart LR
    W[widgets / notebooks / napari] --> P[trainingset]
    F[core.annotation_pipeline<br/>deprecated facade] --> P
    C[core.annotation_config] --> P
    P --> RW[rows]
    P --> PL[planning]
    P --> SE[session]
    P --> R[reconcile]
    SE --> A[annotators]
    SE --> B[bundle]
    R --> SE
    P --> S[sources]
    SE --> O[omero]
    P --> E[events]
    A --> N[(micro-SAM / napari)]
    S --> OM[(OMERO / OME-Zarr)]
    O --> OM
```

| Part | Responsibility |
|---|---|
| `core/annotation_config.py` | A training set's definition (pydantic, schema version 3) and its YAML form: what to annotate and how, plus `identity` (table ID or local ID) and `sessions` (with notes). Rows are not part of it. Removed keys raise `RemovedConfigKey`. |
| `trainingset.py` | `TrainingSet`: create (select images, plan units, create the table) or open (by table ID or folder, repair), annotate, extend, notes, external tools. Lists images and loads pixels. `OmeroWrites` holds every OMERO call it makes (replaceable in tests). Read-only `list_training_sets` / `summarize`. |
| `rows.py` | `Row`, one per annotation unit, and the one codec between rows and a table (`tracking_table.csv` and the OMERO table). |
| `core/annotation_pipeline.py` | Deprecated `create_pipeline` facade over `TrainingSet`, kept for 0.4 notebooks. |
| `session.py` | One annotation session, step by step: `next_batch()`, `commit()`, `release()`, `finish()`. Owns the write order of a unit: mask into the bundle, local accept (CSV), upload label then ROI (each ID saved straight away, tagged), table update per commit. |
| `reconcile.py` | On resume, compares the OMERO table, the local CSV and the mask files per unit (`decide`) and repairs: adopts tagged uploads, uploads what is missing, reports pixel mismatches. Idempotent. |
| `events.py` | What a run reports (`UnitCommitted`, `UploadFailed`, `ReconcileAction`, `Progress`, ...) to an `EventSink`; default: logging. Front-ends pass their own. |
| `planning.py` | Pure planning: which images go to which split (`plan_split`), the units of one image (`units_for_image`), and which images to add to a set (`plan_extension`). Every random choice comes from `training.seed` via `rng_for`, so a plan can be reproduced. |
| `annotators.py` | The `Annotator` protocol and its two implementations: `MicroSamAnnotator` (napari, interactive) and `ScriptedAnnotator` (draws masks with a function; for scripts and tests). |
| `sources/` | Reading pixels and image sizes from OMERO or OME-Zarr behind one `ImageSource` protocol. |
| `omero/` | OMERO access: the tracking table (updated in place, so its ID identifies the training set), uploads of ROIs and labels, connections. |
| `processing/` | Training data export and training helpers. |
| `bundle.py` | Where a training set's files live: one folder per set under the home folder (`$OMERO_ANNOTATE_AI_HOME`, else `~/omero_annotate_ai`), atomic writes, a fresh scratch folder per batch, masks matched to units by exact file name, and pixel digests. |
| `errors.py` | The package's exceptions; all derive from `OmeroAnnotateAIError`. |

**Rules of thumb**

- `planning.py`, `annotators.py`, `bundle.py`, `errors.py`, `events.py`, `session.py` and
  `reconcile.py` are type-checked strictly (pyright); nothing in the core prints. The
  workflow widget only holds form state: it builds a request that `open_or_create`
  turns into a `TrainingSet`, and shows progress from the events.
- A protocol needs two real implementations before it is introduced.
- Tests: unit tests mirror the modules (`tests/test_planning.py`, `tests/test_rows.py`,
  `tests/test_trainingset.py`, `tests/test_session.py`, `tests/test_reconcile.py`, ...) and use the in-memory OMERO in
  `tests/fakes.py`, which can fail or "crash" any call; `tests/test_functional_workflow.py`
  runs the whole workflow against a Docker OMERO and is the acceptance suite.
