# OMERO Annotate AI - Development Guide

## Development Environment

### Before Creating a PR

Always update `pixi.lock` before pushing — CI will fail if it is out of date:

```bash
pixi install
```

Then stage and commit `pixi.lock` together with the other changes.

### Running Tests

To run the test suite, use the dev environment:

```bash
pixi run -e dev pytest tests/ -v
```

For a specific test file:
```bash
pixi run -e dev pytest tests/test_config.py -v
```

### Test Organization Guidelines

Tests should be organized alongside the source code structure. **Do not create separate test files for individual features** - instead, add tests to existing test files that correspond to the module being tested.

#### Test File Mapping

| Source Module | Test File | Description |
|---------------|-----------|-------------|
| `core/annotation_config.py` | `tests/test_config.py` | Configuration classes, schema validation, serialization |
| `core/annotation_pipeline.py` | `tests/test_pipeline.py` | Pipeline methods, workflow logic |
| `core/annotation_pipeline.py` | `tests/test_well_filtering.py` | Container/image retrieval, filtering |
| `omero/omero_functions.py` | `tests/test_omero_functions.py` | OMERO API functions, table management |
| `omero/omero_utils.py` | `tests/test_omero_functions.py` | OMERO utility functions |
| `processing/training_functions.py` | `tests/test_training_functions.py` | Training data preparation |
| `widgets/` | `tests/test_widgets.py` | Widget classes and UI components |

#### Test Class Organization

Within each test file, group related tests into classes:

```python
@pytest.mark.unit
class TestFeatureName:
    """Test description."""

    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        ...
```

#### Test Markers

- `@pytest.mark.unit` - Fast unit tests with mocked dependencies
- `@pytest.mark.integration` - Integration tests with mocked OMERO connection
- `@pytest.mark.slow` - Tests that take longer to run

#### Adding Tests for New Features

1. Identify the source module your feature modifies
2. Find the corresponding test file from the mapping above
3. Add a new test class or add tests to an existing relevant class
4. Use consistent naming: `TestFeatureName` for classes, `test_specific_behavior` for methods

---

## Notebook Structure

Notebooks live in `notebooks/` with three subdirectories:

- **`marimo/`** — Reactive marimo notebooks (`.py` files, preferred for new notebooks)
  - `omero-annotate-ai-annotation.py` — micro-SAM annotation (gold standard/reference pattern)
  - `omero-annotate-ai-cellpose.py` — CellPose annotation
  - `omero-annotate-ai-from-yaml.py` — Run from saved YAML config
  - `omero-training-microsam.py` — micro-SAM training
  - `omero-idr-demo.py` — Full demo on public IDR data (read-only mode)
- **`jupyter/`** — Classic Jupyter notebooks (`annotation/`, `training/`, `inference/`, `other/`)
- **`dev/`** — Developer notebooks (autoreload helpers, not for end users)

When creating new notebooks, prefer marimo. Follow the pattern in `omero-annotate-ai-annotation.py`:
- Connection via `SimpleOMEROConnection.create_connection_from_config()`
- Config via `create_default_config()` + field assignment
- `mo.ui.run_button` + `mo.stop()` for action triggers
- `build_config()` helper function to map UI values → `AnnotationConfig`

IDR connection: `ws://idr.openmicroscopy.org/omero-ws`, user `public`, password `public`, `secure=False`, `read_only_mode=True`.

---

## Marimo Notebook Assistant

When creating or editing marimo notebooks for this project, follow these guidelines.

### Marimo Fundamentals

Marimo is a reactive notebook that differs from traditional notebooks:

- Cells execute automatically when their dependencies change
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update the notebook automatically

### Editing Marimo Notebooks

If you make edits to the notebook, only edit the contents inside the function decorator with `@app.cell`.
Marimo will automatically handle adding the parameters and return statement of the function. For example,
for each edit, just return:

```python
@app.cell
def _():
    return
```

### Code Requirements

1. All code must be complete and runnable
2. Follow consistent coding style throughout
3. Include descriptive variable names and helpful comments
4. Import all modules in the first cell, always including `import marimo as mo`
5. Never redeclare variables across cells
6. Ensure no cycles in notebook dependency graph
7. The last expression in a cell is automatically displayed
8. Don't include comments in markdown cells
9. Don't include comments in SQL cells
10. Never define anything using `global`

### Reactivity

- When a variable changes, all cells that use that variable automatically re-execute
- UI elements trigger updates when their values change without explicit callbacks
- UI element values are accessed through `.value` attribute
- You cannot access a UI element's value in the same cell where it's defined
- Cells prefixed with an underscore (e.g. `_my_var`) are local to the cell and cannot be accessed by other cells

### Best Practices

**Data Handling:**
- Use polars for data manipulation
- Implement proper data validation
- Handle missing values appropriately
- A variable in the last expression of a cell is automatically displayed as a table

**Visualization:**
- For matplotlib: use `plt.gca()` as the last expression instead of `plt.show()`
- For plotly: return the figure object directly
- For altair: return the chart object directly. Add tooltips where appropriate.
- Include proper labels, titles, and color schemes

**UI Elements:**
- Access UI element values with `.value` attribute (e.g., `slider.value`)
- Create UI elements in one cell and reference them in later cells
- Create intuitive layouts with `mo.hstack()`, `mo.vstack()`, and `mo.tabs()`
- Prefer reactive updates over callbacks

### Available UI Elements

- `mo.ui.altair_chart(altair_chart)`
- `mo.ui.button(value=None, kind='primary')`
- `mo.ui.run_button(label=None, tooltip=None, kind='primary')`
- `mo.ui.checkbox(label='', value=False)`
- `mo.ui.date(value=None, label=None, full_width=False)`
- `mo.ui.dropdown(options, value=None, label=None, full_width=False)`
- `mo.ui.file(label='', multiple=False, full_width=False)`
- `mo.ui.number(value=None, label=None, full_width=False)`
- `mo.ui.radio(options, value=None, label=None, full_width=False)`
- `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)`
- `mo.ui.text(value='', label=None, full_width=False)`
- `mo.ui.text_area(value='', label=None, full_width=False)`
- `mo.ui.data_explorer(df)`
- `mo.ui.dataframe(df)`
- `mo.ui.plotly(plotly_figure)`
- `mo.ui.tabs(elements: dict[str, mo.ui.Element])`
- `mo.ui.form(element: mo.ui.Element, label='', bordered=True)`

### Layout and Utility Functions

- `mo.md(text)` - display markdown
- `mo.stop(predicate, output=None)` - stop execution conditionally
- `mo.output.append(value)` - append to the output
- `mo.output.replace(value)` - replace the output
- `mo.Html(html)` - display HTML
- `mo.image(image)` - display an image
- `mo.hstack(elements)` - stack elements horizontally
- `mo.vstack(elements)` - stack elements vertically
- `mo.tabs(elements)` - create a tabbed interface

### Troubleshooting

- **Circular dependencies**: Reorganize code to remove cycles in the dependency graph
- **UI element value access**: Move access to a separate cell from definition
- **Visualization not showing**: Ensure the visualization object is the last expression

After generating a notebook, run `marimo check --fix` to catch and automatically resolve common formatting issues.

---

# Well Filtering Feature Implementation

**Date**: 2025-12-03
**Feature**: Filter datasets based on key-value pair metadata attached to wells in plates

## Summary

Implemented well filtering functionality to allow filtering plate datasets based on map annotation key-value pairs attached to individual wells (e.g., cellline: U2OS, HeLa).

## Changes Made

### 1. Schema Updates ([annotation_config.py:270-280](src/omero_annotate_ai/core/annotation_config.py#L270-L280))

Added two new fields to `OMEROConfig`:
- `well_filters`: Optional Dict[str, List[str]] - Filter criteria with AND logic
- `well_filter_mode`: Literal["include", "exclude"] - Include or exclude matching wells

### 2. Pipeline Methods ([annotation_pipeline.py:836-938](src/omero_annotate_ai/core/annotation_pipeline.py#L836-L938))

Added three new methods to `AnnotationPipeline`:
- `_get_well_map_annotations(well_id)`: Retrieves map annotations from a well as dict
- `_check_well_filter(well_kv_pairs, filters)`: Checks if well matches filter criteria (AND logic)
- Updated `get_image_ids_from_container()`: Applies well filtering when container_type is "plate"

### 3. Tests ([tests/test_well_filtering.py](tests/test_well_filtering.py))

Created comprehensive test suite with 11 tests:
- Unit tests for filter matching logic
- Tests for map annotation retrieval
- Tests for include/exclude modes
- Integration tests for full filtering workflow

## Usage

### In Notebook
```python
config = workflow_widget.get_config()
config.omero.well_filters = {
    "cellline": ["U2OS", "HeLa"],
    "treatment": ["Control"]
}
config.omero.well_filter_mode = "include"
pipeline = create_pipeline(config, conn)
```

### In YAML
```yaml
omero:
  container_type: plate
  container_id: 1
  well_filters:
    cellline: ["U2OS", "HeLa"]
    treatment: ["Control"]
  well_filter_mode: include
```

## Implementation Details

- **Filter Logic**: AND logic - all conditions must be met
- **Metadata Location**: Map annotations on Well objects (navigated via Image → WellSample → Well)
- **Filter Modes**:
  - `include`: Only process wells matching ALL criteria
  - `exclude`: Process all wells EXCEPT those matching criteria
- **No Filtering**: If `well_filters` is None or empty, all wells are processed

## Test Results

All 99 tests pass (93 passed, 6 skipped):
- 11 new well filtering tests
- All existing tests remain passing
- No breaking changes

## Files Modified

1. `src/omero_annotate_ai/core/annotation_config.py` - Added schema fields
2. `src/omero_annotate_ai/core/annotation_pipeline.py` - Added filtering logic
3. `tests/test_well_filtering.py` - New test file (auto-discovered by pytest)

---

# Multi-Container Support Feature Implementation

**Date**: 2025-01-15
**Feature**: Support multiple OMERO containers of the same type with shared table attachment

## Summary

Extended the annotation pipeline to support multiple container IDs of the same type (e.g., multiple plates or datasets), with the tracking table attached to ALL source containers via OMERO AnnotationLink objects.

## Changes Made

### 1. Schema Updates ([annotation_config.py:417-480](src/omero_annotate_ai/core/annotation_config.py#L417-L480))

Added new field and helper methods to `OMEROConfig`:
- `container_ids`: Optional List[int] - List of container IDs (takes precedence over container_id)
- `get_all_container_ids()`: Returns all container IDs with precedence logic
- `get_primary_container_id()`: Returns the first container ID
- `is_multi_container()`: Checks if multiple containers are configured

### 2. Pipeline Methods ([annotation_pipeline.py](src/omero_annotate_ai/core/annotation_pipeline.py))

Added/updated methods in `AnnotationPipeline`:
- `_get_image_ids_from_plate(container_id)`: Extracts plate-specific logic for reuse
- `_get_image_ids_from_single_container(container_type, container_id)`: Gets images from one container
- Updated `get_image_ids_from_container()`: Iterates over all containers, removes duplicates
- Updated `_get_table_title()`: Reflects multi-container in auto-generated titles

### 3. OMERO Functions ([omero_functions.py](src/omero_annotate_ai/omero/omero_functions.py))

Added new function and updated existing:
- `link_table_to_containers(conn, table_id, container_type, container_ids)`: Links FileAnnotation to multiple containers using AnnotationLink objects
- Updated `create_or_replace_tracking_table()`: Accepts `container_ids` list, links to all containers
- Updated `sync_config_to_omero_table()`: Supports multi-container parameter

### 4. Tests

Tests added to existing test files:
- `tests/test_config.py`: TestOMEROConfigMultiContainer class (12 tests)
- `tests/test_well_filtering.py`: TestMultiContainerImageRetrieval, TestMultiContainerTableTitle classes (8 tests)
- `tests/test_omero_functions.py`: TestLinkTableToContainers, TestCreateOrReplaceTrackingTableMultiContainer classes (5 tests)

## Usage

### In Notebook
```python
config = workflow_widget.get_config()
config.omero.container_ids = [123, 456, 789]  # Multiple plates/datasets
config.omero.container_type = "plate"
# Well filters apply to ALL plates
config.omero.well_filters = {"cellline": ["U2OS"]}
pipeline = create_pipeline(config, conn)
```

### In YAML
```yaml
omero:
  container_type: plate
  # Multiple containers (new):
  container_ids: [123, 456, 789]

  # Single container (legacy, still works):
  # container_id: 123

  # Well filtering applies to ALL plates
  well_filters:
    cellline: ["U2OS", "HeLa"]
  well_filter_mode: include
```

## Implementation Details

- **Precedence**: `container_ids` takes precedence over `container_id` when both are set
- **Backward Compatibility**: Single `container_id` configs continue to work unchanged
- **Table Attachment**: Uses OMERO AnnotationLink objects (e.g., PlateAnnotationLinkI) to attach the same FileAnnotation to multiple containers
- **Duplicate Removal**: Images appearing in multiple containers are deduplicated (first occurrence wins)
- **Well Filtering**: Filters apply globally to all plate containers

## OMERO API for Multi-Container Table Linking

The `link_table_to_containers()` function uses OMERO's AnnotationLink classes:
```python
import omero.model as model

link_classes = {
    'dataset': (model.DatasetAnnotationLinkI, model.DatasetI),
    'plate': (model.PlateAnnotationLinkI, model.PlateI),
    'project': (model.ProjectAnnotationLinkI, model.ProjectI),
    'screen': (model.ScreenAnnotationLinkI, model.ScreenI),
}

# table_id is the FileAnnotation ID
file_ann = conn.getObject("FileAnnotation", table_id)
LinkClass, ContainerClass = link_classes[container_type]

for container_id in container_ids:
    link = LinkClass()
    link.setParent(ContainerClass(container_id, False))
    link.setChild(file_ann._obj)
    update_service.saveAndReturnObject(link)
```

## Files Modified

1. `src/omero_annotate_ai/core/annotation_config.py` - Added `container_ids` field and helper methods
2. `src/omero_annotate_ai/core/annotation_pipeline.py` - Multi-container image retrieval and table title
3. `src/omero_annotate_ai/omero/omero_functions.py` - Added `link_table_to_containers()`, updated table creation
4. `tests/test_config.py` - Added TestOMEROConfigMultiContainer tests
5. `tests/test_well_filtering.py` - Added multi-container pipeline tests
6. `tests/test_omero_functions.py` - Added multi-container OMERO function tests

---

# Folder Layout

**Date**: 2026-07-13

## Annotation phase

`AnnotationPipeline`, under `config.output.output_directory`:

- `annotation_input/{id}.tif` — the channel you annotate (e.g. fluorescence)
- `model_input/{id}.tif` — the channel the model consumes (e.g. brightfield); separate-channel workflows only
- `annotation_output/{id}_mask.tif` — the mask you produced
- `sam_embeddings/` — micro-SAM only

Single-channel workflows have no `model_input/`; `annotation_input/` serves both roles.

## Training phase

In a **separate** directory, defaulting to the sibling `<annotation_dir>_training/`:

- `train_input/{id}.tif`, `train_label/{id}.tif`
- `val_input/{id}.tif`, `val_label/{id}.tif`
- `test_*` when `include_test=True`
- `train_annotation_input/`, `val_annotation_input/` for separate-channel workflows

The training directory must NOT be inside the annotation directory — `prepare_training_data*`
raises `ValueError` if it is, because cleaning the training layout there would delete the
source images.

## Rules that hold everywhere

Files are named by `annotation_id`, so an image and its label always pair by name. A record
whose image or label is missing is dropped whole — never written as an orphan. (Consumers such
as micro-SAM pair `raw_paths` to `label_paths` by sorted filename, so a single orphan silently
shifts every subsequent pair.)

`processing/training_layout.py` owns the layout. Both producers
(`prepare_training_data_from_table`, `reorganize_local_data_for_training`) resolve their inputs
into `AnnotationRecord`s and hand them to `write_training_layout()`. `file_mode`
(copy/move/symlink) applies only to the offline producer — the OMERO producer fetches planes,
so there is no file to link. Symlinks fall back to a copy where unavailable (Windows without
Developer Mode). `layout="cellpose"` is specified but raises `NotImplementedError`.

Both producers return `train_input` / `train_label` / `val_input` / `val_label`, which is what
`setup_training()` requires.

## OMERO identifiers

Disk and OMERO names match: `annotation_input_id` (table column),
`openmicroscopy.org/omero/annotate/annotation_input` (namespace),
`upload_annotation_input_image()`. Tables written before the rename carry `label_input_id`;
`from_dataframe()` accepts either name, since those tables live on users' servers and cannot be
migrated. That read-side fallback is the only backwards-compatibility shim in the codebase.

---

When writing an anywidget use vanilla javascript in `_esm` and do not forget about `_css`. The css should look bespoke in light mode and dark mode. Keep the css small unless explicitly asked to go the extra mile. When you display the widget it must be wrapped via `widget = mo.ui.anywidget(OriginalAnywidget())`.

<example title="Example anywidget implementation">
import anywidget
import traitlets


class CounterWidget(anywidget.AnyWidget):
    _esm = """
    // Define the main render function
    function render({ model, el }) {
      let count = () => model.get("number");
      let btn = document.createElement("button");
      btn.innerHTML = `count is ${count()}`;
      btn.addEventListener("click", () => {
        model.set("number", count() + 1);
        model.save_changes();
      });
      model.on("change:number", () => {
        btn.innerHTML = `count is ${count()}`;
      });
      el.appendChild(btn);
    }
    // Important! We must export at the bottom here!
    export default { render };
    """
    _css = """button{
      font-size: 14px;
    }"""
    number = traitlets.Int(0).tag(sync=True)

widget = mo.ui.anywidget(CounterWidget())
widget

# Grabbing the widget from another cell, `.value` is a dictionary.
print(widget.value["number"])
</example>

When sharing the anywidget, keep the example minimal. No need to combine it with marimo ui elements unless explicitly stated to do so.

---

# Napari Plugin

**Branch**: `feat/napari-plugin`
**Worktree**: `../omero_annotate_ai.worktrees/napari-plugin`

Plugin lives in `src/omero_annotate_ai/napari_plugin/`:
- `napari.yaml` — npe2 manifest (points directly at `OMEROAnnotateWidget`)
- `_widget.py` — `OMEROAnnotateWidget(QWidget)`: 3 tabs (Connection / Configure / Run)
- `_worker.py` — `AnnotationWorker(QRunnable)`: OMERO setup stages run in background thread; emits `ready_to_annotate` signal so `run_microsam_annotation()` executes on the main thread (required for Qt/GL)

Key threading rule: `image_series_annotator` creates napari layers and **must run on the main thread**. The worker only handles OMERO I/O (`initialize_workflow`, `define_annotation_schema`, `create_tracking_table`).

`annotation_pipeline.py` has a `_napari_viewer` attribute (default `None`). When set by the plugin, `image_series_annotator` reuses the existing viewer and `napari.run()` is skipped. Notebook behaviour is unchanged.

npe2 entry point requires the annotation to be a live type (not a string): `napari_viewer: napari.Viewer` in `__init__`, no `from __future__ import annotations`.

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

Offline verification is content-addressed: every image (and `.zarr` store) under
`data_dir` is coded, and each stored code must appear in that set. Filenames are never
consulted, so renaming a published file does not break verification. A file that cannot
be read produces a warning ("verification incomplete"), not a mismatch.

Mismatch is an **error**; a code that was never stamped is a **warning** — absence of
evidence is not evidence of tampering.

`verify_config` also refuses (rather than returning a verdict) when it cannot verify at
all: a `data_dir` that is not a directory raises `ValueError`, and a `data_dir` in which
*nothing codeable* is found raises `RuntimeError`. Otherwise an empty scan — a typo'd
path, an empty folder — would make every stored code look absent and be reported as a
mismatch, which is the same "accuse good data" bug in a different disguise. A `data_dir`
pointing directly *at* a `.zarr` store root is coded as that store, not scanned.

## `stamp_config` vs `verify_config`: deliberately asymmetric failure behaviour

When `iscc-bio` is not installed:

- **`stamp_config` stays best-effort and silent.** Codes stay `None`, one warning is
  logged, and the annotation run is never harmed. Stamping is a nice-to-have during a run.
- **`verify_config` raises `RuntimeError`** with an install hint, in *both* `conn` and
  `data_dir` modes. It does not return a `ValidationResult`. Verifying is the whole point
  of this feature: with the library absent nothing can actually be recomputed, so a
  returned "result" would be a verdict with nothing behind it. (An earlier version
  returned a result here and reported every image as a *mismatch* — i.e. it accused good,
  untampered data of being tampered with. A verdict you cannot back up is worse than no
  verdict.)

## Install

`iscc-bio` is optional and pinned `>=0.1,<0.2` (it is PoC software that declares breaking
changes may ship at any time):

```bash
pip install 'omero-annotate-ai[provenance]'
```

Without it, stamping runs unchanged, codes stay `None`, and one warning is logged;
verifying raises instead (see above). Set `iscc_mode: "on"` in the config to enable
stamping from the pipeline — it is **off** by default.

## Pipeline hook

`AnnotationPipeline._stamp_provenance()` is called at the top of `_finalize_workflow()`
(`annotation_pipeline.py`), before `_auto_save_config()`. It is a no-op unless
`config.iscc_mode == "on"`, and it swallows any exception from `stamp_config()` (printing
a warning) so provenance can never cost us the annotations. Both `run_microsam_annotation`
and `run_custom_annotation` converge on `_finalize_workflow`, so this covers both. Note:
`run_cellpose_preparation` does not route through `_finalize_workflow` (it only prepares
local tiles and produces no masks), so it is not stamped — Cellpose datasets get
provenance through the retroactive `stamp_config()` path instead.
