# Resume or repeat a run

Every run is described by its YAML configuration. That file is how you continue an
interrupted run, repeat a run on new data, or let a colleague reproduce your annotation
set.

## Resume an interrupted run

A training set tracked in OMERO is identified by its tracking table's ID, which never
changes. To continue it, open it by that ID:

```python
from omero_annotate_ai import TrainingSet

ts = TrainingSet.open(12345, conn)  # the table ID, shown in OMERO.web on the container
ts.annotate(note="second session")
```

The units are read back from the table and only the unfinished ones are opened, with
the same images and patches as before. A local set (`storage: local`) is opened by its
folder: `TrainingSet.open("~/omero_annotate_ai/local/<id>_<name>")`. In the widget,
choose *Continue Existing* on the table instead.

On an opened set you can change only what does not alter the existing rows:

- more images: `ts.extend(train_n=10, validate_n=4)` adds images to the same table
  (counts can only be raised);
- the model and batch size of a session: `ts.annotate(model="vit_l_lm", batch_size=4)`.

### Session notes

Every session is recorded in the set's `config.yaml` (and its copy on OMERO): who
annotated, when, how many units, and a free-text note.

```python
ts.annotate(note="nuclei only; small debris skipped")
ts.set_session_note(2, "image 14 needs another look")  # add or change a note later
print(ts.config.sessions)
```

A session that stopped without finishing (napari closed, kernel died) is marked
`interrupted` the next time the set is opened on the same computer.

### Correct earlier annotations

`ts.review()` (or *Review annotated* in the widget) opens annotated units with their saved
masks so you can correct them; a changed mask replaces the upload on OMERO. Replacing is
crash-safe too: the new label file and ROI are uploaded before the old ones are deleted,
and opening the set finishes a replacement a crash interrupted. See the
[micro-SAM tutorial](microsam-annotation-pipeline.md#correct-earlier-annotations).

### Several computers, one set

The OMERO tracking table is the source of truth; the local folder is a working copy.
Open the set by its table ID on another computer and carry on: rows come from the
table, and nothing is uploaded twice.

Two people (or two computers) can also annotate the same set **at the same time**.
Before napari shows a batch (also in a review), the session claims its units in the table (columns
`claimed_by`, `claimed_at`), so the other session gets different units. Claims are
released when a unit is saved or handed back, and expire after
`workflow.claim_hours` (default 6) if a computer goes quiet. Opening the set again on
the computer that crashed releases its claims at once.

If a claim expired and someone else saved the unit first, your mask is not uploaded
over theirs: it is kept in `annotation_output/conflicts/` and the session reports it.
Local sets (`storage: local`) have no claims; their folder belongs to one computer.

### What resuming repairs

Each mask is saved in a fixed order: moved into the set's folder, recorded in the local
`tracking_table.csv`, then uploaded (label file, then ROI) with a tag naming the table
and unit, and finally written to the OMERO table. If a run stops halfway - napari
closed, kernel killed, network gone - resuming compares the OMERO table, the local copy
and the mask files for every unit and finishes the job:

- uploads the table never recorded are found by their tag and adopted, never uploaded
  a second time;
- masks saved locally but not uploaded are uploaded;
- a unit whose upload failed is tried again (the error is kept in the local
  `tracking_table.csv`, column `error`).

The local folder can even be lost: the uploads carry their tag, so resuming still
finds them.

### Pixel checks

Every unit records a SHA-256 digest of its pixels: `input_digest` for the image you
annotated and `mask_digest` for your mask. Resuming warns when a local file no longer
matches (for example a mask edited after upload), and
`prepare_training_data_from_table` stops with an `IntegrityError` if a label file on
OMERO changed after it was annotated. Nothing is overwritten silently.

If a mask in your folder and its upload on OMERO differ, the unit is marked
`needs review` and left out of new sessions. Decide which one is right:

- delete the local mask (`annotation_output/<unit id>_mask.tif`) to accept the one on
  OMERO, or
- delete the label file on OMERO to keep your local mask,

then resume the set again.

## Repeat a run from a YAML file

To annotate a different dataset with the same settings, or to run without widgets (for
example from a script), start from a YAML file. Take one you saved from the widget, or
generate a commented template:

```python
from omero_annotate_ai.core.annotation_config import get_config_template

print(get_config_template())
```

If you copy a set's `config.yaml`, remove its `identity` and `sessions` sections, change
`name` and the `omero` container, then:

```python
from omero_annotate_ai import TrainingSet, load_config_from_yaml

config = load_config_from_yaml("my_run.yaml")  # the file itself is not changed
ts = TrainingSet.create(config, conn)
ts.annotate()
```

!!! tip "Typos are errors"
    Unknown keys are refused when the file is loaded, and keys from older versions name
    their replacement (for example `workflow.read_only_mode` → `storage: local`), so a
    misspelt setting never goes unnoticed.

Notebook: [annotate from YAML (marimo)](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/annotate-from-yaml.py)

## Check progress

```python
from omero_annotate_ai.rows import rows_to_frame
from omero_annotate_ai.trainingset import summarize

print(summarize(ts.table_id, conn).progress)  # per split: (annotated, total), read-only
todo = [row for row in ts.units() if not row.processed]
df = rows_to_frame(ts.units())  # one row per unit, as a pandas table
```
