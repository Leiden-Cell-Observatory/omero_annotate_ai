# Share via the BioImage Archive

Annotations are most useful when others can reuse them. The
[BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) accepts annotation datasets
described with the [MIFA](https://arxiv.org/abs/2311.10443) metadata standard.
OMERO Annotate AI writes a ready-to-upload bundle: images, masks, file lists and the MIFA
metadata files.

This needs the BioImage Archive's metadata models, which are not on PyPI:

```bash
pip install "git+https://github.com/BioImage-Archive/bia-mifa-models.git@v1.1.5"
```

## 1. Describe the set

The archive's MIFA metadata comes from the config's [schema sections](../configuration.md#schema-sections).
MIFA requires a study title, description and funding statement, keywords, at least one
link (or a `source_dataset.url`) and a version. Fill them in the YAML before creating the
set:

```yaml
license: CC-BY-4.0               # CC-BY-4.0 or CC0-1.0
version: "1"
keywords: [nuclei, segmentation, cell painting]
links:
  - {url: "https://github.com/example/hela-nuclei", description: Analysis code}
study:
  title: "Nuclei annotations in HeLa Cell Painting images"
  description: "Instance masks of nuclei, annotated with micro-SAM and corrected by hand."
  organism: "Homo sapiens"
  imaging_method: "widefield fluorescence microscopy"
  funding_statement: "Funded by NWO."
source_dataset:
  description: "HeLa cells, Hoechst channel"
methodology:
  annotation_types: [segmentation_mask]
  mask_kind: instance
  method: semi_automatic
  criteria: "Every nucleus fully inside the field; touching nuclei separated."
  coverage: representative
```

Or add them to an existing set with `describe`, which saves them locally and on OMERO:

```python
from omero_annotate_ai import TrainingSet

ts = TrainingSet.open(table_id, conn)
ts.describe(
    version="1",
    keywords=["nuclei", "segmentation", "cell painting"],
    links=[{"url": "https://github.com/example/hela-nuclei"}],
    study={
        "title": "Nuclei annotations in HeLa Cell Painting images",
        "description": "Instance masks of nuclei, annotated with micro-SAM.",
        "funding_statement": "Funded by NWO.",
    },
)
```

When a required field is missing, `save_bia_package` raises `MifaNotReady`
(`from bioimage_annotation_schema.mifa import MifaNotReady`) and lists the missing
fields; it does not fill in placeholders.

## 2. Export from OMERO

```python
from omero_annotate_ai import TrainingSet, prepare_bia_data_from_table
from omero_annotate_ai.schema_export import save_bia_package

ts = TrainingSet.open(table_id, conn)
stats = prepare_bia_data_from_table(conn, table_id, "bia_staging")

summary = save_bia_package(
    ts.config, stats["rows"], "bia_staging", "bia_bundle", move_data=True
)
```

`prepare_bia_data_from_table` downloads the finished units in their original bit depth;
`save_bia_package` writes the bundle. `accession=` sets the name of the metadata files
(default: `source_dataset.identifier`, else the set's name).

## From a local training set

```python
from omero_annotate_ai import TrainingSet
from omero_annotate_ai.schema_export import save_bia_package

ts = TrainingSet.open("~/omero_annotate_ai/local/<id>_hela_nuclei_v1")
save_bia_package(
    ts.config, ts.units(), ts.bundle_dir, "bia_bundle"
)  # annotated units only
```

The MIFA export comes from the
[Bioimage Annotation Schema](https://github.com/Leiden-Cell-Observatory/bioimage-annotation-schema)
library (`bioimage_annotation_schema.mifa`). Its `bas` command checks a set's folder
after `ts.write_ro_crate()` has written `annotation-set.yaml` and
`annotation-records.csv` there:

```bash
bas validate --mifa ~/omero_annotate_ai/local/<id>_hela_nuclei_v1
```

It says *Ready for MIFA* or lists what is missing.

## What is in the bundle

```text
bia_bundle/
├── metadata/
│   ├── Study_<accession>.yaml
│   ├── Annotations_<accession>.yaml
│   └── Version_<accession>.yaml
├── file_list_images.tsv          # with pixel_sha256 and iscc per file
├── file_list_annotations.tsv     # with pixel_sha256 and iscc per file
├── images/<unit id>.tif
└── annotations/<unit id>_mask.tif
```

Upload it following the
[BioImage Archive submission guide](https://www.ebi.ac.uk/bioimage-archive/submit/).

Notebook: [OMERO table to BioImage Archive](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/export_annotations/omero-table-to-bioimage-archive.ipynb)
