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

## 1. Describe the study

The config has three metadata sections used for the archive: `study`, `dataset` and
`annotation_methodology`. Fill them in the YAML before annotating, or set them just before
export:

```yaml
study:
  title: "Nuclei annotations in HeLa Cell Painting images"
  description: "Instance masks of nuclei, annotated with micro-SAM and corrected by hand."
  keywords: [nuclei, segmentation, cell painting]
  organism: "Homo sapiens"
  imaging_method: "widefield fluorescence microscopy"
  funding_statement: "Funded by NWO."
dataset:
  source_description: "HeLa cells, Hoechst channel"
  license: "CC-BY-4.0"           # CC0 or CC-BY
annotation_methodology:
  annotation_type: segmentation_mask
  annotation_method: semi_automatic
  annotation_criteria: "Every nucleus fully inside the field; touching nuclei separated."
  annotation_coverage: representative
```

## 2. Export from OMERO

```python
from omero_annotate_ai import prepare_bia_data_from_table
from omero_annotate_ai.core.mifa_export import save_bia_package
from omero_annotate_ai.omero.omero_functions import find_annotation_config_for_table

config = find_annotation_config_for_table(conn, table_id)  # the set's config
stats = prepare_bia_data_from_table(conn, table_id, "bia_staging")

summary = save_bia_package(
    config, stats["rows"], "bia_staging", "bia_bundle", move_data=True
)
```

`prepare_bia_data_from_table` downloads the finished units in their original bit depth;
`save_bia_package` writes the bundle.

## From a local read-only run

```python
from omero_annotate_ai import TrainingSet
from omero_annotate_ai.core.mifa_export import save_bia_package

ts = TrainingSet.open("~/omero_annotate_ai/local/<id>_hela_nuclei_v1")
done = [row for row in ts.units() if row.processed]  # leave out unfinished units
save_bia_package(ts.config, done, ts.bundle_dir, "bia_bundle")
```

## What is in the bundle

```text
bia_bundle/
├── metadata/
│   ├── Study_<accession>.yaml
│   ├── Annotations_<accession>.yaml
│   └── Version_<accession>.yaml
├── file_list_images.tsv
├── file_list_annotations.tsv
├── images/<unit id>.tif
└── annotations/<unit id>_mask.tif
```

Upload it following the
[BioImage Archive submission guide](https://www.ebi.ac.uk/bioimage-archive/submit/).

Notebook: [OMERO table to BioImage Archive](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/jupyter/export_annotations/omero-table-to-bioimage-archive.ipynb)
