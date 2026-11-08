# Try it on public data

The [Image Data Resource (IDR)](https://idr.openmicroscopy.org) publishes reference
image sets from published studies. Many studies are also available as **OME-Zarr** on
public S3 storage, which OMERO Annotate AI reads directly (`source: zarr`). No OMERO
login is needed, and because nothing can be written to the IDR, the training set is
kept on your computer (`storage: local`).

The example used below is HeLa nuclei (histone H2B-GFP) from the MitoCheck screen,
study [idr0013](https://idr.openmicroscopy.org/webclient/?show=screen-1101): 2D
time-lapses of 1024 × 1344 pixels, with nuclei about 25 pixels across, a good size for
micro-SAM's `vit_b_lm` model.

## With the demo notebook

The [IDR demo notebook](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/idr-zarr-demo.py)
(marimo) checks a screen for OME-Zarr plates, creates the training set, opens napari
and trains a short micro-SAM run, with everything pre-filled.

```bash
pixi run marimo edit notebooks/marimo/idr-zarr-demo.py
```

## With a YAML file

The same example as a config,
[`idr0013-mitocheck-zarr.yaml`](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/configs/idr0013-mitocheck-zarr.yaml):

```yaml
name: idr0013-mitocheck-nuclei
source: zarr
zarr:
  uris:
    - https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0013A/3451.zarr
  level: 0          # full resolution
  fields: first
  wells: [A1, B2, C3, D4]
spatial_coverage:
  channels: [0]     # H2B-GFP: nuclei
  timepoint_mode: random
  n_timepoints: 1   # one random frame of each time-lapse
training:
  train_n: 2
  validate_n: 1
  seed: 42
storage: local
```

Load it in the
[annotate-from-YAML notebook](https://github.com/Leiden-Cell-Observatory/omero_annotate_ai/blob/main/notebooks/marimo/annotate-from-yaml.py),
or in Python:

```python
from omero_annotate_ai import TrainingSet, load_config_from_yaml

ts = TrainingSet.create(load_config_from_yaml("idr0013-mitocheck-zarr.yaml"))
ts.annotate(note="first try on IDR data")
```

Only metadata is read when the set is created; the pixels of each chosen plane are
downloaded when it is annotated.

## Finding other data

Only images converted to OME-Zarr can be used, and often only some plates of a study
are converted (in idr0013A, for example, plate 3451 but not its neighbours). The
[IDR OME-NGFF samples](https://idr.github.io/ome-ngff-samples/) page lists what is
available (studies with plates include idr0001A, idr0013A, idr0056B, idr0072B and
idr0128E). A plate's
OME-Zarr address is

```text
https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/<study folder>/<plate id>.zarr
```

where the study folder is the study name with its screen letter (`idr0013A`) and the
plate ID is the one the IDR shows. Plate IDs of a screen come from the IDR web API, which
needs no login:

```python
import requests

screen_id = 1101  # idr0013-neumann-mitocheck/screenA
plates = requests.get(
    "https://idr.openmicroscopy.org/webclient/api/plates/",
    params={"id": screen_id},
    timeout=30,
).json()["plates"]
for plate in plates[:5]:
    url = f"https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0013A/{plate['id']}.zarr"
    ok = requests.head(url + "/.zattrs", timeout=15).status_code == 200
    print(plate["id"], plate["name"], url if ok else "no OME-Zarr")
```

Pick the channel from the study's description (the OME-Zarr metadata lists channel
labels too), choose wells with `zarr.wells` or a metadata CSV with
[well filters](../configuration.md#zarr-ome-zarr-images-instead-of-omero), and use `zarr.level` to work at a lower
resolution when objects are large.

## What you get

The set's folder (`~/omero_annotate_ai/local/<id>_<name>/`) holds:

```text
annotation_input/     the images you annotated, as TIFF
annotation_output/    your masks, <unit id>_mask.tif
tracking_table.csv    which units were done, and their split
config.yaml           the set's config, sessions and notes
```

These go straight into [training](training-data-prep.md#export-a-training-set).
