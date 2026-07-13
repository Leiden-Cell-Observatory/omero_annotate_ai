# Tutorial: Content Provenance (ISCC)

When you publish an annotation dataset, how does anyone downstream know the images and
masks they received are the ones the config actually describes — without trusting you,
and without an OMERO connection?

OMERO Annotate AI can record [ISO 24138](https://www.iso.org/standard/77899.html)
content-provenance codes ("ISCC") for every annotation. A code is a hash-like fingerprint
of an image's raw pixels. Anyone holding the published files and the `config.yaml` can
recompute the codes themselves and confirm the data matches — fully offline, no server,
no trust required.

## Prerequisites

!!! note "Install the provenance extra"
    ```bash
    pip install 'omero-annotate-ai[provenance]'
    ```
    This installs `iscc-bio`, pinned to `>=0.1,<0.2`. Without it, stamping is silently
    skipped (codes stay `null`) and verification refuses to run — see
    [Error handling](#error-handling) below.

## The one thing to understand first

**Codes are computed on raw source pixels only** — never on the exported 8-bit training
tiles.

The ISCC algorithm used here (IMAGEWALK) is an *exact* code: it packs raw dtype bytes
with no bit-depth or intensity canonicalization, matching OMERO's own pixel
representation bit-for-bit. That gives it a useful property and a sharp limitation:

- It **survives a change of container format** — the same image exported as OME-TIFF,
  OME-Zarr, or read back from OMERO all produce the same code.
- It does **not** survive a change of pixel values. The training-tile export path
  rescales intensities (`img * 255/max`) and restacks channels, so a tile's code will
  **never** match its source image's code. Provenance is about the source image and the
  annotation mask, not the derived training crop.

## 1. Enable stamping during a run

Set `iscc_mode: "on"` (default is `"off"`) before running the pipeline:

```yaml
# config.yaml
iscc_mode: "on"
```

```python
config.iscc_mode = "on"
pipeline = create_pipeline(config, conn)
pipeline.run_microsam_annotation()
```

After the run, each annotation in `config.yaml` carries its codes:

```yaml
annotations:
  - image_id: 101
    source_iscc: "ISCC:..."   # source image's raw OMERO pixels
    label_iscc: "ISCC:..."    # annotation mask
```

Stamping happens once processing is complete (masks already uploaded), and just before
the config is persisted, so the codes land in both `config.yaml` and the OMERO tracking
table.

## 2. Retroactively stamp an old config

`stamp_config` is idempotent, so it can be pointed at a `config.yaml` from a run that
predates this feature (or one where `iscc_mode` was left `"off"`):

```python
from omero_annotate_ai.core.annotation_config import load_config
from omero_annotate_ai.processing.provenance import stamp_config

config = load_config("config.yaml")
stamp_config(config, conn)
config.save_yaml("config.yaml")
```

Codes that are already present are left alone, and each unique `image_id` / `label_id`
is coded once and cached — a source image shared by many patches or z-slices is not
re-coded per row.

`stamp_config` is **best-effort**: if `iscc-bio` is not installed, or a particular image
can't be coded, it logs a warning and leaves that field `None`. It never raises, so
running it never costs you the underlying annotation data.

## 3. Verify as the author (online)

With a live OMERO connection, recompute each stored code from the server and compare:

```python
from omero_annotate_ai.processing.provenance import verify_config

result = verify_config(config, conn=conn)
print(result.summary)
print(result.is_valid)
```

## 4. Verify as a recipient (fully offline)

This is the scenario the feature exists for: a third party has the published images and
`config.yaml`, nothing else — no OMERO, no network.

```python
result = verify_config(config, data_dir="published_dataset/")
print(result.summary)
```

Offline verification is **content-addressed, not filename-addressed**. Every image file
under `data_dir` is coded (`.zarr` stores are matched as directories, not by their
internal chunk files), and each stored `source_iscc` / `label_iscc` only needs to appear
*somewhere* in that set. Filenames are never consulted — renaming a published file does
not break verification.

## Understanding the result: three outcomes

`verify_config` returns a `ValidationResult` with `is_valid`, `errors`, and `warnings`:

| Outcome | Meaning | Effect |
|---|---|---|
| **Match** | A file in `data_dir` (or OMERO) codes to the stored value | Nothing reported |
| **Mismatch** | A code is stored, but no file in the data matches it | **Error** — `is_valid` becomes `False` |
| **Missing** | The annotation was never stamped (`source_iscc`/`label_iscc` is `null`) | **Warning** — `is_valid` stays `True` |

A file that exists but can't be read (corrupt, permission error, unsupported format)
also produces a warning — "verification incomplete" — never a mismatch. Absence of
evidence is not evidence of tampering: only a code that is present and *disagrees* with
the data counts as a real mismatch.

## Error handling

`stamp_config` and `verify_config` are deliberately asymmetric:

- **`stamp_config` is best-effort and silent.** No `iscc-bio`, or a failure computing a
  particular code, means the field stays `None` and one warning is logged. An annotation
  run is never harmed by provenance failing — stamping is a nice-to-have.
- **`verify_config` raises `RuntimeError` if `iscc-bio` is not installed — in both `conn`
  and `data_dir` modes.** It does not return a result. With no library, nothing can
  actually be recomputed, so any verdict (green or red) would be reporting a check that
  never happened. Verifying is the whole point of this feature, so a bogus verdict —
  such as an early version of this code that reported every image as a *mismatch* simply
  because it couldn't check anything — is unacceptable.

`verify_config` also requires **exactly one** of `conn` / `data_dir`; passing neither or
both raises `ValueError`.

## Next steps

- [Configuration Guide](../configuration.md#content-provenance-iscc) — `iscc_mode` and
  the `source_iscc` / `label_iscc` schema fields
- [Installation Guide](../installation.md#optional-content-provenance-iscc) — installing
  the `provenance` extra
