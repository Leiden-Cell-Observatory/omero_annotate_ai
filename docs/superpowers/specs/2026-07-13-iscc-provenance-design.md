# ISCC Content Provenance for Published Annotation Datasets

**Date:** 2026-07-13
**Status:** Approved design, ready for implementation planning

## Problem

When we publish an annotation dataset (original images + annotations), nothing in the record
identifies the image *content* that was annotated. Image identity is carried only by the OMERO
`image_id` — an integer that is meaningless outside the originating server, and that says nothing
about the pixels it points at. A recipient of a published dataset cannot answer:

- Which exact images did these annotations come from?
- Are the masks I received the ones the authors actually produced?

A repo-wide search confirms there is no fingerprint of any kind today
(`grep hashlib|md5|sha256|checksum` → zero matches).

## Goal

Publication-grade, citable provenance. A third party receives the published dataset plus its
`config.yaml` and can verify, **offline and without OMERO access**, both which source images the
annotations derive from and that the annotation masks are unmodified.

Explicitly *not* goals (see Out of Scope).

## Why ISCC, and why the pixel-level code specifically

ISCC (ISO 24138:2024) is a content-based identifier. Two properties matter here:

1. **`iscc-bio` / IMAGEWALK is pixel-level and format-independent.** It traverses planes
   deterministically (Z→C→T), flattens each plane row-major, and packs raw dtype bytes big-endian.
   The same pixels stored as OMERO server pixels, OME-TIFF, or OME-Zarr therefore produce the *same*
   code. This is what makes an OMERO image and a published copy comparable.

2. **`iscc-sum` is byte-level and is the wrong tool here.** It fingerprints file bytes, so the same
   pixels in different container formats produce *different* codes. It cannot compare OMERO against a
   published copy. It is not used in this design.

### Critical property: IMAGEWALK is exact, not transform-invariant

Verified against the `iscc-bio` source. `plane_to_canonical_bytes` (`iscc_bio/imagewalk/common.py`)
performs **no dtype canonicalization and no intensity rescaling** — it packs the raw dtype bytes as-is.
The project's explicit aim is bit-for-bit identity with OMERO's own pixel representation
(`calculate_pixel_sha1_bioio` in `iscc_bio/canonical.py` is built to match OMERO's server-side SHA1,
and `scripts/imagewalk_check.py` verifies this).

Consequences that constrain the design:

- A uint16 image and its uint8 conversion produce **different** codes.
- Selecting or reordering channels changes the plane sequence and therefore the code.
- This pipeline's export is lossy: `img * 255/max` (`training_functions.py:920`, `:1025`) rescales
  intensities and casts to uint8, and channels are restacked (`annotation_pipeline.py:1592`).

Therefore **codes must be computed on raw source pixels, never on the exported training tiles.** This
is not a limitation to work around — it is why the source-side code is canonical and directly
comparable to what OMERO itself stores.

## Scope decisions

**What gets published:** original images + annotations describing which regions were annotated. The
8-bit normalized tiles are *ephemeral working artifacts* for micro-SAM/CellPose and are never shipped.
Hashing them would buy nothing, so we do not.

**Two codes, both via `iscc-bio`:**

| Field | Subject | Purpose |
|---|---|---|
| `source_iscc` | raw pixels of the source OMERO image | proves *which* images the annotations derive from |
| `label_iscc` | the published annotation mask | proves the masks the recipient holds are the ones we made |

**Region is already recorded.** The config already carries `is_patch`, `patch_x/y/width/height`,
`z_slice`, `timepoint`, `channel`. So `source_iscc` plus those coordinates fully specifies lineage:
*"this annotation came from the image with content-code X, region (x, y, w, h), at z/t/c."* Patches
need no separate code.

## Architecture

### Component: `src/omero_annotate_ai/processing/provenance.py`

A self-contained unit with no pipeline coupling. All ISCC logic lives here; every other touchpoint is
a thin call into it.

```
iscc_available() -> bool
    Guarded-import check for iscc-bio.

compute_image_iscc(conn, image_id) -> Optional[str]
    IMAGEWALK code over the raw OMERO pixels, via iscc-bio's OMERO/Blitz path.

compute_file_iscc(path) -> Optional[str]
    IMAGEWALK code for a local image file. Used for masks and for offline verification.

compute_label_iscc(conn, label_id) -> Optional[str]
    Masks are stored as OMERO file annotations (cf. ezomero.get_file_annotation,
    training_functions.py:1094). Fetch, then delegate to compute_file_iscc.

stamp_config(config, conn) -> AnnotationConfig
    Fill source_iscc and label_iscc for every annotation. Caches per image_id and per
    label_id — source images repeat across many annotation rows, so each is coded once.

verify_config(config, conn=None, data_dir=None) -> ValidationResult
    Recompute and diff against stored codes. Exactly one source must be given:
      - conn:     re-fetch from OMERO and recompute (author-side check).
      - data_dir: recompute from local image files (recipient-side, OFFLINE, no OMERO).
    The offline mode is the one that delivers the stated goal, so it is not optional.
    Three outcomes per entry: match / mismatch / missing. Reuses the existing
    ValidationResult shape (annotation_config.py:1145).
    Raises RuntimeError (not a ValidationResult) if iscc-bio is not installed, in both
    modes — see "Error handling" below for why this is intentionally NOT best-effort.
```

The `data_dir` mode is what a third party actually runs: they hold the published images and the
`config.yaml`, and nothing else. It is **content-addressed**: every image file under the directory is
coded, and each stored `source_iscc` / `label_iscc` must appear somewhere in that set. Filenames are
never consulted, so renaming a published file does not break verification — which is exactly the
property a content code is supposed to give us.

### Schema changes: `src/omero_annotate_ai/core/annotation_config.py`

- `ImageAnnotation` (`:58`): add `source_iscc: Optional[str] = None` and
  `label_iscc: Optional[str] = None`.
- Add a provenance toggle on the top-level `AnnotationConfig` (`:683`), not on `OMEROConfig` — it
  governs package behaviour, not OMERO connection details: `iscc_mode: Literal["off", "on"] = "off"`.
- Add both fields to the column lists in `to_dataframe()` (`:855-911`) and `from_dataframe()` (`:945`).
- Bump `schema_version`; document the new fields in `get_config_template()` (`:1323`).

### Storage: `config.yaml` is the provenance record

Adding the fields to `ImageAnnotation` places them in two locations for free:

- **`config.yaml`**, via `to_yaml()` (`:1037`) — the portable artifact that ships with the published
  dataset. A recipient reads it offline, with no OMERO access, and re-verifies. This is the record
  that matters.
- **The OMERO tracking table**, via `to_dataframe()`/`from_dataframe()` — the server-side mirror,
  obtained with no extra machinery.

Both fields must be added to the string-column handling in `_prepare_dataframe_for_omero`
(`omero_functions.py:311-337`) so they round-trip through `ezomero.post_table` / `get_table`.

Size impact is negligible: the `annotations` list is already one entry per annotation; each gains
roughly 80 characters.

### Data flow — three paths, one implementation

1. **Run path.** When `iscc_mode == "on"`, the pipeline calls `compute_image_iscc` once the
   `image_id` is resolved (`annotation_pipeline.py:496`, `:853`) and `compute_label_iscc` once the
   mask is uploaded and `label_id` is known. Codes persist through the existing
   `sync_config_to_omero_table` path (`omero_functions.py:474`) and `save_yaml`. The hook is small
   because the logic lives in the unit.

2. **Retrofit path.** `stamp_config(config, conn)` fills codes into any *existing* `config.yaml` from
   a past run. This matters: without it, every annotation set produced before this feature could
   never be published with provenance.

3. **Verify path.** `verify_config(config, conn)` recomputes and reports.

## Dependencies

`iscc-bio` is a proof of concept (v0.1.0; its own docs warn that "breaking changes may be released at
any time") and pulls in `bioio`. It is therefore an **optional extra**, pinned tightly:

```
[project.optional-dependencies]
provenance = ["iscc-bio>=0.1,<0.2"]
```

Codes are **always computed** via `iscc-bio`. We deliberately do not read codes that the server-side
`omero-iscc` service may have attached: one code path is simpler, and it avoids depending on a
namespace convention and on servers running that service. (Because `iscc-bio` matches OMERO
bit-for-bit, a server-supplied code would agree with ours anyway — so nothing is lost.)

This mirrors the existing `microsam` optional-group + guarded-import pattern already used in the
package.

## Error handling

Stamping and verifying have **deliberately asymmetric** failure behaviour when `iscc-bio` is missing:

- **`stamp_config` is best-effort and silent.** `iscc-bio` not installed → `iscc_available()` is
  False, codes stay `None`, one clear warning is logged, and the pipeline / annotation run is
  otherwise unaffected. Stamping is a nice-to-have during a run, so losing it must never cost us the
  annotations.
- **`verify_config` refuses to run.** If `iscc-bio` is not installed, it **raises `RuntimeError`**
  with an install hint — in both `conn` and `data_dir` modes — instead of returning a
  `ValidationResult`. Verifying is the whole point of this feature: with the library absent, nothing
  can actually be recomputed, so a returned "result" would be a verdict with nothing behind it. (An
  earlier version of this code returned a result in that case and reported every image as a
  *mismatch* — i.e. it accused good, untampered data of being tampered with. A verdict you cannot
  back up is worse than no verdict, so this now raises instead.)
- `iscc_mode == "off"` (the default) → no computation and no import attempt; zero runtime cost and no
  behaviour change for existing users. This only affects the stamping hook; `verify_config` can still
  be called directly regardless of `iscc_mode`.
- An OMERO fetch or compute error on one image (library present, one code fails) → that entry's code
  stays `None` with a warning; the run/verify continues.
- `verify_config` distinguishes **missing** from **mismatch**. A `None` (or unmatched, in `data_dir`
  mode) code is an absence of evidence, not evidence of tampering, and is reported as a warning, not
  an error. In `data_dir` mode, a file that cannot be read is likewise a warning ("verification
  incomplete"), never a mismatch.

## Testing

Following the test-file mapping in `CLAUDE.md`:

- **New `tests/test_provenance.py`** — mock `iscc_bio`. Cover: guarded-import graceful skip when the
  library is absent; per-`image_id` and per-`label_id` caching (assert one call per unique id, not per
  annotation row); `stamp_config` populates both fields; `verify_config` returns each of the three
  states in **both** `conn` and `data_dir` modes; passing neither `conn` nor `data_dir` (or both) is
  an error; a compute error on one image does not abort the pass.
- **`tests/test_config.py`** — the new fields round-trip through YAML, JSON, and
  `to_dataframe`/`from_dataframe`; `get_config_template()` stays valid.
- **`tests/test_omero_functions.py`** — `_prepare_dataframe_for_omero` types the two new string
  columns correctly.

## Verification

1. `pixi run -e dev pytest tests/ -v` — all pass, including the new provenance tests.
2. With `iscc_mode="on"`: run against an OMERO image, confirm `source_iscc` appears in `config.yaml`
   and in the tracking table, computed once per unique `image_id`.
3. **The core claim.** Export a *lossless* copy of a source image in a different container format
   (e.g. OME-TIFF → OME-Zarr) and confirm `compute_file_iscc(copy) == source_iscc`. This is the
   format-independence that the whole design rests on.
4. Alter the pixels of that copy; confirm `verify_config` reports **mismatch**. Clear a stored code;
   confirm it reports **missing**, not mismatch.
5. **Simulate the recipient.** With no OMERO connection at all, run
   `verify_config(config, data_dir=...)` against the published images and `config.yaml` alone, and
   confirm every entry reports **match**. This is the end-user story the feature exists for; if it
   does not work offline, the feature has failed regardless of the other checks.
6. Retrofit: take a `config.yaml` from a previous run with no codes, run `stamp_config`, confirm both
   fields populate.
7. Uninstall the `provenance` extra: confirm the pipeline still runs end to end, codes are `None`, and
   exactly one warning is emitted.

## Out of scope (YAGNI)

- **`iscc-sum` / byte-level codes** — cannot compare across container formats; the wrong tool.
- **Hashing the 8-bit training tiles** — ephemeral, never published.
- **Reading codes from `omero-iscc`** — deliberately rejected in favour of a single compute path.
- **Deduplication and train/val leakage detection** via ISCC similarity (Hamming distance).
- **Drift monitoring** — detecting that an OMERO source image changed after annotation.

Each is a plausible future extension; none is needed for publication-grade provenance, which is the
goal here.

## References

- ISCC-BIO — https://bio.iscc.codes/ (IMAGEWALK spec; PoC v0.1.0)
- `iscc-bio` source — https://github.com/bio-codes/iscc-bio (Apache-2.0)
- `omero-iscc` — https://github.com/bio-codes/omero-iscc (server-side ISCC on import; not used here)
- BIO-CODES project — https://bio-codes.io/
- ISCC-SUM (evaluated and rejected for this use) — https://sum.iscc.codes/userguide/
