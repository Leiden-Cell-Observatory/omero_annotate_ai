# QuPath extension for omero-annotate-ai

Annotate [omero-annotate-ai](../README.md) training sets in QuPath, image by image, with
any QuPath tool (brush, cell detection, Cellpose, InstanSeg, StarDist, SAM). Sets are
kept on OMERO or on this computer, in the formats of the Python package, so both sides
can open each other's sets.

User guide: [docs/qupath-plugin.md](../docs/qupath-plugin.md).

## Build

Java 21 and network access to Maven Central and `maven.scijava.org` are needed.

```bash
cd qupath-extension
./gradlew build          # jar in build/libs/, unit tests included
```

Drag `build/libs/qupath-extension-omero-annotate-ai-<version>.jar` onto QuPath (0.6 or
0.7). The jar is built against QuPath 0.6.0, the oldest version supported.

## Layout

| Package | What it holds |
|---|---|
| `core` | The training set itself, with no QuPath or OMERO class in it: rows and their CSV/table codec (`Row`, `RowCodec`), the config YAML (`SetConfig`), the folder layout (`Bundle`), TIFF and pixel digests (`TiffIO`, `PixelArray`), planning (`Planning`), the set and its sessions (`TrainingSet`, `AnnotationSession`) and the interface to OMERO (`RemoteStore`, `TimeLimitedStore`). |
| `omero` | `OmeroGatewayStore`: `RemoteStore` on a real server through the OMERO Java gateway (tables, label files, ROIs, config). `OmeroAccess` joins the login of the official OMERO extension by reflection, so the jar loads without that extension and without the OMERO libraries. |
| `qp` | QuPath objects: the frame of a unit, objects to label mask and back (`Units`), project entries and image URIs (`Images`), training data export (`Exporter`). |
| `gui` | The *Training set* tab (`TrainingSetPane`), the *New training set* dialog and the detection helpers. |

The formats are defined by the Python package; the Java side mirrors them:

| Python | Java |
|---|---|
| `rows.py` (`Row`, `COLUMNS`, `rows_to_frame`) | `core/Row`, `core/RowCodec` |
| `core/annotation_config.py` (schema 3) | `core/SetConfig` (keeps unknown keys and dates untouched) |
| `bundle.py` (`pixel_digest`, folders, `slug`) | `core/Bundle`, `core/PixelArray.digest()` |
| `sources/uri.py` (`annotation_id`, `uri_key`) | `core/Ids` |
| `session.py`, `claims.py`, `reconcile.py` | `core/AnnotationSession`, `core/TrainingSet` |
| `omero/tracking_table.py`, `omero/omero_functions.py` | `omero/OmeroGatewayStore` |

When a column, a config key or the write order of a unit changes in Python, change the
Java side with it; `FormatsTest` holds reference values computed by Python.

## Tests

```bash
./gradlew test                                  # unit tests, no server needed
./gradlew test -Domero.test.host=localhost      # plus the OMERO backend against a server
```

The OMERO test needs the Docker test server of the Python package
(`tests/docker-compose.yml`, port 6064) with a user `annot` / `annot-pw` and a dataset
holding at least four images; override with `-Domero.test.port`, `-Domero.test.user`,
`-Domero.test.password`, `-Domero.test.dataset`.

- `FormatsTest`: digests, IDs, slugs, CSV, YAML and TIFF against values from Python.
- `SessionTest`: sessions on local sets and on an in-memory OMERO that can fail any
  call: failed and half-finished uploads, repair, review, exclusion, two computers.
- `UnitsTest`, `ExporterTest`: objects to masks and back, the export layouts, timing
  with thousands of objects.
- `OmeroIntegrationTest`: tables, uploads, config and review on a real server.
