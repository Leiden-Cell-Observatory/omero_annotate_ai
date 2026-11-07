# Tests

Run everything from the repository root in the pixi `dev` environment.

| Suite | Command | Needs |
|---|---|---|
| Unit | `pixi run -e dev pytest tests/ -q -m "not (omero or docker)"` | nothing |
| Docker integration + functional | `pixi run -e dev pytest tests/ -m "docker and not slow"` | Docker |
| Slow (micro-SAM training loaders) | `pixi run -e dev pytest tests/ -m "docker and slow"` | Docker |
| Lint, format, types | `pixi run -e dev lint` and `pixi run -e dev typecheck` | nothing |

## Docker OMERO

`tests/docker-compose.yml` runs a throwaway OMERO (server, web, PostgreSQL) on ports
6063/6064 (server) and 5080 (web), login `root` / `omero`.

```bash
pixi run start-omero          # start and wait until healthy (first start: a few minutes)
pixi run -e dev test-omero    # pytest -m docker
pixi run stop-omero           # stop and wipe all data
```

`pytest -m docker` also works without `start-omero`: the fixtures reuse a server that
already answers on the port, and otherwise start the stack themselves and stop it at
the end of the session. Without Docker these tests are skipped.

Each session seeds a uniquely named project and dataset with 6 synthetic images
(2 channels, 3 z, 2 t, known pixel values) and a 2x2 plate imported from a
Bio-Formats `.fake` file (`tests/omero_seed.py`), and deletes them afterwards.

After the pinned image versions change, recreate the stack (`stop-omero`, then
`start-omero`): an older PostgreSQL data volume cannot be reused.

## Functional workflow suite

`tests/test_functional_workflow.py` is the acceptance suite. It runs the real workflow
(annotation, upload, resume, extend, training export, training loaders, delete)
against the Docker OMERO; only napari is replaced, by a scripted annotator that writes
masks like micro-SAM does. Each mask encodes the image it was drawn on, so a label or
ROI on the wrong image fails the test.

Known-wrong behaviour is marked `xfail(strict=True)` with the reason and the change
that fixes it; when such a test starts passing, the run fails until the marker is
removed.

## Where tests live

Unit tests mirror the source modules, e.g. `planning.py` → `tests/test_planning.py`,
`annotators.py` → `tests/test_annotators.py`, `session.py` → `tests/test_session.py`,
`reconcile.py` → `tests/test_reconcile.py`, `core/annotation_pipeline.py` →
`tests/test_pipeline.py`. Add tests to the file of the module you change.

`tests/fakes.py` holds an in-memory OMERO (`FakeOmero`: tracking table, label files,
ROIs and their tags) and image source. `FakeOmero(fail={...}, crash={...})` makes a call
fail with `ConnectionError` or "kill the process" with `Crash`, which the crash-matrix
tests in `tests/test_reconcile.py` use to check that resuming repairs every state.

## Markers

| Marker | Meaning |
|---|---|
| `unit` | fast, dependencies mocked |
| `integration` | exercises several modules together |
| `omero` | needs an OMERO server |
| `docker` | needs the Docker OMERO above |
| `slow` | takes more than a few seconds |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OAI_OMERO_HOST` / `OAI_OMERO_PORT` | `localhost` / `6064` | server to test against |
| `OAI_OMERO_USER` / `OAI_OMERO_PASSWORD` | `root` / `omero` | login |
| `OAI_KEEP_OMERO` | unset | `1` keeps the stack and the seeded data after the session |
| `OAI_FULL_TRAINING` | unset | `1` runs a full micro-SAM training (over 30 min on CPU) |
