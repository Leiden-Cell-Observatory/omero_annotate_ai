# Testing Guide

This guide provides instructions for running the tests for the `omero-annotate-ai` package. The test suite is built with `pytest` and managed with `pixi`.

## Test Environments

All test commands should be run from the root of the project using the `dev` environment in `pixi`.

## Running Tests

### Unit tests (fast, no server)

```bash
pixi run -e dev pytest tests/ -q -m "not (omero or docker)"
```

### Integration tests against a local OMERO (Docker)

`tests/docker-compose.yml` runs a throwaway OMERO (server, web, PostgreSQL) on
ports 6063/6064 (server) and 5080 (web), login `root` / `omero`.

```bash
pixi run start-omero          # start and wait until healthy (first start: a few minutes)
pixi run -e dev test-omero    # pytest -m docker
pixi run stop-omero           # stop and wipe all data
```

`pytest -m docker` also works without `start-omero`: the fixtures reuse a server
that already answers on the port, and otherwise start the stack themselves and
stop it at the end of the session. Without Docker the tests are skipped.

Each session seeds a uniquely named project/dataset with 6 synthetic images
(2 channels, 3 z, 2 t, 64x80, known pixel values) and a 2x2 plate imported from a
Bio-Formats `.fake` file inside the server container (`tests/omero_seed.py`), and
deletes them afterwards.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `OAI_OMERO_HOST` / `OAI_OMERO_PORT` | `localhost` / `6064` | Server to test against |
| `OAI_OMERO_USER` / `OAI_OMERO_PASSWORD` | `root` / `omero` | Login |
| `OAI_KEEP_OMERO` | unset | `1` keeps the stack and the seeded data after the session |

After the pinned image versions change, recreate the stack (`stop-omero`, then
`start-omero`): the database volume of an older PostgreSQL cannot be reused.

## Test Markers

The test suite uses `pytest` markers to categorize tests:

-   `@pytest.mark.unit`: For fast, self-contained unit tests.
-   `@pytest.mark.integration`: For tests that require external services, like a running OMERO server.
-   `@pytest.mark.omero`: A specific marker for tests that interact with an OMERO server.
-   `@pytest.mark.docker`: For tests that need the Docker OMERO test server (see above).

## Test Suite Status (As of last update)

The test suite has recently undergone a cleanup to remove obsolete tests and improve clarity. As a result, some tests may be temporarily failing while they are being updated to match the latest codebase.

| File | Purpose | Type | Status |
|---|---|---|---|
| `test_config.py` | Tests the configuration management system. | Unit | Passing |
| `test_file_io_functions.py` | Obsolete; functions removed. | - | No Tests |
| `test_image_functions.py` | Tests the image processing functions. | Unit | Passing |
| `test_installation.py` | Smoke tests for package installation and imports. | Unit | Passing |
| `test_omero_functions.py` | Tests the OMERO utility functions. | Unit | Passing |
| `test_omero_integration.py` | Tests the OMERO integration with a real server. | Integration/Unit | In Progress |
| `test_pipeline.py` | Tests the annotation pipeline. | Unit | In Progress |
| `test_training_functions.py` | Tests the training data preparation functions. | Unit | In Progress |
| `test_utils.py` | Obsolete; functions removed. | - | No Tests |
| `test_widgets.py` | Tests the widget classes. | Unit | In Progress |