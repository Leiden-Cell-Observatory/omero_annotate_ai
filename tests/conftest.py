"""Simple pytest configuration and fixtures for testing."""

import numpy as np
import pytest
import time
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

try:
    import ezomero
    from omero.gateway import BlitzGateway
    OMERO_AVAILABLE = True
except ImportError:
    OMERO_AVAILABLE = False

try:
    from omero_annotate_ai.core.annotation_config import create_default_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


# =============================================================================
# Simple Shared Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test in a throwaway working directory.

    Several config paths are relative - OutputConfig.output_directory defaults
    to ./annotations, and SimpleOMEROConnection reads ./.env - so a test that
    triggers an auto-save would otherwise write into the checked-out repo and
    dirty the working tree. Isolating the cwd makes that structurally impossible.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def sample_config():
    """Basic configuration for testing."""
    if not CONFIG_AVAILABLE:
        pytest.skip("Config module not available")
    return create_default_config()


@pytest.fixture
def fake_omero_connection():
    """Mock OMERO connection that appears connected."""
    mock_conn = Mock()
    mock_conn.isConnected.return_value = True
    mock_conn.getUser.return_value = Mock(getName=lambda: "testuser")
    mock_conn.getUserId.return_value = 1
    mock_conn.getGroupId.return_value = 1
    return mock_conn


@pytest.fixture
def temp_work_dir():
    """Temporary directory for test file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Simple Test Helper Functions
# =============================================================================

def create_mock_image(image_id=123, name="test_image.tiff"):
    """Helper to create a fake OMERO image."""
    mock_image = Mock()
    mock_image.getId.return_value = image_id
    mock_image.getName.return_value = name
    mock_image.getSizeX.return_value = 512
    mock_image.getSizeY.return_value = 512
    mock_image.getSizeZ.return_value = 1
    mock_image.getSizeC.return_value = 1
    mock_image.getSizeT.return_value = 1
    return mock_image


def create_mock_dataset(dataset_id=456, name="test_dataset"):
    """Helper to create a fake OMERO dataset."""
    mock_dataset = Mock()
    mock_dataset.getId.return_value = dataset_id
    mock_dataset.getName.return_value = name
    mock_dataset.getDescription.return_value = "Test dataset"
    return mock_dataset


def create_mock_project(project_id=789, name="test_project"):
    """Helper to create a fake OMERO project."""
    mock_project = Mock()
    mock_project.getId.return_value = project_id
    mock_project.getName.return_value = name
    mock_project.getDescription.return_value = "Test project"
    return mock_project


def assert_config_has_required_fields(config):
    """Check that config has all required fields."""
    assert hasattr(config, 'omero')
    assert hasattr(config, 'micro_sam')
    assert hasattr(config, 'batch_processing')
    assert hasattr(config, 'training')


# =============================================================================
# Docker OMERO fixtures (pytest -m docker)
# =============================================================================
#
# A server already answering on the configured port is reused and left running.
# Otherwise tests/docker-compose.yml is started and stopped again at the end of
# the session, unless OAI_KEEP_OMERO=1. Without Docker the tests skip.

OMERO_READY_TIMEOUT = 600  # seconds; a cold start takes a few minutes


@pytest.fixture(scope="session")
def omero_params() -> Dict[str, Any]:
    """Connection parameters of the test server (overridable by environment)."""
    return {
        "host": os.environ.get("OAI_OMERO_HOST", "localhost"),
        "port": int(os.environ.get("OAI_OMERO_PORT", "6064")),
        "user": os.environ.get("OAI_OMERO_USER", "root"),
        "password": os.environ.get("OAI_OMERO_PASSWORD", "omero"),
        "secure": True,
    }


def _connect(params):
    conn = BlitzGateway(params["user"], params["password"],
                        host=params["host"], port=params["port"],
                        secure=params["secure"])
    try:
        if conn.connect():
            return conn
    except Exception:
        pass
    conn.close()
    return None


def _server_ready(params) -> bool:
    conn = _connect(params)
    if conn is None:
        return False
    conn.close()
    return True


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "compose", "version"], check=True,
                       capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def docker_omero_server(omero_params):
    """A reachable OMERO server; starts the Docker stack when none is running."""
    if not OMERO_AVAILABLE:
        pytest.skip("omero-py not installed")
    if _server_ready(omero_params):
        yield omero_params
        return
    if not _docker_available():
        pytest.skip("No OMERO server running and docker compose not available")

    from tests.omero_seed import compose_cmd

    subprocess.run(compose_cmd("up", "-d"), check=True)
    try:
        deadline = time.monotonic() + OMERO_READY_TIMEOUT
        while not _server_ready(omero_params):
            if time.monotonic() > deadline:
                pytest.fail(f"OMERO did not accept logins within {OMERO_READY_TIMEOUT}s")
            time.sleep(10)
        yield omero_params
    finally:
        if os.environ.get("OAI_KEEP_OMERO") != "1":
            subprocess.run(compose_cmd("down", "-v"), check=False)


@pytest.fixture
def omero_conn(docker_omero_server):
    """A fresh BlitzGateway connection per test."""
    conn = _connect(docker_omero_server)
    if conn is None:
        pytest.fail("Could not log in to the test OMERO server")
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def omero_seed(docker_omero_server):
    """Project/dataset with synthetic images and a small plate, removed afterwards.

    See tests/omero_seed.py. ``plate_id`` is None when the plate could not be
    imported (e.g. a reused server that is not the compose stack).
    """
    from tests import omero_seed as seed_module

    conn = _connect(docker_omero_server)
    if conn is None:
        pytest.fail("Could not log in to the test OMERO server")
    try:
        data = seed_module.seed(conn, docker_omero_server["user"],
                                docker_omero_server["password"])
    except Exception:
        conn.close()
        raise
    try:
        yield data
    finally:
        if os.environ.get("OAI_KEEP_OMERO") != "1":
            seed_module.cleanup(conn, data)
        conn.close()


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, with mocks)"
    )
    config.addinivalue_line(
        "markers", "omero: mark test as requiring OMERO server"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (takes >5 seconds)"
    )
    config.addinivalue_line(
        "markers", "docker: mark test as needing the Docker OMERO test server"
    )


def pytest_collection_modifyitems(config, items):
    """Skip OMERO tests if OMERO is not available."""
    if not OMERO_AVAILABLE:
        skip_omero = pytest.mark.skip(reason="OMERO not available")
        for item in items:
            if "omero" in item.keywords:
                item.add_marker(skip_omero)

@pytest.fixture(scope="session")
def ngff_stores(tmp_path_factory):
    """Small OME-Zarr stores: NGFF 0.4 image (c,y,x), 0.5 image (t,c,z,y,x), 0.4 plate.

    Each image has 2 levels. The plate is named "testplate" with wells A/1, A/2, B/3,
    two fields ("0", "1") each, every field holding the 0.4 image data.
    """
    pytest.importorskip("ome_zarr")
    import zarr
    from ome_zarr.format import FormatV04, FormatV05
    from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

    root = tmp_path_factory.mktemp("ngff")
    arr04 = np.arange(2 * 64 * 80, dtype=np.uint16).reshape(2, 64, 80)
    arr05 = np.arange(2 * 2 * 3 * 32 * 40, dtype=np.uint16).reshape(2, 2, 3, 32, 40)

    g04 = zarr.open_group(str(root / "img04.zarr"), mode="w", zarr_format=2)
    write_image(arr04, g04, axes="cyx", fmt=FormatV04(), scale_factors=(2,))
    g05 = zarr.open_group(str(root / "img05.zarr"), mode="w", zarr_format=3)
    write_image(arr05, g05, axes="tczyx", fmt=FormatV05(), scale_factors=(2,))

    plate = zarr.open_group(str(root / "plate.zarr"), mode="w", zarr_format=2)
    well_paths = ["A/1", "A/2", "B/3"]
    write_plate_metadata(plate, ["A", "B"], ["1", "2", "3"], well_paths,
                         name="testplate", fmt=FormatV04())
    for well_path in well_paths:
        row, col = well_path.split("/")
        well = plate.require_group(row).require_group(col)
        write_well_metadata(well, ["0", "1"], fmt=FormatV04())
        for field_path in ("0", "1"):
            write_image(arr04, well.require_group(field_path), axes="cyx",
                        fmt=FormatV04(), scale_factors=(2,))

    return {
        "img04": (root / "img04.zarr").as_posix(),
        "img05": (root / "img05.zarr").as_posix(),
        "plate": (root / "plate.zarr").as_posix(),
        "arr04": arr04,
        "arr05": arr05,
    }
