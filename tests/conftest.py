"""Simple pytest configuration and fixtures for testing."""

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
# OMERO Integration Test Fixtures (for real OMERO server)
# =============================================================================

@pytest.fixture(scope="session")
def omero_params() -> Dict[str, Any]:
    """OMERO connection parameters for testing."""
    return {
        "host": "localhost",
        "port": 6063,
        "user": "root", 
        "password": "omero",
        "group": "",
        "secure": False
    }


@pytest.fixture(scope="session") 
def docker_omero_server(omero_params):
    """Start OMERO server via docker-compose for testing."""
    # Check if docker-compose is available
    try:
        subprocess.run(["docker-compose", "--version"], 
                      check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("docker-compose not available")
    
    # Start the OMERO server
    compose_file = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
    
    try:
        # Start services
        subprocess.run([
            "docker-compose", "-f", compose_file, "up", "-d"
        ], check=True, cwd=os.path.dirname(__file__))
        
        # Wait for OMERO to be ready
        max_wait = 300  # 5 minutes
        wait_interval = 10
        waited = 0
        
        while waited < max_wait:
            try:
                if OMERO_AVAILABLE:
                    conn = ezomero.connect(**omero_params)
                    if conn and conn.isConnected():
                        conn.close()
                        break
                    if conn:
                        conn.close()
            except Exception:
                pass
            
            time.sleep(wait_interval)
            waited += wait_interval
            print(f"Waiting for OMERO server... ({waited}s)")
        
        if waited >= max_wait:
            pytest.fail("OMERO server did not start within timeout")
        
        yield omero_params
        
    finally:
        # Cleanup
        try:
            subprocess.run([
                "docker-compose", "-f", compose_file, "down", "-v"
            ], check=True, cwd=os.path.dirname(__file__))
        except subprocess.CalledProcessError:
            pass


@pytest.fixture
def omero_connection(docker_omero_server):
    """Create an OMERO connection for testing."""
    if not OMERO_AVAILABLE:
        pytest.skip("OMERO not available")
    
    conn = ezomero.connect(**docker_omero_server)
    if not conn or not conn.isConnected():
        pytest.fail("Could not connect to OMERO server")
    
    yield conn
    
    if conn:
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


def pytest_collection_modifyitems(config, items):
    """Skip OMERO tests if OMERO is not available."""
    if not OMERO_AVAILABLE:
        skip_omero = pytest.mark.skip(reason="OMERO not available")
        for item in items:
            if "omero" in item.keywords:
                item.add_marker(skip_omero)