"""Pytest configuration and fixtures for OMERO testing."""

import pytest
import time
import subprocess
import os
from typing import Dict, Any, Optional

try:
    import ezomero
    from omero.gateway import BlitzGateway
    OMERO_AVAILABLE = True
except ImportError:
    OMERO_AVAILABLE = False


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


@pytest.fixture
def sample_image_id(omero_connection):
    """Create a sample image for testing."""
    # This would create a test image in OMERO
    # For now, we'll skip if no test images are available
    pytest.skip("Sample image creation not implemented yet")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "omero: mark test as requiring OMERO server"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "docker: mark test as requiring Docker"
    )


def pytest_collection_modifyitems(config, items):
    """Skip OMERO tests if OMERO is not available."""
    if not OMERO_AVAILABLE:
        skip_omero = pytest.mark.skip(reason="OMERO not available")
        for item in items:
            if "omero" in item.keywords:
                item.add_marker(skip_omero)