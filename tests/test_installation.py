"""Test package installation and basic imports with modern package structure."""

import importlib.util

import pytest


def test_package_import():
    """
    Tests that the main package can be imported.
    This is a simple smoke test to ensure that the package is installed correctly
    and that there are no syntax errors in the `__init__.py` file.
    """
    import omero_annotate_ai

    assert omero_annotate_ai.__version__


def test_core_imports():
    """
    Tests that the core modules can be imported.
    This test ensures that the core modules of the application, `annotation_config`
    and `annotation_pipeline`, can be imported without errors.
    """
    from omero_annotate_ai.core.annotation_config import (
        AnnotationConfig,
        create_default_config,
        load_config,
    )
    from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

    assert callable(load_config)
    assert isinstance(AnnotationPipeline, type)

    # Test that we can create instances
    config = create_default_config()
    assert isinstance(config, AnnotationConfig)


def test_omero_functions_import():
    """
    Tests that the OMERO functions can be imported.
    This test ensures that the `omero_functions` and `omero_utils` modules can be
    imported without errors.
    """
    from omero_annotate_ai.omero import omero_functions, omero_utils

    # Test that key functions exist
    assert hasattr(omero_functions, "upload_rois_and_labels")
    assert hasattr(omero_utils, "list_user_tables")


def test_processing_functions_import():
    """
    Tests that the processing functions can be imported.
    This test ensures that the `image_functions` module can be imported without errors.
    """
    from omero_annotate_ai.processing import image_functions

    # Test that key functions exist
    assert hasattr(image_functions, "generate_patch_coordinates")
    assert hasattr(image_functions, "label_to_rois")


def test_public_api():
    """The core is importable from the package top level."""
    import omero_annotate_ai

    for name in ("TrainingSet", "Row", "AnnotationConfig", "create_pipeline"):
        assert hasattr(omero_annotate_ai, name), name


def test_optional_dependencies():
    """
    Tests the behavior of the package when optional dependencies are missing.
    This test ensures that the application handles the absence of optional
    dependencies, such as `ezomero`, gracefully.
    """
    # Test ezomero import handling
    ezomero_available = importlib.util.find_spec("ezomero") is not None

    # Should be able to import omero functions even without ezomero
    from omero_annotate_ai.omero import omero_functions

    assert hasattr(omero_functions, "upload_rois_and_labels")

    if not ezomero_available:
        # Functions should raise ImportError when ezomero is needed
        from omero_annotate_ai.omero.omero_functions import upload_rois_and_labels

        with pytest.raises(ImportError, match="ezomero is required"):
            upload_rois_and_labels(None, 123, "test.tif")
