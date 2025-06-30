"""Test package installation and basic imports with modern package structure."""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    import omero_annotate_ai
    # Note: Version testing removed as it may not be set in development


def test_core_imports():
    """Test core module imports."""
    from omero_annotate_ai.core.config import (
        AnnotationConfig, 
        create_default_config, 
        load_config
    )
    from omero_annotate_ai.core.pipeline import AnnotationPipeline
    
    # Test that we can create instances
    config = create_default_config()
    assert isinstance(config, AnnotationConfig)


def test_widget_import():
    """Test widget imports (might fail in headless environment)."""
    try:
        from omero_annotate_ai.widgets.config_widget import ConfigWidget
        from omero_annotate_ai import create_config_widget
        # Just test import, don't create widget (requires ipywidgets/jupyter)
    except ImportError:
        pytest.skip("ipywidgets not available in test environment")


def test_omero_functions_import():
    """Test OMERO function imports."""
    from omero_annotate_ai.omero import omero_functions, omero_utils
    
    # Test that key functions exist
    assert hasattr(omero_functions, 'initialize_tracking_table')
    assert hasattr(omero_functions, 'upload_rois_and_labels')
    assert hasattr(omero_utils, 'list_user_tables')
    assert hasattr(omero_utils, 'delete_table')


def test_processing_functions_import():
    """Test processing function imports."""
    from omero_annotate_ai.processing import image_functions, file_io_functions
    
    # Test that key functions exist
    assert hasattr(image_functions, 'generate_patch_coordinates')
    assert hasattr(image_functions, 'label_to_rois')
    assert hasattr(file_io_functions, 'load_annotation_file')


def test_config_functionality():
    """Test basic config functionality with modern structure."""
    from omero_annotate_ai.core.config import create_default_config
    
    config = create_default_config()
    
    # Test updated defaults
    assert config.batch_processing.batch_size == 0  # New default
    assert config.omero.container_type == "dataset"
    assert config.microsam.model_type == "vit_b_lm"  # New default
    assert config.training.trainingset_name == "default_training_set"
    
    # Test that removed parameters are gone
    assert not hasattr(config.training, 'group_by_image')
    
    # Test YAML conversion
    yaml_str = config.to_yaml()
    assert "batch_processing:" in yaml_str
    assert "microsam:" in yaml_str  # Updated section name
    assert "model_type: vit_b_lm" in yaml_str
    
    # Test dictionary conversion
    config_dict = config.to_dict()
    assert "batch_processing" in config_dict
    assert "microsam" in config_dict  # Updated section name
    
    # Test legacy params (for backward compatibility)
    legacy_params = config.get_legacy_params()
    assert "batch_size" in legacy_params
    assert legacy_params["batch_size"] == 0  # New default
    assert legacy_params["model_type"] == "vit_b_lm"  # New default
    assert "trainingset_name" in legacy_params
    assert "group_by_image" not in legacy_params  # Removed parameter


def test_pipeline_creation():
    """Test that pipeline can be created with mock connection."""
    from omero_annotate_ai.core.pipeline import AnnotationPipeline
    from omero_annotate_ai.core.config import create_default_config
    from unittest.mock import Mock
    
    config = create_default_config()
    config.omero.container_id = 123  # Set required field
    
    # Create pipeline with mock connection (pipeline requires non-None connection)
    mock_conn = Mock()
    pipeline = AnnotationPipeline(config, conn=mock_conn)
    assert pipeline.config == config
    assert pipeline.conn == mock_conn


def test_optional_dependencies():
    """Test behavior when optional dependencies are missing."""
    # Test ezomero import handling
    try:
        import ezomero
        ezomero_available = True
    except ImportError:
        ezomero_available = False
    
    # Should be able to import omero functions even without ezomero
    from omero_annotate_ai.omero import omero_functions
    assert hasattr(omero_functions, 'initialize_tracking_table')
    
    if not ezomero_available:
        # Functions should raise ImportError when ezomero is needed
        from omero_annotate_ai.omero.omero_functions import initialize_tracking_table
        
        with pytest.raises(ImportError, match="ezomero is required"):
            initialize_tracking_table(None, "test", [], "dataset", 1, "test")


def test_all_submodules_importable():
    """Test that all submodules can be imported without errors."""
    # Core modules
    from omero_annotate_ai.core import config, pipeline
    
    # OMERO modules
    from omero_annotate_ai.omero import omero_functions, omero_utils
    
    # Processing modules
    from omero_annotate_ai.processing import image_functions, file_io_functions, utils
    
    # Widget modules (may fail in headless environment)
    try:
        from omero_annotate_ai.widgets import config_widget
    except ImportError:
        # Expected in headless environment
        pass


def test_main_package_exports():
    """Test that main package exports expected functions."""
    import omero_annotate_ai
    
    # Should have create_config_widget function
    assert hasattr(omero_annotate_ai, 'create_config_widget')
    
    # Test that it can be called (may fail without ipywidgets)
    try:
        widget = omero_annotate_ai.create_config_widget()
        assert widget is not None
    except ImportError:
        # Expected without ipywidgets
        pass