#!/usr/bin/env python3
"""
Simple test script to verify the omero-annotate-ai package works.
This script tests the basic functionality without requiring OMERO connection.
"""

def test_basic_imports():
    """Test that the package imports correctly."""
    print("🧪 Testing basic imports...")
    
    try:
        import omero_annotate_ai
        print(f"✅ Package imported successfully! Version: {omero_annotate_ai.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import package: {e}")
        return False
    
    try:
        from omero_annotate_ai import create_default_config, AnnotationConfig, create_config_widget
        print("✅ Core functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core functions: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration functionality."""
    print("\n🧪 Testing configuration...")
    
    try:
        from omero_annotate_ai import create_default_config
        
        # Create default config
        config = create_default_config()
        print(f"✅ Default config created: {type(config).__name__}")
        
        # Test basic properties
        print(f"   - Model type: {config.microsam.model_type}")
        print(f"   - Container type: {config.omero.container_type}")
        print(f"   - Batch size: {config.batch_processing.batch_size}")
        
        # Test YAML conversion
        yaml_str = config.to_yaml()
        print("✅ YAML conversion works")
        
        # Test dictionary conversion
        config_dict = config.to_dict()
        print("✅ Dictionary conversion works")
        
        # Test validation (should pass with container_id set)
        config.omero.container_id = 123
        config.validate()
        print("✅ Configuration validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_widget_creation():
    """Test widget creation (without displaying)."""
    print("\n🧪 Testing widget creation...")
    
    try:
        from omero_annotate_ai import create_config_widget
        
        # Create widget (but don't display it)
        widget = create_config_widget()
        print("✅ Config widget created successfully")
        
        # Test getting config from widget
        config = widget.get_config()
        print(f"✅ Got config from widget: {type(config).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Widget test failed: {e}")
        return False


def test_pipeline_creation():
    """Test pipeline creation (without OMERO connection)."""
    print("\n🧪 Testing pipeline creation...")
    
    try:
        from omero_annotate_ai import create_pipeline, create_default_config
        
        config = create_default_config()
        config.omero.container_id = 123
        
        # This should fail without a connection, but let's test the error handling
        try:
            pipeline = create_pipeline(config, conn=None)
            # This should raise an error when we try to validate
            pipeline._validate_setup()
            print("❌ Pipeline should have failed without connection")
            return False
        except ValueError as expected_error:
            if "OMERO connection is required" in str(expected_error):
                print("✅ Pipeline correctly requires OMERO connection")
                return True
            else:
                print(f"❌ Unexpected error: {expected_error}")
                return False
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def test_microsam_configs():
    """Test different micro-SAM model configurations."""
    print("\n🧪 Testing micro-SAM configurations...")
    
    try:
        from omero_annotate_ai import AnnotationConfig
        
        # Test MicroSAM config
        microsam_dict = {
            "microsam": {
                "model_type": "vit_l"
            },
            "omero": {"container_id": 123}
        }
        
        config = AnnotationConfig.from_dict(microsam_dict)
        print(f"✅ MicroSAM config: {config.microsam.model_type}")
        
        # Test vit_b_lm model
        vit_b_lm_dict = {
            "microsam": {
                "model_type": "vit_b_lm"
            },
            "omero": {"container_id": 123}
        }
        
        config = AnnotationConfig.from_dict(vit_b_lm_dict)
        print(f"✅ VIT-B-LM config: {config.microsam.model_type}")
        
        # Test backward compatibility
        old_config_dict = {
            "image_processing": {
                "model_type": "vit_h",
                "three_d": True
            },
            "omero": {"container_id": 123}
        }
        
        config = AnnotationConfig.from_dict(old_config_dict)
        print(f"✅ Backward compatibility: {config.microsam.model_type}")
        print(f"   - 3D mode: {config.microsam.three_d}")
        
        return True
        
    except Exception as e:
        print(f"❌ MicroSAM config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing omero-annotate-ai package...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_widget_creation,
        test_pipeline_creation,
        test_microsam_configs
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The package is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Install OMERO dependencies: pip install -e .[omero]")
        print("   2. Try the notebook: notebooks/omero-annotate-ai-batch.ipynb")
        print("   3. Configure your OMERO connection in .env file")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)