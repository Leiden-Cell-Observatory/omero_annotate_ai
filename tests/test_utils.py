"""Tests for utility functions."""

import pytest
import numpy as np

from omero_annotate_ai.processing.utils import (
    interleave_arrays,
    validate_image_dimensions,
    calculate_optimal_batch_size
)


class TestArrayOperations:
    """Test array manipulation utilities."""
    
    def test_interleave_arrays_equal_length(self):
        """Test interleaving arrays of equal length."""
        array1 = [1, 3, 5]
        array2 = [2, 4, 6]
        
        result = interleave_arrays(array1, array2)
        
        assert result == [1, 2, 3, 4, 5, 6]
    
    def test_interleave_arrays_first_longer(self):
        """Test interleaving when first array is longer."""
        array1 = [1, 3, 5, 7, 9]
        array2 = [2, 4]
        
        result = interleave_arrays(array1, array2)
        
        assert result == [1, 2, 3, 4, 5, 7, 9]
    
    def test_interleave_arrays_second_longer(self):
        """Test interleaving when second array is longer."""
        array1 = [1, 3]
        array2 = [2, 4, 6, 8, 10]
        
        result = interleave_arrays(array1, array2)
        
        assert result == [1, 2, 3, 4, 6, 8, 10]
    
    def test_interleave_arrays_empty_arrays(self):
        """Test interleaving empty arrays."""
        result = interleave_arrays([], [])
        assert result == []
    
    def test_interleave_arrays_one_empty(self):
        """Test interleaving with one empty array."""
        array1 = [1, 2, 3]
        array2 = []
        
        result1 = interleave_arrays(array1, array2)
        result2 = interleave_arrays(array2, array1)
        
        assert result1 == [1, 2, 3]
        assert result2 == [1, 2, 3]
    
    def test_interleave_arrays_different_types(self):
        """Test interleaving arrays with different element types."""
        array1 = ["a", "c", "e"]
        array2 = ["b", "d", "f"]
        
        result = interleave_arrays(array1, array2)
        
        assert result == ["a", "b", "c", "d", "e", "f"]
    
    def test_interleave_arrays_mixed_types(self):
        """Test interleaving arrays with mixed types."""
        array1 = [1, "hello", 3.14]
        array2 = ["world", 2, True]
        
        result = interleave_arrays(array1, array2)
        
        assert result == [1, "world", "hello", 2, 3.14, True]


class TestImageValidation:
    """Test image dimension validation utilities."""
    
    def test_validate_image_dimensions_valid(self):
        """Test validation with valid dimensions."""
        image_shape = (1000, 1000)
        patch_size = (512, 512)
        
        assert validate_image_dimensions(image_shape, patch_size) is True
    
    def test_validate_image_dimensions_exact_fit(self):
        """Test validation when patch exactly fits image."""
        image_shape = (512, 512)
        patch_size = (512, 512)
        
        assert validate_image_dimensions(image_shape, patch_size) is True
    
    def test_validate_image_dimensions_patch_too_large(self):
        """Test validation when patch is larger than image."""
        image_shape = (256, 256)
        patch_size = (512, 512)
        
        assert validate_image_dimensions(image_shape, patch_size) is False
    
    def test_validate_image_dimensions_height_too_small(self):
        """Test validation when image height is too small."""
        image_shape = (100, 1000)
        patch_size = (512, 256)
        
        assert validate_image_dimensions(image_shape, patch_size) is False
    
    def test_validate_image_dimensions_width_too_small(self):
        """Test validation when image width is too small."""
        image_shape = (1000, 100)
        patch_size = (256, 512)
        
        assert validate_image_dimensions(image_shape, patch_size) is False
    
    def test_validate_image_dimensions_zero_dimensions(self):
        """Test validation with zero dimensions."""
        image_shape = (0, 0)
        patch_size = (1, 1)
        
        assert validate_image_dimensions(image_shape, patch_size) is False
    
    def test_validate_image_dimensions_zero_patch(self):
        """Test validation with zero patch size."""
        image_shape = (100, 100)
        patch_size = (0, 0)
        
        assert validate_image_dimensions(image_shape, patch_size) is True


class TestBatchSizeCalculation:
    """Test batch size calculation utilities."""
    
    def test_calculate_optimal_batch_size_default(self):
        """Test batch size calculation with default memory."""
        batch_size = calculate_optimal_batch_size(100)
        
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 100
    
    def test_calculate_optimal_batch_size_small_dataset(self):
        """Test batch size calculation with small dataset."""
        batch_size = calculate_optimal_batch_size(5)
        
        assert batch_size <= 5
        assert batch_size > 0
    
    def test_calculate_optimal_batch_size_large_memory(self):
        """Test batch size calculation with large memory."""
        batch_size_small_mem = calculate_optimal_batch_size(100, available_memory_gb=2.0)
        batch_size_large_mem = calculate_optimal_batch_size(100, available_memory_gb=32.0)
        
        assert batch_size_large_mem >= batch_size_small_mem
    
    def test_calculate_optimal_batch_size_zero_images(self):
        """Test batch size calculation with zero images."""
        batch_size = calculate_optimal_batch_size(0)
        
        assert batch_size == 0  # Returns min(n_images, max_batch)
    
    def test_calculate_optimal_batch_size_negative_images(self):
        """Test batch size calculation with negative image count."""
        batch_size = calculate_optimal_batch_size(-5)
        
        assert batch_size == -5  # Returns min(n_images, max_batch)
    
    def test_calculate_optimal_batch_size_very_low_memory(self):
        """Test batch size calculation with very low memory."""
        batch_size = calculate_optimal_batch_size(100, available_memory_gb=0.1)
        
        assert batch_size == 1  # max(1, int(0.1)) = 1, min(100, 1) = 1
    
    def test_calculate_optimal_batch_size_zero_memory(self):
        """Test batch size calculation with zero memory."""
        batch_size = calculate_optimal_batch_size(100, available_memory_gb=0.0)
        
        assert batch_size == 1  # max(1, int(0.0)) = 1, min(100, 1) = 1
    
    def test_calculate_optimal_batch_size_negative_memory(self):
        """Test batch size calculation with negative memory."""
        batch_size = calculate_optimal_batch_size(100, available_memory_gb=-1.0)
        
        assert batch_size == 1  # max(1, int(-1.0)) = 1, min(100, 1) = 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_interleave_arrays_with_none_elements(self):
        """Test interleaving arrays containing None elements."""
        array1 = [1, None, 3]
        array2 = [None, 2, None]
        
        result = interleave_arrays(array1, array2)
        
        assert result == [1, None, None, 2, 3, None]
    
    def test_validate_image_dimensions_negative_dimensions(self):
        """Test validation with negative dimensions."""
        image_shape = (-100, 100)
        patch_size = (50, 50)
        
        # Should handle gracefully (probably return False)
        result = validate_image_dimensions(image_shape, patch_size)
        assert isinstance(result, bool)
    
    def test_validate_image_dimensions_float_dimensions(self):
        """Test validation with float dimensions."""
        image_shape = (100.5, 100.7)
        patch_size = (50.2, 50.8)
        
        # Should handle gracefully
        result = validate_image_dimensions(image_shape, patch_size)
        assert isinstance(result, bool)
    
    def test_calculate_optimal_batch_size_float_images(self):
        """Test batch size calculation with float image count."""
        batch_size = calculate_optimal_batch_size(10.5)
        
        assert isinstance(batch_size, int)
        assert batch_size > 0
    
    def test_interleave_arrays_very_large_arrays(self):
        """Test interleaving very large arrays."""
        array1 = list(range(0, 10000, 2))  # Even numbers
        array2 = list(range(1, 10000, 2))  # Odd numbers
        
        result = interleave_arrays(array1, array2)
        
        # Should be sorted sequence 0, 1, 2, 3, ..., 9999
        assert len(result) == 10000
        assert result[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def test_batch_size_calculation_consistency(self):
        """Test that batch size calculation is consistent."""
        n_images = 50
        memory_gb = 8.0
        
        # Multiple calls should return same result
        batch1 = calculate_optimal_batch_size(n_images, memory_gb)
        batch2 = calculate_optimal_batch_size(n_images, memory_gb)
        
        assert batch1 == batch2