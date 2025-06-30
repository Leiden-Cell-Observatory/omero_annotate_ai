"""Tests for image processing functions."""

import pytest
import numpy as np
from omero_annotate_ai.processing.image_functions import (
    generate_patch_coordinates,
    _rectangles_overlap
)

# Test label_to_rois if OpenCV is available
try:
    from omero_annotate_ai.processing.image_functions import (
        label_to_rois,
        process_label_plane,
        mask_to_contour
    )
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class TestPatchGeneration:
    """Test patch coordinate generation functions."""
    
    def test_generate_patch_coordinates_grid(self):
        """Test grid-based patch generation."""
        image_shape = (1000, 1000)
        patch_size = (200, 200)
        n_patches = 4
        
        coordinates = generate_patch_coordinates(
            image_shape, patch_size, n_patches, random=False
        )
        
        assert len(coordinates) == 4
        
        # Check all patches are within bounds
        for x, y in coordinates:
            assert 0 <= x <= 800  # 1000 - 200
            assert 0 <= y <= 800  # 1000 - 200
        
        # Check no overlaps
        rectangles = [(x, y, x + patch_size[1], y + patch_size[0]) for x, y in coordinates]
        for i, rect1 in enumerate(rectangles):
            for j, rect2 in enumerate(rectangles[i+1:], i+1):
                assert not _rectangles_overlap(rect1, rect2)
    
    def test_generate_patch_coordinates_random(self):
        """Test random patch generation with non-overlapping constraint."""
        image_shape = (1000, 1000)
        patch_size = (200, 200)
        n_patches = 5
        
        coordinates = generate_patch_coordinates(
            image_shape, patch_size, n_patches, random=True
        )
        
        assert len(coordinates) <= n_patches  # May be fewer if can't fit without overlap
        
        # Check all patches are within bounds
        for x, y in coordinates:
            assert 0 <= x <= 800
            assert 0 <= y <= 800
        
        # Check no overlaps (CRUCIAL requirement)
        rectangles = [(x, y, x + patch_size[1], y + patch_size[0]) for x, y in coordinates]
        for i, rect1 in enumerate(rectangles):
            for j, rect2 in enumerate(rectangles[i+1:], i+1):
                assert not _rectangles_overlap(rect1, rect2), f"Patches {i} and {j} overlap"
    
    def test_patch_generation_small_image(self):
        """Test patch generation when image is smaller than patch."""
        small_shape = (100, 100)
        large_patch = (150, 150)
        
        coordinates = generate_patch_coordinates(small_shape, large_patch, 3, random=True)
        
        # Should return only one patch at (0, 0)
        assert len(coordinates) == 1
        assert coordinates[0] == (0, 0)
    
    def test_patch_generation_exact_fit(self):
        """Test patch generation when patch exactly fits image."""
        image_shape = (200, 200)
        patch_size = (200, 200)
        
        coordinates = generate_patch_coordinates(image_shape, patch_size, 1, random=False)
        
        assert len(coordinates) == 1
        assert coordinates[0] == (0, 0)
    
    def test_patch_generation_multiple_exact_fit(self):
        """Test patch generation with multiple exact-fit patches."""
        image_shape = (400, 400)
        patch_size = (200, 200)
        n_patches = 4  # Should fit exactly 2x2 grid
        
        coordinates = generate_patch_coordinates(image_shape, patch_size, n_patches, random=False)
        
        assert len(coordinates) == 4
        expected_coords = [(0, 0), (200, 0), (0, 200), (200, 200)]
        
        for coord in coordinates:
            assert coord in expected_coords


class TestRectangleOverlap:
    """Test rectangle overlap detection."""
    
    def test_overlapping_rectangles(self):
        """Test detection of overlapping rectangles."""
        rect1 = (0, 0, 100, 100)
        rect2 = (50, 50, 150, 150)
        
        assert _rectangles_overlap(rect1, rect2)
        assert _rectangles_overlap(rect2, rect1)  # Commutative
    
    def test_non_overlapping_rectangles(self):
        """Test detection of non-overlapping rectangles."""
        rect1 = (0, 0, 50, 50)
        rect2 = (60, 60, 110, 110)
        
        assert not _rectangles_overlap(rect1, rect2)
        assert not _rectangles_overlap(rect2, rect1)  # Commutative
    
    def test_adjacent_rectangles(self):
        """Test that adjacent rectangles are not considered overlapping."""
        rect1 = (0, 0, 50, 50)
        rect2 = (50, 0, 100, 50)  # Adjacent, not overlapping
        
        assert not _rectangles_overlap(rect1, rect2)
        assert not _rectangles_overlap(rect2, rect1)
    
    def test_touching_corner_rectangles(self):
        """Test rectangles that only touch at a corner."""
        rect1 = (0, 0, 50, 50)
        rect2 = (50, 50, 100, 100)  # Touch at corner (50, 50)
        
        assert not _rectangles_overlap(rect1, rect2)
    
    def test_identical_rectangles(self):
        """Test that identical rectangles overlap completely."""
        rect = (10, 10, 60, 60)
        
        assert _rectangles_overlap(rect, rect)
    
    def test_contained_rectangles(self):
        """Test that contained rectangles overlap."""
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        
        assert _rectangles_overlap(outer, inner)
        assert _rectangles_overlap(inner, outer)


# Grid and random patch generation are tested through the main generate_patch_coordinates function


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
class TestROICreation:
    """Test ROI creation functions (requires OpenCV)."""
    
    def test_mask_to_contour_simple(self):
        """Test converting simple mask to contours."""
        # Create a simple rectangular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1
        
        contours = mask_to_contour(mask)
        
        assert len(contours) == 1
        assert len(contours[0]) >= 4  # At least 4 points for rectangle
    
    def test_mask_to_contour_multiple_objects(self):
        """Test converting mask with multiple objects."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # First object
        mask[60:80, 60:80] = 1  # Second object
        
        contours = mask_to_contour(mask)
        
        assert len(contours) == 2
    
    def test_mask_to_contour_empty(self):
        """Test converting empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        contours = mask_to_contour(mask)
        
        assert len(contours) == 0
    
    def test_process_label_plane_basic(self):
        """Test processing a basic label plane."""
        # Create label image with two objects
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[10:30, 10:30] = 1  # Object 1
        labels[60:80, 60:80] = 2  # Object 2
        
        shapes = process_label_plane(
            label_plane=labels,
            z_slice=0,
            channel=0,
            timepoint=0,
            model_type="vit_b_lm"
        )
        
        assert len(shapes) == 2  # Two objects
        
        # Check shape properties
        for shape in shapes:
            assert hasattr(shape, 'points')
            assert shape.z == 0
            assert shape.c == 0
            assert shape.t == 0
            assert "vit_b_lm" in shape.label
            assert "micro_sam" in shape.label
    
    def test_process_label_plane_with_offset(self):
        """Test processing label plane with patch offset."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[10:30, 10:30] = 1
        
        patch_offset = (100, 200)
        
        shapes = process_label_plane(
            label_plane=labels,
            z_slice=0,
            channel=0,
            timepoint=0,
            model_type="vit_b_lm",
            x_offset=patch_offset[0],
            y_offset=patch_offset[1]
        )
        
        assert len(shapes) == 1
        
        # Check that coordinates are offset
        shape = shapes[0]
        points = shape.points
        
        # All points should be offset by patch_offset
        for x, y in points:
            assert x >= 100  # Should be offset by patch_offset[0]
            assert y >= 200  # Should be offset by patch_offset[1]
    
    def test_label_to_rois_basic(self):
        """Test converting label image to ROI shapes."""
        # Create 3D label image (single plane) with volumetric=True
        label_img = np.zeros((1, 100, 100), dtype=np.uint16)
        label_img[0, 25:75, 25:75] = 1
        
        shapes = label_to_rois(
            label_img=label_img,
            z_slice=0,
            channel=0,
            timepoint=0,
            model_type="vit_b_lm",
            is_volumetric=True  # Changed to True since it's 3D
        )
        
        assert len(shapes) == 1
        
        shape = shapes[0]
        assert shape.z == 0
        assert shape.c == 0
        assert shape.t == 0
        assert "vit_b_lm" in shape.label
        assert "micro_sam" in shape.label
    
    def test_label_to_rois_2d(self):
        """Test converting 2D label image to ROI shapes."""
        # Create 2D label image
        label_img = np.zeros((100, 100), dtype=np.uint16)
        label_img[25:75, 25:75] = 1
        
        shapes = label_to_rois(
            label_img=label_img,
            z_slice=0,
            channel=0,
            timepoint=0,
            model_type="vit_b_lm",
            is_volumetric=False
        )
        
        assert len(shapes) == 1
    
    def test_label_to_rois_empty(self):
        """Test converting empty label image."""
        label_img = np.zeros((100, 100), dtype=np.uint16)
        
        shapes = label_to_rois(
            label_img=label_img,
            z_slice=0,
            channel=0,
            timepoint=0,
            model_type="vit_b_lm",
            is_volumetric=False
        )
        
        assert len(shapes) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_patch_generation_zero_patches(self):
        """Test patch generation with zero patches requested."""
        coordinates = generate_patch_coordinates((100, 100), (50, 50), 0, random=True)
        assert len(coordinates) == 0
    
    def test_patch_generation_negative_patches(self):
        """Test patch generation with negative patch count."""
        coordinates = generate_patch_coordinates((100, 100), (50, 50), -1, random=True)
        assert len(coordinates) == 0
    
    def test_rectangle_overlap_invalid_rectangles(self):
        """Test rectangle overlap with invalid rectangles."""
        # Rectangle with negative dimensions
        rect1 = (0, 0, -10, -10)
        rect2 = (5, 5, 15, 15)
        
        # Should handle gracefully (probably return False)
        result = _rectangles_overlap(rect1, rect2)
        assert isinstance(result, bool)
    
    def test_patch_generation_very_large_patch(self):
        """Test patch generation with patch larger than image."""
        image_shape = (100, 100)
        patch_size = (200, 300)
        
        coordinates = generate_patch_coordinates(image_shape, patch_size, 5, random=True)
        
        # Should return at most one patch at (0, 0)
        assert len(coordinates) <= 1
        if coordinates:
            assert coordinates[0] == (0, 0)