import pytest
import numpy as np
from shapely.geometry import Polygon
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cfs_filler import generate_iso_contours

def test_generate_iso_contours_square():
    """Test iso-contour generation for a simple square."""
    # Create a square polygon
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Generate iso-contours with a toolpath width of 1.0
    contours = generate_iso_contours(square, 1.0)
    
    # Check that we have the expected number of contour levels
    # For a 10x10 square with 1.0 width, we should have about 5-6 levels
    # (original boundary + inward offsets until we reach the center)
    assert len(contours) >= 5
    
    # Check that the first contour is the original boundary
    assert contours[1][0].equals(square)
    
    # Check that each subsequent contour is smaller than the previous
    for i in range(2, len(contours)):
        assert contours[i][0].area < contours[i-1][0].area

def test_generate_iso_contours_complex():
    """Test iso-contour generation for a more complex shape."""
    # Create an L-shaped polygon
    l_shape = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    
    # Generate iso-contours with a toolpath width of 0.5
    contours = generate_iso_contours(l_shape, 0.5)
    
    # Check that we have contours
    assert len(contours) > 1
    
    # The innermost contour should be much smaller than the original
    innermost_level = max(contours.keys())
    assert contours[innermost_level][0].area < l_shape.area * 0.25

def test_generate_iso_contours_invalid_input():
    """Test iso-contour generation with invalid inputs."""
    # Test with None
    with pytest.raises(Exception):
        generate_iso_contours(None, 1.0)
    
    # Test with invalid polygon
    invalid_polygon = Polygon([(0, 0), (1, 1), (0, 1)])  # Degenerate polygon
    contours = generate_iso_contours(invalid_polygon, 1.0)
    assert len(contours) <= 1  # Should have at most the original boundary
