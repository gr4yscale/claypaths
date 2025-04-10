import pytest
import numpy as np
from shapely.geometry import Polygon, LineString
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cfs_filler import generate_cfs_fill

def test_generate_cfs_fill_square():
    """Test generating a CFS fill for a simple square."""
    # Create a square polygon
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Generate CFS fill
    toolpath = generate_cfs_fill(square, 1.0)
    
    # Check that the toolpath is not None
    assert toolpath is not None
    
    # Check that the toolpath is a LineString
    assert isinstance(toolpath, LineString)
    
    # Check that the toolpath has a reasonable number of points
    assert len(toolpath.coords) > 10
    
    # Check that the toolpath is within the original square
    for x, y in toolpath.coords:
        assert -0.1 <= x <= 10.1  # Allow small epsilon for numerical precision
        assert -0.1 <= y <= 10.1

def test_generate_cfs_fill_complex():
    """Test generating a CFS fill for a more complex shape."""
    # Create an L-shaped polygon
    l_shape = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    
    # Generate CFS fill
    toolpath = generate_cfs_fill(l_shape, 0.5)
    
    # Check that the toolpath is not None
    assert toolpath is not None
    
    # Check that the toolpath is a LineString
    assert isinstance(toolpath, LineString)
    
    # Check that the toolpath has a reasonable number of points
    assert len(toolpath.coords) > 10

def test_generate_cfs_fill_invalid():
    """Test generating a CFS fill with invalid inputs."""
    # Test with None
    toolpath = generate_cfs_fill(None, 1.0)
    assert toolpath is None
    
    # Test with invalid polygon
    invalid_polygon = Polygon([(0, 0), (1, 1), (0, 1)])  # Degenerate polygon
    toolpath = generate_cfs_fill(invalid_polygon, 1.0)
    assert toolpath is None
