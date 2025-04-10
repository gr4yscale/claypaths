import pytest
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cfs_filler import generate_iso_contours, build_spiral_contour_tree, perform_recursive_rerouting

def test_perform_recursive_rerouting_square():
    """Test recursive rerouting for a simple square."""
    # Create a square polygon
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Generate iso-contours
    contours = generate_iso_contours(square, 1.0)
    
    # Build the spiral-contour tree
    _, mst = build_spiral_contour_tree(contours)
    
    # Perform recursive rerouting
    final_path = perform_recursive_rerouting(contours, mst, 1.0)
    
    # Check that the final path is not None
    assert final_path is not None
    
    # Check that the final path is a LineString
    assert isinstance(final_path, LineString)
    
    # Check that the final path has a reasonable number of points
    assert len(final_path.coords) > 10
    
    # Check that the final path is within the original square
    for x, y in final_path.coords:
        assert 0 <= x <= 10
        assert 0 <= y <= 10

def test_perform_recursive_rerouting_complex():
    """Test recursive rerouting for a more complex shape."""
    # Create an L-shaped polygon
    l_shape = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    
    # Generate iso-contours
    contours = generate_iso_contours(l_shape, 0.5)
    
    # Build the spiral-contour tree
    _, mst = build_spiral_contour_tree(contours)
    
    # Perform recursive rerouting
    final_path = perform_recursive_rerouting(contours, mst, 0.5)
    
    # Check that the final path is not None
    assert final_path is not None
    
    # Check that the final path is a LineString
    assert isinstance(final_path, LineString)
    
    # Check that the final path has a reasonable number of points
    assert len(final_path.coords) > 10

def test_perform_recursive_rerouting_empty():
    """Test recursive rerouting with empty inputs."""
    # Create empty contours and MST
    contours = {}
    mst = nx.Graph()  # Empty graph instead of None
    
    # Perform recursive rerouting
    final_path = perform_recursive_rerouting(contours, mst, 1.0)
    
    # Should return None due to invalid inputs
    assert final_path is None
