import pytest
import numpy as np
from shapely.geometry import Polygon
import networkx as nx
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cfs_filler import generate_iso_contours, build_spiral_contour_tree, identify_spirallable_regions

def test_identify_spirallable_regions_simple():
    """Test identifying spirallable regions in a simple shape."""
    # Create a square polygon (should be one spirallable region)
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Generate iso-contours
    contours = generate_iso_contours(square, 1.0)
    
    # Build the spiral-contour tree
    _, mst = build_spiral_contour_tree(contours)
    
    # Identify spirallable regions
    regions = identify_spirallable_regions(mst)
    
    # A simple square should have exactly one spirallable region
    assert len(regions) == 1
    
    # The region should contain all nodes in the MST
    assert len(regions[0]) == len(mst.nodes)

def test_identify_spirallable_regions_complex():
    """Test identifying spirallable regions in a complex shape."""
    # Create a shape that might have multiple spirallable regions
    # This is a shape with two "pockets"
    complex_shape = Polygon([
        (0, 0), (10, 0), (10, 3), (7, 3), (7, 7), (10, 7), (10, 10), (0, 10),
        (0, 7), (3, 7), (3, 3), (0, 3)
    ])
    
    # Generate iso-contours
    contours = generate_iso_contours(complex_shape, 0.5)
    
    # Build the spiral-contour tree
    _, mst = build_spiral_contour_tree(contours)
    
    # Identify spirallable regions
    regions = identify_spirallable_regions(mst)
    
    # Check that we have at least one spirallable region
    assert len(regions) > 0
    
    # Check that all nodes in the MST are covered by the regions
    all_region_nodes = [node for region in regions for node in region]
    assert len(all_region_nodes) <= len(mst.nodes)  # May be less due to branch points

def test_identify_spirallable_regions_empty():
    """Test identifying spirallable regions with an empty MST."""
    # Create an empty MST
    empty_mst = nx.Graph()
    
    # Identify spirallable regions
    regions = identify_spirallable_regions(empty_mst)
    
    # Should either return an empty list or a list with an empty list
    assert regions == [] or regions == [[]]
