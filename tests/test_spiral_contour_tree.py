import pytest
import numpy as np
from shapely.geometry import Polygon
import networkx as nx
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cfs_filler import generate_iso_contours, build_spiral_contour_tree

def test_build_spiral_contour_tree_square():
    """Test building the spiral-contour tree for a simple square."""
    # Create a square polygon
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Generate iso-contours
    contours = generate_iso_contours(square, 1.0)
    
    # Build the spiral-contour tree
    graph, mst = build_spiral_contour_tree(contours)
    
    # Check that the graph and MST are not None
    assert graph is not None
    assert mst is not None
    
    # Check that the MST is a tree (no cycles)
    assert nx.is_tree(mst)
    
    # Check that the MST has the expected number of nodes
    # Should have one node per contour
    expected_nodes = sum(len(contours[level]) for level in contours)
    assert len(mst.nodes) == expected_nodes
    
    # Check that the root node (1,0) is in the MST
    assert (1, 0) in mst.nodes

def test_build_spiral_contour_tree_complex():
    """Test building the spiral-contour tree for a more complex shape."""
    # Create an L-shaped polygon
    l_shape = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    
    # Generate iso-contours
    contours = generate_iso_contours(l_shape, 0.5)
    
    # Build the spiral-contour tree
    graph, mst = build_spiral_contour_tree(contours)
    
    # Check that the graph and MST are not None
    assert graph is not None
    assert mst is not None
    
    # Check that the MST is a tree (no cycles)
    assert nx.is_tree(mst)
    
    # The MST should connect all contours
    assert len(mst.nodes) > 0

def test_build_spiral_contour_tree_empty():
    """Test building the spiral-contour tree with empty contours."""
    # Create empty contours dictionary
    contours = {}
    
    # Build the spiral-contour tree
    graph, mst = build_spiral_contour_tree(contours)
    
    # Graph should be empty but not None
    assert graph is not None
    assert len(graph.nodes) == 0
    
    # MST should be None since there are no nodes to connect
    assert mst is None
