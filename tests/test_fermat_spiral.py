import pytest
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cfs_filler import generate_iso_contours, generate_fermat_spiral_segment

def test_generate_fermat_spiral_segment():
    """Test generating a Fermat spiral segment for a simple shape."""
    # Create a square polygon
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Generate iso-contours
    contours = generate_iso_contours(square, 1.0)
    
    # Extract polygons and exteriors for the spiral generation
    polygons = {}
    exteriors = {}
    
    for level in contours:
        polygons[level] = {}
        exteriors[level] = {}
        for j, contour in enumerate(contours[level]):
            polygons[level][j] = contour
            exteriors[level][j] = contour.exterior
    
    # Define parameters for the spiral
    innermost_contour_idx = max(contours.keys())
    outermost_contour_idx = 1  # The original boundary
    
    # Define entry and exit points on the outermost contour
    pin = Point(0, 5)  # Middle of left edge
    pout = Point(5, 0)  # Middle of bottom edge
    
    # Define center point (centroid of innermost contour)
    center_point = Point(polygons[innermost_contour_idx][0].centroid)
    
    # Generate the Fermat spiral
    spiral = generate_fermat_spiral_segment(
        polygons, exteriors, innermost_contour_idx, outermost_contour_idx,
        pin, pout, center_point, 1.0
    )
    
    # Check that the spiral is not None
    assert spiral is not None
    
    # Check that the spiral is a LineString
    assert isinstance(spiral, LineString)
    
    # Check that the spiral has a reasonable number of points
    assert len(spiral.coords) > 10
    
    # Check that the spiral starts near pin and ends near pout
    start_point = Point(spiral.coords[0])
    end_point = Point(spiral.coords[-1])
    assert start_point.distance(pin) < 1.0
    assert end_point.distance(pout) < 1.0

def test_generate_fermat_spiral_segment_invalid():
    """Test generating a Fermat spiral segment with invalid inputs."""
    # Create empty dictionaries
    polygons = {}
    exteriors = {}
    
    # Try to generate a spiral with invalid parameters
    spiral = generate_fermat_spiral_segment(
        polygons, exteriors, 1, 2,
        Point(0, 0), Point(1, 1), Point(0.5, 0.5), 1.0
    )
    
    # Should return None due to invalid inputs
    assert spiral is None
