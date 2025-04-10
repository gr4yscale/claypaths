import pytest
from shapely.geometry import Polygon

@pytest.fixture
def square_polygon():
    """Return a simple square polygon."""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

@pytest.fixture
def l_shape_polygon():
    """Return an L-shaped polygon."""
    return Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])

@pytest.fixture
def complex_polygon():
    """Return a more complex polygon with two pockets."""
    return Polygon([
        (0, 0), (10, 0), (10, 3), (7, 3), (7, 7), (10, 7), (10, 10), (0, 10),
        (0, 7), (3, 7), (3, 3), (0, 3)
    ])
