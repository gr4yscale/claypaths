import numpy as np
import matplotlib.pyplot as plt
import logging
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from src.hilbert import coordinates_from_distance, distance_from_coordinates

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_continuous_fill(polygon, toolpath_width=1.0, prev_end_point=None):
    """
    Generate a continuous fill pattern using Hilbert space-filling curves.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple): The end point of the previous layer's path (x, y)
        
    Returns:
        list: List of points representing the continuous toolpath
    """
    # Validate input polygon
    logger.debug("Validating input polygon")
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        logger.error("Invalid polygon for region fill (not a polygon or empty)")
        return []
    
    if not polygon.is_valid:
        logger.warning("Polygon is not valid, attempting to fix...")
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            logger.error("Failed to fix invalid polygon")
            return []

    # Get bounding box of polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    logger.debug(f"Polygon bounds: min_x={min_x:.2f}, min_y={min_y:.2f}, "
                f"max_x={max_x:.2f}, max_y={max_y:.2f}")
    
    # Calculate required order for the Hilbert curve
    size = max(max_x - min_x, max_y - min_y)
    order = int(np.ceil(np.log2(size / toolpath_width)))
    logger.debug(f"Calculated Hilbert curve order: {order} (size={size:.2f}, "
                f"toolpath_width={toolpath_width:.2f})")
    
    # Generate Hilbert curve points
    num_points = 2 ** (2 * order)
    logger.debug(f"Generating {num_points} Hilbert curve points")
    hilbert_points = []
    
    for i in range(num_points):
        # Get normalized coordinates
        x, y = coordinates_from_distance(i, order)
        
        # Scale to polygon bounds
        x_scaled = min_x + x * (max_x - min_x)
        y_scaled = min_y + y * (max_y - min_y)
        
        # Check if point is inside polygon
        point = Point(x_scaled, y_scaled)
        if polygon.contains(point):
            hilbert_points.append((x_scaled, y_scaled))
            logger.debug(f"Point {i}: ({x_scaled:.2f}, {y_scaled:.2f}) - inside polygon")
        else:
            logger.debug(f"Point {i}: ({x_scaled:.2f}, {y_scaled:.2f}) - outside polygon")
    
    # Optimize starting point if previous end point is provided
    if prev_end_point and len(hilbert_points) > 1:
        logger.debug(f"Optimizing starting point relative to previous end point: {prev_end_point}")
        # Find nearest point in Hilbert curve to previous end point
        prev_point = np.array(prev_end_point)
        distances = [np.linalg.norm(np.array(p) - prev_point) 
                    for p in hilbert_points]
        start_idx = np.argmin(distances)
        min_dist = distances[start_idx]
        logger.debug(f"Nearest point index: {start_idx}, distance: {min_dist:.2f}")
        
        # Reorder points to start from nearest point
        hilbert_points = hilbert_points[start_idx:] + hilbert_points[:start_idx]
        logger.debug(f"Reordered points to start from index {start_idx}")
    
    logger.debug(f"Generated {len(hilbert_points)} points in fill path")
    return hilbert_points


def visualize_fill_path(polygon, path, title="Continuous Fill Path"):
    """
    Visualize the polygon and the fill path.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon
        path (list): List of points representing the path
        title (str): Title for the plot
    """
    # Check if visualization is enabled
    from src.config import get_config
    config = get_config()
    if not config.get('visualize_fill_paths', True):
        print("Fill path visualization disabled in config")
        return
        
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the polygon
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon Boundary')
    
    # Plot holes if any
    for interior in polygon.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'b-', linewidth=2)
    
    # Plot the fill path
    if path:
        path_x, path_y = zip(*path)
        
        # Use a colormap to show the direction of the path
        points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a colorful line collection
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=1.5)
        lc.set_array(np.linspace(0, 1, len(path_x)-1))
        ax.add_collection(lc)
        
        # Mark start and end points
        ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')
        
        # Add a colorbar to show progression
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Path Direction')
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
