import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, LinearRing
from shapely.ops import unary_union
from shapely.affinity import scale, translate
from matplotlib.collections import LineCollection # Import moved here as it's used by visualize_fill_path

from src.config import get_config # Import config getter
from src.fill_smooth_contour import generate_smooth_contour_fill # Import contour fill
from src.fill_zigzag import generate_zigzag_fill # Import zigzag fill

# --- Public Fill Function ---

def generate_continuous_fill(polygon, toolpath_width, prev_end_point=None):
    """
    Generate a continuous fill pattern for a polygon based on the configured algorithm.
    Acts as a dispatcher to the specific fill algorithm implementations.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple): The end point of the previous layer's path (x, y) - not used here
                               as the optimizer will handle path ordering
        
    Returns:
        list: List of points representing the continuous toolpath
    """
    # Get polygon properties for logging
    area = 0
    perimeter = 0
    num_interiors = 0
    is_valid = False
    is_ccw = False
    
    try:
        if isinstance(polygon, Polygon):
            area = polygon.area
            perimeter = polygon.length
            num_interiors = len(list(polygon.interiors))
            is_valid = polygon.is_valid
            is_ccw = polygon.exterior.is_ccw
    except Exception as e:
        print(f"Error getting polygon properties: {e}")
    
    print(f"\nGenerating fill for polygon:")
    print(f"  Area: {area:.2f} sq units")
    print(f"  Perimeter: {perimeter:.2f} units")
    print(f"  Number of holes: {num_interiors}")
    print(f"  Is valid: {is_valid}")
    print(f"  Exterior orientation: {'CCW' if is_ccw else 'CW'}")
    print(f"  Toolpath width: {toolpath_width:.3f}")
    
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        print("ERROR: Invalid polygon for region fill (not a polygon or empty)")
        return []
    
    if not polygon.is_valid:
        print("WARNING: Polygon is not valid, attempting to fix...")
        try:
            # Try to fix the polygon
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                print("ERROR: Failed to fix invalid polygon")
                return []
            print("  Successfully fixed polygon")
        except Exception as e:
            print(f"ERROR: Exception while fixing polygon: {e}")
            return []
    
    # Check if the polygon is a hole (has interiors)
    # if len(list(polygon.interiors)) > 0:
    #     # This is a polygon with holes - process it normally
    #     contour_path = generate_contour_fill(polygon, toolpath_width)
    #     return contour_path
    
    # Check if this polygon might be a hole itself
    # A hole typically has a counterclockwise orientation
    # if not polygon.exterior.is_ccw:
    #     print("Skipping fill for hole polygon (counterclockwise exterior)")
    #     return []
    
    # Get the selected fill algorithm from config
    config = get_config()
    algorithm = config.get('region_fill_algorithm', 'contour') # Default to contour
    print(f"Using region fill algorithm: {algorithm}")

    # Dispatch to the appropriate fill function
    fill_path = []
    if algorithm == 'contour':
        print("Generating contour-based fill pattern...")
        fill_path = generate_smooth_contour_fill(polygon, toolpath_width) # Call the imported function
    elif algorithm == 'zigzag':
        print("Generating zigzag fill pattern...")
        # TODO: Make angle configurable? Defaulting to 45 degrees.
        fill_path = generate_zigzag_fill(polygon, toolpath_width, angle=45)
    # Add other algorithms here with 'elif algorithm == "other_algo":'
    else:
        print(f"ERROR: Unknown region fill algorithm specified in config: {algorithm}")
        return [] # Return empty path for unknown algorithm
    if fill_path:
        print(f"Successfully generated fill path with {len(fill_path)} points")
    else:
        print("WARNING: Failed to generate fill path (empty result)")
    
    return fill_path


# --- General Path Utilities ---
# (These functions might be useful for other fill algorithms or visualization)

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
        # Note: LineCollection is imported at the top now
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
