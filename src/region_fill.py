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
        list | list[list[tuple[float, float]]]: 
            For 'contour', returns a single list of points.
            For 'zigzag', returns a list of paths (each path is a list of points), 
            split where continuity is broken by holes.
            Returns an empty list on failure.
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
    fill_result = []
    if algorithm == 'contour':
        print("Generating contour-based fill pattern...")
        fill_result = generate_smooth_contour_fill(polygon, toolpath_width) # Returns a single path (list)
        if fill_result:
             print(f"Successfully generated contour fill path with {len(fill_result)} points")
        else:
             print("WARNING: Failed to generate contour fill path (empty result)")

    elif algorithm == 'zigzag':
        print("Generating zigzag fill pattern...")
        # TODO: Make angle configurable? Defaulting to 45 degrees.
        fill_result = generate_zigzag_fill(polygon, toolpath_width, angle=45) # Returns list[list[tuple]]
        if fill_result:
             num_paths = len(fill_result)
             num_points = sum(len(p) for p in fill_result)
             print(f"Successfully generated {num_paths} zigzag path(s) with {num_points} total points")
        else:
             print("WARNING: Failed to generate zigzag fill path(s) (empty result)")
             
    # Add other algorithms here with 'elif algorithm == "other_algo":'
    else:
        print(f"ERROR: Unknown region fill algorithm specified in config: {algorithm}")
        return [] # Return empty list/path for unknown algorithm

    return fill_result


# --- General Path Utilities ---
# (These functions might be useful for other fill algorithms or visualization)

def visualize_fill_path(polygon, path, title="Continuous Fill Path"):
    """
    Visualize the polygon and the fill path(s).
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon
        paths (list | list[list]): Either a single path (list of points) 
                                   or a list of paths (list of lists of points).
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

    # Determine if we have a single path or a list of paths
    if path:
        # Check if the first element of path is itself a list (indicating list of paths)
        is_list_of_paths = isinstance(path[0], list) if path else False
    else:
        is_list_of_paths = False

    # Plot the fill path(s)
    if path:
        if is_list_of_paths:
            print(f"  Visualizing {len(path)} separate path segments...")
            all_segments = []
            start_points = []
            end_points = []
            total_points = 0
            for single_path in path: # Iterate through the list of paths
                if len(single_path) > 1:
                    path_x, path_y = zip(*single_path)
                    points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    all_segments.extend(segments)
                    start_points.append((path_x[0], path_y[0]))
                    end_points.append((path_x[-1], path_y[-1]))
                    total_points += len(single_path)
                elif len(single_path) == 1: # Handle single-point paths if they occur
                    start_points.append((single_path[0][0], single_path[0][1]))
                    end_points.append((single_path[0][0], single_path[0][1]))
                    total_points += 1


            if all_segments:
                 # Create a line collection for all segments
                 # Use a single color as directionality across segments is less meaningful
                 lc = LineCollection(all_segments, colors='orange', linewidth=1.5)
                 ax.add_collection(lc)
                 
                 # Mark start and end points of each segment
                 start_x, start_y = zip(*start_points)
                 end_x, end_y = zip(*end_points)
                 ax.plot(start_x, start_y, 'go', markersize=5, label='Segment Starts')
                 ax.plot(end_x, end_y, 'ro', markersize=5, label='Segment Ends')
            elif total_points > 0: # Only single points were generated
                 start_x, start_y = zip(*start_points)
                 ax.plot(start_x, start_y, 'go', markersize=5, label='Points')


        else: # It's a single path
             path_x, path_y = zip(*path) # Unpack the single path
             points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
             segments = np.concatenate([points[:-1], points[1:]], axis=1)

             # Create a colorful line collection for the single path
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
