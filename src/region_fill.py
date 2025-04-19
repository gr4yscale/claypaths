import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, LinearRing, MultiPolygon # Added MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import scale, translate
from matplotlib.collections import LineCollection # Import moved here as it's used by visualize_fill_path

from src.config import get_config # Import config getter
from shapely.validation import make_valid

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
            For 'contour', returns a single list of points (list[tuple]).
            For 'zigzag', returns a list of paths (list[list[tuple]]), split by holes.
            For 'hybrid_contour_zigzag', returns a list of paths (list[list[tuple]]), 
            containing the contour path first, followed by zigzag paths for unfilled regions.
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
    unfilled_regions_for_viz = [] # Store unfilled regions for visualization if hybrid

    if algorithm == 'contour':
        print("Generating contour-based fill pattern...")
        # Note: generate_smooth_contour_fill now returns (path, last_polygon)
        contour_path, _ = generate_smooth_contour_fill(polygon, toolpath_width) 
        fill_result = contour_path # Keep return type as single path for pure contour
        if fill_result:
             print(f"Successfully generated contour fill path with {len(fill_result)} points")
        else:
             print("WARNING: Failed to generate contour fill path (empty result)")

    elif algorithm == 'zigzag':
        print("Generating zigzag fill pattern...")
        # TODO: Make angle configurable? Defaulting to 45 degrees.
        zigzag_paths = generate_zigzag_fill(polygon, toolpath_width, angle=45) # Returns list[list[tuple]]
        fill_result = zigzag_paths # Keep return type as list of paths
        if fill_result:
             num_paths = len(fill_result)
             num_points = sum(len(p) for p in fill_result)
             print(f"Successfully generated {num_paths} zigzag path(s) with {num_points} total points")
        else:
             print("WARNING: Failed to generate zigzag fill path(s) (empty result)")

    elif algorithm == 'hybrid_contour_zigzag':
        print("Generating hybrid contour + zigzag fill pattern...")
        all_paths = []
        
        # 1. Generate contour fill
        contour_path, last_inner_polygon = generate_smooth_contour_fill(polygon, toolpath_width)
        if contour_path:
            all_paths.append(contour_path) # Add contour path as the first path
            print(f"  Generated contour part with {len(contour_path)} points.")
        else:
            print("  WARNING: Failed to generate contour part.")

        # 2. Detect unfilled regions
        unfilled_regions = _detect_unfilled_regions(polygon, last_inner_polygon)
        unfilled_regions_for_viz = unfilled_regions # Save for visualization
        print(f"  Detected {len(unfilled_regions)} unfilled region(s).")

        # 3. Generate zigzag fill for unfilled regions
        total_zigzag_points = 0
        for i, region in enumerate(unfilled_regions):
            print(f"    Generating zigzag fill for unfilled region {i+1}...")
            # TODO: Make angle configurable?
            region_zigzag_paths = generate_zigzag_fill(region, toolpath_width, angle=45)
            if region_zigzag_paths:
                num_region_paths = len(region_zigzag_paths)
                num_region_points = sum(len(p) for p in region_zigzag_paths)
                print(f"      Generated {num_region_paths} zigzag path(s) with {num_region_points} points.")
                all_paths.extend(region_zigzag_paths) # Add zigzag paths to the list
                total_zigzag_points += num_region_points
            else:
                print(f"      WARNING: Failed to generate zigzag fill for region {i+1}.")
        
        fill_result = all_paths # Return type is list of paths
        total_points = sum(len(p) for p in fill_result)
        print(f"Successfully generated hybrid fill with {len(fill_result)} total path(s) and {total_points} total points.")
             
    # Add other algorithms here with 'elif algorithm == "other_algo":'
    else:
        print(f"ERROR: Unknown region fill algorithm specified in config: {algorithm}")
        return [] # Return empty list/path for unknown algorithm

    # Visualize the result (passing unfilled regions if generated)
    visualize_fill_path(polygon, fill_result, 
                        title=f"Fill Path(s) - Algorithm: {algorithm}", 
                        unfilled_regions=unfilled_regions_for_viz)

    return fill_result


# --- Internal Helper Functions ---

def _detect_unfilled_regions(original_polygon, inner_boundary_polygon):
    """
    Calculates the region(s) between the original polygon and the inner boundary.

    Args:
        original_polygon (Polygon): The initial polygon for the layer slice.
        inner_boundary_polygon (Polygon | None): The innermost polygon reached by contour filling.

    Returns:
        list[Polygon]: A list of polygons representing the unfilled areas.
                       Returns an empty list if inner_boundary is None or invalid.
    """
    unfilled = []
    if inner_boundary_polygon is None or not inner_boundary_polygon.is_valid or inner_boundary_polygon.is_empty:
        # If no inner boundary, the whole original polygon might be considered unfilled 
        # (or contour fill failed), but for zigzag fill, let's return empty for now.
        # Alternatively, could return [original_polygon] if contour path was empty.
        print("  No valid inner boundary polygon provided for difference calculation.")
        return []

    try:
        # Calculate the difference: Original - Inner = Unfilled Area
        difference = original_polygon.difference(inner_boundary_polygon)
        
        # Ensure the result is valid
        if not difference.is_valid:
            difference = make_valid(difference) # Requires shapely >= 1.8
            # difference = difference.buffer(0) # Older shapely versions

        if difference.is_empty:
            print("  Difference calculation resulted in empty geometry.")
        elif isinstance(difference, Polygon):
            unfilled.append(difference)
        elif isinstance(difference, MultiPolygon):
            unfilled.extend(list(difference.geoms))
        else:
            print(f"  Difference calculation resulted in unexpected type: {difference.geom_type}")
            
    except Exception as e:
        print(f"  Error calculating difference for unfilled regions: {e}")

    return unfilled


# --- Visualization Utility ---

def visualize_fill_path(polygon, path, title="Continuous Fill Path", unfilled_regions=None):
    """
    Visualize the polygon, the fill path(s), and optionally the unfilled regions.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon.
        path (list | list[list]): Either a single path (list of points) 
                                   or a list of paths (list of lists of points).
        title (str): Title for the plot.
        unfilled_regions (list[Polygon], optional): List of polygons representing unfilled areas.
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
        ax.plot(x, y, 'b--', linewidth=1) # Use dashed line for holes

    # Plot unfilled regions if provided
    if unfilled_regions:
        print(f"  Visualizing {len(unfilled_regions)} unfilled region(s)...")
        for i, region in enumerate(unfilled_regions):
            if isinstance(region, Polygon):
                x, y = region.exterior.xy
                ax.fill(x, y, alpha=0.3, fc='yellow', ec='orange', linewidth=1, linestyle='--', label='Unfilled Region' if i == 0 else "")
                for interior in region.interiors:
                    x_int, y_int = interior.xy
                    ax.fill(x_int, y_int, alpha=1.0, fc='white', ec='orange', linewidth=1, linestyle='--') # Punch holes visually

    # Determine if we have a single path or a list of paths
    # Handle cases: empty list, list with one path, list with multiple paths
    is_list_of_paths = False
    if path:
        # Check if the first element is a list (of coordinates) or a tuple (coordinate)
        # This distinguishes between list[list[tuple]] and list[tuple]
        if isinstance(path[0], list):
             is_list_of_paths = True
        elif isinstance(path[0], tuple):
             # It's a single path (list of tuples), treat as list containing one path
             path = [path] # Wrap the single path in a list for consistent processing below
             is_list_of_paths = True # Now it is technically a list of paths (albeit one)
        # If path[0] is neither list nor tuple, something is wrong, but proceed assuming list of paths

    # Plot the fill path(s)
    if path and is_list_of_paths: # Now always process as list of paths
            print(f"  Visualizing {len(path)} path(s)...")
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
                 # Use different colors/markers for start/end of each path
                 ax.plot(start_x, start_y, 'go', markersize=5, alpha=0.7, label='Path Starts' if not ax.get_legend() else "")
                 ax.plot(end_x, end_y, 'ro', markersize=5, alpha=0.7, label='Path Ends' if not ax.get_legend() else "")
            elif total_points > 0: # Only single points were generated
                 start_x, start_y = zip(*start_points)
                 ax.plot(start_x, start_y, 'go', markersize=5, label='Single Points' if not ax.get_legend() else "")

    # Add legend if labels were added
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Remove duplicate labels
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
