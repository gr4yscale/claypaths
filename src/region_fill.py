import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# Use a specific cap style constant
from shapely.geometry import Polygon, LineString, Point, LinearRing, MultiPolygon, CAP_STYLE 
from shapely.ops import unary_union
from shapely.affinity import scale, translate
from matplotlib.collections import LineCollection # Import moved here as it's used by visualize_fill_path

from src.config import get_config # Import config getter
from shapely.validation import make_valid

from src.config import get_config # Import config getter
from shapely.validation import make_valid

from src.config import get_config # Import config getter
from src.fill_smooth_contour import generate_smooth_contour_fill # Import contour fill
from src.fill_zigzag import generate_zigzag_fill # Import zigzag fill
from src.fill_fermat_spiral import generate_fermat_spiral_fill # Import Fermat spiral fill

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
            For 'fermat_spiral', returns a single list of points (list[tuple]).
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

    elif algorithm == 'fermat_spiral':
        print("Generating Fermat spiral fill pattern...")
        spiral_path = generate_fermat_spiral_fill(polygon, toolpath_width)
        fill_result = spiral_path # Keep return type as single path
        if fill_result:
             print(f"Successfully generated Fermat spiral fill path with {len(fill_result)} points")
        else:
             print("WARNING: Failed to generate Fermat spiral fill path (empty result)")

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
            # If contour fails, should we fill the whole area with zigzag?
            # For now, we proceed, and _detect_unfilled_regions will likely find the whole area.

        # 2. Detect unfilled regions based on the generated contour path
        unfilled_regions = _detect_unfilled_regions(polygon, contour_path, toolpath_width)
        unfilled_regions_for_viz = unfilled_regions # Save for visualization
        print(f"  Detected {len(unfilled_regions)} unfilled region(s) not covered by contour path.")

        # 3. Generate zigzag fill for detected unfilled regions
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

    # Visualization is now handled in main.py after perimeters are also generated
    # visualize_fill_path(polygon, fill_result, 
    #                     title=f"Fill Path(s) - Algorithm: {algorithm}", 
    #                     unfilled_regions=unfilled_regions_for_viz)

    return fill_result


# --- Internal Helper Functions ---

def _detect_unfilled_regions(original_polygon, contour_path, toolpath_width):
    """
    Calculates the region(s) within the original polygon (excluding holes) 
    that are not covered by the contour toolpath.

    Args:
        original_polygon (Polygon): The initial polygon for the layer slice.
        contour_path (list[tuple]): The list of points representing the contour fill path.
        toolpath_width (float): The width of the toolpath.

    Returns:
        list[Polygon]: A list of polygons representing the unfilled areas.
    """
    unfilled = []
    if not original_polygon.is_valid or original_polygon.is_empty:
        print("  Original polygon invalid or empty for unfilled region detection.")
        return []

    if not contour_path or len(contour_path) < 2:
        print("  No valid contour path provided; considering entire polygon (minus holes) as unfilled.")
        # If the original polygon is simple (no holes), return it directly.
        # If it has holes, the difference calculation below handles it implicitly.
        # However, returning the original directly might be faster if no path exists.
        if not original_polygon.interiors:
             return [original_polygon]
        else:
             # Proceed with difference calculation against an empty geometry
             contour_coverage_area = Polygon() 
    else:
        try:
            # Create area covered by contour path
            path_line = LineString(contour_path)
            # Buffer the line by half the toolpath width on each side
            # Use CAP_STYLE.flat to prevent rounded ends from over-covering
            contour_coverage_area = path_line.buffer(toolpath_width / 2.0, cap_style=CAP_STYLE.flat)
            
            if not contour_coverage_area.is_valid:
                 print("  Warning: Contour coverage area is invalid, attempting fix.")
                 contour_coverage_area = make_valid(contour_coverage_area)
                 # contour_coverage_area = contour_coverage_area.buffer(0) # Older shapely

        except Exception as e:
            print(f"  Error creating contour coverage area: {e}")
            return [] # Cannot determine unfilled regions

    try:
        # Calculate the difference: Original Polygon - Contour Coverage = Unfilled Area
        # This automatically respects holes in the original_polygon
        difference = original_polygon.difference(contour_coverage_area)
        
        # Ensure the resulting difference is valid
        if not difference.is_valid:
            difference = make_valid(difference) # Requires shapely >= 1.8
            # difference = difference.buffer(0) # Older shapely versions

        if difference.is_empty:
            print("  Difference calculation resulted in empty geometry.")
        elif isinstance(difference, Polygon):
            unfilled.append(difference)
        elif isinstance(difference, MultiPolygon):
            # Add only valid polygons from the MultiPolygon
            for poly in difference.geoms:
                if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                    unfilled.append(poly)
        else:
            print(f"  Difference calculation resulted in unexpected type: {difference.geom_type}")
            
    except Exception as e:
        print(f"  Error calculating difference for unfilled regions: {e}")

    # Final check for validity just in case
    valid_unfilled = [p for p in unfilled if p.is_valid and not p.is_empty and p.area > 1e-6]
    
    return valid_unfilled


# --- Visualization Utility ---

def visualize_fill_path(polygon, perimeter_paths, fill_paths, title="Layer Paths", unfilled_regions=None):
    """
    Visualize the original polygon, perimeter paths, fill paths, and optionally unfilled regions.
    
    Args:
        polygon (shapely.geometry.Polygon): The original polygon for the layer slice.
        perimeter_paths (list[list[tuple]]): List of perimeter paths.
        fill_paths (list | list[list]): Fill path(s) (single list or list of lists).
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

    # --- Plot Perimeter Paths ---
    if perimeter_paths:
        print(f"  Visualizing {len(perimeter_paths)} perimeter path(s)...")
        perim_segments = []
        perim_starts = []
        perim_ends = []
        for p_path in perimeter_paths:
             if len(p_path) > 1:
                  px, py = zip(*p_path)
                  points = np.array([px, py]).T.reshape(-1, 1, 2)
                  segments = np.concatenate([points[:-1], points[1:]], axis=1)
                  perim_segments.extend(segments)
                  perim_starts.append((px[0], py[0]))
                  perim_ends.append((px[-1], py[-1])) # Note: for closed loops, start=end

        if perim_segments:
             lc_perim = LineCollection(perim_segments, colors='blue', linewidth=1.5, label='Perimeters')
             ax.add_collection(lc_perim)
             # Optionally mark start/end points for perimeters if needed
             # start_x, start_y = zip(*perim_starts)
             # ax.plot(start_x, start_y, 'co', markersize=4, alpha=0.5) 

    # --- Plot Fill Paths ---
    # Process fill_paths (could be list or list of lists) into a consistent list of paths format
    fill_paths_list = []
    if fill_paths:
        if isinstance(fill_paths[0], list):
            fill_paths_list = fill_paths # Already list of lists
        elif isinstance(fill_paths[0], tuple):
            fill_paths_list = [fill_paths] # Wrap single path
            
    if fill_paths_list: 
            print(f"  Visualizing {len(fill_paths_list)} fill path(s)...")
            fill_segments = []
            start_points = []
            end_points = []
            total_points = 0
            for single_path in fill_paths_list: # Iterate through the list of fill paths
                if len(single_path) > 1:
                    path_x, path_y = zip(*single_path)
                    points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    fill_segments.extend(segments) # Add to fill segments list
                    start_points.append((path_x[0], path_y[0]))
                    end_points.append((path_x[-1], path_y[-1]))
                    total_points += len(single_path)
                elif len(single_path) == 1: # Handle single-point paths if they occur
                    start_points.append((single_path[0][0], single_path[0][1]))
                    end_points.append((single_path[0][0], single_path[0][1]))
                    total_points += 1


            if fill_segments:
                 # Create a line collection for fill segments
                 lc_fill = LineCollection(fill_segments, colors='orange', linewidth=1.0, linestyle='--', label='Fill Paths')
                 ax.add_collection(lc_fill)
                 
                 # Mark start and end points of each fill path segment
                 if start_points:
                      start_x, start_y = zip(*start_points)
                      ax.plot(start_x, start_y, 'go', markersize=4, alpha=0.7, label='Fill Starts' if not ax.get_legend() else "")
                 if end_points:
                      end_x, end_y = zip(*end_points)
                      ax.plot(end_x, end_y, 'ro', markersize=4, alpha=0.7, label='Fill Ends' if not ax.get_legend() else "")
            elif total_points > 0: # Only single points were generated in fill
                 if start_points:
                      start_x, start_y = zip(*start_points)
                      ax.plot(start_x, start_y, 'go', markersize=4, label='Single Fill Points' if not ax.get_legend() else "")

    # Add legend if labels were added
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Create legend with unique labels
        by_label = dict(zip(labels, handles)) # Use dict to automatically handle duplicates
        ax.legend(by_label.values(), by_label.keys())

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
