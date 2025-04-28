import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LinearRing # Added Point, LinearRing
from shapely.ops import unary_union
from src.config import get_config
# Removed import of _detect_unfilled_regions from region_fill
# Removed: from src.region_fill import connect_contours

def generate_smooth_contour_fill(polygon, toolpath_width):
    """
    Generates a continuous fill pattern using inward contours.
    This is the actual implementation for the 'contour' algorithm.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill.
        toolpath_width (float): Width of the toolpath.

    Returns:
        tuple[list, Polygon | None]: A tuple containing:
            - list: List of points representing the continuous toolpath.
            - Polygon | None: The innermost polygon boundary reached, or None if no fill generated.
    """
    # The code below this comment block was the core logic of the original contour fill
    if not polygon.is_valid:
        print("Invalid polygon for contour fill")
        return []
    
    # Get the configuration to check the polygonization method
    config = get_config()
    polygonization_method = config.get('polygonization_method', 'standard')
    
    print(f"  Using polygonization method: {polygonization_method}")
    
    # Start with the original polygon boundary
    current_polygon = polygon
    contours = []
    
    # Generate inward contours until the polygon becomes too small
    min_area = toolpath_width * toolpath_width * 4  # Minimum area threshold
    print(f"  Minimum area threshold: {min_area:.4f} sq units")
    
    contour_count = 0
    
    while current_polygon.area > min_area:
        # Add the current contour
        contours.append(current_polygon.exterior)
        contour_count += 1
        
        # Generate the next inward contour
        # Adjust buffer distance based on polygonization method
        buffer_distance = -toolpath_width
        
        # For small_buffer and large_buffer methods, we need to be more careful
        # with the buffer operation to avoid invalid geometries
        if polygonization_method in ['small_buffer', 'large_buffer']:
            # Use a more conservative buffer distance for these methods
            # to prevent issues with the already buffered polygons
            buffer_distance = -max(toolpath_width * 0.9, 0.1)
            
            # For very small polygons, be even more conservative
            if current_polygon.area < min_area * 2:
                buffer_distance = -max(toolpath_width * 0.8, 0.05)
        
        print(f"  Contour {contour_count}: Area={current_polygon.area:.4f}, Buffer={buffer_distance:.4f}")
        
        try:
            next_polygon = current_polygon.buffer(buffer_distance, join_style=2)
            
            # Check if the buffer operation resulted in a valid polygon
            if next_polygon.is_empty:
                print(f"  Stopping: Buffer resulted in empty polygon")
                break
                
            # Handle potential MultiPolygon results
            if next_polygon.geom_type == 'MultiPolygon':
                print(f"  Buffer resulted in MultiPolygon with {len(next_polygon.geoms)} parts")
                # Take the largest polygon from the multipolygon
                largest_area = 0
                largest_poly = None
                for poly in next_polygon.geoms:
                    if poly.area > largest_area:
                        largest_area = poly.area
                        largest_poly = poly
                
                if largest_poly is None or largest_poly.area < min_area:
                    print(f"  Stopping: Largest polygon too small ({largest_area if largest_poly else 0:.4f} < {min_area:.4f})")
                    break
                    
                next_polygon = largest_poly
                print(f"  Selected largest polygon with area {largest_area:.4f}")
            elif next_polygon.geom_type != 'Polygon':
                print(f"  Stopping: Buffer resulted in {next_polygon.geom_type} instead of Polygon")
                break
                
            current_polygon = next_polygon
        except Exception as e:
            print(f"  Error during buffer operation: {e}")
            break
            
    # The 'current_polygon' at this point is the boundary of the unfilled region
    # (or the last valid polygon before it became too small/invalid)
    
    # Connect the contours to form a continuous spiral path
    # We don't optimize for previous end point here as the toolpath optimizer will handle that
    path = connect_contours(contours, toolpath_width)
    
    # Return the path and the final inner polygon
    # Return None for the polygon if the buffer failed early or area was too small initially
    last_inner_polygon = current_polygon if current_polygon.area > 1e-6 else None

    return path, last_inner_polygon


# --- Internal Helper Function (Moved from region_fill.py) ---

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


# --- Enhanced Contour Fill ---

def generate_enhanced_contour_fill(polygon, toolpath_width):
    """
    Generates a fill pattern using inward contours, and then fills any remaining
    unfilled regions with additional local contour paths.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill.
        toolpath_width (float): Width of the toolpath.

    Returns:
        list[list[tuple]]: A list containing:
            - The main continuous contour path (if generated).
            - Additional paths for locally filled regions (if any).
            Returns an empty list if generation fails.
    """
    all_paths = []

    # 1. Generate the main contour fill path
    print("  Generating main contour path...")
    main_contour_path, _ = generate_smooth_contour_fill(polygon, toolpath_width)

    if main_contour_path:
        all_paths.append(main_contour_path)
        print(f"    Generated main contour path with {len(main_contour_path)} points.")
    else:
        print("    WARNING: Failed to generate main contour path.")
        # If the main contour fails, the entire polygon is considered unfilled.

    # 2. Detect unfilled regions based on the generated main contour path
    #    (or the original polygon if the main path failed)
    print("  Detecting unfilled regions...")
    unfilled_regions = _detect_unfilled_regions(polygon, main_contour_path, toolpath_width)
    print(f"    Detected {len(unfilled_regions)} unfilled region(s).")

    # 3. Generate contour fill for each detected unfilled region
    total_local_points = 0
    for i, region in enumerate(unfilled_regions):
        print(f"    Generating local contour fill for unfilled region {i+1} (Area: {region.area:.2f})...")
        # Recursively call the standard contour fill for the small region
        local_path, _ = generate_smooth_contour_fill(region, toolpath_width)

        if local_path:
            num_local_points = len(local_path)
            print(f"      Generated local path with {num_local_points} points.")
            all_paths.append(local_path) # Add the local path to the list
            total_local_points += num_local_points
        else:
            print(f"      WARNING: Failed to generate local contour fill for region {i+1}.")

    total_points = sum(len(p) for p in all_paths)
    print(f"  Enhanced contour fill finished. Generated {len(all_paths)} total path(s) with {total_points} total points.")

    return all_paths


# --- Contour Connection Logic (Moved from region_fill.py) ---

def connect_contours(contours, toolpath_width, prev_end_point=None):
    """
    Connect contours to form a continuous spiral path.
    
    Args:
        contours (list): List of LinearRings representing contours
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple, optional): The end point of the previous layer's path (x, y)
        
    Returns:
        list: List of points representing the continuous path
    """
    if not contours:
        print("  No contours to connect")
        return []
    
    print(f"  Connecting {len(contours)} contours to form continuous path")
    path = []
    
    # Start with the outermost contour
    outer_contour_coords = list(contours[0].coords)[:-1]  # Exclude the last point as it's the same as the first
    print(f"  Outer contour has {len(outer_contour_coords)} points")
    
    # Add the outer contour points to the path
    for x, y in outer_contour_coords:
        path.append((x, y))
    
    # Connect to inner contours
    for i in range(1, len(contours)):
        # Find the closest point on the next contour to the current position
        current_point = Point(path[-1])
        next_contour = contours[i]
        
        # Sample points along the contour to find the closest
        next_coords = list(next_contour.coords)[:-1]  # Exclude the last duplicate point
        print(f"  Contour {i} has {len(next_coords)} points")
        
        # Find the closest point
        min_dist = float('inf')
        closest_idx = 0
        
        for j, point in enumerate(next_coords):
            dist = current_point.distance(Point(point))
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        
        print(f"  Connecting to contour {i} at point {closest_idx} (distance: {min_dist:.4f})")
        
        # Create a smooth connection to the next contour
        connection_points = create_smooth_connection(
            path[-1], 
            next_coords[closest_idx], 
            toolpath_width
        )
        
        print(f"  Added {len(connection_points)} connection points")
        
        for point in connection_points:
            path.append(point)
        
        # Add the next contour, starting from the closest point and wrapping around
        for j in range(len(next_coords)):
            idx = (closest_idx + j) % len(next_coords)
            path.append(next_coords[idx])
    
    return path

def create_smooth_connection(start_point, end_point, toolpath_width):
    """
    Create a smooth connection between two points.
    
    Args:
        start_point (tuple): Starting point (x, y)
        end_point (tuple): Ending point (x, y)
        toolpath_width (float): Width of the toolpath
        
    Returns:
        list: List of points forming a smooth connection
    """
    # Calculate direction vector
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    # If points are very close, just return a direct line
    if distance < toolpath_width:
        return [end_point]
    
    # Create a smooth curve using a few intermediate points
    num_points = max(3, int(distance / toolpath_width))
    
    # Use a simple curve approximation
    connection_points = []
    
    for i in range(1, num_points):
        t = i / num_points
        # Simple quadratic curve
        x = start_point[0] + t * dx
        y = start_point[1] + t * dy
        connection_points.append((x, y))
    
    return connection_points
