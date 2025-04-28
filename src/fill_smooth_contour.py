import numpy as np
# Use a specific cap style constant
from shapely.geometry import Polygon, MultiPolygon, Point, LinearRing, LineString, CAP_STYLE 
from shapely.ops import unary_union
from shapely.validation import make_valid # Import make_valid
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
        tuple[list[list[tuple]], Polygon | None]: A tuple containing:
            - list[list[tuple]]: A list of paths, where each path is a list of points
                                 representing a single contour loop.
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
    min_area_multiplier = config.get('contour_min_area_multiplier', 4.0) # Default to 4.0 if not set
    min_area = toolpath_width * toolpath_width * min_area_multiplier  # Minimum area threshold
    print(f"  Minimum area threshold (multiplier={min_area_multiplier}): {min_area:.4f} sq units")
    
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
            
    # The 'current_polygon' at this point is the boundary of the potential unfilled region
    # (or the last valid polygon before it became too small/invalid)

    # Convert the collected contours (LinearRings) into a list of paths (list of points)
    contour_paths = []
    if contours:
        print(f"  Converting {len(contours)} contours into separate paths.")
        for contour_ring in contours:
            # Exclude the last point as it's the same as the first for LinearRing
            path_points = list(contour_ring.coords)[:-1]
            if len(path_points) >= 2: # Need at least two points for a path segment
                contour_paths.append(path_points)
            else:
                print(f"    Warning: Skipping contour with less than 2 points.")
    else:
        print("  No valid contours generated.")

    # Return the list of contour paths and the final inner polygon
    last_inner_polygon = current_polygon if current_polygon.area > 1e-6 else None

    return contour_paths, last_inner_polygon


# --- Internal Helper Function (Moved from region_fill.py) ---

def _detect_unfilled_regions(original_polygon, coverage_geometry, toolpath_width):
    """
    Calculates the region(s) within the original polygon (excluding holes)
    that are not covered by the provided coverage geometry (e.g., buffered toolpaths).

    Args:
        original_polygon (Polygon): The initial polygon for the layer slice.
        coverage_geometry (Polygon | MultiPolygon | None): The geometry representing the area
                                                           covered by the toolpaths.
        toolpath_width (float): The width of the toolpath (used for logging/context).

    Returns:
        list[Polygon]: A list of polygons representing the unfilled areas.
    """
    unfilled = []
    if not original_polygon.is_valid or original_polygon.is_empty:
        print("  Original polygon invalid or empty for unfilled region detection.")
        return []

    # Use the provided coverage_geometry, or an empty polygon if None/invalid
    if coverage_geometry is None or not coverage_geometry.is_valid or coverage_geometry.is_empty:
        print("  No valid coverage geometry provided; considering entire polygon (minus holes) as unfilled.")
        # If the original polygon is simple (no holes), return it directly.
        if not original_polygon.interiors:
             return [original_polygon]
        else:
             # Proceed with difference calculation against an empty geometry
             coverage_geometry = Polygon() # Use empty polygon for difference calculation

    try:
        # Calculate the difference: Original Polygon - Coverage Geometry = Unfilled Area
        # This automatically respects holes in the original_polygon
        difference = original_polygon.difference(coverage_geometry)
        
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

    # 1. Generate the main contour fill paths
    print("  Generating main contour paths...")
    # generate_smooth_contour_fill now returns a list of paths
    main_contour_paths, _ = generate_smooth_contour_fill(polygon, toolpath_width)

    if main_contour_paths:
        all_paths.extend(main_contour_paths) # Use extend for list of paths
        num_main_paths = len(main_contour_paths)
        num_main_points = sum(len(p) for p in main_contour_paths)
        print(f"    Generated {num_main_paths} main contour path(s) with {num_main_points} total points.")
    else:
        print("    WARNING: Failed to generate main contour path.")
        # If the main contour fails, the entire polygon is considered unfilled.

    # 2. Detect unfilled regions based on the generated main contour paths
    #    (or the original polygon if the main path failed)
    print("  Detecting unfilled regions...")
    # _detect_unfilled_regions expects coverage geometry.
    # Calculate coverage from the list of main_contour_paths.
    coverage_geometry = None
    if main_contour_paths: # Use the plural form from the previous block
        try:
            buffered_paths = []
            for path in main_contour_paths:
                 if len(path) >= 2:
                      line = LineString(path)
                      buffered_paths.append(line.buffer(toolpath_width / 2.0, cap_style=CAP_STYLE.flat))
            if buffered_paths:
                 coverage_geometry = unary_union(buffered_paths)
                 if not coverage_geometry.is_valid:
                      print("    Warning: Union of buffered paths resulted in invalid geometry, attempting fix.")
                      coverage_geometry = make_valid(coverage_geometry)
        except Exception as e:
            print(f"    Warning: Error creating coverage geometry from paths: {e}")

    unfilled_regions = _detect_unfilled_regions(polygon, coverage_geometry, toolpath_width)
    print(f"    Detected {len(unfilled_regions)} unfilled region(s).")

    # 3. Generate contour fill for each detected unfilled region
    total_local_points = 0
    for i, region in enumerate(unfilled_regions):
        print(f"    Generating local contour fill for unfilled region {i+1} (Area: {region.area:.2f})...")
        # Recursively call the standard contour fill for the small region
        # This will return a list of paths for the local region
        local_paths, _ = generate_smooth_contour_fill(region, toolpath_width)

        if local_paths:
            num_local_paths = len(local_paths)
            num_local_points = sum(len(p) for p in local_paths)
            print(f"      Generated {num_local_paths} local path(s) with {num_local_points} points.")
            all_paths.extend(local_paths) # Add the local paths to the list
            total_local_points += num_local_points
        else:
            print(f"      WARNING: Failed to generate local contour fill for region {i+1}.")

    total_points = sum(len(p) for p in all_paths)
    print(f"  Enhanced contour fill finished. Generated {len(all_paths)} total path(s) with {total_points} total points.")

    return all_paths


# --- Contour Connection Logic (REMOVED) ---
# The functions connect_contours and create_smooth_connection have been removed
# as the contour fill algorithms now return lists of separate paths instead of
# a single connected path. Path connection/optimization is handled by the
# ToolpathOptimizer class.
