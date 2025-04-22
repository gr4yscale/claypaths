from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.validation import make_valid

def generate_perimeter_paths(polygon, perimeter_count, toolpath_width):
    """
    Generates perimeter paths (walls) for a given polygon slice.

    Args:
        polygon (Polygon): The input polygon for the layer.
        perimeter_count (int): The number of perimeter lines to generate.
        toolpath_width (float): The width of a single toolpath line.

    Returns:
        tuple[list[list[tuple]], Polygon | None]: A tuple containing:
            - A list of perimeter paths (each path is a list of points).
            - The innermost polygon remaining after offsetting for perimeters, 
              or None if the polygon becomes invalid or too small.
    """
    if not polygon.is_valid or polygon.is_empty or perimeter_count <= 0:
        return [], polygon # Return original polygon if invalid or no perimeters needed

    perimeter_paths = []
    inner_fill_polygon = polygon # Start with the original polygon

    print(f"  Generating {perimeter_count} perimeter(s)...")

    # Store the polygon whose boundaries will form the paths for the current iteration
    poly_for_paths = polygon 

    for i in range(perimeter_count):
        # Calculate the inward offset distance from the original boundary
        # Offset 1: -width/2
        # Offset 2: -width/2 - width
        # Offset i: -width/2 - (i-1)*width  (for i >= 1)
        # Using 0-based index 'i': offset = -(i + 0.5) * toolpath_width
        offset_dist = -(i + 0.5) * toolpath_width
        
        print(f"    Calculating offset {i+1} at distance {offset_dist:.3f}")
        
        try:
            # Generate the paths from the *previous* offset polygon's boundaries
            if poly_for_paths:
                 # Add exterior path
                 perimeter_paths.append(list(poly_for_paths.exterior.coords))
                 # Add interior (hole) paths
                 for interior in poly_for_paths.interiors:
                      perimeter_paths.append(list(interior.coords))
                 print(f"      Added paths from boundary of polygon with area {poly_for_paths.area:.2f}")
            else:
                 print(f"      Skipping path generation for perimeter {i+1} as previous polygon was invalid.")
                 inner_fill_polygon = None # No fill possible if a perimeter fails
                 break # Stop generating perimeters

            # Calculate the next inner polygon by offsetting the original polygon
            offset_poly = polygon.buffer(offset_dist, join_style=2) # MITRE join style

            # --- Validation and Handling of Offset Result ---
            if offset_poly.is_empty:
                 print(f"    Stopping perimeter generation at offset {i+1}: Result is empty.")
                 inner_fill_polygon = None 
                 break
            
            if not offset_poly.is_valid:
                 print(f"    Warning: Offset {i+1} resulted in invalid geometry, attempting fix.")
                 offset_poly = make_valid(offset_poly)
                 # offset_poly = offset_poly.buffer(0) # Older shapely
                 if not offset_poly.is_valid or offset_poly.is_empty:
                      print(f"    Stopping perimeter generation at offset {i+1}: Could not fix invalid geometry.")
                      inner_fill_polygon = None
                      break

            # Handle MultiPolygon result (take the largest valid polygon)
            if isinstance(offset_poly, MultiPolygon):
                 print(f"    Warning: Offset {i+1} resulted in MultiPolygon. Selecting largest valid part.")
                 valid_polygons = [p for p in offset_poly.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
                 if not valid_polygons:
                      print(f"    Stopping perimeter generation at offset {i+1}: No valid polygons in MultiPolygon result.")
                      inner_fill_polygon = None
                      break
                 offset_poly = max(valid_polygons, key=lambda p: p.area)
                 print(f"      Selected largest polygon with area {offset_poly.area:.2f}")
            
            # Check if the result is a valid Polygon
            if not isinstance(offset_poly, Polygon):
                 print(f"    Stopping perimeter generation at offset {i+1}: Result is not a Polygon (type: {type(offset_poly)}).")
                 inner_fill_polygon = None
                 break
            # --- End Validation ---

            # Update for the next iteration
            poly_for_paths = offset_poly # This polygon's boundaries are the paths for the *next* perimeter
            inner_fill_polygon = offset_poly # This is the potential fill area after this offset

            print(f"    Offset {i+1} successful. New inner area: {inner_fill_polygon.area:.2f}")

        except Exception as e:
            print(f"    Error during offset {i+1} calculation: {e}")
            inner_fill_polygon = None # Mark as invalid
            break

    # Final check on the inner fill polygon
    if inner_fill_polygon and (not inner_fill_polygon.is_valid or inner_fill_polygon.is_empty):
         inner_fill_polygon = None

    print(f"  Finished generating {len(perimeter_paths)} perimeter paths.")
    if inner_fill_polygon:
        print(f"  Inner polygon remaining for fill: Area={inner_fill_polygon.area:.2f}")
    else:
        print("  No inner polygon remaining for fill.")
        
    return perimeter_paths, inner_fill_polygon
