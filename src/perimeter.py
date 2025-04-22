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
    current_polygon = polygon

    print(f"  Generating {perimeter_count} perimeter(s)...")

    for i in range(perimeter_count):
        # Calculate offset distance for this perimeter line
        # The first perimeter is offset by half width, subsequent ones by full width
        offset = -toolpath_width / 2.0 if i == 0 else -toolpath_width
        
        # Apply the offset to the current polygon boundary
        try:
            # Offset the exterior inwards
            inner_boundary = current_polygon.buffer(offset, join_style=2) # MITRE join style

            if not inner_boundary.is_valid:
                 print(f"    Warning: Perimeter {i+1} offset resulted in invalid geometry, attempting fix.")
                 inner_boundary = make_valid(inner_boundary)
                 # inner_boundary = inner_boundary.buffer(0) # Older shapely

            if inner_boundary.is_empty or not isinstance(inner_boundary, (Polygon, MultiPolygon)):
                 print(f"    Stopping perimeter generation at line {i+1}: Offset resulted in empty or non-polygon geometry.")
                 current_polygon = None # Mark as invalid/too small
                 break

            # Handle MultiPolygon result (can happen if offset splits the shape)
            # We typically want the largest resulting area for the next iteration
            if isinstance(inner_boundary, MultiPolygon):
                 print(f"    Warning: Perimeter {i+1} offset resulted in MultiPolygon. Selecting largest part.")
                 largest_poly = max(inner_boundary.geoms, key=lambda p: p.area)
                 if not isinstance(largest_poly, Polygon):
                      print(f"    Stopping perimeter generation at line {i+1}: Largest part is not a Polygon.")
                      current_polygon = None
                      break
                 inner_boundary = largest_poly

            # Extract the path for this perimeter
            # The path is the exterior of the *previous* polygon boundary
            # and the interiors (holes) of the *previous* polygon boundary
            perimeter_line_outer = list(current_polygon.exterior.coords)
            perimeter_paths.append(perimeter_line_outer)
            
            for hole in current_polygon.interiors:
                 # Offset holes outwards by the same amount
                 # Note: Offsetting holes requires careful handling, 
                 # simply adding hole paths might not be correct for multiple perimeters.
                 # For now, we add the original hole boundaries offset inwards.
                 # A more robust approach might involve difference operations.
                 # Let's just add the exterior path for now for simplicity.
                 # TODO: Revisit hole handling for multiple perimeters.
                 pass # Simplified: Only adding exterior perimeter for now

            # Update the current polygon for the next iteration
            current_polygon = inner_boundary
            print(f"    Generated perimeter {i+1}. Remaining area: {current_polygon.area:.2f}")

        except Exception as e:
            print(f"    Error generating perimeter {i+1}: {e}")
            current_polygon = None # Mark as invalid
            break

    # The final 'current_polygon' is the area left for infill
    inner_fill_polygon = current_polygon if current_polygon and current_polygon.is_valid and not current_polygon.is_empty else None

    print(f"  Finished generating {len(perimeter_paths)} perimeter paths.")
    if inner_fill_polygon:
        print(f"  Inner polygon remaining for fill: Area={inner_fill_polygon.area:.2f}")
    else:
        print("  No inner polygon remaining for fill.")
        
    return perimeter_paths, inner_fill_polygon
