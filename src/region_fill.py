import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, LinearRing
from shapely.ops import unary_union
from shapely.affinity import scale, translate

def generate_continuous_fill(polygon, toolpath_width=1.0, prev_end_point=None):
    """
    Generate a continuous fill pattern for a polygon using smooth contour-based paths.
    Only generates fill for polygons that are not holes.
    
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
    
    # Generate contour-based fill without optimizing for previous end point
    # The toolpath optimizer will handle the optimization across layers
    print("Generating contour-based fill pattern...")
    contour_path, zigzag_paths = generate_contour_fill(polygon, toolpath_width)
    
    if contour_path:
        print(f"Successfully generated contour path with {len(contour_path)} points")
    else:
        print("WARNING: Failed to generate contour path (empty result)")
    
    if zigzag_paths:
        total_zigzag_points = sum(len(z) for z in zigzag_paths)
        print(f"Generated {len(zigzag_paths)} zig-zag fills with {total_zigzag_points} total points")
    else:
        print("No zig-zag fills generated")
    
    # Return both path types separately for visualization and later combination
    return contour_path, zigzag_paths


def generate_zigzag_fill(polygon, toolpath_width):
    """
    Generate zig-zag fill pattern for a polygon, clipped to the exact polygon boundaries.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath

    Returns:
        list: List of points representing the zig-zag path, or empty list if fill is not possible/needed.
    """
    if not polygon.is_valid or polygon.is_empty:
        # print(f"  Skipping zig-zag for invalid/empty polygon area: {polygon.area:.4f}")
        return []

    # Get polygon bounds
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y

    # Don't generate zig-zag for extremely thin slivers
    # Check if both dimensions are smaller than half the toolpath width
    if width < toolpath_width / 2 and height < toolpath_width / 2:
        # print(f"  Skipping zig-zag for very small polygon area: {polygon.area:.4f}")
        return []

    # Generate lines across the bounding box
    # Choose direction based on aspect ratio (fill along the longer dimension)
    zigzag_lines = []
    spacing = toolpath_width

    if width >= height:
        # Fill vertically (lines parallel to y-axis)
        num_passes = int(width / spacing)
        if num_passes < 1 and width > 0: num_passes = 1 # Ensure at least one pass if width > 0

        for i in range(num_passes + 1):
            x = min_x + i * spacing + spacing / 2 # Center lines within spacing
            if x > max_x + spacing/2: continue # Avoid lines significantly outside bounds
            # Extend lines slightly beyond bounds to ensure full intersection
            line = LineString([(x, min_y - height*0.1 - spacing), (x, max_y + height*0.1 + spacing)])
            zigzag_lines.append(line)
    else:
        # Fill horizontally (lines parallel to x-axis)
        num_passes = int(height / spacing)
        if num_passes < 1 and height > 0: num_passes = 1 # Ensure at least one pass if height > 0

        for i in range(num_passes + 1):
            y = min_y + i * spacing + spacing / 2 # Center lines within spacing
            if y > max_y + spacing/2: continue # Avoid lines significantly outside bounds
            # Extend lines slightly beyond bounds to ensure full intersection
            line = LineString([(min_x - width*0.1 - spacing, y), (max_x + width*0.1 + spacing, y)])
            zigzag_lines.append(line)

    # Clip lines to the polygon
    clipped_path_segments = []
    for line in zigzag_lines:
        try:
            intersection = polygon.intersection(line)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    clipped_path_segments.append(list(intersection.coords))
                elif intersection.geom_type == 'MultiLineString':
                    for segment in intersection.geoms:
                        clipped_path_segments.append(list(segment.coords))
        except Exception as e:
            print(f"  Warning: Error during zig-zag line intersection: {e}")
            # Optionally try buffering the polygon slightly if intersection fails
            try:
                intersection = polygon.buffer(1e-9).intersection(line)
                if not intersection.is_empty:
                     if intersection.geom_type == 'LineString':
                         clipped_path_segments.append(list(intersection.coords))
                     elif intersection.geom_type == 'MultiLineString':
                         for segment in intersection.geoms:
                             clipped_path_segments.append(list(segment.coords))
            except Exception as e2:
                 print(f"  Warning: Intersection failed even after buffer: {e2}")


    if not clipped_path_segments:
        # print(f"  No valid zig-zag segments generated for area {polygon.area:.4f}")
        return []

    # Connect segments into a zig-zag path (simple alternating connection)
    zigzag_path = []
    for i, segment in enumerate(clipped_path_segments):
        if i % 2 == 1: # Reverse every other segment for zig-zag
            segment.reverse()
        # Add segment points, avoiding duplicates between segments
        if zigzag_path and segment and zigzag_path[-1] == segment[0]:
            zigzag_path.extend(segment[1:])
        else:
            zigzag_path.extend(segment)

    # print(f"  Generated zig-zag with {len(zigzag_path)} points for area {polygon.area:.4f}")
    return zigzag_path


def generate_contour_fill(polygon, toolpath_width, prev_end_point=None):
    """
    Generate a continuous fill pattern using inward contours and zig-zag fills for underfill regions.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple, optional): The end point of the previous layer's path (x, y)
        
    Returns:
        tuple: (contour_path, zigzag_paths) where:
            contour_path: List of points representing the continuous contour path
            zigzag_paths: List of lists of points representing zig-zag fill paths
    """
    if not polygon.is_valid:
        print("Invalid polygon for contour fill")
        return []
    
    # Get the configuration to check the polygonization method
    from src.config import get_config
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
    
    # Connect the contours to form a continuous spiral path
    contour_path = connect_contours(contours, toolpath_width)
    
    # Create a union of all buffered contours to represent the filled area
    # Buffer by half toolpath width on each side
    buffered_area = Polygon() # Initialize as empty
    if contours:
        try:
            # Buffer each contour LINESTRING (converted from LinearRing), not the polygon it came from
            # Use join_style=2 (BEVEL) and cap_style=2 (FLAT) for better results with line buffering
            buffered_lines = [LineString(c.coords).buffer(toolpath_width / 2, join_style=2, cap_style=2) for c in contours]
            
            # Combine the buffered lines into a single area
            combined_buffered_area = unary_union(buffered_lines)

            # Ensure the buffered area is valid
            if not combined_buffered_area.is_valid:
                print("  Warning: Combined buffered contour area is invalid, attempting fix...")
                combined_buffered_area = combined_buffered_area.buffer(0) # Attempt to fix validity

            # Intersect with the original polygon to constrain the filled area
            # This prevents the buffered area from extending outside the original bounds
            buffered_area = polygon.intersection(combined_buffered_area)
            if not buffered_area.is_valid:
                 print("  Warning: Final buffered area (intersection) is invalid, attempting fix...")
                 buffered_area = buffered_area.buffer(0)

        except Exception as e:
            print(f"  Error calculating contour filled area: {e}")
            # Fallback: Use the simpler buffer approach if LineString buffering fails
            try:
                 buffered_area = unary_union([c.buffer(toolpath_width/2) for c in contours])
                 if not buffered_area.is_valid: buffered_area = buffered_area.buffer(0)
                 buffered_area = polygon.intersection(buffered_area) # Still intersect
                 if not buffered_area.is_valid: buffered_area = buffered_area.buffer(0)
            except Exception as e2:
                 print(f"  Fallback buffer method also failed: {e2}")
                 buffered_area = Polygon() # Assign empty polygon on failure

    # Calculate underfill regions as the difference between the original polygon and the calculated filled area
    try:
        underfill_regions = polygon.difference(buffered_area)
        # Clean up potential small artifacts from the difference operation
        if not underfill_regions.is_valid:
             print("  Warning: Initial underfill region is invalid, attempting fix...")
             underfill_regions = underfill_regions.buffer(0)
    except Exception as e:
        print(f"  Error calculating underfill regions: {e}")
        underfill_regions = Polygon() # Empty polygon on error

    # Generate zig-zag fills for each underfill region
    zigzag_paths = []
    # Set a minimum area slightly larger than zero to avoid tiny artifacts
    min_zigzag_area = (toolpath_width * toolpath_width) * 0.1

    if underfill_regions.is_empty:
        print("  No underfill regions found")
    else:
        regions_to_fill = []
        if underfill_regions.geom_type == 'Polygon':
            regions_to_fill.append(underfill_regions)
        elif underfill_regions.geom_type == 'MultiPolygon':
            regions_to_fill.extend(list(underfill_regions.geoms))
        
        print(f"  Found {len(regions_to_fill)} potential underfill region(s)")
        fill_count = 0
        skipped_count = 0
        for region in regions_to_fill:
            # Ensure region is valid before checking area and filling
            if not region.is_valid:
                region = region.buffer(0)
            if not region.is_valid:
                 print(f"    Skipping invalid underfill region geometry.")
                 skipped_count += 1
                 continue

            if region.area > min_zigzag_area:  # Only fill if area is significant
                # print(f"    Filling underfill region with area {region.area:.4f}")
                zigzag_fill = generate_zigzag_fill(region, toolpath_width)
                if zigzag_fill: # Only add if fill was actually generated
                    zigzag_paths.append(zigzag_fill)
                    fill_count += 1
                else:
                    # print(f"    Skipping region: zig-zag generation returned empty path (area: {region.area:.4f})")
                    skipped_count += 1
            else:
                # print(f"    Skipping small underfill region with area {region.area:.4f}")
                skipped_count += 1
        print(f"  Generated zig-zag fill for {fill_count} underfill region(s). Skipped {skipped_count}.")

    return contour_path, zigzag_paths

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
    # Simple linear connection for now
    # return [end_point] # Keep the smooth connection logic
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

def visualize_fill_path(polygon, contour_path, zigzag_paths=None, title="Continuous Fill Path"):
    """
    Visualize the polygon and the fill paths.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon
        contour_path (list): List of points representing the contour path
        zigzag_paths (list): List of lists of points representing zig-zag fill paths
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
    
    # Plot the contour path
    if contour_path:
        path_x, path_y = zip(*contour_path)
        
        # Use a colormap to show the direction of the path
        points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a colorful line collection
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=1.5)
        lc.set_array(np.linspace(0, 1, len(path_x)-1))
        ax.add_collection(lc)
        
        # Mark start and end points
        ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Contour Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='Contour End')
        
        # Add a colorbar to show progression
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Contour Direction')
    
    # Plot zig-zag paths if any
    if zigzag_paths:
        for i, zigzag in enumerate(zigzag_paths):
            if zigzag and len(zigzag) >= 2:  # Need at least 2 points to plot a line
                zx, zy = zip(*zigzag)
                ax.plot(zx, zy, 'm-', linewidth=1, alpha=0.7, label=f'Zig-Zag {i+1}' if i == 0 else "")
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
