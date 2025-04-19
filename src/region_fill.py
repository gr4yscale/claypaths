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
    
    # Combine paths, only including non-empty zig-zag paths
    combined_path = contour_path.copy()
    for zigzag in zigzag_paths:
        if zigzag:  # Only add non-empty zig-zag paths
            combined_path.extend(zigzag)
    return combined_path

def generate_zigzag_fill(polygon, toolpath_width):
    """
    Generate zig-zag fill pattern for a polygon.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        
    Returns:
        list: List of points representing the zig-zag path
    """
    if not polygon.is_valid:
        return []
    
    # Get polygon bounds
    min_x, min_y, max_x, max_y = polygon.bounds
    
    # Calculate number of passes needed
    spacing = toolpath_width
    num_passes = int((max_x - min_x) / spacing)
    
    # Generate zig-zag lines
    zigzag_path = []
    for i in range(num_passes + 1):
        x = min_x + i * spacing
        # Alternate direction for each pass
        if i % 2 == 0:
            zigzag_path.append((x, min_y))
            zigzag_path.append((x, max_y))
        else:
            zigzag_path.append((x, max_y))
            zigzag_path.append((x, min_y))
    
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
    
    # Create a union of all buffered contours
    buffered_contours = unary_union([contour.buffer(toolpath_width/2) for contour in contours])
    
    # Calculate underfill regions as the difference between the original polygon and buffered contours
    underfill_regions = polygon.difference(buffered_contours)
    
    # Generate zig-zag fills for each underfill region
    zigzag_paths = []
    if underfill_regions.is_empty:
        print("  No underfill regions found")
    else:
        if underfill_regions.geom_type == 'Polygon':
            if underfill_regions.area > (toolpath_width * toolpath_width):  # Only fill if area is significant
                print(f"  Found underfill region with area {underfill_regions.area:.4f}")
                zigzag_paths.append(generate_zigzag_fill(underfill_regions, toolpath_width))
        elif underfill_regions.geom_type == 'MultiPolygon':
            for region in underfill_regions.geoms:
                if region.area > (toolpath_width * toolpath_width):  # Only fill if area is significant
                    print(f"  Found underfill region with area {region.area:.4f}")
                    zigzag_paths.append(generate_zigzag_fill(region, toolpath_width))
    
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
