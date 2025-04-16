import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString
from shapely.ops import polygonize, unary_union
from src.config import get_config

def slice_mesh(stl_mesh, layer_height=None):
    """
    Slice the mesh into horizontal layers.
    
    Args:
        stl_mesh: The mesh object to slice
        layer_height (float): Height of each layer in mm
        
    Returns:
        list: List of layer contours, where each layer is a list of shapely Polygons
    """
    if stl_mesh is None:
        print("No mesh to slice")
        return []
    
    # Get mesh bounds
    min_coords = np.min(stl_mesh.vectors.reshape([-1, 3]), axis=0)
    max_coords = np.max(stl_mesh.vectors.reshape([-1, 3]), axis=0)
    
    # Use configuration value if layer_height is not provided
    if layer_height is None:
        config = get_config()
        layer_height = config['layer_height']
    
    # Calculate number of layers
    z_min, z_max = min_coords[2], max_coords[2]
    num_layers = int((z_max - z_min) / layer_height) + 1
    
    print(f"Slicing mesh into {num_layers} layers (z: {z_min:.2f}mm to {z_max:.2f}mm)")
    
    # Initialize layers
    layers = []
    
    # Get max layers to process from config
    config = get_config()
    max_layers_to_process = config.get('max_layers_to_process', 6)
    
    # Process each layer (limited by max_layers_to_process)
    for i in range(min(num_layers, max_layers_to_process)):
        z = z_min + i * layer_height
        layer_contours = slice_at_height(stl_mesh, z)
        layers.append(layer_contours)
        print(f"Layer {i+1}/{num_layers} at z={z:.2f}mm: {len(layer_contours)} contours")
    
    return layers

def slice_at_height(stl_mesh, z_height):
    """
    Slice the mesh at a specific height.
    
    Args:
        stl_mesh: The mesh object to slice
        z_height (float): Height at which to slice
        
    Returns:
        list: List of shapely Polygons representing the contours at this height
    """
    # Get all triangles from the mesh
    triangles = stl_mesh.vectors
    
    # Find all line segments where triangles intersect with the z plane
    segments = []
    
    for triangle in triangles:
        # Get the three vertices of the triangle
        vertices = triangle
        
        # Check if the triangle intersects with the z plane
        above = vertices[:, 2] > z_height
        below = vertices[:, 2] < z_height
        on_plane = np.isclose(vertices[:, 2], z_height, atol=1e-6)
        
        # If all vertices are above or below the plane, no intersection
        if np.all(above) or np.all(below):
            continue
        
        # Find the line segments where the triangle intersects the plane
        intersections = []
        
        # Check each edge of the triangle
        for i in range(3):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 3]
            
            # If both vertices are on the plane, add the edge
            if on_plane[i] and on_plane[(i + 1) % 3]:
                intersections.append((v1[0], v1[1]))
                intersections.append((v2[0], v2[1]))
                continue
            
            # If one vertex is on the plane, add it
            if on_plane[i]:
                intersections.append((v1[0], v1[1]))
                continue
            
            # If one vertex is above and one is below, find the intersection
            if (above[i] and below[(i + 1) % 3]) or (below[i] and above[(i + 1) % 3]):
                # Calculate the intersection point
                t = (z_height - v1[2]) / (v2[2] - v1[2])
                x = v1[0] + t * (v2[0] - v1[0])
                y = v1[1] + t * (v2[1] - v1[1])
                intersections.append((x, y))
        
        # If we found exactly two intersection points, add the segment
        if len(intersections) == 2:
            segments.append(LineString(intersections))
    
    # Convert segments to polygons
    if not segments:
        return []
    
    try:
        # Create polygons from the segments
        # First, ensure the segments are properly connected
        merged_lines = unary_union(segments)
        
        # Debug information
        print(f"  Found {len(segments)} segments at z={z_height:.2f}")
        
        # Get configuration for polygonization method
        config = get_config()
        polygonization_method = config.get('polygonization_method', 'standard')
        
        # Initialize polygons list
        polygons = []
        
        # Apply the selected polygonization method
        if polygonization_method == 'standard':
            # Standard polygonize method
            print(f"  Using standard polygonization method at z={z_height:.2f}")
            polygons = list(polygonize(merged_lines))
            
        elif polygonization_method == 'small_buffer':
            # Small buffer method
            buffer_size = config.get('buffer_size_small', 0.001)
            print(f"  Using small buffer polygonization method (size={buffer_size}) at z={z_height:.2f}")
            
            # Get the bounding box of the merged lines to estimate expected polygon size
            minx, miny, maxx, maxy = merged_lines.bounds
            expected_area = (maxx - minx) * (maxy - miny) * 0.5  # Rough estimate
            print(f"  Expected polygon area based on bounding box: {expected_area:.4f}")
            
            # Try with increasing buffer sizes until we get reasonable polygons
            buffer_sizes = [buffer_size, buffer_size*2, buffer_size*5, buffer_size*10, buffer_size*20, buffer_size*50]
            min_reasonable_area = max(1.0, expected_area * 0.1)  # Minimum area to consider a polygon "reasonable"
            
            best_polygons = []
            best_area_sum = 0
            
            for try_buffer in buffer_sizes:
                print(f"  Trying buffer size: {try_buffer}")
                try:
                    buffered_lines = merged_lines.buffer(try_buffer)
                    
                    if not buffered_lines.is_empty:
                        current_polygons = []
                        current_area_sum = 0
                        
                        if isinstance(buffered_lines, Polygon):
                            if buffered_lines.area > min_reasonable_area:
                                print(f"  Created polygon with area: {buffered_lines.area:.4f}")
                                current_polygons = [buffered_lines]
                                current_area_sum = buffered_lines.area
                        elif isinstance(buffered_lines, MultiPolygon):
                            # Filter out tiny polygons
                            reasonable_polys = [p for p in buffered_lines.geoms if p.area > min_reasonable_area]
                            if reasonable_polys:
                                print(f"  Created {len(reasonable_polys)} polygons with total area: {sum(p.area for p in reasonable_polys):.4f}")
                                current_polygons = reasonable_polys
                                current_area_sum = sum(p.area for p in reasonable_polys)
                        
                        # Keep track of the best result so far
                        if current_area_sum > best_area_sum:
                            best_polygons = current_polygons
                            best_area_sum = current_area_sum
                            print(f"  New best result with total area: {best_area_sum:.4f}")
                            
                        # If we've found a good result, stop trying larger buffers
                        if current_area_sum > expected_area * 0.5:
                            print(f"  Found good result (area: {current_area_sum:.4f} vs expected: {expected_area:.4f})")
                            break
                except Exception as e:
                    print(f"  Error with buffer size {try_buffer}: {e}")
            
            # Use the best result we found
            if best_polygons:
                print(f"  Using best result with {len(best_polygons)} polygons, total area: {best_area_sum:.4f}")
                polygons = best_polygons
            else:
                print(f"  Warning: Could not create reasonable sized polygons")
                # Try one last approach - create a convex hull from all line endpoints
                try:
                    from shapely.geometry import MultiPoint
                    
                    # Extract all endpoints from the lines
                    points = []
                    if isinstance(merged_lines, LineString):
                        points.extend(list(merged_lines.coords))
                    elif isinstance(merged_lines, MultiLineString):
                        for line in merged_lines.geoms:
                            points.extend(list(line.coords))
                    
                    if points:
                        hull = MultiPoint(points).convex_hull
                        if isinstance(hull, Polygon) and hull.area > min_reasonable_area:
                            print(f"  Created convex hull with area: {hull.area:.4f}")
                            polygons = [hull]
                except Exception as e:
                    print(f"  Error creating convex hull: {e}")
                    
        elif polygonization_method == 'large_buffer':
            # Large buffer method
            buffer_size = config.get('buffer_size_large', 0.01)
            print(f"  Using large buffer polygonization method (size={buffer_size}) at z={z_height:.2f}")
            
            # Try with increasing buffer sizes until we get reasonable polygons
            buffer_sizes = [buffer_size, buffer_size*2, buffer_size*5, buffer_size*10]
            min_reasonable_area = 1.0  # Minimum area to consider a polygon "reasonable"
            
            for try_buffer in buffer_sizes:
                print(f"  Trying buffer size: {try_buffer}")
                buffered_lines = merged_lines.buffer(try_buffer)
                
                if not buffered_lines.is_empty:
                    if isinstance(buffered_lines, Polygon):
                        if buffered_lines.area > min_reasonable_area:
                            print(f"  Created polygon with area: {buffered_lines.area:.4f}")
                            polygons = [buffered_lines]
                            break
                        else:
                            print(f"  Polygon too small (area: {buffered_lines.area:.4f}), trying larger buffer")
                    elif isinstance(buffered_lines, MultiPolygon):
                        # Filter out tiny polygons
                        reasonable_polys = [p for p in buffered_lines.geoms if p.area > min_reasonable_area]
                        if reasonable_polys:
                            print(f"  Created {len(reasonable_polys)} polygons with reasonable area")
                            polygons = reasonable_polys
                            break
                        else:
                            print(f"  All polygons too small, trying larger buffer")
            
            # If we still don't have reasonable polygons, use the last buffer result anyway
            if not polygons:
                print(f"  Warning: Could not create reasonable sized polygons, using last buffer result")
                if isinstance(buffered_lines, Polygon):
                    polygons = [buffered_lines]
                elif isinstance(buffered_lines, MultiPolygon):
                    polygons = list(buffered_lines.geoms)
                    
        elif polygonization_method == 'manual_close':
            # Manual close method - only works with MultiLineString
            print(f"  Using manual close polygonization method at z={z_height:.2f}")
            if isinstance(merged_lines, MultiLineString):
                # Extract all endpoints
                endpoints = []
                for line in merged_lines.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        endpoints.append(coords[0])
                        endpoints.append(coords[-1])
                
                # Find endpoints that are very close to each other
                tolerance = config.get('buffer_size_small', 0.001)
                connected_lines = list(merged_lines.geoms)
                
                # Try to connect very close endpoints
                for i in range(0, len(endpoints), 2):
                    if i+1 >= len(endpoints):
                        break
                    p1 = endpoints[i]
                    for j in range(i+1, len(endpoints), 2):
                        if j+1 >= len(endpoints):
                            break
                        p2 = endpoints[j]
                        # Calculate distance
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        if dist < tolerance:
                            # Connect these points with a new line segment
                            connected_lines.append(LineString([p1, p2]))
                
                # Try polygonize with the connected lines
                connected_merged = unary_union(connected_lines)
                polygons = list(polygonize(connected_merged))
            else:
                print(f"  Warning: Manual close method requires MultiLineString but got {type(merged_lines)}")
                # Fall back to standard method
                polygons = list(polygonize(merged_lines))
        elif polygonization_method == 'hybrid':
            # Hybrid method - try standard polygonization first, then fall back to buffering if needed
            print(f"  Using hybrid polygonization method at z={z_height:.2f}")
            
            # Get the bounding box of the merged lines to estimate expected polygon size
            minx, miny, maxx, maxy = merged_lines.bounds
            expected_area = (maxx - minx) * (maxy - miny) * 0.5  # Rough estimate
            print(f"  Expected polygon area based on bounding box: {expected_area:.4f}")
            
            # Try standard polygonization first
            standard_polygons = list(polygonize(merged_lines))
            
            # Check if we got reasonable polygons
            min_reasonable_area = max(1.0, expected_area * 0.1)
            reasonable_standard_polys = [p for p in standard_polygons if p.area > min_reasonable_area]
            
            if reasonable_standard_polys and sum(p.area for p in reasonable_standard_polys) > expected_area * 0.3:
                print(f"  Standard polygonization successful, created {len(reasonable_standard_polys)} polygons")
                print(f"  Total area: {sum(p.area for p in reasonable_standard_polys):.4f}")
                polygons = reasonable_standard_polys
            else:
                print(f"  Standard polygonization failed or created only tiny polygons, trying buffering")
                # Try buffering with increasing sizes
                buffer_sizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
                
                best_polygons = []
                best_area_sum = 0
                
                for try_buffer in buffer_sizes:
                    print(f"  Trying buffer size: {try_buffer}")
                    try:
                        buffered_lines = merged_lines.buffer(try_buffer)
                        
                        if not buffered_lines.is_empty:
                            current_polygons = []
                            current_area_sum = 0
                            
                            if isinstance(buffered_lines, Polygon):
                                if buffered_lines.area > min_reasonable_area:
                                    print(f"  Created polygon with area: {buffered_lines.area:.4f}")
                                    current_polygons = [buffered_lines]
                                    current_area_sum = buffered_lines.area
                            elif isinstance(buffered_lines, MultiPolygon):
                                # Filter out tiny polygons
                                reasonable_polys = [p for p in buffered_lines.geoms if p.area > min_reasonable_area]
                                if reasonable_polys:
                                    print(f"  Created {len(reasonable_polys)} polygons with total area: {sum(p.area for p in reasonable_polys):.4f}")
                                    current_polygons = reasonable_polys
                                    current_area_sum = sum(p.area for p in reasonable_polys)
                            
                            # Keep track of the best result so far
                            if current_area_sum > best_area_sum:
                                best_polygons = current_polygons
                                best_area_sum = current_area_sum
                                print(f"  New best result with total area: {best_area_sum:.4f}")
                                
                            # If we've found a good result, stop trying larger buffers
                            if current_area_sum > expected_area * 0.5:
                                print(f"  Found good result (area: {current_area_sum:.4f} vs expected: {expected_area:.4f})")
                                break
                    except Exception as e:
                        print(f"  Error with buffer size {try_buffer}: {e}")
                
                # Use the best result we found
                if best_polygons:
                    print(f"  Using best result with {len(best_polygons)} polygons, total area: {best_area_sum:.4f}")
                    polygons = best_polygons
                else:
                    # If both standard and buffering failed, try convex hull as a last resort
                    try:
                        from shapely.geometry import MultiPoint
                        
                        # Extract all endpoints from the lines
                        points = []
                        if isinstance(merged_lines, LineString):
                            points.extend(list(merged_lines.coords))
                        elif isinstance(merged_lines, MultiLineString):
                            for line in merged_lines.geoms:
                                points.extend(list(line.coords))
                        
                        if points:
                            hull = MultiPoint(points).convex_hull
                            if isinstance(hull, Polygon) and hull.area > min_reasonable_area:
                                print(f"  Created convex hull with area: {hull.area:.4f}")
                                polygons = [hull]
                                print(f"  Using convex hull as last resort")
                            else:
                                print(f"  Warning: Convex hull too small or invalid, falling back to standard result")
                                polygons = standard_polygons
                        else:
                            print(f"  Warning: No points found for convex hull, falling back to standard result")
                            polygons = standard_polygons
                    except Exception as e:
                        print(f"  Error creating convex hull: {e}")
                        print(f"  Warning: All methods failed, falling back to standard result")
                        polygons = standard_polygons
        else:
            # Unknown method, use standard
            print(f"  Warning: Unknown polygonization method '{polygonization_method}', using standard")
            polygons = list(polygonize(merged_lines))
        
        # If the selected method failed, log the issue
        if not polygons:
            print(f"  Warning: Polygonization method '{polygonization_method}' failed at z={z_height:.2f}")
            
            # Check if visualization of problematic segments is enabled
            if config.get('visualize_problematic_segments', False):
                # Visualize the problematic segments
                if isinstance(merged_lines, (LineString, MultiLineString)):
                    fig, ax = plt.subplots()
                    if isinstance(merged_lines, LineString):
                        x, y = merged_lines.xy
                        ax.plot(x, y, 'r-')
                    else: # MultiLineString
                        for line in merged_lines.geoms:
                            x, y = line.xy
                            ax.plot(x, y, 'r-')
                    ax.set_title(f"Problematic Segments at z={z_height:.2f}")
                    ax.set_aspect('equal')
                    plt.show(block=False)

        # Ensure all returned geometries are valid Polygons
        valid_polygons = [p for p in polygons if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
        if len(valid_polygons) != len(polygons):
             print(f"  Warning: Filtered out {len(polygons) - len(valid_polygons)} invalid/non-polygon geometries.")

        # Determine which polygons are holes and which are solid areas
        # For models with holes, we need to identify the outer contour as the solid area
        if len(valid_polygons) > 1:
            # Find the polygon with the largest area - this is likely the outer contour
            areas = [p.area for p in valid_polygons]
            largest_idx = areas.index(max(areas))
            largest_poly = valid_polygons[largest_idx]
            
            print(f"  Largest polygon has area: {largest_poly.area:.4f}")
            
            # If the largest polygon is too small, it might not be the actual outer contour
            # In this case, we should try to create a better representation of the outer contour
            if largest_poly.area < 10.0:  # Arbitrary threshold for "too small"
                print(f"  Warning: Largest polygon may be too small, attempting to create better outer contour")
                
                # Try to create a convex hull of all polygons to get a better outer boundary
                all_points = []
                for poly in valid_polygons:
                    all_points.extend(list(poly.exterior.coords)[:-1])  # Exclude last point (duplicate of first)
                
                if all_points:
                    from shapely.geometry import MultiPoint
                    convex_hull = MultiPoint(all_points).convex_hull
                    
                    if isinstance(convex_hull, Polygon) and convex_hull.area > largest_poly.area * 1.5:
                        print(f"  Created convex hull with area: {convex_hull.area:.4f}")
                        # Replace the largest polygon with the convex hull
                        valid_polygons[largest_idx] = convex_hull
                        largest_poly = convex_hull
            
            # Improved hole detection algorithm
            final_polygons = []
            
            # First, identify potential holes (smaller polygons inside larger ones)
            potential_holes = {}  # Dictionary mapping container polygons to lists of hole polygons
            standalone_polygons = []  # Polygons that aren't contained by any other polygon
            
            # For each polygon, check if it's contained within another polygon
            for i, poly in enumerate(valid_polygons):
                is_contained = False
                
                # Check if this polygon is contained within any other polygon
                for j, container in enumerate(valid_polygons):
                    if i != j and container.contains(poly):
                        # This polygon is contained within another polygon
                        if j not in potential_holes:
                            potential_holes[j] = []
                        potential_holes[j].append(i)
                        is_contained = True
                        break
                
                if not is_contained:
                    standalone_polygons.append(i)
            
            print(f"  Found {len(standalone_polygons)} standalone polygons and {len(potential_holes)} polygons with potential holes")
            
            # Process each standalone polygon
            for idx in standalone_polygons:
                poly = valid_polygons[idx]
                
                # Check if this polygon has any holes
                if idx in potential_holes:
                    # Get the holes for this polygon
                    hole_indices = potential_holes[idx]
                    holes = [valid_polygons[h].exterior.coords for h in hole_indices]
                    
                    # Create a new polygon with holes
                    new_poly = Polygon(poly.exterior.coords, holes)
                    print(f"  Created polygon with {len(holes)} holes")
                    final_polygons.append(new_poly)
                else:
                    # No holes, just add the polygon as is
                    final_polygons.append(poly)
            
            # If we didn't find any valid polygons with the improved algorithm, fall back to the original method
            if not final_polygons:
                print("  Warning: Improved hole detection failed, falling back to original method")
                final_polygons = [largest_poly]
                holes = []
                
                # Then check all other polygons
                for i, poly in enumerate(valid_polygons):
                    if i == largest_idx:
                        continue  # Skip the largest polygon, already added
                    
                    # Check if this polygon is contained within the largest polygon
                    if largest_poly.contains(poly):
                        # This is a hole - collect it for later
                        print(f"  Detected a hole in the largest polygon at z={z_height:.2f}")
                        holes.append(poly.exterior.coords)
                    else:
                        # This is a separate solid area
                        final_polygons.append(poly)
                
                # Create a new polygon with holes if needed
                if holes:
                    # Create a new polygon with holes
                    new_poly = Polygon(largest_poly.exterior.coords, holes)
                    # Replace the largest polygon with the new one that has holes
                    final_polygons[0] = new_poly
                    print(f"  Created polygon with {len(holes)} holes")
            
            valid_polygons = final_polygons

        print(f"  Created {len(valid_polygons)} valid polygons at z={z_height:.2f}")
        return valid_polygons
    except Exception as e:
        print(f"Error during polygonization at z={z_height:.2f}: {e}")
        return []

def visualize_layer(layer_contours, layer_num, z_height):
    """
    Visualize a single layer's contours.
    
    Args:
        layer_contours: List of shapely Polygons for the layer
        layer_num (int): Layer number
        z_height (float): Height of the layer
    """
    # Check if visualization is enabled
    from src.config import get_config
    config = get_config()
    if not config.get('visualize_layer_contours', True):
        print(f"Layer contour visualization disabled in config")
        return
        
    if not layer_contours:
        print(f"No contours to visualize for layer {layer_num}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each contour
    for polygon in layer_contours:
        # Plot exterior boundary in blue
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='Exterior' if polygon == layer_contours[0] else "")
        
        # Plot holes in red with dashed lines
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, 'r--', linewidth=1.5, label='Interior/Hole' if interior == list(polygon.interiors)[0] else "")
    
    # Add legend if we have both exterior and interior
    has_interior = any(len(list(polygon.interiors)) > 0 for polygon in layer_contours)
    if has_interior:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    ax.set_aspect('equal')
    ax.set_title(f"Layer {layer_num} (z={z_height:.2f}mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    plt.tight_layout()
    plt.show()

def visualize_layers(layers, min_z, layer_height, num_to_show=5):
    """
    Visualize a subset of layers.
    
    Args:
        layers: List of layer contours
        min_z (float): Minimum z height
        layer_height (float): Height of each layer
        num_to_show (int): Number of layers to visualize
    """
    # Check if visualization is enabled
    from src.config import get_config
    config = get_config()
    if not config.get('visualize_layer_contours', True):
        print("Layer contour visualization disabled in config")
        return
        
    if not layers:
        print("No layers to visualize")
        return

    print("\nVisualizing sample layers (original contours)...")
    
    # Select layers to visualize (evenly distributed)
    total_layers = len(layers)
    if num_to_show >= total_layers:
        indices = list(range(total_layers))
    else:
        indices = [int(i * (total_layers - 1) / (num_to_show - 1)) for i in range(num_to_show)]
    
    # Visualize selected layers
    for i in indices:
        z_height = min_z + i * layer_height
        visualize_layer(layers[i], i+1, z_height)
