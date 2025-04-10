import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import polygonize, unary_union

def slice_mesh(stl_mesh, layer_height=0.2):
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
    
    # Calculate number of layers
    z_min, z_max = min_coords[2], max_coords[2]
    num_layers = int((z_max - z_min) / layer_height) + 1
    
    print(f"Slicing mesh into {num_layers} layers (z: {z_min:.2f}mm to {z_max:.2f}mm)")
    
    # Initialize layers
    layers = []
    
    # Process each layer (limited to 4 for testing)
    for i in range(min(num_layers, 4)):
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

        # Attempt to form polygons using polygonize
        polygons_raw = list(polygonize(merged_lines))
        polygons = []

        if polygons_raw:
            # If polygonize returns results, combine them using unary_union
            # This handles cases where polygonize might return separate outer/inner boundaries
            combined_geometry = unary_union(polygons_raw)
            if isinstance(combined_geometry, Polygon):
                polygons = [combined_geometry]
            elif isinstance(combined_geometry, MultiPolygon):
                polygons = list(combined_geometry.geoms)
            else:
                print(f"  Warning: Unexpected geometry type after unary_union: {type(combined_geometry)}")
                polygons = [] # Fallback to empty if union result is weird
        else:
            # Fallback: If polygonize created nothing, try a small buffer on the lines
            # Use a very small buffer to minimize distortion
            print(f"  Polygonize failed, attempting buffer fallback...")
            try:
                buffered = merged_lines.buffer(1e-6, join_style=2) # MITRE join style
                if isinstance(buffered, Polygon):
                    polygons = [buffered]
                elif isinstance(buffered, MultiPolygon):
                    polygons = list(buffered.geoms)
            except Exception as buffer_err:
                 print(f"  Buffer fallback also failed: {buffer_err}")
                 polygons = [] # Ensure polygons is empty list on failure

        # Final validation and cleanup
        valid_polygons = [p for p in polygons if p.is_valid and not p.is_empty]

        print(f"  Created {len(valid_polygons)} valid polygons at z={z_height:.2f}")
        return valid_polygons
    except Exception as e:
        print(f"Error during polygon creation at z={z_height:.2f}: {e}")
        # Optionally plot problematic segments for debugging
        # fig, ax = plt.subplots()
        # for seg in segments: ax.plot(*seg.xy, 'r-')
        # plt.title(f"Problem Segments at z={z_height:.2f}")
        # plt.show()
        return []

def visualize_layer(layer_contours, layer_num, z_height):
    """
    Visualize a single layer's contours.
    
    Args:
        layer_contours: List of shapely Polygons for the layer
        layer_num (int): Layer number
        z_height (float): Height of the layer
    """
    if not layer_contours:
        print(f"No contours to visualize for layer {layer_num}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each contour
    for polygon in layer_contours:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-')
        
        # Plot holes if any
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, 'r-')
    
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
    if not layers:
        print("No layers to visualize")
        return
    
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
