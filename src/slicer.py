import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from src.config import get_config

def slice_mesh(stl_mesh, layer_height):
    """
    Slice a 3D mesh into horizontal layers at specified height intervals.
    
    Args:
        stl_mesh: The input mesh (numpy-stl mesh object)
        layer_height: Distance between layers in mm
        
    Returns:
        list: List of layers, where each layer is a list of Shapely Polygons
    """
    config = get_config()
    
    # Convert numpy-stl mesh to trimesh object
    mesh = trimesh.Trimesh(vertices=stl_mesh.vectors.reshape(-1, 3),
                          faces=np.arange(len(stl_mesh.vectors)*3).reshape(-1, 3))
    
    # Get mesh bounds and calculate slicing parameters
    min_z, max_z = mesh.bounds[0][2], mesh.bounds[1][2]
    num_layers = int(np.ceil((max_z - min_z) / layer_height))
    max_layers = config.get('max_layers_to_process', num_layers)
    num_layers = min(num_layers, max_layers)
    
    print(f"Slicing mesh from z={min_z:.2f} to {max_z:.2f} in {num_layers} layers")
    
    layers = []
    for i in range(num_layers):
        z_height = min_z + (i * layer_height)
        
        try:
            # Get 2D cross-section at this height
            section = mesh.section(plane_origin=[0, 0, z_height],
                                 plane_normal=[0, 0, 1])
            
            if section is None:
                continue
                
            # Convert to 2D polygons
            section_2d, _ = section.to_planar()
            layer_polygons = []
            
            for polygon in section_2d.polygons_full:
                if isinstance(polygon, shapely.geometry.Polygon):
                    # Already a Shapely polygon
                    processed = polygon
                elif isinstance(polygon, (list, tuple)) and len(polygon) == 2:
                    # Handle (exterior, interiors) format
                    exterior, interiors = polygon
                    if len(exterior) >= 3:  # Need at least 3 points for a polygon
                        processed = Polygon(exterior, interiors)
                    else:
                        continue
                else:
                    continue
                
                # Validate and clean the polygon
                if not processed.is_valid:
                    processed = processed.buffer(0)
                
                if processed and not processed.is_empty:
                    if isinstance(processed, MultiPolygon):
                        layer_polygons.extend([p for p in processed.geoms if isinstance(p, Polygon)])
                    else:
                        layer_polygons.append(processed)
            
            if layer_polygons:
                layers.append(layer_polygons)
                
        except Exception as e:
            print(f"Error slicing at height {z_height:.2f}: {str(e)}")
            continue
    
    return layers

def visualize_layers(layers, min_z, layer_height):
    """
    Visualize the sliced layers as 2D contours.
    
    Args:
        layers: List of layers from slice_mesh()
        min_z: Minimum Z coordinate of the mesh
        layer_height: Distance between layers in mm
    """
    config = get_config()
    if not config.get('visualize_layer_contours', False):
        print("Layer contour visualization disabled in config")
        return
        
    if not layers:
        print("No layers to visualize")
        return
    
    print("\nVisualizing layer contours...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_title('Layer Contours')
    
    # Create a colormap for the layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    
    patches = []
    for i, layer in enumerate(layers):
        z = min_z + (i * layer_height)
        color = colors[i]
        
        for polygon in layer:
            # Plot exterior
            x, y = polygon.exterior.xy
            ax.plot(x, y, color=color, label=f'Layer {i} (z={z:.2f})' if i == 0 else "")
            
            # Plot interiors (holes) in red
            for interior in polygon.interiors:
                xi, yi = interior.xy
                ax.plot(xi, yi, color='red', linewidth=1)
    
    # Add legend and show
    ax.legend()
    plt.tight_layout()
    plt.show()
