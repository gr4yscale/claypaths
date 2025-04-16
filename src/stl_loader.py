import numpy as np
from stl import mesh
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

matplotlib.use("GTK3Agg")

def load_stl(file_path):
    """
    Load an STL file and return the mesh object.
    
    Args:
        file_path (str): Path to the STL file
        
    Returns:
        mesh: The loaded mesh object
    """
    try:
        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(file_path)
        print(f"Successfully loaded STL file: {file_path}")
        print(f"Number of triangles: {len(stl_mesh.vectors)}")
        return stl_mesh
    except Exception as e:
        print(f"Error loading STL file: {e}")
        return None

def visualize_stl(stl_mesh):
    """
    Visualize the STL mesh using matplotlib.
    
    Args:
        stl_mesh: The mesh object to visualize
    """
    # Check if visualization is enabled
    from src.config import get_config
    config = get_config()
    if not config.get('visualize_stl_mesh', True):
        print("STL mesh visualization disabled in config")
        return
        
    if stl_mesh is None:
        print("No mesh to visualize")
        return

    print("\nVisualizing the STL mesh...")

    # Create a new plot
    figure = plt.figure(figsize=(10, 10))
    axes = figure.add_subplot(111, projection='3d')
    
    # Add the mesh to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
    
    # Auto scale to the mesh size
    scale = stl_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    
    # Show the plot
    plt.show()

def get_mesh_info(stl_mesh):
    """
    Get basic information about the mesh.
    
    Args:
        stl_mesh: The mesh object
        
    Returns:
        dict: Dictionary containing mesh information
    """
    if stl_mesh is None:
        return None
    
    # Calculate mesh properties
    volume, cog, inertia = stl_mesh.get_mass_properties()
    
    # Get min and max coordinates
    min_coords = np.min(stl_mesh.vectors.reshape([-1, 3]), axis=0)
    max_coords = np.max(stl_mesh.vectors.reshape([-1, 3]), axis=0)
    dimensions = max_coords - min_coords
    
    return {
        "num_triangles": len(stl_mesh.vectors),
        "volume": volume,
        "center_of_gravity": cog,
        "dimensions": dimensions,
        "min_coords": min_coords,
        "max_coords": max_coords
    }
