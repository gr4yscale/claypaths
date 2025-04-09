import os
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers

def main():
    print("Welcome to claypaths - Fermat Spiral 3D Printing Toolpath Generator")
    
    # Path to the test STL file
    stl_file_path = os.path.join("models", "stretchrite3.stl")
    
    # Step 1: Load the STL file
    print(f"\nStep 1: Loading STL file from {stl_file_path}")
    stl_mesh = load_stl(stl_file_path)
    
    if stl_mesh is not None:
        # Get and display mesh information
        mesh_info = get_mesh_info(stl_mesh)
        print("\nMesh Information:")
        print(f"  Number of triangles: {mesh_info['num_triangles']}")
        print(f"  Volume: {mesh_info['volume']:.2f} cubic units")
        print(f"  Dimensions (x,y,z): {mesh_info['dimensions']}")
        print(f"  Min coordinates: {mesh_info['min_coords']}")
        print(f"  Max coordinates: {mesh_info['max_coords']}")
        
        # Visualize the STL file (commented out)
        # print("\nVisualizing the STL mesh...")
        # visualize_stl(stl_mesh)
        
        # Step 2: Slice the model into layers
        print("\nStep 2: Slicing the model into layers")
        layer_height = 0.2  # Default layer height in mm
        layers = slice_mesh(stl_mesh, layer_height)
        
        # Visualize a few layers
        if layers:
            print(f"\nGenerated {len(layers)} layers")
            print("Visualizing sample layers...")
            visualize_layers(layers, mesh_info['min_coords'][2], layer_height, num_to_show=3)
    else:
        print(f"Failed to load STL file: {stl_file_path}")
        print("Please ensure the file exists and is a valid STL file.")


if __name__ == "__main__":
    main()
