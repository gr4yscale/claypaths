import os
import matplotlib.pyplot as plt # Ensure matplotlib is imported
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
from src.cfs_filler import generate_cfs_fill, visualize_cfs_fill

def main():
    print("Welcome to claypaths - Fermat Spiral 3D Printing Toolpath Generator")
    
    # Paths to test STL files
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    stl_file_path = os.path.join("models", "t-shape.stl")

    #confirmed working
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "cuboid.stl")
    #stl_file_path = os.path.join("models", "extruded-rounded-rectangle.stl")
    #stl_file_path = os.path.join("models", "right-triangular-prism.stl")
    #stl_file_path = os.path.join("models", "stack-of-cuboids.stl")
    #stl_file_path = os.path.join("models", "stack-of-cylinders.stl")


    # complex shapes, problematic
    # holes are detected as solid rather than the cuboid
    #stl_file_path = os.path.join("models", "cuboid-with-holes.stl")

    # cuboid detected as solid rather than cylinder
    #stl_file_path = os.path.join("models", "cylinder-minus-cuboid.stl")

    # cuboid inside which should be empty is detected as solid
    #stl_file_path = os.path.join("models", "hollow-cuboid.stl")

    # empty hex is detected as solid
    #stl_file_path = os.path.join("models", "wrench.stl")
    
    # complex shapes, problematic
    #stl_file_path = os.path.join("models", "u-shape.stl")
    #stl_file_path = os.path.join("models", "l-shape.stl")
    #stl_file_path = os.path.join("models", "rectangular-cuboid-with-hole.stl")
    #stl_file_path = os.path.join("models", "cylinder-with-cone.stl")
    
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
        visualize_stl(stl_mesh)
        
        # Step 2: Slice the model into layers
        print("\nStep 2: Slicing the model into layers")
        layer_height = 1.0  # Default layer height in mm
        layers = slice_mesh(stl_mesh, layer_height)
        
        # Optionally visualize the original layers again if needed
        print("\nVisualizing sample layers (original contours)...")
        visualize_layers(layers, mesh_info['min_coords'][2], layer_height, num_to_show=min(len(layers), 3))
        
        # Step 3: Generate CFS fill for each layer
        print("\nStep 3: Generating CFS fill for sample layers")
        toolpath_width = 0.4  # Default toolpath width in mm
        
        # Process a sample layer
        if layers and len(layers) > 0:
            sample_layer_idx = 2
            sample_layer = layers[sample_layer_idx]
            
            if sample_layer and len(sample_layer) > 0:
                # Take the first polygon in the layer
                contour_poly = sample_layer[0]
                
                print(f"\nGenerating CFS fill for layer {sample_layer_idx+1}, polygon 1")
                cfs_result = generate_cfs_fill(contour_poly, toolpath_width)
                
                if cfs_result:
                    print(f"Successfully generated CFS fill with {len(cfs_result.coords)} points")
                    
                    # Visualize the CFS fill
                    print("\nVisualizing CFS fill...")
                    visualize_cfs_fill(contour_poly, cfs_result, toolpath_width)
                else:
                    print("Failed to generate CFS fill for the layer")
            else:
                print(f"No valid polygons in layer {sample_layer_idx+1}")

    else:
        print(f"Failed to load STL file: {stl_file_path}")
        print("Please ensure the file exists and is a valid STL file.")

    # Keep all plot windows open until manually closed
    if plt.get_fignums(): # Check if any figures were created
        print("\nDisplaying generated plots. Close plot windows to exit.")
        plt.show() # Blocking call to display all figures

if __name__ == "__main__":
    main()
