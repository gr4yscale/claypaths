import os
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
from src.cfs_filler import generate_cfs_fill # Import the CFS filler function

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
        layer_height = 1.0  # Default layer height in mm
        layers = slice_mesh(stl_mesh, layer_height)
        
        # Step 3: Generate Fill Paths for each layer's contours
        if layers:
            print(f"\nGenerated {len(layers)} layers (processing up to 4 for testing)")
            print("\nStep 3: Generating CFS fill paths (currently visualizes contours/MST)...")
            toolpath_width = 0.8 # Reduced toolpath width for potentially finer details

            for i, layer_contours in enumerate(layers):
                print(f"\n--- Processing Layer {i+1} ---")
                if not layer_contours:
                    print("  No contours in this layer.")
                    continue

                for j, contour_poly in enumerate(layer_contours):
                    print(f"  Processing Contour {j+1} of Layer {i+1} (Area: {contour_poly.area:.4f})")

                    # Define a minimum area threshold to attempt filling
                    # A contour smaller than roughly the square of the toolpath width is likely too small
                    min_area_threshold = (toolpath_width ** 2)

                    if contour_poly.area < min_area_threshold:
                        print(f"    -> Contour area is below threshold ({min_area_threshold:.4f}). Skipping CFS fill.")
                        continue

                    # Call the CFS filler for each polygon contour
                    # Currently, this will print progress and plot the contours/MST
                    cfs_result = generate_cfs_fill(contour_poly, toolpath_width)
                    if cfs_result:
                        print(f"    -> CFS Path generated (length: {cfs_result.length:.2f})")
                        # TODO: Store or visualize the actual path later
                    else:
                        print(f"    -> CFS Path generation incomplete for Contour {j+1}.")

            # Optionally visualize the original layers again if needed
            # print("\nVisualizing sample layers (original contours)...")
            # visualize_layers(layers, mesh_info['min_coords'][2], layer_height, num_to_show=min(len(layers), 3))

    else:
        print(f"Failed to load STL file: {stl_file_path}")
        print("Please ensure the file exists and is a valid STL file.")


if __name__ == "__main__":
    main()
