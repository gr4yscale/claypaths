import os
import matplotlib.pyplot as plt # Ensure matplotlib is imported
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
from src.cfs_filler import generate_cfs_fill # Import the CFS filler function

def main():
    print("Welcome to claypaths - Fermat Spiral 3D Printing Toolpath Generator")
    
    # Path to the test STL file
  

    #confirmed working
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
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
                    # Call the CFS filler for each polygon contour
                    cfs_result = generate_cfs_fill(contour_poly, toolpath_width)
                    if cfs_result:
                        print(f"    -> CFS Path generated (length: {cfs_result.length:.2f}). Visualizing...")
                        # Visualize the contour and the generated path
                        fig, ax = plt.subplots(figsize=(8, 8))
                        # Plot original contour
                        x_orig, y_orig = contour_poly.exterior.xy
                        ax.plot(x_orig, y_orig, 'k--', linewidth=0.8, label='Original Contour')
                        for interior in contour_poly.interiors:
                            x_int, y_int = interior.xy
                            ax.plot(x_int, y_int, 'k:', linewidth=0.8)
                        # Plot CFS path
                        x_path, y_path = cfs_result.xy
                        ax.plot(x_path, y_path, 'b-', linewidth=1.0, label='CFS Path')
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_title(f"Layer {i+1}, Contour {j+1} - CFS Fill (w={toolpath_width})")
                        ax.set_xlabel("X (mm)")
                        ax.set_ylabel("Y (mm)")
                        ax.legend()
                        plt.grid(True, linestyle=':', alpha=0.5)
                        plt.show(block=False) # Show plot non-blockingly
                    else:
                        print(f"    -> CFS Path generation incomplete for Contour {j+1}.")

            # Optionally visualize the original layers again if needed
            # print("\nVisualizing sample layers (original contours)...")
            # visualize_layers(layers, mesh_info['min_coords'][2], layer_height, num_to_show=min(len(layers), 3))

    else:
        print(f"Failed to load STL file: {stl_file_path}")
        print("Please ensure the file exists and is a valid STL file.")

    # Keep all plot windows open until manually closed
    if plt.get_fignums(): # Check if any figures were created
        print("\nDisplaying generated plots. Close plot windows to exit.")
        plt.show() # Blocking call to display all figures

if __name__ == "__main__":
    main()
