import os
import numpy as np
import matplotlib.pyplot as plt # Ensure matplotlib is imported
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
from src.region_fill import generate_continuous_fill, visualize_fill_path

def calculate_travel_distance(prev_layer, current_layer, current_path):
    """
    Calculate the travel distance between the end of the previous layer's path
    and the start of the current layer's path.
    
    Args:
        prev_layer: List of polygons in the previous layer
        current_layer: List of polygons in the current layer
        current_path: The current layer's path
        
    Returns:
        float: The travel distance in mm
    """
    if not prev_layer or not current_layer or not current_path:
        return 0.0
    
    # Get the end point of the previous layer's path
    # This is a simplification - in a real implementation, you'd store the actual path
    prev_end_point = getattr(calculate_travel_distance, 'prev_end_point', None)
    
    if prev_end_point and current_path:
        # Calculate Euclidean distance between the previous end point and current start point
        start_point = current_path[0]
        dx = prev_end_point[0] - start_point[0]
        dy = prev_end_point[1] - start_point[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Store the current end point for the next calculation
        calculate_travel_distance.prev_end_point = current_path[-1]
        
        return distance
    
    # Store the current end point for the next calculation
    if current_path:
        calculate_travel_distance.prev_end_point = current_path[-1]
    
    return 0.0

def main():
    print("Welcome to claypaths - Fermat Spiral 3D Printing Toolpath Generator")
    
    # Paths to test STL files
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")

    #confirmed working
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    stl_file_path = os.path.join("models", "t-shape.stl")
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
        #visualize_stl(stl_mesh)
        
        # Step 2: Slice the model into layers
        print("\nStep 2: Slicing the model into layers")
        layer_height = 1.0  # Default layer height in mm
        layers = slice_mesh(stl_mesh, layer_height)
        
        # Optionally visualize the original layers again if needed
        #print("\nVisualizing sample layers (original contours)...")
        #visualize_layers(layers, mesh_info['min_coords'][2], layer_height, num_to_show=min(len(layers), 3))
        
        # Step 3: Generate region fill for each layer
        print("\nStep 3: Generating region fill for layers")
        toolpath_width = 0.4  # Default toolpath width in mm
        
        # Process multiple layers
        if layers and len(layers) > 0:
            # Initialize previous end point to None (for the first layer)
            prev_end_point = None
            
            # Process a few sample layers
            num_layers_to_process = min(5, len(layers))
            for layer_idx in range(num_layers_to_process):
                layer = layers[layer_idx]
                
                if layer and len(layer) > 0:
                    # Take the first polygon in the layer
                    contour_poly = layer[0]
                    
                    print(f"\nGenerating region fill for layer {layer_idx+1}, polygon 1")
                    
                    # Generate continuous fill path for the polygon, optimizing for previous layer's end point
                    fill_path = generate_continuous_fill(contour_poly, toolpath_width, prev_end_point)
                    
                    if fill_path:
                        print(f"Generated continuous fill path with {len(fill_path)} points")
                        
                        # Update the previous end point for the next layer
                        prev_end_point = fill_path[-1] if fill_path else None
                        
                        # Visualize the fill path
                        visualize_fill_path(contour_poly, fill_path, 
                                          f"Layer {layer_idx+1} Continuous Fill Path")
                        
                        # If this isn't the first layer, print the travel distance
                        if layer_idx > 0:
                            print(f"Travel distance between layers {layer_idx} and {layer_idx+1}: "
                                  f"{calculate_travel_distance(layers[layer_idx-1], layer, fill_path):.2f}mm")
                    else:
                        print("Failed to generate fill path")
                        prev_end_point = None
                else:
                    print(f"No valid polygons in layer {layer_idx+1}")
                    prev_end_point = None

    else:
        print(f"Failed to load STL file: {stl_file_path}")
        print("Please ensure the file exists and is a valid STL file.")

    # Keep all plot windows open until manually closed
    if plt.get_fignums(): # Check if any figures were created
        print("\nDisplaying generated plots. Close plot windows to exit.")
        plt.show() # Blocking call to display all figures

if __name__ == "__main__":
    main()
