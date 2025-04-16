import os
import numpy as np
import matplotlib.pyplot as plt # Ensure matplotlib is imported
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
from src.region_fill import generate_continuous_fill, visualize_fill_path
from src.toolpath_optimizer import ToolpathOptimizer
from src.gcode_generator import GCodeGenerator
from src.config import get_config


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
        visualize_stl(stl_mesh)
        
        # Step 2: Slice the model into layers
        print("\nStep 2: Slicing the model into layers")
        config = get_config()
        layer_height = config['layer_height']
        layers = slice_mesh(stl_mesh, layer_height)
        
        # Optionally visualize the original layers again if needed
        visualize_layers(layers, mesh_info['min_coords'][2], layer_height)
        
        # Step 3: Generate and optimize region fill for each layer
        print("\nStep 3: Generating and optimizing region fill for layers")
        config = get_config()
        toolpath_width = config['toolpath_width']
        
        # Create a toolpath optimizer
        optimizer = ToolpathOptimizer(toolpath_width)
        
        # Process all layers
        if layers and len(layers) > 0:
            # Get max layers to process from config
            config = get_config()
            max_layers = config['max_layers_to_process']
            
            # Limit the number of layers to process
            num_layers_to_process = min(max_layers, len(layers))
            layers_to_process = layers[:num_layers_to_process]
            
            print(f"Processing {num_layers_to_process} layers...")
            
            # Generate fill paths for all layers first
            all_layer_paths = []
            
            for layer_idx, layer in enumerate(layers_to_process):
                print(f"\nGenerating fill paths for layer {layer_idx+1}/{len(layers_to_process)}...")
                
                # Generate fill paths for each polygon in the layer
                layer_paths = []
                for j, polygon in enumerate(layer):
                    # Generate the fill path for this polygon
                    fill_path = generate_continuous_fill(polygon, toolpath_width)
                    
                    if fill_path and len(fill_path) > 1:
                        # Visualize the fill path for this polygon
                        visualize_fill_path(polygon, fill_path, f"Layer {layer_idx+1}, Polygon {j+1} Fill Path")
                        layer_paths.append(fill_path)
                
                all_layer_paths.append(layer_paths)
            
            # Now optimize the paths across all layers
            print("\nOptimizing paths across all layers...")
            optimized_paths = optimizer.optimize_layers(layers_to_process, all_layer_paths)
            
            # Visualize the optimized paths
            for layer_idx, (layer, path) in enumerate(zip(layers_to_process, optimized_paths)):
                if path:
                    print(f"\nLayer {layer_idx+1}: Optimized path with {len(path)} points")
                    
                    # Visualize the optimized path
                    optimizer.visualize_optimized_path(layer, path, layer_idx)
                    
                    # If this isn't the first layer, print the travel distance
                    if layer_idx > 0 and optimized_paths[layer_idx-1]:
                        prev_end = optimized_paths[layer_idx-1][-1]
                        curr_start = path[0]
                        travel_dist = np.sqrt((prev_end[0] - curr_start[0])**2 + 
                                             (prev_end[1] - curr_start[1])**2)
                        print(f"Travel distance between layers {layer_idx} and {layer_idx+1}: "
                              f"{travel_dist:.2f}mm")
                else:
                    print(f"\nLayer {layer_idx+1}: No valid path generated")
            
            # Visualize the transitions between layers
            print("\nVisualizing layer transitions...")
            optimizer.visualize_layer_transitions(layers_to_process, optimized_paths)
            
            # Generate GCode from optimized paths
            if config.get('generate_gcode', True):
                print("\nGenerating GCode from optimized paths...")
                gcode_gen = GCodeGenerator()  # Will use flavor from config
                gcode = gcode_gen.generate_gcode(layers_to_process, optimized_paths, mesh_info['min_coords'][2])
                
                # Save GCode to file
                gcode_file = gcode_gen.save_gcode(gcode)
                print(f"GCode saved to: {gcode_file}")
            else:
                print("\nGCode generation is disabled in config")

    else:
        print(f"Failed to load STL file: {stl_file_path}")
        print("Please ensure the file exists and is a valid STL file.")

    # Keep all plot windows open until manually closed
    if plt.get_fignums(): # Check if any figures were created
        print("\nDisplaying generated plots. Close plot windows to exit.")
        plt.show() # Blocking call to display all figures

if __name__ == "__main__":
    main()
