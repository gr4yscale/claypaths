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
    print("Welcome to claypaths - continuous deposition slicer")
    
    # Paths to test STL files
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")

    # testing (stlparts)
    #stl_file_path = os.path.join("models", "test", "hollow-cuboid.stl") #kinda works
    #stl_file_path = os.path.join("models", "test", "5cm-cube-with-80-diameter-hole.stl")
    #stl_file_path = os.path.join("models", "test", "hollow-cylinder.stl")
    #stl_file_path = os.path.join("models", "test", "hollow-cylinder-with-floor.stl")
    #stl_file_path = os.path.join("models", "test", "hollow-stadium.stl")
    #stl_file_path = os.path.join("models", "test", "mountainbike-cable-holder.stl")
    #stl_file_path = os.path.join("models", "test", "ring.stl")
    #stl_file_path = os.path.join("models", "test", "truncated-cone.stl")
    #stl_file_path = os.path.join("models", "test", "truncated-cone-with-hole.stl")
    
    # testing (mine, freecad)
    #stl_file_path = os.path.join("models", "mine", "hex.stl")
    #stl_file_path = os.path.join("models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")
    
    #confirmed working, simple models
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "cuboid.stl")
    #stl_file_path = os.path.join("models", "extruded-rounded-rectangle.stl")
    #stl_file_path = os.path.join("models", "right-triangular-prism.stl")
    #stl_file_path = os.path.join("models", "stack-of-cuboids.stl")
    #stl_file_path = os.path.join("models", "stack-of-cylinders.stl")

    # complex shapes, problematic
    # holes are detected as solid rather than the cuboid
    stl_file_path = os.path.join("models", "cuboid-with-holes.stl")

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

        # Visualize the sliced layers (contours)
        if layers:
             visualize_layers(layers, mesh_info['min_coords'][2], layer_height)
        else:
             print("No layers were generated, skipping visualization.")

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
                layer_paths = [] # This will now store all paths (segments) for the layer
                for j, polygon in enumerate(layer):
                    # Generate the fill path(s) for this polygon
                    # Note: fill_result can be a single path (list) or a list of paths (list of lists)
                    fill_result = generate_continuous_fill(polygon, toolpath_width)
                    
                    if fill_result:
                        # Determine if we got a single path or a list of paths
                        is_list_of_paths = isinstance(fill_result[0], list) if fill_result else False
                        
                        # Visualize the fill path(s) for this polygon
                        visualize_fill_path(polygon, fill_result, f"Layer {layer_idx+1}, Polygon {j+1} Fill Path(s)")
                        
                        if is_list_of_paths:
                            # Extend layer_paths with the list of paths (segments)
                            # Filter out very short paths if necessary
                            valid_paths = [p for p in fill_result if len(p) > 1]
                            layer_paths.extend(valid_paths)
                        elif len(fill_result) > 1:
                            # Append the single path
                            layer_paths.append(fill_result)
                            
                all_layer_paths.append(layer_paths) # Add all paths for this layer
            
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
