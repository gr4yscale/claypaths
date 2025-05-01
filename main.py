import os
import numpy as np
import matplotlib.pyplot as plt # Ensure matplotlib is imported
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
from src.perimeter import generate_perimeter_paths # Import perimeter generation
from src.region_fill import generate_continuous_fill, visualize_fill_path # visualize_fill_path might be removed later
from src.optimizer_a import OptimizerA
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

        # Step 3: Generate perimeters and region fill for each layer
        print("\nStep 3: Generating perimeters and region fill for layers")
        config = get_config()
        toolpath_width = config['toolpath_width']
        perimeter_count = config.get('perimeter_count', 1) # Get perimeter count from config
        
        # Get the configured optimizer
        config = get_config()
        optimizer_class = config.get('toolpath_optimizer', 'OptimizerA')
        
        # Create the appropriate optimizer instance
        if optimizer_class == 'OptimizerB':
            optimizer = OptimizerB(toolpath_width)
        else:
            optimizer = OptimizerA(toolpath_width)
            
        print(f"Using {optimizer_class} for toolpath optimization")
        
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
                
                # Generate paths for each polygon in the layer
                layer_paths = [] # Stores all paths (perimeters + fill) for the layer
                for j, polygon in enumerate(layer):
                    print(f"\n  Processing polygon {j+1} in layer {layer_idx+1}...")
                    
                    # 1. Generate Perimeter Paths
                    perimeter_paths, inner_fill_polygon = generate_perimeter_paths(
                        polygon, perimeter_count, toolpath_width
                    )
                    
                    # Add valid perimeter paths to the layer paths
                    valid_perimeter_paths = [p for p in perimeter_paths if len(p) > 1]
                    layer_paths.extend(valid_perimeter_paths)
                    
                    # 2. Generate Fill Paths (if inner polygon exists)
                    fill_paths_for_polygon = []
                    if inner_fill_polygon:
                        fill_result = generate_continuous_fill(inner_fill_polygon, toolpath_width) 
                        
                        # Process fill_result (can be list or list of lists)
                        if fill_result:
                            is_list_of_paths = (isinstance(fill_result, list) and 
                                                len(fill_result) > 0 and 
                                                isinstance(fill_result[0], list))

                            if is_list_of_paths:
                                valid_fill_paths = [p for p in fill_result if len(p) > 1]
                                fill_paths_for_polygon.extend(valid_fill_paths)
                            elif isinstance(fill_result, list) and len(fill_result) > 1:
                                fill_paths_for_polygon.append(fill_result)
                                
                    # Add valid fill paths to the layer paths
                    layer_paths.extend(fill_paths_for_polygon)

                    # Visualize the results for this polygon (perimeters and fill)
                    # Note: visualize_fill_path needs adaptation to show perimeters distinctly
                    visualize_fill_path(
                        polygon=polygon, 
                        perimeter_paths=valid_perimeter_paths, 
                        fill_paths=fill_paths_for_polygon, 
                        title=f"Layer {layer_idx+1}, Polygon {j+1} - Perimeters & Fill"
                        # unfilled_regions could be passed if generated and needed for viz
                    )
                            
                all_layer_paths.append(layer_paths) # Add all paths (perimeters + fill) for this layer
            
            # Step 4: Optimize paths (optional)
            enable_optimization = config.get('enable_toolpath_optimization', True)
            paths_for_gcode = [] # This will hold either optimized or unoptimized paths

            if enable_optimization:
                print("\nStep 4: Optimizing paths across all layers...")
                optimized_paths = optimizer.optimize_layers(layers_to_process, all_layer_paths)
                paths_for_gcode = optimized_paths # Use optimized paths for GCode

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
                        print(f"\nLayer {layer_idx+1}: No valid path generated after optimization")
                
                # Visualize the transitions between layers
                print("\nVisualizing layer transitions...")
                optimizer.visualize_layer_transitions(layers_to_process, optimized_paths)
            else:
                print("\nStep 4: Toolpath optimization is disabled.")
                # Use the unoptimized paths directly for GCode generation
                # The optimizer returns a single path per layer, while all_layer_paths
                # contains a list of paths per layer. We need to adapt the GCode generator
                # or format the paths here. For now, let's assume GCode generator
                # can handle the list of paths per layer format.
                # TODO: Verify GCode generator handles list of paths per layer.
                paths_for_gcode = all_layer_paths 
            
            # Step 5: Generate GCode
            if config.get('generate_gcode', True):
                print("\nStep 5: Generating GCode...")
                gcode_gen = GCodeGenerator()  # Will use flavor from config
                # Pass the selected paths (optimized or unoptimized)
                gcode = gcode_gen.generate_gcode(layers_to_process, paths_for_gcode, mesh_info['min_coords'][2])
                
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
