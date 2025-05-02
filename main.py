import os
import numpy as np
import matplotlib.pyplot as plt
from src.stl_loader import load_stl, visualize_stl, get_mesh_info
from src.slicer import slice_mesh, visualize_layers
# from src.perimeter import generate_perimeter_paths # Keep if needed later
from src.fill_continuous_offset import generate_continuous_offset_fill, visualize_contours_and_path # Import the new function
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

        print("\nStep 3: Generating continuous toolpath for layers")
        config = get_config()
        toolpath_width = config['toolpath_width']
        perimeter_count = config.get('perimeter_count', 1) # Get perimeter count from config
        
       
        # Process all layers
        if layers and len(layers) > 0:
            # Get max layers to process from config
            config = get_config()
            max_layers = config['max_layers_to_process']
            
            # Limit the number of layers to process
            num_layers_to_process = min(max_layers, len(layers))
            layers_to_process = layers[:num_layers_to_process]
            
            print(f"Processing {num_layers_to_process} layers...")
            
            all_layer_paths = []
            
            paths_for_gcode = [] # Store the final path for each layer for GCode generation

            for layer_idx, layer_polygons in enumerate(layers_to_process):
                print(f"\nProcessing layer {layer_idx+1}/{len(layers_to_process)}...")

                layer_paths = []
                # Process each polygon in the layer (usually one, but could be more for complex slices)
                for poly_idx, polygon in enumerate(layer_polygons):
                    print(f"  Generating continuous offset fill for polygon {poly_idx+1}...")

                    # Generate the continuous path using the new method
                    continuous_path = generate_continuous_offset_fill(polygon, toolpath_width)

                    if continuous_path:
                        layer_paths.append(continuous_path)
                        # Optional: Visualize each layer's path
                        # visualize_contours_and_path(polygon, continuous_path, f"Layer {layer_idx+1} Polygon {poly_idx+1} Path")
                    else:
                        print(f"  Warning: Could not generate path for polygon {poly_idx+1} in layer {layer_idx+1}")

                if layer_paths:
                     # Combine paths if multiple polygons were processed in the layer
                     # For now, assume we just take the paths as a list (or combine if needed by optimizer later)
                     # If the fill function returns a single LineString per polygon,
                     # we might need an optimizer step here to connect them if there are multiple.
                     # For now, store them as a list.
                     all_layer_paths.append(layer_paths) # Store paths for this layer
                     paths_for_gcode.append(layer_paths) # Add to list for GCode
                else:
                     all_layer_paths.append([]) # Keep layer count consistent
                     paths_for_gcode.append([])


            # Step 4: (Optional) Optimization - Skipped for now, using direct output
            print("\nStep 4: Toolpath Optimization (Skipped - Using direct continuous path)")
            # optimized_paths = paths_for_gcode # Use the generated paths directly


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
