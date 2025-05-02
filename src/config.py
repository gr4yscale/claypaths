import os
import yaml

# Default configuration values
DEFAULT_CONFIG = {
    'toolpath_width': 0.4,  # Default toolpath width in mm
    'layer_height': 1.0,    # Default layer height in mm
    'perimeter_count': 2,   # Number of perimeter walls to generate
    'max_layers_to_process': 6,  # Maximum number of layers to process
    'enable_toolpath_optimization': True, # Whether to run the toolpath optimization step
    'optimization_method': 'greedy',  # Preferred optimization method
    'region_fill_algorithm': 'contour', # Algorithm for filling regions ('contour', 'zigzag', 'hybrid_contour_zigzag', 'fermat_spiral')
    
    'visualize_stl_mesh': False,
    'visualize_layer_contours': False,
    'visualize_fill_paths': False,
    'visualize_optimized_paths': False,
    'visualize_layer_transitions': False,
    'visualize_problematic_segments': False,

    # OptimizerC settings
    'optimizer_c_resolution': 0.2,      # Pixel size for image processing (mm)
    'optimizer_c_gaussian_sigma': 1.5,  # Sigma for Gaussian blur
    'visualize_optimizer_c_binary_image': False, # Show the rasterized layer shape
    'visualize_optimizer_c_filtered_image': False, # Show the Gaussian blurred image
    'visualize_optimizer_c_graph': False, # Show the graph nodes and start/end points

    # GCode generator settings
    'generate_gcode': True,     # Whether to generate GCode output
    'gcode_flavor': 'klipper',  # GCode flavor, either "klipper" or "marlin"
    'print_speed': 10,          # Print speed in mm/s
    'acceleration': 250,        # Acceleration in mm/s²
}

_config = None

def load_config(config_file='config.yaml'):
    """
    Load configuration from a YAML file.
    If the file doesn't exist, create it with default values.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    global _config
    
    if _config is not None:
        return _config
    
    # Check if config file exists
    if not os.path.exists(config_file):
        # Create default config file
        with open(config_file, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
        _config = DEFAULT_CONFIG
    else:
        # Load existing config file
        try:
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
            # Merge with defaults to ensure all required keys exist
            _config = DEFAULT_CONFIG.copy()
            if loaded_config and isinstance(loaded_config, dict):
                _config.update(loaded_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
            _config = DEFAULT_CONFIG
    
    return _config

def get_config():
    """
    Get the current configuration.
    Loads the configuration if it hasn't been loaded yet.
    
    Returns:
        dict: Configuration dictionary
    """
    if _config is None:
        return load_config()
    return _config
