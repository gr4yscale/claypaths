import os
import yaml

# Default configuration values
DEFAULT_CONFIG = {
    'toolpath_width': 0.4,  # Default toolpath width in mm
    'layer_height': 1.0,    # Default layer height in mm
    'max_layers_to_process': 6,  # Maximum number of layers to process
    'optimization_method': 'greedy',  # Preferred optimization method
   
    'visualize_stl_mesh': False,
    'visualize_layer_contours': False,
    'visualize_fill_paths': False,
    'visualize_optimized_paths': False,
    'visualize_layer_transitions': False,
    'visualize_problematic_segments': False,
    
    # GCode generator settings
    'gcode_flavor': 'klipper',  # GCode flavor, either "klipper" or "marlin"
    'travel_speed': 100,        # Travel speed in mm/s
    'print_speed': 50,          # Print speed in mm/s
    'acceleration': 1000,       # Acceleration in mm/s²
    'z_lift': 5,                # Z lift height in mm
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
