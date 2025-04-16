import os
import yaml

# Default configuration values
DEFAULT_CONFIG = {
    'toolpath_width': 0.4,  # Default toolpath width in mm
    'layer_height': 1.0,    # Default layer height in mm
    'max_layers_to_process': 6,  # Maximum number of layers to process
    'optimization_method': 'greedy',  # Preferred optimization method
    
    # Polygonization method options: 'standard', 'small_buffer', 'large_buffer', 'manual_close', 'hybrid'
    'polygonization_method': 'hybrid',
    'buffer_size_small': 0.001,  # Size for small buffer method
    'buffer_size_large': 0.01,   # Size for large buffer method
   
    'visualize_stl_mesh': False,
    'visualize_layer_contours': True,  # Enable layer contour visualization by default
    'visualize_fill_paths': True,      # Enable fill path visualization
    'visualize_optimized_paths': True, # Enable optimized path visualization
    'visualize_layer_transitions': False,
    'visualize_problematic_segments': True, # Enable visualization of problematic segments
    
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
