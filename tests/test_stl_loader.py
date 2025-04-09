import os
import unittest
import numpy as np
from src.stl_loader import load_stl, get_mesh_info

class TestSTLLoader(unittest.TestCase):
    def setUp(self):
        # Path to the test STL file
        self.valid_stl_path = os.path.join("models", "stretchrite3.stl")
        self.invalid_stl_path = os.path.join("models", "nonexistent.stl")
    
    def test_load_valid_stl(self):
        """Test loading a valid STL file"""
        # Load the STL file
        stl_mesh = load_stl(self.valid_stl_path)
        
        # Check if the mesh was loaded successfully
        self.assertIsNotNone(stl_mesh, "Failed to load a valid STL file")
        
        # Check if the mesh has triangles
        self.assertGreater(len(stl_mesh.vectors), 0, "Loaded mesh has no triangles")
    
    def test_load_invalid_stl(self):
        """Test loading an invalid/nonexistent STL file"""
        # Try to load a nonexistent file
        stl_mesh = load_stl(self.invalid_stl_path)
        
        # Check if the function returns None for invalid file
        self.assertIsNone(stl_mesh, "Should return None for invalid STL file")
    
    def test_get_mesh_info(self):
        """Test extracting information from a loaded mesh"""
        # Load the STL file
        stl_mesh = load_stl(self.valid_stl_path)
        
        # Get mesh info
        mesh_info = get_mesh_info(stl_mesh)
        
        # Check if mesh_info is not None
        self.assertIsNotNone(mesh_info, "Failed to get mesh information")
        
        # Check if mesh_info contains expected keys
        expected_keys = ["num_triangles", "volume", "center_of_gravity", 
                         "dimensions", "min_coords", "max_coords"]
        for key in expected_keys:
            self.assertIn(key, mesh_info, f"Missing key in mesh_info: {key}")
        
        # Check if dimensions are positive
        self.assertTrue(np.all(mesh_info["dimensions"] > 0), 
                        "Mesh dimensions should be positive")
        
        # Check if min_coords are less than max_coords
        self.assertTrue(np.all(mesh_info["min_coords"] < mesh_info["max_coords"]), 
                        "Min coordinates should be less than max coordinates")
    
    def test_get_mesh_info_none(self):
        """Test get_mesh_info with None input"""
        # Get mesh info with None input
        mesh_info = get_mesh_info(None)
        
        # Check if the function returns None
        self.assertIsNone(mesh_info, "Should return None for None input")

if __name__ == "__main__":
    unittest.main()
