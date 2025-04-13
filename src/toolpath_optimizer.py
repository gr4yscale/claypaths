import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile
import re

class ToolpathOptimizer:
    """
    Optimizes toolpaths using TSP solver and greedy algorithms to minimize travel distance.
    """
    
    def __init__(self, toolpath_width=0.4):
        """
        Initialize the toolpath optimizer.
        
        Args:
            toolpath_width (float): Width of the toolpath in mm
        """
        self.toolpath_width = toolpath_width
        self.total_cost = 0.0
        self.layer_paths = []
    
    def optimize_layers(self, layers, fill_generator):
        """
        Optimize toolpaths across all layers.
        
        Args:
            layers (list): List of layer contours, where each layer is a list of shapely Polygons
            fill_generator (function): Function to generate fill paths for a polygon
            
        Returns:
            list: List of optimized paths for each layer
        """
        if not layers:
            return []
        
        print("\nOptimizing toolpaths across layers...")
        optimized_paths = []
        prev_end_point = None
        
        for i, layer in enumerate(layers):
            print(f"Optimizing layer {i+1}/{len(layers)}...")
            
            if not layer or len(layer) == 0:
                print(f"  No polygons in layer {i+1}, skipping")
                optimized_paths.append([])
                continue
            
            # Generate fill paths for each polygon in the layer
            layer_curves = []
            for j, polygon in enumerate(layer):
                # Generate the fill path for this polygon
                fill_path = fill_generator(polygon, self.toolpath_width)
                
                if fill_path and len(fill_path) > 1:
                    # Split the path into segments if it's too long
                    segments = self._split_path_into_segments(fill_path)
                    layer_curves.extend(segments)
            
            # Optimize the ordering of curves within this layer
            if layer_curves:
                optimized_layer_path = self._optimize_layer(layer_curves, prev_end_point)
                optimized_paths.append(optimized_layer_path)
                
                # Update the previous end point for the next layer
                if optimized_layer_path:
                    prev_end_point = optimized_layer_path[-1]
            else:
                print(f"  No valid curves in layer {i+1}, skipping")
                optimized_paths.append([])
        
        print(f"Toolpath optimization complete. Total cost: {self.total_cost:.2f}")
        return optimized_paths
    
    def _split_path_into_segments(self, path, max_points=1000):
        """
        Split a long path into smaller segments for optimization.
        
        Args:
            path (list): List of points representing the path
            max_points (int): Maximum number of points per segment
            
        Returns:
            list: List of path segments
        """
        if len(path) <= max_points:
            return [path]
        
        segments = []
        num_segments = (len(path) + max_points - 1) // max_points
        
        for i in range(num_segments):
            start_idx = i * max_points
            end_idx = min((i + 1) * max_points, len(path))
            segment = path[start_idx:end_idx]
            
            if len(segment) > 1:
                segments.append(segment)
        
        return segments
    
    def _optimize_layer(self, curves, prev_end_point=None):
        """
        Optimize the ordering of curves within a layer using TSP.
        
        Args:
            curves (list): List of curves, where each curve is a list of points
            prev_end_point (tuple): The end point of the previous layer's path
            
        Returns:
            list: Optimized path for the layer
        """
        if not curves:
            return []
        
        # Construct a complete graph with nodes for each curve
        # Each curve can be traversed in either direction (A->B or B->A)
        num_curves = len(curves)
        
        # Extract endpoints for each curve
        endpoints = []
        for curve in curves:
            endpoints.append((curve[0], curve[-1]))
        
        # Create a temporary file for the TSP problem
        with tempfile.NamedTemporaryFile(suffix='.tsp', delete=False) as tsp_file:
            tsp_filename = tsp_file.name
            
            # Write the TSP problem in TSPLIB format
            tsp_file.write(f"NAME: Layer_TSP\n".encode())
            tsp_file.write(f"TYPE: TSP\n".encode())
            tsp_file.write(f"DIMENSION: {num_curves}\n".encode())
            tsp_file.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n".encode())
            tsp_file.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n".encode())
            tsp_file.write(f"EDGE_WEIGHT_SECTION\n".encode())
            
            # Calculate the distance matrix
            distance_matrix = np.zeros((num_curves, num_curves))
            
            for i in range(num_curves):
                for j in range(num_curves):
                    if i == j:
                        distance_matrix[i, j] = 0
                    else:
                        # Calculate distances between all possible endpoint combinations
                        # We'll use the minimum distance when optimizing
                        start_i, end_i = endpoints[i]
                        start_j, end_j = endpoints[j]
                        
                        # Calculate distances between all endpoint combinations
                        d1 = self._euclidean_distance(end_i, start_j)  # i forward -> j forward
                        d2 = self._euclidean_distance(end_i, end_j)    # i forward -> j backward
                        d3 = self._euclidean_distance(start_i, start_j) # i backward -> j forward
                        d4 = self._euclidean_distance(start_i, end_j)   # i backward -> j backward
                        
                        # Use the minimum distance
                        distance_matrix[i, j] = min(d1, d2, d3, d4)
            
            # Write the distance matrix to the TSP file
            for i in range(num_curves):
                row = " ".join([str(int(distance_matrix[i, j] * 1000)) for j in range(num_curves)])
                tsp_file.write(f"{row}\n".encode())
            
            tsp_file.write(f"EOF\n".encode())
        
        # Call Concorde TSP solver if available, otherwise use a simple greedy approach
        try:
            # Try to run Concorde
            tsp_tour = self._run_concorde(tsp_filename)
            
            if not tsp_tour:
                # Fall back to greedy approach
                tsp_tour = self._greedy_tsp(distance_matrix, prev_end_point, endpoints)
        except Exception as e:
            print(f"  Error running TSP solver: {e}")
            # Fall back to greedy approach
            tsp_tour = self._greedy_tsp(distance_matrix, prev_end_point, endpoints)
        
        # Clean up the temporary file
        try:
            os.remove(tsp_filename)
        except:
            pass
        
        # Now we have the TSP tour, but we need to determine the direction for each curve
        optimized_path = []
        
        # If we have a previous end point, find the best starting curve and direction
        if prev_end_point:
            # Find the best starting curve and direction
            best_start_idx = 0
            best_start_cost = float('inf')
            best_start_reverse = False
            
            for i in range(len(tsp_tour)):
                curve_idx = tsp_tour[i]
                start, end = endpoints[curve_idx]
                
                # Calculate cost to start from this curve
                cost_forward = self._euclidean_distance(prev_end_point, start)
                cost_backward = self._euclidean_distance(prev_end_point, end)
                
                if cost_forward < best_start_cost:
                    best_start_idx = i
                    best_start_cost = cost_forward
                    best_start_reverse = False
                
                if cost_backward < best_start_cost:
                    best_start_idx = i
                    best_start_cost = cost_backward
                    best_start_reverse = True
            
            # Reorder the tour to start from the best curve
            tsp_tour = tsp_tour[best_start_idx:] + tsp_tour[:best_start_idx]
            
            # Add the cost of traveling from the previous layer
            self.total_cost += best_start_cost
        
        # Build the optimized path by connecting curves in the TSP tour order
        prev_point = prev_end_point
        
        for i, curve_idx in enumerate(tsp_tour):
            curve = curves[curve_idx]
            start, end = endpoints[curve_idx]
            
            # Determine whether to traverse the curve forward or backward
            reverse = False
            
            if prev_point:
                # Calculate costs for both directions
                cost_forward = self._euclidean_distance(prev_point, start)
                cost_backward = self._euclidean_distance(prev_point, end)
                
                # Choose the direction with the lower cost
                reverse = cost_backward < cost_forward
                
                # Add the travel cost
                self.total_cost += min(cost_forward, cost_backward)
            
            # Add the curve to the optimized path
            if reverse:
                # Add the curve in reverse order
                for point in reversed(curve):
                    optimized_path.append(point)
                prev_point = start
            else:
                # Add the curve in forward order
                for point in curve:
                    optimized_path.append(point)
                prev_point = end
        
        return optimized_path
    
    def _run_concorde(self, tsp_filename):
        """
        Run the Concorde TSP solver using Docker.
        
        Args:
            tsp_filename (str): Path to the TSP problem file
            
        Returns:
            list: Optimal tour as a list of indices
        """
        try:
            # Get the directory and filename
            tsp_dir = os.path.dirname(tsp_filename)
            tsp_basename = os.path.basename(tsp_filename)
            
            # Create the Docker command
            docker_cmd = [
                'docker', 'run', '--rm', '-t',
                '-v', f'{tsp_dir}:/usr/local/opt/concorde/',
                'alehkot/concorde-tsp:1.1',
                f'/data/{tsp_basename}'
            ]
            
            # Run Concorde via Docker
            print("  Running Concorde TSP solver via Docker...")
            result = subprocess.run(docker_cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print(f"  Error running Concorde via Docker: {result.stderr.decode('utf-8')}")
                print("  Falling back to greedy approach")
                return None
            
            # Parse the output to get the tour
            tour_file = tsp_filename.replace('.tsp', '.sol')
            
            if not os.path.exists(tour_file):
                print("  Tour file not found, using greedy approach")
                return None
            
            with open(tour_file, 'r') as f:
                lines = f.readlines()
                
                if len(lines) < 2:
                    print("  Invalid tour file, using greedy approach")
                    return None
                
                # Parse the tour
                tour_line = lines[1].strip()
                tour = [int(x) for x in tour_line.split()]
                
                # Clean up the tour file
                try:
                    os.remove(tour_file)
                except:
                    pass
                
                return tour
        except Exception as e:
            print(f"  Error running Concorde: {e}")
            return None
    
    def _greedy_tsp(self, distance_matrix, prev_end_point=None, endpoints=None):
        """
        Solve the TSP problem using a greedy approach.
        
        Args:
            distance_matrix (numpy.ndarray): Distance matrix
            prev_end_point (tuple): The end point of the previous layer's path
            endpoints (list): List of (start, end) points for each curve
            
        Returns:
            list: Tour as a list of indices
        """
        num_nodes = distance_matrix.shape[0]
        
        # If we have a previous end point, find the best starting node
        if prev_end_point and endpoints:
            start_node = 0
            min_dist = float('inf')
            
            for i in range(num_nodes):
                start, end = endpoints[i]
                
                dist_to_start = self._euclidean_distance(prev_end_point, start)
                dist_to_end = self._euclidean_distance(prev_end_point, end)
                
                if dist_to_start < min_dist:
                    min_dist = dist_to_start
                    start_node = i
                
                if dist_to_end < min_dist:
                    min_dist = dist_to_end
                    start_node = i
        else:
            # Start from node 0
            start_node = 0
        
        # Initialize the tour with the starting node
        tour = [start_node]
        unvisited = set(range(num_nodes))
        unvisited.remove(start_node)
        
        # Greedily build the tour
        while unvisited:
            current = tour[-1]
            next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(next_node)
            unvisited.remove(next_node)
        
        return tour
    
    
    def _euclidean_distance(self, p1, p2):
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            p1 (tuple): First point (x, y)
            p2 (tuple): Second point (x, y)
            
        Returns:
            float: Euclidean distance
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def visualize_optimized_path(self, layer_polygons, optimized_path, layer_idx):
        """
        Visualize the optimized path for a layer.
        
        Args:
            layer_polygons (list): List of polygons in the layer
            optimized_path (list): Optimized path for the layer
            layer_idx (int): Layer index
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the polygons
        for polygon in layer_polygons:
            x, y = polygon.exterior.xy
            ax.plot(x, y, 'b-', linewidth=2, label='Polygon Boundary')
            
            # Plot holes if any
            for interior in polygon.interiors:
                x, y = interior.xy
                ax.plot(x, y, 'b-', linewidth=2)
        
        # Plot the optimized path
        if optimized_path:
            path_x, path_y = zip(*optimized_path)
            
            # Use a colormap to show the direction of the path
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a colorful line collection
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis', linewidth=1.5)
            lc.set_array(np.linspace(0, 1, len(path_x)-1))
            ax.add_collection(lc)
            
            # Mark start and end points
            ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')
            
            # Add a colorbar to show progression
            cbar = plt.colorbar(lc, ax=ax)
            cbar.set_label('Path Direction')
        
        ax.set_aspect('equal')
        ax.set_title(f"Layer {layer_idx+1} Optimized Path")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend()
        
        plt.tight_layout()
        plt.show(block=False)
