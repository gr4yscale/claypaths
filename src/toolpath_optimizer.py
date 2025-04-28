import numpy as np
from shapely.geometry import Point, LineString, Polygon # Added LineString and Polygon
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile
import re
from src.config import get_config

class ToolpathOptimizer:
    """
    Optimizes toolpaths using TSP solver and greedy algorithms to minimize travel distance.
    """
    
    def __init__(self, toolpath_width=None):
        """
        Initialize the toolpath optimizer.
        
        Args:
            toolpath_width (float, optional): Width of the toolpath in mm. 
                                             If None, uses value from config.
        """
        # Use configuration value if toolpath_width is not provided
        if toolpath_width is None:
            config = get_config()
            toolpath_width = config['toolpath_width']
            
        self.toolpath_width = toolpath_width
        self.total_cost = 0.0
        self.layer_paths = []
    
    def optimize_layers(self, layers, layer_paths):
        """
        Optimize toolpaths across all layers.
        
        Args:
            layers (list): List of layer contours, where each layer is a list of shapely Polygons
            layer_paths (list): List of paths for each layer, where each path is a list of points
            
        Returns:
            list: List of optimized paths for each layer
        """
        if not layers or not layer_paths:
            return []
        
        print("\nOptimizing toolpaths across layers...")
        optimized_paths = []
        prev_end_point = None
        
        for i, (layer, paths) in enumerate(zip(layers, layer_paths)):
            print(f"Optimizing layer {i+1}/{len(layers)}...")
            
            if not paths:
                print(f"  No paths in layer {i+1}, skipping")
                optimized_paths.append([])
                continue
            
            # Split the paths into segments if they're too long
            layer_curves = []
            for path in paths:
                if path and len(path) > 1:
                    segments = self._split_path_into_segments(path)
                    layer_curves.extend(segments)
            
            # Optimize the ordering of curves within this layer
            if layer_curves:
                # Pass the actual layer polygons for intersection checks
                optimized_layer_path = self._optimize_layer(layer_curves, layer, prev_end_point)

                # Smooth the optimized path, passing polygons for intersection checks
                smoothed_path = self.smooth_path(optimized_layer_path, layer_polygons=layer)
                optimized_paths.append(smoothed_path)

                # Check for self-intersection in the final smoothed path
                self._check_self_intersection(smoothed_path, i + 1)

                # Update the previous end point for the next layer
                if smoothed_path:
                    prev_end_point = smoothed_path[-1]
                    print(f"  Layer {i+1}: Generated smooth path with {len(smoothed_path)} points")
            else:
                print(f"  No valid curves in layer {i+1}, skipping")
                optimized_paths.append([])
        
        print(f"Toolpath optimization complete. Total cost: {self.total_cost:.2f}")
        return optimized_paths
    
    def _split_path_into_segments(self, path, max_points=None):
        """
        Split a long path into smaller segments for optimization.
        
        Args:
            path (list): List of points representing the path
            max_points (int, optional): Maximum number of points per segment.
                                       If None, uses value from config.
            
        Returns:
            list: List of path segments
        """
        # Use configuration value if max_points is not provided
        if max_points is None:
            config = get_config()
            max_points = config['max_points_per_segment']
            
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

    def _optimize_layer(self, curves, layer_polygons, prev_end_point=None):
        """
        Optimize the ordering of curves within a layer using TSP, avoiding hole intersections.

        Args:
            curves (list): List of curves, where each curve is a list of points.
            layer_polygons (list): List of Shapely Polygons for the current layer.
            prev_end_point (tuple, optional): End point of the previous layer's path.

        Returns:
            list: Optimized path for the layer (before smoothing).
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
                        # Check for intersections with holes for each potential connection
                        d1 = self._calculate_connection_cost(end_i, start_j, layer_polygons)   # i forward -> j forward
                        d2 = self._calculate_connection_cost(end_i, end_j, layer_polygons)     # i forward -> j backward
                        d3 = self._calculate_connection_cost(start_i, start_j, layer_polygons)  # i backward -> j forward
                        connections = [
                            self._calculate_connection_cost(end_i, start_j, layer_polygons),   # 0: i forward -> j forward
                            self._calculate_connection_cost(end_i, end_j, layer_polygons),     # 1: i forward -> j backward
                            self._calculate_connection_cost(start_i, start_j, layer_polygons),  # 2: i backward -> j forward
                            self._calculate_connection_cost(start_i, end_j, layer_polygons)     # 3: i backward -> j backward
                        ]

                        # Find the connection with the minimum cost
                        min_cost = float('inf')
                        best_connection_info = None # Will store (cost, p1, p2)
                        for cost, p1_conn, p2_conn in connections:
                             if cost < min_cost:
                                  min_cost = cost
                                  best_connection_info = (cost, p1_conn, p2_conn)

                        # Store the minimum cost in the distance matrix
                        distance_matrix[i, j] = min_cost
                        # We also need to store which connection was best for path reconstruction later
                        # Let's use another matrix or dictionary for this.
                        # For simplicity now, we'll recalculate the best connection when building the path.
                        # TODO: Store best_connection_info efficiently if performance becomes an issue.


            # Scale the distance matrix to avoid "edge too long" errors
            # Concorde has limits on edge lengths, so we'll scale to a reasonable range
            # Note: Scaling infinity might cause issues, handle large numbers carefully
            config = get_config()
            tsp_scale_factor = config['tsp_scale_factor']
            
            if np.max(distance_matrix) > 0:
                # Scale to a range that Concorde can handle (typically max of 32767)
                scale_factor = min(tsp_scale_factor, 30000 / max(1, np.max(distance_matrix)))
            else:
                scale_factor = tsp_scale_factor
                
            print(f"  Scaling distances by factor {scale_factor:.2f}")
            # Write the scaled distance matrix to the TSP file, handling potential infinities
            max_finite_dist = 0
            for i in range(num_curves):
                 for j in range(num_curves):
                      if i != j and np.isfinite(distance_matrix[i, j]):
                           max_finite_dist = max(max_finite_dist, distance_matrix[i, j])

            # Define a large integer representation for infinity
            # Ensure it's within Concorde's typical limits (e.g., < 2^31)
            large_int_penalty = int(max(1, max_finite_dist * scale_factor * 100)) # Significantly larger than other costs
            large_int_penalty = min(large_int_penalty, 2**30) # Cap it

            for i in range(num_curves):
                row_values = []
                for j in range(num_curves):
                    cost = distance_matrix[i, j]
                    if np.isinf(cost):
                        scaled_cost = large_int_penalty
                    else:
                        scaled_cost = int(cost * scale_factor)
                    row_values.append(str(scaled_cost))
                row = " ".join(row_values)
                tsp_file.write(f"{row}\n".encode())

            tsp_file.write(f"EOF\n".encode())
        
        # Get the preferred optimization method from config
        config = get_config()
        optimization_method = config.get('optimization_method', 'greedy')
        
        # Use the specified optimization method
        # Pass layer_polygons to greedy if used
        if optimization_method.lower() == 'greedy':
            print("  Using greedy TSP optimization method")
            tsp_tour = self._greedy_tsp(distance_matrix, layer_polygons, prev_end_point, endpoints)
        else:  # Default to Concorde
            try:
                # Try to run Concorde
                print("  Using Concorde TSP optimization method")
                tsp_tour = self._run_concorde(tsp_filename)
                
                if not tsp_tour:
                    # Fall back to greedy approach
                    print("  Concorde failed, falling back to greedy approach")
                    tsp_tour = self._greedy_tsp(distance_matrix, layer_polygons, prev_end_point, endpoints)
            except Exception as e:
                print(f"  Error running TSP solver: {e}")
                # Fall back to greedy approach
                tsp_tour = self._greedy_tsp(distance_matrix, layer_polygons, prev_end_point, endpoints)

        # Clean up the temporary file
        try:
            os.remove(tsp_filename)
        except:
            pass
        
        # Now we have the TSP tour, but we need to determine the direction for each curve
        optimized_path = []
        best_start_reverse = False # Initialize default direction

        # If we have a previous end point, find the best starting curve and direction
        if prev_end_point:
            # Find the best starting curve and direction, considering hole intersections
            best_start_idx = 0
            best_start_cost = float('inf')
            best_start_reverse = False
            best_start_connection_points = (None, None) # Store (prev_end_point, actual_start_point)

            for i in range(len(tsp_tour)):
                curve_idx = tsp_tour[i]
                start, end = endpoints[curve_idx]

                # Calculate cost to start from this curve, penalizing intersections
                cost_forward, p1f, p2f = self._calculate_connection_cost(prev_end_point, start, layer_polygons)
                cost_backward, p1b, p2b = self._calculate_connection_cost(prev_end_point, end, layer_polygons)

                if cost_forward < best_start_cost:
                    best_start_idx = i
                    best_start_cost = cost_forward
                    best_start_reverse = False
                    best_start_connection_points = (p1f, p2f)

                if cost_backward < best_start_cost:
                    best_start_idx = i
                    best_start_cost = cost_backward
                    best_start_reverse = True
                    best_start_connection_points = (p1b, p2b)

            # Reorder the tour to start from the best curve
            tsp_tour = tsp_tour[best_start_idx:] + tsp_tour[:best_start_idx]

            # Add the cost of traveling from the previous layer
            if np.isfinite(best_start_cost):
                 self.total_cost += best_start_cost
                 # Add the initial connection segment if valid
                 if best_start_connection_points[0] and best_start_connection_points[1]:
                      # Add points between prev_end and actual start if needed (e.g., for smoothing)
                      # For now, just ensure the start point is correct
                      # optimized_path.append(best_start_connection_points[0]) # prev_end_point
                      optimized_path.append(best_start_connection_points[1]) # actual start point
            else:
                 print(f"  Warning: Initial connection from previous layer has infinite cost.")
                 # Decide how to handle this - maybe start path without connection?

        # Build the optimized path by connecting curves in the TSP tour order
        # Ensure the first point of the first curve is added correctly based on best_start_reverse
        first_curve_idx = tsp_tour[0]
        first_curve = curves[first_curve_idx]
        if best_start_reverse:
             # If starting reversed, the first point added should be the end of the first curve
             if not optimized_path or optimized_path[-1] != first_curve[-1]:
                  optimized_path.append(first_curve[-1])
             prev_point = first_curve[-1] # The point we are starting *from* on the first curve
        else:
             # If starting forward, the first point added should be the start of the first curve
             if not optimized_path or optimized_path[-1] != first_curve[0]:
                  optimized_path.append(first_curve[0])
             prev_point = first_curve[0] # The point we are starting *from* on the first curve


        for i, curve_idx in enumerate(tsp_tour):
            curve = curves[curve_idx]
            start, end = endpoints[curve_idx]

            # Determine the connection to the *next* curve (if not the last one)
            # And determine the traversal direction for the *current* curve
            reverse = False
            connection_cost = 0.0
            connection_points = (None, None) # (from_point, to_point)

            # Calculate connection cost from the *current* curve's potential endpoints
            # to the *next* curve's potential start points
            if i < len(tsp_tour) - 1:
                 next_curve_idx = tsp_tour[i+1]
                 next_start, next_end = endpoints[next_curve_idx]

                 # Consider 4 connection possibilities (end -> next_start, end -> next_end, start -> next_start, start -> next_end)
                 connections = [
                      self._calculate_connection_cost(end, next_start, layer_polygons),   # 0: current forward -> next forward
                      self._calculate_connection_cost(end, next_end, layer_polygons),     # 1: current forward -> next backward
                      self._calculate_connection_cost(start, next_start, layer_polygons),  # 2: current backward -> next forward
                      self._calculate_connection_cost(start, next_end, layer_polygons)     # 3: current backward -> next backward
                 ]

                 min_cost = float('inf')
                 best_conn_idx = -1
                 for idx, (cost, p1_conn, p2_conn) in enumerate(connections):
                      if cost < min_cost:
                           min_cost = cost
                           best_conn_idx = idx
                           connection_points = (p1_conn, p2_conn) # Store the points for the best connection

                 connection_cost = min_cost
                 # Determine reversal based on the best connection index
                 # If best connection starts from 'start' (indices 2 or 3), current curve should be reversed
                 reverse = best_conn_idx >= 2

                 # Add the travel cost (only if finite)
                 if np.isfinite(connection_cost):
                      self.total_cost += connection_cost
                 else:
                      print(f"  Warning: Infinite cost connection selected between curves {curve_idx} and {next_curve_idx}")

            else:
                 # Last curve in the tour - determine direction based on connection from previous
                 # This logic needs refinement - the direction of the last curve depends
                 # on how the *previous* curve connected to *it*.
                 # Let's recalculate based on prev_point connection to this curve's start/end
                 cost_forward, _, _ = self._calculate_connection_cost(prev_point, start, layer_polygons)
                 cost_backward, _, _ = self._calculate_connection_cost(prev_point, end, layer_polygons)
                 reverse = cost_backward < cost_forward


            # Add the points of the current curve
            current_curve_points = list(reversed(curve)) if reverse else list(curve)

            # Add the connection point (start of the curve) if it's not already the last point
            if not optimized_path or optimized_path[-1] != current_curve_points[0]:
                 # Add the connection segment points if needed (e.g., for visualization/smoothing)
                 # For now, just add the start point of the curve segment
                 optimized_path.append(current_curve_points[0])

            # Add the rest of the curve points (excluding the first one already added)
            optimized_path.extend(current_curve_points[1:])

            # Update prev_point to the end of the traversed curve
            prev_point = current_curve_points[-1]

            # Add the connection segment to the next curve if applicable and valid
            if i < len(tsp_tour) - 1 and connection_points[0] and connection_points[1] and np.isfinite(connection_cost):
                 # Add points for the travel move if needed for smoothing/visualization
                 # For now, we assume the next loop iteration will add the 'to_point'
                 # optimized_path.append(connection_points[1]) # Add the start point of the next curve
                 pass # The next iteration handles adding the start point of the next curve

            # --- Old Logic ---
            # --- End Old Logic ---

        # Remove consecutive duplicate points
        final_path = []
        if optimized_path:
             # Ensure the first point is always added
             if optimized_path:
                 final_path.append(optimized_path[0])
                 for k in range(1, len(optimized_path)):
                      # Add subsequent points only if they are different enough from the previous one
                      if self._euclidean_distance(final_path[-1], optimized_path[k]) > 1e-6:
                           final_path.append(optimized_path[k])

        return final_path
    
    def _run_concorde(self, tsp_filename):
        """
        Run the Concorde TSP solver using Docker.
        
        Args:
            tsp_filename (str): Path to the TSP problem file
            
        Returns:
            list: Optimal tour as a list of indices
        """
        try:
            # Read the TSP file to determine the problem size
            with open(tsp_filename, 'r') as f:
                for line in f:
                    if line.startswith("DIMENSION"):
                        dimension = int(line.split(":")[1].strip())
                        break
            
            # For very small problems (<=4 nodes), just use the greedy approach
            # Concorde sometimes has issues with very small problems
            if dimension <= 4:
                print(f"  Small problem (dimension={dimension}), using greedy approach")
                return None
            # Get the directory and filename
            tsp_dir = os.path.abspath(os.path.dirname(tsp_filename))
            tsp_basename = os.path.basename(tsp_filename)
            
            # Create the Docker command
            docker_cmd = [
                'docker', 'run', '--rm', '-t',
                '-v', f'{tsp_dir}:/data',  # Map to /data inside container
                '-w', '/data',  # Set working directory to /data
                'alehkot/concorde-tsp',
                f'{tsp_basename}'  # Use relative path since we set the working directory
            ]
            
            # Run Concorde via Docker
            print("  Running Concorde TSP solver via Docker...")
            result = subprocess.run(docker_cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
            
            # Log stdout and stderr for debugging
            stdout_output = result.stdout.decode('utf-8')
            stderr_output = result.stderr.decode('utf-8')
            
            if stdout_output:
                print(f"  Concorde stdout: {stdout_output}")
            if stderr_output:
                print(f"  Concorde stderr: {stderr_output}")
            
            # Check for specific error patterns in the output
            if "edge too long" in stdout_output or "edge too long" in stderr_output:
                print("  Concorde error: Edge too long. Try reducing the scale factor.")
                print("  Falling back to greedy approach")
                return None
            
            if result.returncode != 0:
                print(f"  Error running Concorde via Docker (return code: {result.returncode})")
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

    def _greedy_tsp(self, distance_matrix, layer_polygons, prev_end_point=None, endpoints=None):
        """
        Solve the TSP problem using a greedy approach, avoiding hole intersections.

        Args:
            distance_matrix (numpy.ndarray): Pre-calculated distance matrix (with penalties).
            layer_polygons (list): List of Shapely Polygons for the layer.
            prev_end_point (tuple, optional): End point of the previous layer's path.
            endpoints (list, optional): List of (start, end) points for each curve.

        Returns:
            list: Tour as a list of indices.
        """
        num_nodes = distance_matrix.shape[0]
        
        # If we have a previous end point, find the best starting node
        if prev_end_point and endpoints:
            start_node = 0
            min_dist = float('inf')

            for i in range(num_nodes):
                start, end = endpoints[i]

                # Calculate connection cost, considering intersections
                dist_to_start = self._calculate_connection_cost(prev_end_point, start, layer_polygons)
                dist_to_end = self._calculate_connection_cost(prev_end_point, end, layer_polygons)

                if dist_to_start < min_dist:
                    min_dist = dist_to_start
                    start_node = i

                if dist_to_end < min_dist:
                    min_dist = dist_to_end
                    start_node = i

            if np.isinf(min_dist):
                 print("  Warning (Greedy TSP): All starting connections intersect holes. Choosing node 0.")
                 start_node = 0 # Fallback if all connections are bad

        else:
            # Start from node 0 if no previous point
            start_node = 0
        
        # Initialize the tour with the starting node
        tour = [start_node]
        unvisited = set(range(num_nodes))
        unvisited.remove(start_node)
        
        # Greedily build the tour
        while unvisited:
            current = tour[-1]
            # Find the nearest unvisited node using the pre-calculated (penalized) distance matrix
            best_next_node = -1
            min_dist = float('inf')

            # Iterate through unvisited nodes to find the minimum valid distance
            possible_next_nodes = list(unvisited)
            np.random.shuffle(possible_next_nodes) # Add randomness if multiple nodes have same min dist

            for node in possible_next_nodes:
                 dist = distance_matrix[current, node]
                 if dist < min_dist:
                      min_dist = dist
                      best_next_node = node

            if best_next_node == -1 or np.isinf(min_dist):
                 print(f"  Warning (Greedy TSP): Cannot find valid next node from {current}. Stopping tour early.")
                 # This might happen if 'current' is isolated due to hole intersections
                 break # Stop the tour here

            tour.append(best_next_node)
            unvisited.remove(best_next_node)

        return tour

    def _check_intersection(self, p1, p2, layer_polygons):
        """
        Check if the line segment between p1 and p2 intersects any hole
        in the provided layer polygons.

        Args:
            p1 (tuple): Start point (x, y).
            p2 (tuple): End point (x, y).
            layer_polygons (list): List of Shapely Polygons for the layer.

        Returns:
            bool: True if the segment intersects a hole, False otherwise.
        """
        if not layer_polygons or p1 is None or p2 is None:
            return False # Cannot check intersection

        segment = LineString([p1, p2])

        for poly in layer_polygons:
            if isinstance(poly, Polygon): # Ensure it's a Polygon
                for interior in poly.interiors:
                    # Check if the segment intersects the boundary of the hole
                    if segment.intersects(interior):
                        # Optional: Add a small buffer check to avoid issues with points exactly on boundary
                        # if segment.buffer(1e-6).intersects(interior):
                        # print(f"    Intersection detected: Segment {p1} -> {p2} crosses hole.")
                        return True
        return False

    def _calculate_connection_cost(self, p1, p2, layer_polygons):
        """
        Calculate the cost of connecting p1 to p2, returning infinity
        if the connection intersects a hole.

        Args:
            p1 (tuple): Start point (x, y).
            p2 (tuple): End point (x, y).
            layer_polygons (list): List of Shapely Polygons for the layer.

        Returns:
            tuple: (cost, p1, p2) where cost is Euclidean distance or float('inf'),
                   and p1, p2 are the points defining the connection.
        """
        cost = self._euclidean_distance(p1, p2)
        if self._check_intersection(p1, p2, layer_polygons):
            cost = float('inf')
        # Return the cost and the points defining this specific connection attempt
        return cost, p1, p2

    def _euclidean_distance(self, p1, p2):
        """
        Calculate the Euclidean distance between two points. Handles None inputs.
        
        Args:
            p1 (tuple): First point (x, y)
            p2 (tuple): Second point (x, y)
            
        Returns:
            float: Euclidean distance, or 0.0 if points are identical or invalid.
        """
        if p1 is None or p2 is None:
             return 0.0 # Or perhaps float('inf') depending on context? Returning 0 for now.
        if p1 == p2:
             return 0.0
        try:
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        except (TypeError, IndexError):
             print(f"  Warning: Invalid points for distance calculation: {p1}, {p2}")
             return float('inf') # Penalize invalid points heavily

    def smooth_path(self, path, segment_length=None, layer_polygons=None):
        """
        Smooth a path by resampling it with uniform point spacing, avoiding hole intersections.

        Args:
            path (list): List of (x, y) points defining the path.
            segment_length (float, optional): Desired distance between points.
                                             If None, uses value from config.
            layer_polygons (list, optional): List of Shapely Polygons for the current layer
                                             to check against for hole intersections.

        Returns:
            list: Smoothed path with uniform point spacing.
        """
        if not path or len(path) < 2:
            return path
            
        # If segment_length is not provided, use value from config
        if segment_length is None:
            config = get_config()
            segment_length = config.get('path_smoothing_segment_length', self.toolpath_width / 2)
            
        # Calculate the total path length
        total_length = 0
        for i in range(len(path) - 1):
            total_length += self._euclidean_distance(path[i], path[i+1])
            
        # Calculate the number of segments needed
        num_segments = max(2, int(total_length / segment_length))
        
        # Create a new path with uniform spacing
        smoothed_path = []
        
        # Always include the first point
        smoothed_path.append(path[0])
        
        # Current position along the path
        current_length = 0
        current_segment = 0
        target_length = segment_length
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            segment_length_i = self._euclidean_distance(p1, p2)
            
            # Skip zero-length segments
            if segment_length_i < 1e-6:
                continue
                
            # Parametric position along this segment
            t_start = 0
            
            # Process this segment until we move to the next one
            while current_length + segment_length_i >= target_length:
                # Calculate how far along this segment the next point should be
                t = (target_length - current_length) / segment_length_i
                
                # Ensure t is between t_start and 1
                t = min(1, max(t_start, t))
                
                # Interpolate to find the point
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                next_point = (x, y)

                # --- Intersection Check ---
                valid_segment = True
                if layer_polygons and len(smoothed_path) > 0:
                    last_point = smoothed_path[-1]
                    # Avoid zero-length segments for checking
                    if self._euclidean_distance(last_point, next_point) > 1e-6:
                        segment_line = LineString([last_point, next_point])
                        for poly in layer_polygons:
                            if isinstance(poly, Polygon): # Ensure it's a Polygon
                                for interior in poly.interiors:
                                    if segment_line.intersects(interior):
                                        # print(f"  Warning: Smoothing segment intersects hole boundary. "
                                        #       f"Segment: {last_point} -> {next_point}. Skipping point.")
                                        valid_segment = False
                                        break # Stop checking interiors for this poly
                            if not valid_segment:
                                break # Stop checking other polygons

                # Add the point only if the segment is valid
                if valid_segment:
                    smoothed_path.append(next_point)
                    # Update tracking variables only if point was added
                    current_segment += 1
                    target_length = current_segment * segment_length
                    t_start = t
                else:
                    # If intersection occurs, we skip this point and try the next target_length
                    # This might lead to slightly uneven spacing near holes.
                    # An alternative would be to stop smoothing for this p1-p2 segment.
                    target_length += segment_length # Move target to next point

                # If we've reached the end of this segment, break
                if abs(t - 1) < 1e-6:
                    break
                target_length = current_segment * segment_length
                t_start = t
                
                # If we've reached the end of this segment, break
                if abs(t - 1) < 1e-6:
                    break
            
            # Update the current length along the path
            current_length += segment_length_i
        
        # Always include the last point, checking the final segment
        last_point_original = path[-1]
        if smoothed_path and smoothed_path[-1] != last_point_original:
             valid_segment = True
             if layer_polygons and len(smoothed_path) > 0:
                 last_added_point = smoothed_path[-1]
                 if self._euclidean_distance(last_added_point, last_point_original) > 1e-6:
                     segment_line = LineString([last_added_point, last_point_original])
                     for poly in layer_polygons:
                         if isinstance(poly, Polygon):
                             for interior in poly.interiors:
                                 if segment_line.intersects(interior):
                                     valid_segment = False
                                     break
                         if not valid_segment:
                             break
             if valid_segment:
                 smoothed_path.append(last_point_original)
             # else:
             #      print(f"  Warning: Final segment to endpoint intersects hole. Endpoint not added.")

        return smoothed_path
    
    def visualize_optimized_path(self, layer, optimized_path, layer_idx):
        """
        Visualize the optimized path for a layer.
        
        Args:
            layer (list): List of polygons in the layer
            optimized_path (list): Optimized path for the layer
            layer_idx (int): Layer index
        """
        # Check if visualization is enabled
        config = get_config()
        if not config.get('visualize_optimized_paths', True):
            print(f"Optimized path visualization disabled in config")
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the polygons
        for polygon in layer:
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

    def _check_self_intersection(self, path, layer_index):
        """
        Checks if a path self-intersects using Shapely.

        Args:
            path (list): List of (x, y) points defining the path.
            layer_index (int): The index of the layer for logging purposes.
        """
        if not path or len(path) < 4:
            return # Need at least 4 points for a potential intersection

        try:
            line = LineString(path)
            if not line.is_simple:
                print(f"  WARNING: Layer {layer_index} - Smoothed path may self-intersect.")
                # Optionally, visualize the intersection points if needed for debugging
                # intersections = line.intersection(line)
                # if not intersections.is_empty:
                #     print(f"    Intersection points/segments: {intersections}")
        except Exception as e:
            print(f"  Warning: Could not check self-intersection for layer {layer_index}: {e}")
    def visualize_layer_transitions(self, layers, optimized_paths):
        """
        Visualize the transitions between layers.
        
        Args:
            layers (list): List of layer polygons
            optimized_paths (list): List of optimized paths for each layer
        """
        # Check if visualization is enabled
        config = get_config()
        if not config.get('visualize_layer_transitions', True):
            print("Layer transition visualization disabled in config")
            return
            
        if not layers or not optimized_paths or len(layers) < 2:
            print("Not enough layers to visualize transitions")
            return
        
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set colors for different layers
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        
        # Plot each layer and its path
        for i, (layer, path, color) in enumerate(zip(layers, optimized_paths, colors)):
            if not path:
                continue
            
            # Extract path coordinates
            path_x, path_y = zip(*path)
            z_height = i  # Use layer index as z-height for visualization
            
            # Plot the layer path
            ax.plot(path_x, path_y, [z_height] * len(path_x), 
                   color=color, linewidth=2, label=f'Layer {i+1}')
            
            # Mark start and end points
            ax.scatter(path_x[0], path_y[0], z_height, 
                      color='green', s=100, marker='o', label=f'Start {i+1}' if i==0 else "")
            ax.scatter(path_x[-1], path_y[-1], z_height, 
                      color='red', s=100, marker='o', label=f'End {i+1}' if i==0 else "")
            
            # If not the first layer, draw a line connecting to the previous layer
            if i > 0 and optimized_paths[i-1]:
                prev_end = optimized_paths[i-1][-1]
                curr_start = path[0]
                
                # Draw a line connecting the layers
                ax.plot([prev_end[0], curr_start[0]], 
                       [prev_end[1], curr_start[1]], 
                       [i-1, i], 'k--', linewidth=1.5)
                
                # Calculate and display the travel distance
                travel_dist = self._euclidean_distance(prev_end, curr_start)
                mid_x = (prev_end[0] + curr_start[0]) / 2
                mid_y = (prev_end[1] + curr_start[1]) / 2
                mid_z = i - 0.5
                ax.text(mid_x, mid_y, mid_z, f"{travel_dist:.2f}mm", 
                       color='black', fontsize=9, ha='center')
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Layer')
        ax.set_title('Layer Transitions Visualization')
        
        # Add a legend for the first few items only (to avoid clutter)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        plt.tight_layout()
        plt.show(block=False)
