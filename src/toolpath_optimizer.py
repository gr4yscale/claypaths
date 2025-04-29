import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union # Added unary_union
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile
import re
from src.config import get_config

class OptimizerA:
    """
    Optimizes toolpaths using TSP solver and greedy algorithms to minimize travel distance.
    Ensures travel moves do not cross holes or go outside layer boundaries.
    """

    def __init__(self, toolpath_width=None):
        """
        Initialize OptimizerA.

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
        Optimize the ordering of curves within a layer using TSP, avoiding invalid connections.

        Args:
            curves (list): List of curves, where each curve is a list of points.
            layer_polygons (list): List of Shapely Polygons for the current layer.
            prev_end_point (tuple, optional): End point of the previous layer's path.

        Returns:
            list: Optimized path for the layer (before smoothing).
        """
        if not curves:
            return []

        # Calculate the combined boundary of the layer for checking "outside" moves
        layer_boundary_buffered = None
        if layer_polygons:
            try:
                # Ensure all elements are valid polygons before union
                valid_polygons = [p for p in layer_polygons if isinstance(p, Polygon) and p.is_valid]
                if valid_polygons:
                    layer_boundary = unary_union(valid_polygons)
                    # Buffer slightly to handle points exactly on the boundary during checks
                    layer_boundary_buffered = layer_boundary.buffer(1e-6)
                else:
                    print("  Warning: No valid polygons found in layer for boundary check.")
            except Exception as e:
                print(f"  Warning: Could not create unary union for layer boundary checks: {e}")


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
                        start_i, end_i = endpoints[i]
                        start_j, end_j = endpoints[j]

                        # Check for intersections with holes and going outside bounds
                        connections = [
                            self._calculate_connection_cost(end_i, start_j, layer_polygons, layer_boundary_buffered),   # 0: i forward -> j forward
                            self._calculate_connection_cost(end_i, end_j, layer_polygons, layer_boundary_buffered),     # 1: i forward -> j backward
                            self._calculate_connection_cost(start_i, start_j, layer_polygons, layer_boundary_buffered),  # 2: i backward -> j forward
                            self._calculate_connection_cost(start_i, end_j, layer_polygons, layer_boundary_buffered)     # 3: i backward -> j backward
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

            if np.max(distance_matrix[np.isfinite(distance_matrix)]) > 0:
                # Scale to a range that Concorde can handle (typically max of 32767)
                scale_factor = min(tsp_scale_factor, 30000 / max(1, np.max(distance_matrix[np.isfinite(distance_matrix)])))
            else:
                scale_factor = tsp_scale_factor

            print(f"  Scaling distances by factor {scale_factor:.2f}")
            # Write the scaled distance matrix to the TSP file, handling potential infinities
            max_finite_dist = 0
            finite_dists = distance_matrix[np.isfinite(distance_matrix)]
            if finite_dists.size > 0:
                 max_finite_dist = np.max(finite_dists)

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
        # Pass layer_polygons and boundary to greedy if used
        if optimization_method.lower() == 'greedy':
            print("  Using greedy TSP optimization method")
            tsp_tour = self._greedy_tsp(distance_matrix, layer_polygons, layer_boundary_buffered, prev_end_point, endpoints)
        else:  # Default to Concorde
            try:
                # Try to run Concorde
                print("  Using Concorde TSP optimization method")
                tsp_tour = self._run_concorde(tsp_filename)

                if not tsp_tour:
                    # Fall back to greedy approach
                    print("  Concorde failed, falling back to greedy approach")
                    tsp_tour = self._greedy_tsp(distance_matrix, layer_polygons, layer_boundary_buffered, prev_end_point, endpoints)
            except Exception as e:
                print(f"  Error running TSP solver: {e}")
                # Fall back to greedy approach
                tsp_tour = self._greedy_tsp(distance_matrix, layer_polygons, layer_boundary_buffered, prev_end_point, endpoints)

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
            # Find the best starting curve and direction, considering invalid connections
            best_start_idx = 0
            best_start_cost = float('inf')
            best_start_reverse = False # Initialize here
            best_start_connection_points = (None, None) # Store (prev_end_point, actual_start_point)

            for i in range(len(tsp_tour)):
                curve_idx = tsp_tour[i]
                start, end = endpoints[curve_idx]

                # Calculate cost to start from this curve, penalizing invalid connections
                cost_forward, p1f, p2f = self._calculate_connection_cost(prev_end_point, start, layer_polygons, layer_boundary_buffered)
                cost_backward, p1b, p2b = self._calculate_connection_cost(prev_end_point, end, layer_polygons, layer_boundary_buffered)

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
        if not tsp_tour: # Handle case where TSP solver returned empty tour
             print("  Warning: TSP solver returned empty tour.")
             return []

        first_curve_idx = tsp_tour[0]
        first_curve = curves[first_curve_idx]
        if best_start_reverse:
             # If starting reversed, the first point added should be the end of the first curve
             if not optimized_path or self._euclidean_distance(optimized_path[-1], first_curve[-1]) > 1e-6:
                  optimized_path.append(first_curve[-1])
             prev_point = first_curve[-1] # The point we are starting *from* on the first curve
        else:
             # If starting forward, the first point added should be the start of the first curve
             if not optimized_path or self._euclidean_distance(optimized_path[-1], first_curve[0]) > 1e-6:
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

                 # Consider 4 connection possibilities, checking validity
                 connections = [
                      self._calculate_connection_cost(end, next_start, layer_polygons, layer_boundary_buffered),   # 0: current forward -> next forward
                      self._calculate_connection_cost(end, next_end, layer_polygons, layer_boundary_buffered),     # 1: current forward -> next backward
                      self._calculate_connection_cost(start, next_start, layer_polygons, layer_boundary_buffered),  # 2: current backward -> next forward
                      self._calculate_connection_cost(start, next_end, layer_polygons, layer_boundary_buffered)     # 3: current backward -> next backward
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
                 cost_forward, _, _ = self._calculate_connection_cost(prev_point, start, layer_polygons, layer_boundary_buffered)
                 cost_backward, _, _ = self._calculate_connection_cost(prev_point, end, layer_polygons, layer_boundary_buffered)
                 # If both costs are infinite, default to forward (or handle error)
                 if np.isinf(cost_forward) and np.isinf(cost_backward):
                      print(f"  Warning: Both forward and backward connections for last curve {curve_idx} are invalid. Defaulting to forward.")
                      reverse = False
                 else:
                      # Choose the direction with the finite (or lower) cost
                      reverse = cost_backward < cost_forward


            # Add the points of the current curve
            current_curve_points = list(reversed(curve)) if reverse else list(curve)

            # Add the connection point (start of the curve) if it's not already the last point
            if not optimized_path or self._euclidean_distance(optimized_path[-1], current_curve_points[0]) > 1e-6:
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
            list: Optimal tour as a list of indices, or None on failure.
        """
        try:
            # Read the TSP file to determine the problem size
            dimension = 0
            with open(tsp_filename, 'r') as f:
                for line in f:
                    if line.startswith("DIMENSION"):
                        dimension = int(line.split(":")[1].strip())
                        break

            if dimension == 0:
                 print("  Error: Could not read DIMENSION from TSP file.")
                 return None

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
                                   stderr=subprocess.PIPE,
                                   timeout=60) # Add a timeout

            # Log stdout and stderr for debugging
            stdout_output = result.stdout.decode('utf-8', errors='ignore')
            stderr_output = result.stderr.decode('utf-8', errors='ignore')

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
                print("  Tour file (.sol) not found after Concorde run.")
                # Check if output contains the tour directly (sometimes happens)
                tour_match = re.search(r'Optimal Solution: ([\d\s]+)', stdout_output)
                if tour_match:
                     tour_str = tour_match.group(1).strip()
                     tour = [int(x) for x in tour_str.split()]
                     print("  Parsed tour directly from Concorde stdout.")
                     return tour
                else:
                     print("  Could not find tour in stdout either. Falling back to greedy.")
                     return None

            with open(tour_file, 'r') as f:
                lines = f.readlines()

                if len(lines) < 2:
                    print("  Invalid tour file format, using greedy approach")
                    return None

                # Parse the tour (assuming format: first line is count, second is tour)
                try:
                    tour_line = lines[1].strip()
                    tour = [int(x) for x in tour_line.split()]
                except (ValueError, IndexError):
                    print("  Error parsing tour file, using greedy approach")
                    return None

                # Clean up the tour file
                try:
                    os.remove(tour_file)
                except:
                    pass

                return tour
        except subprocess.TimeoutExpired:
            print("  Concorde timed out. Falling back to greedy approach.")
            return None
        except FileNotFoundError:
            print("  Error: Docker command not found. Is Docker installed and in PATH?")
            print("  Falling back to greedy approach.")
            return None
        except Exception as e:
            print(f"  Error running Concorde: {e}")
            return None

    def _greedy_tsp(self, distance_matrix, layer_polygons, layer_boundary_buffered, prev_end_point=None, endpoints=None):
        """
        Solve the TSP problem using a greedy approach, avoiding invalid connections.

        Args:
            distance_matrix (numpy.ndarray): Pre-calculated distance matrix (with penalties).
            layer_polygons (list): List of Shapely Polygons for the layer.
            layer_boundary_buffered (Polygon | MultiPolygon | None): Buffered layer boundary.
            prev_end_point (tuple, optional): End point of the previous layer's path.
            endpoints (list, optional): List of (start, end) points for each curve.

        Returns:
            list: Tour as a list of indices.
        """
        num_nodes = distance_matrix.shape[0]
        if num_nodes == 0:
            return []

        start_node = 0 # Default start node

        # If we have a previous end point, find the best starting node
        if prev_end_point and endpoints:
            min_dist = float('inf')
            found_valid_start = False

            for i in range(num_nodes):
                start, end = endpoints[i]

                # Calculate connection cost, considering invalid segments
                cost_to_start, _, _ = self._calculate_connection_cost(prev_end_point, start, layer_polygons, layer_boundary_buffered)
                cost_to_end, _, _ = self._calculate_connection_cost(prev_end_point, end, layer_polygons, layer_boundary_buffered)

                if cost_to_start < min_dist:
                    min_dist = cost_to_start
                    start_node = i
                    found_valid_start = True

                if cost_to_end < min_dist:
                    min_dist = cost_to_end
                    start_node = i
                    found_valid_start = True

            if not found_valid_start:
                 print("  Warning (Greedy TSP): All starting connections are invalid. Choosing node 0.")
                 start_node = 0 # Fallback if all connections are bad

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
                 # Use the pre-calculated distance which already includes validity checks
                 dist = distance_matrix[current, node]
                 if dist < min_dist:
                      min_dist = dist
                      best_next_node = node

            if best_next_node == -1 or np.isinf(min_dist):
                 print(f"  Warning (Greedy TSP): Cannot find valid next node from {current}. Stopping tour early.")
                 # This might happen if 'current' is isolated due to invalid connections
                 break # Stop the tour here

            tour.append(best_next_node)
            unvisited.remove(best_next_node)

        return tour

    def _is_invalid_segment(self, p1, p2, layer_polygons, layer_boundary_buffered):
        """
        Check if the line segment between p1 and p2 is invalid because it:
        1. Intersects any hole in the layer_polygons.
        2. Is not contained within the overall layer_boundary_buffered area.

        Args:
            p1 (tuple): Start point (x, y).
            p2 (tuple): End point (x, y).
            layer_polygons (list): List of original Shapely Polygons for the layer (for hole check).
            layer_boundary_buffered (Polygon | MultiPolygon | None): The buffered unary_union of layer polygons.

        Returns:
            bool: True if the segment is invalid, False otherwise.
        """
        if p1 is None or p2 is None:
            return True # Invalid if points are missing

        # Avoid checking zero-length segments
        if self._euclidean_distance(p1, p2) < 1e-6:
             return False # Zero-length segment is not invalid by itself

        try:
            segment = LineString([p1, p2])

            # Check 1: Intersection with holes
            if layer_polygons:
                for poly in layer_polygons:
                    if isinstance(poly, Polygon) and poly.is_valid: # Ensure it's a valid Polygon
                        for interior in poly.interiors:
                            # Check if the segment intersects the boundary of the hole
                            if segment.intersects(interior):
                                # print(f"    Intersection detected: Segment {p1} -> {p2} crosses hole.")
                                return True # Invalid due to hole intersection

            # Check 2: Segment goes outside the layer boundary
            # Use the buffered boundary for contains check
            if layer_boundary_buffered and not layer_boundary_buffered.contains(segment):
                 # Optional: Check if endpoints are outside too, might indicate issue earlier
                 # if not layer_boundary_buffered.contains(Point(p1)) or not layer_boundary_buffered.contains(Point(p2)):
                 #      print(f"    Segment endpoints outside boundary: {p1}, {p2}")

                 # print(f"    Outside boundary detected: Segment {p1} -> {p2} goes outside layer area.")
                 return True # Invalid because it goes outside

        except Exception as e:
            print(f"  Warning: Error during segment validation ({p1} -> {p2}): {e}")
            return True # Treat errors during check as invalid

        return False # Segment is valid

    def _calculate_connection_cost(self, p1, p2, layer_polygons, layer_boundary_buffered):
        """
        Calculate the cost of connecting p1 to p2, returning infinity
        if the connection is invalid (intersects hole or goes outside boundary).

        Args:
            p1 (tuple): Start point (x, y).
            p2 (tuple): End point (x, y).
            layer_polygons (list): List of original Shapely Polygons for the layer.
            layer_boundary_buffered (Polygon | MultiPolygon | None): The buffered unary_union of layer polygons.

        Returns:
            tuple: (cost, p1, p2) where cost is Euclidean distance or float('inf'),
                   and p1, p2 are the points defining the connection.
        """
        cost = self._euclidean_distance(p1, p2)
        # Check if the segment is invalid (crosses hole OR goes outside)
        if self._is_invalid_segment(p1, p2, layer_polygons, layer_boundary_buffered):
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
             # Returning infinity might be better if None indicates an impossible connection
             return float('inf')
        if p1 == p2:
             return 0.0
        try:
            # Ensure points are tuples/lists of numbers
            if not (isinstance(p1, (tuple, list)) and len(p1) == 2 and
                    isinstance(p2, (tuple, list)) and len(p2) == 2 and
                    all(isinstance(coord, (int, float)) for coord in p1 + p2)):
                 print(f"  Warning: Invalid point format for distance calculation: {p1}, {p2}")
                 return float('inf')
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        except (TypeError, IndexError) as e:
             print(f"  Warning: Error in distance calculation ({p1}, {p2}): {e}")
             return float('inf') # Penalize invalid points heavily

    def smooth_path(self, path, segment_length=None, layer_polygons=None):
        """
        Smooth a path by resampling it with uniform point spacing, avoiding invalid segments.

        Args:
            path (list): List of (x, y) points defining the path.
            segment_length (float, optional): Desired distance between points.
                                             If None, uses value from config.
            layer_polygons (list, optional): List of Shapely Polygons for the current layer
                                             to check against for invalid segments (holes/outside).

        Returns:
            list: Smoothed path with uniform point spacing.
        """
        if not path or len(path) < 2:
            return path

        # Pre-calculate layer boundary for smoothing checks
        layer_boundary_buffered = None
        if layer_polygons:
             try:
                  # Ensure all elements are valid polygons before union
                  valid_polygons = [p for p in layer_polygons if isinstance(p, Polygon) and p.is_valid]
                  if valid_polygons:
                       layer_boundary = unary_union(valid_polygons)
                       layer_boundary_buffered = layer_boundary.buffer(1e-6)
                  else:
                       print("  Warning (Smoothing): No valid polygons found in layer for boundary check.")
             except Exception as e:
                  print(f"  Warning (Smoothing): Could not create unary union for boundary checks: {e}")


        # If segment_length is not provided, use value from config
        if segment_length is None:
            config = get_config()
            segment_length = config.get('path_smoothing_segment_length', self.toolpath_width / 2)
            # Ensure segment length is positive
            segment_length = max(1e-3, segment_length)


        # Calculate the total path length
        total_length = 0
        valid_path_segments = []
        for i in range(len(path) - 1):
             dist = self._euclidean_distance(path[i], path[i+1])
             # Only consider finite distances for total length
             if np.isfinite(dist):
                  total_length += dist
                  valid_path_segments.append((path[i], path[i+1], dist))
             else:
                  print(f"  Warning (Smoothing): Skipping infinite distance segment {path[i]} -> {path[i+1]}")

        if total_length < 1e-6:
             print("  Warning (Smoothing): Path has zero valid length.")
             return path # Return original path if no length

        # Calculate the number of segments needed based on valid length
        num_segments_target = max(2, int(total_length / segment_length))

        # Create a new path with uniform spacing
        smoothed_path = []

        # Always include the first point of the original path
        smoothed_path.append(path[0])

        # Current position along the valid path segments
        current_length_along_path = 0.0
        target_length = segment_length # Start aiming for the first segment length

        for p1, p2, segment_len_i in valid_path_segments:
            # Process this segment until the target length exceeds the current position + segment length
            while current_length_along_path + segment_len_i >= target_length - 1e-9: # Use tolerance
                # Calculate how far along this segment the next point should be
                # Avoid division by zero for zero-length segments (already filtered, but safety)
                if segment_len_i < 1e-9:
                     break
                t = (target_length - current_length_along_path) / segment_len_i

                # Clamp t to [0, 1] due to potential floating point inaccuracies
                t = min(1.0, max(0.0, t))

                # Interpolate to find the point
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                next_point = (x, y)

                # --- Validity Check ---
                is_valid = True
                if len(smoothed_path) > 0:
                    last_point = smoothed_path[-1]
                    # Check if the segment from last_point to next_point is invalid
                    if self._is_invalid_segment(last_point, next_point, layer_polygons, layer_boundary_buffered):
                         # print(f"  Warning: Smoothing segment invalid. "
                         #       f"Segment: {last_point} -> {next_point}. Skipping point.")
                         is_valid = False

                # Add the point only if the segment is valid
                if is_valid:
                    # Avoid adding duplicate points
                    if not smoothed_path or self._euclidean_distance(smoothed_path[-1], next_point) > 1e-6:
                         smoothed_path.append(next_point)

                # Move to the next target length
                target_length += segment_length

                # If we've interpolated to the end of this segment, break the inner loop
                if t >= 1.0 - 1e-9:
                    break

            # Update the current length along the path
            current_length_along_path += segment_len_i

        # Always include the last point of the original path, checking the final segment's validity
        last_point_original = path[-1]
        # Check if last point is valid and different from the last added point
        if smoothed_path and self._euclidean_distance(smoothed_path[-1], last_point_original) > 1e-6:
             is_valid = True
             last_added_point = smoothed_path[-1]
             if self._is_invalid_segment(last_added_point, last_point_original, layer_polygons, layer_boundary_buffered):
                  is_valid = False
                  # print(f"  Warning: Final segment to endpoint is invalid. Endpoint not added.")

             if is_valid:
                 smoothed_path.append(last_point_original)

        # Final check for duplicates
        final_smoothed_path = []
        if smoothed_path:
             final_smoothed_path.append(smoothed_path[0])
             for k in range(1, len(smoothed_path)):
                  if self._euclidean_distance(final_smoothed_path[-1], smoothed_path[k]) > 1e-6:
                       final_smoothed_path.append(smoothed_path[k])

        return final_smoothed_path


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
            # print(f"Optimized path visualization disabled in config") # Reduce noise
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the polygons
        if layer:
            for polygon in layer:
                 if isinstance(polygon, Polygon) and polygon.is_valid:
                      x, y = polygon.exterior.xy
                      ax.plot(x, y, 'b-', linewidth=1, alpha=0.7, label='Polygon Boundary' if 'Polygon Boundary' not in ax.get_legend_handles_labels()[1] else "")

                      # Plot holes if any
                      for interior in polygon.interiors:
                           x_int, y_int = interior.xy
                           ax.plot(x_int, y_int, 'r-', linewidth=1, alpha=0.7, label='Hole Boundary' if 'Hole Boundary' not in ax.get_legend_handles_labels()[1] else "")
                 else:
                      print(f"  Warning (Viz): Skipping invalid/non-polygon geometry in layer {layer_idx+1}")

        # Plot the optimized path
        if optimized_path and len(optimized_path) > 1:
            path_x, path_y = zip(*optimized_path)

            # Use a colormap to show the direction of the path
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a colorful line collection
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis', linewidth=1.5)
            lc.set_array(np.linspace(0, 1, len(segments))) # Use len(segments)
            line_coll = ax.add_collection(lc) # Store the collection

            # Mark start and end points
            ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')

            # Add a colorbar to show progression, only if segments were plotted
            if len(segments) > 0:
                 cbar = plt.colorbar(line_coll, ax=ax, shrink=0.8) # Use the returned collection
                 cbar.set_label('Path Direction')
        elif optimized_path and len(optimized_path) == 1:
             # Plot single point if path has only one point
             ax.plot(optimized_path[0][0], optimized_path[0][1], 'go', markersize=8, label='Start/End (Single Point)')
        else:
             print(f"  Layer {layer_idx+1}: No optimized path to visualize.")


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
            # is_simple checks for self-intersection (excluding start/end touching in a loop)
            if not line.is_simple:
                # Further check if it's a closed loop touching only at ends
                is_closed_loop = Point(path[0]).equals_exact(Point(path[-1]), 1e-6)
                if not is_closed_loop:
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
            # print("Layer transition visualization disabled in config") # Reduce noise
            return

        valid_paths_indices = [i for i, p in enumerate(optimized_paths) if p and len(p) > 0]

        if len(valid_paths_indices) < 1:
             print("No valid paths found to visualize transitions.")
             return

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Set colors for different layers
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        min_z_overall = float('inf')
        max_z_overall = float('-inf')

        # Plot each layer and its path
        for i in valid_paths_indices:
            layer = layers[i] # Get corresponding layer polygons
            path = optimized_paths[i]
            color = colors[i]

            # Determine Z height (use index for now, could use actual Z later)
            z_height = i
            min_z_overall = min(min_z_overall, z_height)
            max_z_overall = max(max_z_overall, z_height)

            # Extract path coordinates
            path_x, path_y = zip(*path)

            # Plot the layer path
            ax.plot(path_x, path_y, [z_height] * len(path_x),
                   color=color, linewidth=2, label=f'Layer {i+1}' if i == valid_paths_indices[0] else "") # Label only first plotted layer

            # Mark start and end points
            ax.scatter(path_x[0], path_y[0], z_height,
                      color='green', s=50, marker='o', label='Start' if i == valid_paths_indices[0] else "")
            ax.scatter(path_x[-1], path_y[-1], z_height,
                      color='red', s=50, marker='x', label='End' if i == valid_paths_indices[0] else "") # Use 'x' for end

            # If not the first *plotted* layer, draw a line connecting to the previous *plotted* layer
            current_plot_index = valid_paths_indices.index(i)
            if current_plot_index > 0:
                prev_plotted_layer_idx = valid_paths_indices[current_plot_index - 1]
                prev_path = optimized_paths[prev_plotted_layer_idx]
                prev_end = prev_path[-1]
                curr_start = path[0]
                prev_z_height = prev_plotted_layer_idx

                # Draw a line connecting the layers
                ax.plot([prev_end[0], curr_start[0]],
                       [prev_end[1], curr_start[1]],
                       [prev_z_height, z_height], 'k--', linewidth=1.5)

                # Calculate and display the travel distance
                travel_dist = self._euclidean_distance(prev_end, curr_start)
                if np.isfinite(travel_dist): # Only display finite distances
                     mid_x = (prev_end[0] + curr_start[0]) / 2
                     mid_y = (prev_end[1] + curr_start[1]) / 2
                     mid_z = (prev_z_height + z_height) / 2
                     ax.text(mid_x, mid_y, mid_z, f"{travel_dist:.2f}mm",
                            color='black', fontsize=8, ha='center', zorder=10) # Ensure text is visible

        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Layer Index') # Changed label
        ax.set_title('Layer Transitions Visualization')

        # Set Z limits for better visualization
        if np.isfinite(min_z_overall) and np.isfinite(max_z_overall):
             ax.set_zlim(min_z_overall - 1, max_z_overall + 1)

        # Add a legend for the first few items only (to avoid clutter)
        handles, labels = ax.get_legend_handles_labels()
        if handles: # Only show legend if there are labeled items
             by_label = dict(zip(labels, handles))
             ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        # Improve layout
        ax.view_init(elev=20., azim=-65) # Adjust viewing angle
        plt.tight_layout()
        plt.show(block=False)
class OptimizerB:
    """
    A stub implementation of an alternative toolpath optimizer.
    """
    
    def __init__(self, toolpath_width=None):
        """
        Initialize OptimizerB.

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
        Stub implementation for layer optimization.

        Args:
            layers (list): List of layer contours
            layer_paths (list): List of paths for each layer

        Returns:
            list: Empty list since this is a stub implementation
        """
        print("OptimizerB: No optimization implemented yet")
        return []

    # Add other required methods as stubs to match OptimizerA's interface
    def _split_path_into_segments(self, path, max_points=None):
        return [path]

    def _optimize_layer(self, curves, layer_polygons, prev_end_point=None):
        return []

    def _run_concorde(self, tsp_filename):
        return None

    def _greedy_tsp(self, distance_matrix, layer_polygons, layer_boundary_buffered, prev_end_point=None, endpoints=None):
        return []

    def _is_invalid_segment(self, p1, p2, layer_polygons, layer_boundary_buffered):
        return False

    def _calculate_connection_cost(self, p1, p2, layer_polygons, layer_boundary_buffered):
        return float('inf'), p1, p2

    def _euclidean_distance(self, p1, p2):
        return 0.0

    def smooth_path(self, path, segment_length=None, layer_polygons=None):
        return path

    def visualize_optimized_path(self, layer, optimized_path, layer_idx):
        pass

    def _check_self_intersection(self, path, layer_index):
        pass

    def visualize_layer_transitions(self, layers, optimized_paths):
        pass
