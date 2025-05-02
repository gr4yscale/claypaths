import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from src.config import get_config
import time # For timing operations

class OptimizerC:
    """
    Optimizes toolpaths using an image processing approach (SED with Gaussian filter).
    Generates a single continuous path for each layer.
    """

    def __init__(self, toolpath_width=None):
        """
        Initialize OptimizerC.

        Args:
            toolpath_width (float, optional): Width of the toolpath in mm.
                                             If None, uses value from config.
        """
        config = get_config()
        if toolpath_width is None:
            toolpath_width = config['toolpath_width']

        self.toolpath_width = toolpath_width
        self.resolution = config.get('optimizer_c_resolution', self.toolpath_width / 2.0) # Pixel size in mm
        self.gaussian_sigma = config.get('optimizer_c_gaussian_sigma', 1.5) # Sigma for Gaussian filter
        self.total_cost = 0.0 # Note: Cost calculation might differ from OptimizerA

    def optimize_layers(self, layers, layer_paths):
        """
        Optimize toolpaths for all layers using SED with Gaussian filtering.

        Args:
            layers (list): List of layer contours (list of shapely Polygons per layer).
            layer_paths (list): List of *unoptimized* paths for each layer (used for start/end points).

        Returns:
            list: List of optimized paths (single continuous path per layer).
        """
        if not layers:
            return []

        print("\nOptimizing toolpaths using OptimizerC (Image Processing for Connections)...")
        optimized_paths_final = []
        prev_layer_end_point = None # Keep track of the previous layer's end point

        for layer_idx, (layer_polygons, unoptimized_paths_layer) in enumerate(zip(layers, layer_paths)):
            print(f"Optimizing layer {layer_idx+1}/{len(layers)}...")
            start_time = time.time()

            # Filter out empty or invalid paths
            valid_unoptimized_paths = [p for p in unoptimized_paths_layer if p and len(p) > 1]

            if not layer_polygons:
                print(f"  No polygons in layer {layer_idx+1}, skipping.")
                optimized_paths_final.append([])
                continue

            if not valid_unoptimized_paths:
                 print(f"  No valid unoptimized paths provided for layer {layer_idx+1}. Skipping.")
                 optimized_paths_final.append([])
                 continue

            # --- 1. Prepare Image and Graph for Travel Moves ---
            try:
                # Combine all polygons in the layer into one representative shape for travel space
                layer_union = unary_union([p for p in layer_polygons if isinstance(p, Polygon) and p.is_valid])
                if layer_union.is_empty:
                     print(f"  Layer {layer_idx+1} union is empty. Skipping.")
                     optimized_paths_final.append([])
                     continue
                # Buffer slightly less than pixel size for image generation
                buffer_dist = max(self.resolution * 0.6, 1e-3)
                layer_polygon_buffered = layer_union.buffer(buffer_dist)

                # Create binary image, filtered image, and graph ONCE per layer
                binary_image, transform = self._create_binary_image(layer_polygon_buffered, self.resolution)
                if binary_image is None or transform is None or binary_image.size == 0:
                    raise ValueError("Empty or invalid binary image created.")

                filtered_image = self._apply_gaussian_filter(binary_image, self.gaussian_sigma)
                if filtered_image is None: raise ValueError("Gaussian filter returned None.")
                max_val = filtered_image.max()
                if max_val > 1e-9: filtered_image /= max_val
                else: print(f"  Warning: Filtered image has near-zero max value.")

                graph, node_map = self._create_graph(filtered_image)
                if graph is None or node_map is None or graph.shape[0] == 0:
                    raise ValueError("Empty or invalid graph created.")

                # --- Optional Visualizations ---
                if get_config().get('visualize_optimizer_c_binary_image', False):
                    self._visualize_image(binary_image, f"Layer {layer_idx+1} - Binary Image", transform)
                if get_config().get('visualize_optimizer_c_filtered_image', False):
                    self._visualize_image(filtered_image, f"Layer {layer_idx+1} - Filtered Image", transform)
                # Graph visualization might be too slow/cluttered here, consider disabling by default

            except Exception as e:
                print(f"  Error preparing image/graph for layer {layer_idx+1}: {e}")
                optimized_paths_final.append([])
                continue

            # --- 2. Iteratively Connect Paths using Graph ---
            num_segments = len(valid_unoptimized_paths)
            used_indices = set()
            current_optimized_path = []
            current_point = None

            # --- Find Starting Segment ---
            start_segment_idx = 0
            start_reverse = False
            if prev_layer_end_point:
                min_start_cost = float('inf')
                found_valid_start = False
                # Find segment closest to previous layer's end point using graph path cost
                prev_layer_end_node = self._find_closest_node(
                    self._world_to_image_coords(prev_layer_end_point, transform, binary_image.shape),
                    node_map, binary_image.shape
                )

                if prev_layer_end_node is not None:
                    for i, path in enumerate(valid_unoptimized_paths):
                        path_start_node = self._find_closest_node(
                            self._world_to_image_coords(path[0], transform, binary_image.shape),
                            node_map, binary_image.shape
                        )
                        path_end_node = self._find_closest_node(
                            self._world_to_image_coords(path[-1], transform, binary_image.shape),
                            node_map, binary_image.shape
                        )

                        if path_start_node is not None:
                            _, cost_to_start = self._find_shortest_path(graph, prev_layer_end_node, path_start_node)
                            if cost_to_start < min_start_cost:
                                min_start_cost = cost_to_start
                                start_segment_idx = i
                                start_reverse = False
                                found_valid_start = True

                        if path_end_node is not None:
                            _, cost_to_end = self._find_shortest_path(graph, prev_layer_end_node, path_end_node)
                            if cost_to_end < min_start_cost:
                                min_start_cost = cost_to_end
                                start_segment_idx = i
                                start_reverse = True
                                found_valid_start = True

                if found_valid_start and np.isfinite(min_start_cost):
                    print(f"  Starting with segment {start_segment_idx} (reverse={start_reverse}) based on previous layer end point.")
                    # Add the travel path from previous layer end
                    target_node = path_end_node if start_reverse else path_start_node
                    travel_indices, _ = self._find_shortest_path(graph, prev_layer_end_node, target_node)
                    if travel_indices:
                        travel_pixel_path = [node_map[idx] for idx in travel_indices]
                        travel_world_path = self._image_to_world_coords(travel_pixel_path, transform)
                        # Add travel path (skip first point as it's prev_layer_end_point)
                        if len(travel_world_path) > 1:
                             current_optimized_path.extend(travel_world_path[1:])
                else:
                    print(f"  Warning: Could not find valid start connection from previous layer. Starting with segment 0.")
                    start_segment_idx = 0
                    start_reverse = False # Default direction
            else:
                # No previous layer, start with segment 0
                start_segment_idx = 0
                start_reverse = False

            # Add the first segment
            first_segment = valid_unoptimized_paths[start_segment_idx]
            segment_to_add = list(reversed(first_segment)) if start_reverse else list(first_segment)

            # Ensure the first point is added correctly
            if not current_optimized_path or self._euclidean_distance(current_optimized_path[-1], segment_to_add[0]) > 1e-6:
                 current_optimized_path.append(segment_to_add[0])
            # Add the rest of the first segment
            current_optimized_path.extend(segment_to_add[1:])
            used_indices.add(start_segment_idx)
            current_point = current_optimized_path[-1]

            # --- Connection Loop ---
            while len(used_indices) < num_segments:
                if current_point is None: # Should not happen if starting segment was valid
                     print("  Error: Current point is None during connection loop. Aborting layer.")
                     break

                current_node = self._find_closest_node(
                    self._world_to_image_coords(current_point, transform, binary_image.shape),
                    node_map, binary_image.shape
                )
                if current_node is None:
                     print(f"  Error: Could not map current point {current_point} to graph node. Aborting layer.")
                     break

                best_next_segment_idx = -1
                best_reverse_next = False
                min_connection_cost = float('inf')
                best_target_node = None

                # Find the closest unused segment endpoint based on graph path cost
                for i in range(num_segments):
                    if i not in used_indices:
                        path = valid_unoptimized_paths[i]
                        path_start_node = self._find_closest_node(
                            self._world_to_image_coords(path[0], transform, binary_image.shape),
                            node_map, binary_image.shape
                        )
                        path_end_node = self._find_closest_node(
                            self._world_to_image_coords(path[-1], transform, binary_image.shape),
                            node_map, binary_image.shape
                        )

                        # Check connection to start point
                        if path_start_node is not None:
                            _, cost_to_start = self._find_shortest_path(graph, current_node, path_start_node)
                            if cost_to_start < min_connection_cost:
                                min_connection_cost = cost_to_start
                                best_next_segment_idx = i
                                best_reverse_next = False
                                best_target_node = path_start_node

                        # Check connection to end point
                        if path_end_node is not None:
                            _, cost_to_end = self._find_shortest_path(graph, current_node, path_end_node)
                            if cost_to_end < min_connection_cost:
                                min_connection_cost = cost_to_end
                                best_next_segment_idx = i
                                best_reverse_next = True
                                best_target_node = path_end_node

                # If a valid connection was found
                if best_next_segment_idx != -1 and np.isfinite(min_connection_cost):
                    # Get the travel path on the graph
                    travel_indices, _ = self._find_shortest_path(graph, current_node, best_target_node)
                    if travel_indices:
                        travel_pixel_path = [node_map[idx] for idx in travel_indices]
                        travel_world_path = self._image_to_world_coords(travel_pixel_path, transform)
                        # Add travel path (skip first point as it's current_point)
                        if len(travel_world_path) > 1:
                             current_optimized_path.extend(travel_world_path[1:])
                        elif len(travel_world_path) == 1 and self._euclidean_distance(current_point, travel_world_path[0]) > 1e-6:
                             # Add if the single point is different
                             current_optimized_path.append(travel_world_path[0])


                    # Add the chosen segment
                    chosen_segment = valid_unoptimized_paths[best_next_segment_idx]
                    segment_to_add = list(reversed(chosen_segment)) if best_reverse_next else list(chosen_segment)

                    # Add segment points (skip first point as it should match end of travel path)
                    if len(segment_to_add) > 1:
                         current_optimized_path.extend(segment_to_add[1:])
                    elif len(segment_to_add) == 1 and self._euclidean_distance(current_optimized_path[-1], segment_to_add[0]) > 1e-6:
                         # Add if the single point is different
                         current_optimized_path.append(segment_to_add[0])


                    # Update state
                    used_indices.add(best_next_segment_idx)
                    current_point = current_optimized_path[-1]
                else:
                    print(f"  Warning: Could not find valid connection from point {current_point} to any remaining segments. Stopping layer optimization.")
                    break # No more valid connections found

            # --- Finalize Layer ---
            # Optional Smoothing could be applied here to current_optimized_path
            # smoothed_path = self.smooth_path(current_optimized_path, layer_polygons=layer_polygons)
            smoothed_path = current_optimized_path # Placeholder

            optimized_paths_final.append(smoothed_path)
            if smoothed_path:
                prev_layer_end_point = smoothed_path[-1] # Update for next layer
                print(f"  Layer {layer_idx+1} optimized in {time.time() - start_time:.2f}s. Path length: {len(smoothed_path)} points.")
            else:
                print(f"  Layer {layer_idx+1} resulted in an empty path after processing.")
                prev_layer_end_point = None # Reset if path is empty


        print(f"\nOptimizerC processing complete.")
        # Note: self.total_cost is not accurately tracked here yet.
        return optimized_paths_final

    def _create_binary_image(self, polygon, resolution):
        """
        Create a binary image representation of the polygon.

        Args:
            polygon (shapely.Polygon): The polygon to rasterize (should be buffered).
            resolution (float): Pixel size in mm.

        Returns:
            tuple: (binary_image, transform) where transform maps image coords to world coords.
                   Returns (None, None) on error.
        """
        if not polygon or polygon.is_empty or not polygon.is_valid:
             print("  Error (_create_binary_image): Invalid or empty polygon provided.")
             return None, None

        try:
            min_x, min_y, max_x, max_y = polygon.bounds

            # Add a small margin to ensure bounds cover pixel centers
            margin = resolution * 0.5
            min_x -= margin
            min_y -= margin
            max_x += margin
            max_y += margin

            # Calculate image dimensions (ensure at least 1x1)
            width = max(1, int(np.ceil((max_x - min_x) / resolution)))
            height = max(1, int(np.ceil((max_y - min_y) / resolution)))

            # Create coordinate grids for pixel centers
            # Y-axis is inverted in image space (origin top-left)
            x_coords = np.linspace(min_x + resolution / 2.0, max_x - resolution / 2.0, width)
            y_coords = np.linspace(max_y - resolution / 2.0, min_y + resolution / 2.0, height) # Flipped Y
            xx, yy = np.meshgrid(x_coords, y_coords)

            # Create points for each pixel center
            points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

            # Check which points are inside the polygon
            # This can be slow for large images/complex polygons
            # Consider using rasterio or other libraries for faster rasterization if needed
            image = np.zeros((height, width), dtype=np.uint8)
            for idx, p in enumerate(points):
                if polygon.contains(p):
                    row, col = np.unravel_index(idx, (height, width))
                    image[row, col] = 1 # Mark pixel as inside

            # Define the transformation from image (row, col) to world (x, y)
            # Based on Affine library convention: https://affine.readthedocs.io/en/latest/
            # transform = Affine(resolution, 0, min_x,
            #                    0, -resolution, max_y) # a, b, c, d, e, f
            # Simpler dict representation:
            transform = {
                'offset': (min_x, max_y), # World coords of top-left pixel's top-left corner
                'scale': (resolution, -resolution) # Pixel size (x, y), y is negative
            }

            return image, transform

        except Exception as e:
            print(f"  Error (_create_binary_image): {e}")
            return None, None


    def _apply_gaussian_filter(self, image, sigma):
        """Apply Gaussian filter to the binary image."""
        if image is None: return None
        try:
            return gaussian_filter(image.astype(float), sigma=sigma)
        except Exception as e:
            print(f"  Error (_apply_gaussian_filter): {e}")
            return None

    def _create_graph(self, filtered_image):
        """
        Create a sparse graph from the filtered image.
        Nodes are pixels with intensity > 0.
        Edges connect adjacent pixels (4-connectivity).
        Edge weights are inversely proportional to the average intensity of connected pixels.

        Returns:
            tuple: (csr_matrix, node_map)
                   node_map: dict mapping graph node index to (row, col) pixel coordinate.
                   Returns (None, None) on error.
        """
        if filtered_image is None or filtered_image.size == 0:
            return None, None

        try:
            rows, cols = filtered_image.shape
            # Consider only pixels with significant intensity (e.g., > threshold)
            threshold = 1e-3
            valid_pixels = np.argwhere(filtered_image > threshold) # Array of [row, col]

            if valid_pixels.shape[0] == 0:
                 print("  Warning (_create_graph): No valid pixels found after filtering/thresholding.")
                 return None, None

            # Map (row, col) to a unique node index
            node_map = {i: tuple(coords) for i, coords in enumerate(valid_pixels)}
            coord_to_node_idx = {tuple(coords): i for i, coords in enumerate(valid_pixels)}
            num_nodes = len(node_map)

            row_ind = []
            col_ind = []
            data = []

            # Define 4-connectivity neighbors (right, left, down, up)
            dr = [0, 0, 1, -1] # row offsets
            dc = [1, -1, 0, 0] # col offsets

            for node_idx, (r, c) in node_map.items():
                current_intensity = filtered_image[r, c]
                for i in range(4): # Check 4 neighbors
                    nr, nc = r + dr[i], c + dc[i]

                    # Check bounds
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_coord = (nr, nc)
                        # Check if neighbor is a valid node
                        if neighbor_coord in coord_to_node_idx:
                            neighbor_idx = coord_to_node_idx[neighbor_coord]
                            neighbor_intensity = filtered_image[nr, nc]

                            # Ensure we only add edge once (e.g., from lower index to higher)
                            if node_idx < neighbor_idx:
                                # Calculate weight: Combine distance and intensity.
                                # Higher intensity = lower cost multiplier.
                                avg_intensity = (current_intensity + neighbor_intensity) / 2.0
                                intensity_cost_multiplier = 1.0 / max(avg_intensity, 1e-6)

                                # Calculate Euclidean distance between pixel centers (scaled by resolution)
                                # dr[i], dc[i] are pixel offsets (0 or +/-1)
                                pixel_dist = np.sqrt(dr[i]**2 + dc[i]**2)
                                world_dist = pixel_dist * self.resolution # Approximate distance in mm

                                # Combine: base cost + distance * intensity_penalty
                                base_cost = 0.01 # Smaller base cost
                                weight = base_cost + world_dist * (1 + intensity_cost_multiplier) # Penalize low intensity more

                                # Add edge in both directions for undirected graph
                                row_ind.extend([node_idx, neighbor_idx])
                                col_ind.extend([neighbor_idx, node_idx])
                                data.extend([weight, weight])

            if not row_ind:
                 print("  Warning (_create_graph): No edges created in the graph.")
                 # This might happen if valid pixels are isolated.
                 # Return a graph with nodes but no edges? Or None? Let's return None.
                 return None, None


            graph = csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))
            return graph, node_map

        except Exception as e:
            print(f"  Error (_create_graph): {e}")
            return None, None

    def _world_to_image_coords(self, world_point, transform, image_shape):
        """Convert world (x, y) coordinates to image (row, col) coordinates, clamping to bounds."""
        if world_point is None or transform is None: return None
        try:
            height, width = image_shape
            wx, wy = world_point
            offset_x, offset_y = transform['offset']
            scale_x, scale_y = transform['scale'] # scale_y is negative

            if abs(scale_x) < 1e-9 or abs(scale_y) < 1e-9:
                 raise ValueError("Transform scale is zero or near-zero.")

            # Inverse transformation
            col_f = (wx - offset_x) / scale_x
            row_f = (wy - offset_y) / scale_y # scale_y is negative

            # Clamp and convert to integer indices
            col = max(0, min(width - 1, int(np.floor(col_f))))
            row = max(0, min(height - 1, int(np.floor(row_f))))

            return row, col
        except Exception as e:
            print(f"  Error (_world_to_image_coords): {e}")
            return None

    def _find_closest_node(self, pixel_coord, node_map, image_shape):
        """Find the graph node index corresponding to the valid pixel closest to the target pixel_coord."""
        if pixel_coord is None or not node_map: return None
        target_r, target_c = pixel_coord
        height, width = image_shape

        # Target coordinates should already be clamped by _world_to_image_coords
        target_coord = (target_r, target_c)

        # Check if the exact target pixel is a node
        coord_to_node_idx = {v: k for k, v in node_map.items()}
        if target_coord in coord_to_node_idx:
            return coord_to_node_idx[target_coord]

        # If not, find the closest node by Euclidean distance in pixel space
        min_dist_sq = float('inf')
        closest_node_idx = None

        # Create a NumPy array of node coordinates for faster distance calculation
        node_coords_array = np.array(list(node_map.values())) # Shape: (num_nodes, 2)

        # Calculate squared Euclidean distances
        distances_sq = np.sum((node_coords_array - np.array(target_coord))**2, axis=1)

        # Find the index of the minimum distance
        min_idx_in_array = np.argmin(distances_sq)

        # Map this index back to the original graph node index
        # The order in node_coords_array corresponds to the keys 0..N-1 of node_map
        closest_node_idx = min_idx_in_array # Assuming node_map keys are 0 to N-1

        if closest_node_idx is None:
             print(f"  Warning (_find_closest_node): Could not find any node near {target_coord}")

        return closest_node_idx


    def _find_shortest_path(self, graph, start_node_idx, end_node_idx):
        """
        Find the shortest path in the graph using Dijkstra's algorithm.

        Returns:
            tuple: (list_of_node_indices, total_cost) or (None, float('inf')) if no path.
        """
        if graph is None or start_node_idx is None or end_node_idx is None:
            return None, float('inf')
        if start_node_idx == end_node_idx:
             # Return path with single node if start and end are the same valid node
             return [start_node_idx], 0.0

        try:
            num_nodes = graph.shape[0]
            if not (0 <= start_node_idx < num_nodes and 0 <= end_node_idx < num_nodes):
                 print(f"  Error (_find_shortest_path): Start ({start_node_idx}) or End ({end_node_idx}) node index out of bounds ({num_nodes}).")
                 return None, float('inf')

            # Compute shortest paths from start_node_idx to all other nodes
            # Rely on default method selection (should be Dijkstra for sparse, positive weights)
            distances, predecessors = shortest_path(csgraph=graph,
                                                   directed=False,
                                                   indices=start_node_idx,
                                                   return_predecessors=True)

            # Check if end_node_idx is reachable
            if np.isinf(distances[end_node_idx]):
                return None, float('inf') # No path exists

            # Reconstruct the path
            path = []
            current_node = end_node_idx
            while current_node != start_node_idx:
                if current_node == -9999: # Indicates no predecessor / unreachable
                    print(f"  Error (_find_shortest_path): Path reconstruction failed (predecessor is -9999).")
                    return None, float('inf')
                path.append(current_node)
                # Check for potential infinite loop
                if len(path) > num_nodes * 2:
                     print(f"  Error (_find_shortest_path): Path reconstruction seems stuck in a loop.")
                     return None, float('inf')
                prev_node = predecessors[current_node]
                if prev_node == current_node: # Should not happen with Dijkstra unless start=end
                     print(f"  Error (_find_shortest_path): Path reconstruction loop detected (node {current_node}).")
                     return None, float('inf')
                current_node = prev_node


            path.append(start_node_idx)
            path.reverse() # Path from start to end

            return path, distances[end_node_idx]

        except ValueError as ve:
             # Catch specific errors like "negative weights not allowed" if they occur
             print(f"  Error (_find_shortest_path - ValueError): {ve}")
             return None, float('inf')
        except Exception as e:
            print(f"  Error (_find_shortest_path): {e}")
            return None, float('inf')

    def _image_to_world_coords(self, pixel_path, transform):
        """Convert a path of (row, col) pixel coordinates to world (x, y) coordinates."""
        if not pixel_path or transform is None: return []
        world_path = []
        offset_x, offset_y = transform['offset']
        scale_x, scale_y = transform['scale'] # scale_y is negative

        for r, c in pixel_path:
            # Calculate world coordinates corresponding to the *center* of the pixel
            wx = offset_x + (c + 0.5) * scale_x
            wy = offset_y + (r + 0.5) * scale_y # scale_y is negative
            world_path.append((wx, wy))

        # Remove consecutive duplicates (can happen with pixel paths)
        final_path = []
        if world_path:
             final_path.append(world_path[0])
             for k in range(1, len(world_path)):
                  if self._euclidean_distance(final_path[-1], world_path[k]) > 1e-6:
                       final_path.append(world_path[k])

        return final_path

    def _euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance, handling None."""
        if p1 is None or p2 is None: return float('inf')
        try:
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        except (TypeError, IndexError):
            # print(f"  Warning: Invalid points for distance: {p1}, {p2}") # Reduce noise
            return float('inf') # Invalid points

    # --- Visualization Methods ---

    def visualize_optimized_path(self, layer, optimized_path, layer_idx):
        """Visualize the optimized path for a layer."""
        config = get_config()
        if not config.get('visualize_optimized_paths', True): return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot polygons
        if layer:
            for polygon in layer:
                 if isinstance(polygon, Polygon) and polygon.is_valid:
                      x, y = polygon.exterior.xy
                      ax.plot(x, y, 'b-', linewidth=1, alpha=0.5, label='Boundary' if 'Boundary' not in ax.get_legend_handles_labels()[1] else "")
                      for interior in polygon.interiors:
                           x_int, y_int = interior.xy
                           ax.plot(x_int, y_int, 'r-', linewidth=1, alpha=0.5, label='Hole' if 'Hole' not in ax.get_legend_handles_labels()[1] else "")

        # Plot the optimized path
        if optimized_path and len(optimized_path) > 1:
            path_x, path_y = zip(*optimized_path)
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis', linewidth=1.5)
            lc.set_array(np.linspace(0, 1, len(segments)))
            line_coll = ax.add_collection(lc)

            ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')

            if len(segments) > 0:
                 cbar = plt.colorbar(line_coll, ax=ax, shrink=0.8)
                 cbar.set_label('Path Direction')
        elif optimized_path:
             ax.plot(optimized_path[0][0], optimized_path[0][1], 'go', markersize=8, label='Start/End')

        ax.set_aspect('equal')
        ax.set_title(f"Layer {layer_idx+1} Optimized Path (OptimizerC)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)

    def _check_self_intersection(self, path, layer_index):
        """Checks if a path self-intersects using Shapely."""
        # Reusing OptimizerA's implementation logic
        if not path or len(path) < 4: return
        try:
            line = LineString(path)
            if not line.is_simple:
                is_closed_loop = Point(path[0]).equals_exact(Point(path[-1]), 1e-6)
                if not is_closed_loop:
                     print(f"  WARNING (OptimizerC): Layer {layer_index} - Path may self-intersect.")
        except Exception as e:
            print(f"  Warning (OptimizerC): Could not check self-intersection for layer {layer_index}: {e}")

    def visualize_layer_transitions(self, layers, optimized_paths):
        """Visualize the transitions between layers."""
        # Reusing OptimizerA's implementation logic
        config = get_config()
        if not config.get('visualize_layer_transitions', True): return

        valid_paths_indices = [i for i, p in enumerate(optimized_paths) if p and len(p) > 0]
        if len(valid_paths_indices) < 1: return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        min_z, max_z = float('inf'), float('-inf')

        for i in valid_paths_indices:
            path = optimized_paths[i]
            color = colors[i]
            z_height = i # Use index as Z for now
            min_z, max_z = min(min_z, z_height), max(max_z, z_height)
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, [z_height] * len(path_x), color=color, linewidth=2, label=f'Layer {i+1}' if i == valid_paths_indices[0] else "")
            ax.scatter(path_x[0], path_y[0], z_height, color='green', s=50, marker='o', label='Start' if i == valid_paths_indices[0] else "")
            ax.scatter(path_x[-1], path_y[-1], z_height, color='red', s=50, marker='x', label='End' if i == valid_paths_indices[0] else "")

            current_plot_index = valid_paths_indices.index(i)
            if current_plot_index > 0:
                prev_plotted_layer_idx = valid_paths_indices[current_plot_index - 1]
                prev_path = optimized_paths[prev_plotted_layer_idx]
                prev_end = prev_path[-1]
                curr_start = path[0]
                prev_z_height = prev_plotted_layer_idx
                ax.plot([prev_end[0], curr_start[0]], [prev_end[1], curr_start[1]], [prev_z_height, z_height], 'k--', linewidth=1.5)
                travel_dist = self._euclidean_distance(prev_end, curr_start)
                if np.isfinite(travel_dist):
                     mid_x, mid_y, mid_z = (prev_end[0] + curr_start[0]) / 2, (prev_end[1] + curr_start[1]) / 2, (prev_z_height + z_height) / 2
                     ax.text(mid_x, mid_y, mid_z, f"{travel_dist:.2f}mm", color='black', fontsize=8, ha='center', zorder=10)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Layer Index')
        ax.set_title('Layer Transitions Visualization (OptimizerC)')
        if np.isfinite(min_z) and np.isfinite(max_z): ax.set_zlim(min_z - 1, max_z + 1)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
             by_label = dict(zip(labels, handles))
             ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        ax.view_init(elev=20., azim=-65)
        plt.tight_layout()
        plt.show(block=False)

    # Placeholder for smoothing if needed later
    # def smooth_path(self, path, segment_length=None, layer_polygons=None):
    #     # Implement smoothing similar to OptimizerA if required
    #     # Consider if smoothing is necessary given the pixel grid nature
    #     print("  Smoothing not yet implemented for OptimizerC path.")
    #     return path

    # --- Debug Visualization Methods ---

    def _visualize_image(self, image, title, transform):
        """Visualize a numpy array image with world coordinates."""
        if image is None: return
        try:
            plt.figure(figsize=(10, 10))
            offset_x, offset_y = transform['offset']
            scale_x, scale_y = transform['scale'] # scale_y is negative
            height, width = image.shape

            # Calculate extent for imshow [left, right, bottom, top] in world coords
            extent = [
                offset_x, offset_x + width * scale_x,
                offset_y + height * scale_y, offset_y
            ]

            plt.imshow(image, cmap='gray', extent=extent, origin='upper') # Origin upper because scale_y is negative
            plt.colorbar()
            plt.title(title)
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show(block=False)
        except Exception as e:
            print(f"  Error visualizing image '{title}': {e}")

    def _visualize_graph(self, graph, node_map, start_node_idx, end_node_idx, transform, title):
        """Visualize the graph nodes and start/end points."""
        if graph is None or node_map is None: return
        try:
            plt.figure(figsize=(10, 10))
            num_nodes = graph.shape[0]
            print(f"  Visualizing graph with {num_nodes} nodes...")

            # Get world coordinates for all nodes
            pixel_coords = list(node_map.values())
            world_coords = self._image_to_world_coords(pixel_coords, transform)

            if not world_coords:
                 print("  Cannot visualize graph: Failed to convert node coordinates.")
                 return

            world_coords_np = np.array(world_coords)

            # Plot all nodes
            plt.scatter(world_coords_np[:, 0], world_coords_np[:, 1], s=1, c='gray', alpha=0.5, label='Graph Nodes')

            # Highlight start and end nodes
            if start_node_idx is not None and 0 <= start_node_idx < len(world_coords):
                start_coord = world_coords[start_node_idx]
                plt.scatter(start_coord[0], start_coord[1], s=50, c='lime', marker='o', label='Start Node', zorder=5)
            if end_node_idx is not None and 0 <= end_node_idx < len(world_coords):
                end_coord = world_coords[end_node_idx]
                plt.scatter(end_coord[0], end_coord[1], s=50, c='red', marker='x', label='End Node', zorder=5)

            plt.title(title)
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            plt.show(block=False)
            print("  Graph visualization complete.")

        except Exception as e:
            print(f"  Error visualizing graph '{title}': {e}")
