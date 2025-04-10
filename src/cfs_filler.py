from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, nearest_points, substring # Import substring
import matplotlib.pyplot as plt
import networkx as nx
import math
from typing import List, Dict, Tuple, Optional, Any

# Define type aliases for clarity
ContourData = Dict[str, Any] # {'polygon': Polygon, 'i': int, 'j': int, 'id': str}
MST = nx.Graph
PathSegment = List[str] # List of node IDs forming a path in the MST

def generate_cfs_fill(region: Polygon, toolpath_width: float = 0.4) -> Optional[LineString]:
    """
    Generates a continuous Fermat spiral (CFS) fill path for a given 2D region.

    Based on the algorithm described in "Connected Fermat Spirals for Layered Fabrication".

    Args:
        region (Polygon): The singly connected 2D region (layer contour) to fill.
                          Assumes the exterior is counter-clockwise and interiors (holes) are clockwise.
        toolpath_width (float): The desired spacing between adjacent toolpath segments (w).

    Returns:
        LineString: The continuous CFS toolpath, or None if generation fails.
    """
    print(f"Starting CFS fill generation for region: {region.wkt[:100]}...")
    print(f"Toolpath width (w): {toolpath_width}")

    if not isinstance(region, Polygon) or not region.is_valid or region.is_empty:
        print("Error: Input must be a valid, non-empty Shapely Polygon.")
        return None

    # --- Step 3: Generate Iso-contours ---
    # Compute successive inward offsets (negative buffers)
    # L = {c_i,j} where d(∂R, c_i,j) = (i - 0.5)w
    print("\nStep 3: Generating Iso-contours...")
    iso_contours = []
    current_offset_region = region
    i = 1
    while True:
        offset_distance = (i - 0.5) * toolpath_width
        # Perform negative buffer (inward offset)
        try:
            # Use a small positive buffer first to handle potential self-intersections near the boundary
            # Then apply the negative buffer
            buffered_region = current_offset_region.buffer(1e-6).buffer(-offset_distance, join_style=2) # MITRE join style
        except Exception as e:
             print(f"  Error during buffering at offset {offset_distance}: {e}")
             break # Stop if buffering fails

        if buffered_region.is_empty or not buffered_region.is_valid:
            print(f"  Offset {offset_distance} resulted in empty or invalid geometry. Stopping.")
            break

        # Handle MultiPolygons - each polygon is a contour c_i,j
        contours_at_level_i = []
        if isinstance(buffered_region, Polygon):
            contours_at_level_i.append(buffered_region)
        elif isinstance(buffered_region, MultiPolygon):
            contours_at_level_i.extend(list(buffered_region.geoms))

        if not contours_at_level_i:
             print(f"  No valid contours generated at offset {offset_distance}. Stopping.")
             break

        # Store contours with their level index i and sub-index j
        level_contours = []
        for j, contour in enumerate(contours_at_level_i):
             # Ensure contour is valid before adding
             if contour.is_valid and not contour.is_empty:
                 level_contours.append({'polygon': contour, 'i': i, 'j': j, 'id': f'c_{i}_{j}'})
             else:
                 print(f"  Warning: Invalid or empty contour c_{i}_{j} generated at offset {offset_distance}. Skipping.")

        if not level_contours:
            print(f"  No valid contours remaining at offset {offset_distance} after validation. Stopping.")
            break

        iso_contours.extend(level_contours)
        print(f"  Generated {len(level_contours)} contour(s) at level i={i} (offset={offset_distance:.3f})")

        # Prepare for next iteration - use the boundary of the current buffer
        # We need the boundary to calculate the *next* offset relative to the *original* boundary
        # The paper implies offsetting from the original boundary ∂R each time.
        # Let's recalculate the offset from the original region boundary for the next step.
        i += 1
        # We don't update current_offset_region here, as the next offset is also from the original boundary.

    print(f"Total iso-contours generated (excluding boundary): {len(iso_contours)}")

    # Add the original boundary as c_0,0 (or adjust indexing based on paper c_1,1 = ∂R)
    # Let's follow the paper: c_1,1 is the boundary ∂R. Adjust 'i' index accordingly.
    # Rerun generation with i starting from 1, offset (i-0.5)w.
    # The code above already does this. Need to add the boundary c_1,1 explicitly?
    # The paper says "c1,1 is always the outer region boundary ∂R".
    # Let's add the original region boundary with i=1, j=0 (or 1?)
    # Let's adjust the loop: i starts at 2, offset (i-0.5)w. Add boundary as c_1,1.
    # RETHINK: The distance is d(∂R, ci,j) = (i - 0.5)w.
    # For i=1, distance is 0.5w. This is the first *offset* contour.
    # ∂R itself is the contour at distance 0. Let's represent it as c_0,0 for graph building.
    # Or, stick to paper's c_1,1 = ∂R. Then first offset is c_2,j at distance 1.5w? No, (2-0.5)w = 1.5w.
    # Let's stick to the formula: d = (i - 0.5)w.
    # i=1 -> d=0.5w. First offset contour.
    # i=2 -> d=1.5w. Second offset contour.
    # The boundary ∂R is not explicitly generated by this loop. Add it manually.
    # We need a node for the boundary in the graph. Let's call it c_0,0.
    boundary_node = {'polygon': region, 'i': 0, 'j': 0, 'id': 'c_0_0'} # Represent boundary as level 0

    # Combine boundary with generated iso-contours
    all_contours_data = [boundary_node] + iso_contours
    print(f"Total contours including boundary (c_0_0): {len(all_contours_data)}")


    # --- Step 4: Build Initial Connectivity Graph ---
    print("\nStep 4: Building Initial Connectivity Graph...")
    # Nodes: iso-contours c_i,j (including boundary c_0,0)
    # Edges: between c_i,j and c_i+1,j' if connecting segment O_i,j,j' exists.
    # Edge Weight: length(O_i,j,j')

    connectivity_graph = nx.Graph()

    # Add nodes
    for contour_data in all_contours_data:
        connectivity_graph.add_node(contour_data['id'], data=contour_data)

    print(f"  Added {connectivity_graph.number_of_nodes()} nodes to the graph.")

    # Add edges (requires calculating connecting segments O_i,j,j')
    # This is complex: O_i,j,j' = {p ∈ c_i,j | d(p, c_i+1,j') < d(p, c_i+1,k), k != j'}
    # Requires finding points on contour c_i,j closest to c_i+1,j'.
    # This might involve discretizing contours or using Voronoi concepts.
    # For now, let's simplify: connect c_i,j to *all* c_i+1,j' if they are "close" (e.g., overlap slightly after small buffer).
    # Weight can be simplified initially, e.g., distance between centroids or minimum distance.

    # Simplified edge connection: Connect if polygons at level i and i+1 are close.
    # Weight: Use distance between centroids as a proxy for now.
    for idx1, contour_data1 in enumerate(all_contours_data):
        for idx2 in range(idx1 + 1, len(all_contours_data)):
            contour_data2 = all_contours_data[idx2]

            # Check if contours are from adjacent levels (i and i+1)
            if contour_data2['i'] == contour_data1['i'] + 1:
                node1_id = contour_data1['id']
                node2_id = contour_data2['id']
                poly1 = contour_data1['polygon']
                poly2 = contour_data2['polygon']

                # Simplified check: Add edge if centroids are reasonably close
                # A better check might involve minimum distance or intersection of small buffers
                # distance = poly1.centroid.distance(poly2.centroid)
                min_dist = poly1.exterior.distance(poly2.exterior) # Approx distance

                # Add edge with weight (using min_dist for now, lower is better)
                # The paper uses length of connecting segment O. Let's use min_dist as placeholder.
                # We need a robust way to calculate O_i,j,j' and its length later.
                # For now, just connect based on adjacency. Weight placeholder = 1.
                # Let's use min_dist as weight, as lower distance might imply easier connection.
                connectivity_graph.add_edge(node1_id, node2_id, weight=min_dist)
                print(f"  Adding potential edge between {node1_id} and {node2_id} (weight={min_dist:.3f})")


    print(f"  Added {connectivity_graph.number_of_edges()} edges to the graph (simplified).")

    # --- Step 5: Construct Spiral-Contour Tree (MST) ---
    print("\nStep 5: Constructing Spiral-Contour Tree (MST)...")
    # Compute MST rooted at the boundary node c_0_0.
    # Need to ensure the graph is connected. If not, the region might be complex or offsets failed.
    if not nx.is_connected(connectivity_graph):
         print("  Warning: Connectivity graph is not connected. Cannot compute single MST.")
         # Handle disconnected components if necessary, or error out.
         # For now, proceed assuming the relevant component is connected to the root.
         pass # Or return None

    # Use Prim's or Kruskal's algorithm. NetworkX has MST functions.
    # Rooting at c_0_0 means we prefer paths starting from the boundary.
    # nx.minimum_spanning_tree handles weights.
    try:
        # Ensure the root node exists before proceeding
        root_node_id = boundary_node['id']
        if root_node_id not in connectivity_graph:
            print(f"Error: Root node {root_node_id} not found in the graph.")
            return None

        # Check connectivity to the root node
        if not nx.has_path(connectivity_graph, root_node_id, list(connectivity_graph.nodes())[0]): # Check path to an arbitrary node
             print(f"Warning: Graph might be disconnected from the root node {root_node_id}.")
             # Attempt MST on the component containing the root
             root_component_nodes = nx.node_connected_component(connectivity_graph, root_node_id)
             subgraph = connectivity_graph.subgraph(root_component_nodes)
             if not subgraph.nodes:
                 print("Error: Root component is empty.")
                 return None
             print(f"  Computing MST on the connected component containing the root ({len(root_component_nodes)} nodes).")
             spiral_contour_tree = nx.minimum_spanning_tree(subgraph, weight='weight')
             mst_nodes = subgraph.number_of_nodes()
        else:
             print("  Computing MST on the full graph.")
             spiral_contour_tree = nx.minimum_spanning_tree(connectivity_graph, weight='weight')
             mst_nodes = connectivity_graph.number_of_nodes()

        print(f"  MST computed with {spiral_contour_tree.number_of_nodes()} nodes and {spiral_contour_tree.number_of_edges()} edges.")

        # --- MST Validation ---
        print("  Validating MST properties...")
        is_valid_mst = True
        num_nodes = spiral_contour_tree.number_of_nodes()
        num_edges = spiral_contour_tree.number_of_edges()
        expected_edges = num_nodes - 1 if num_nodes > 0 else 0

        if num_nodes == 0:
             print("    - Validation Skipped: MST is empty.")
             is_valid_mst = False # Treat empty MST as invalid for further processing
        else:
            # 1. Check number of edges
            if num_edges != expected_edges:
                print(f"    - FAIL: Incorrect number of edges. Expected {expected_edges}, found {num_edges}.")
                is_valid_mst = False
            else:
                print(f"    - OK: Correct number of edges ({num_edges}).")

            # 2. Check if it's a tree (connected and acyclic)
            # Note: nx.minimum_spanning_tree should always return a tree if the input graph component is connected.
            # is_tree checks for both connectivity and acyclicity within the context of the MST graph itself.
            if not nx.is_tree(spiral_contour_tree):
                 # Check specific reasons if it's not a tree
                 if not nx.is_connected(spiral_contour_tree):
                     print("    - FAIL: MST is not connected.")
                     is_valid_mst = False
                 # Check for cycles (shouldn't happen with MST algorithm)
                 try:
                     cycle = nx.find_cycle(spiral_contour_tree)
                     print(f"    - FAIL: MST contains a cycle: {cycle}")
                     is_valid_mst = False
                 except nx.NetworkXNoCycle:
                     # If it's not connected, it might also report no cycle here.
                     # The primary issue is likely connectivity if is_tree failed but no cycle found.
                     if is_valid_mst: # Only print OK if no other failure occurred
                          print("    - OK: MST is acyclic.")
            else:
                 print("    - OK: MST is a valid tree (connected and acyclic).")

            # 3. Check total weight (optional info)
            total_weight = spiral_contour_tree.size(weight='weight')
            print(f"    - Info: Total weight of MST: {total_weight:.4f}")

        if not is_valid_mst:
             print("  Error: MST validation failed. Cannot proceed with CFS generation.")
             plot_contours_and_mst(all_contours_data, spiral_contour_tree, toolpath_width, None) # Plot potentially problematic MST
             return None
        # --- End MST Validation ---

    except Exception as e:
        print(f"Error computing or validating MST: {e}")
        # Consider visualization or further debugging here
        # plot_graph(connectivity_graph, "Connectivity Graph")
        return None


    # --- Step 6: Identify Spirallable Regions & Branch Points ---
    print("\nStep 6: Identifying Spirallable Regions & Branch Points...")
    mst_structure = identify_mst_structure(spiral_contour_tree, boundary_node['id'])
    if not mst_structure:
        print("Error: Failed to analyze MST structure.")
        # Pass None for mst_structure as it failed to be created
        plot_contours_and_mst(all_contours_data, spiral_contour_tree, toolpath_width, None)
        return None

    branch_points = mst_structure['branch_points']
    leaf_nodes = mst_structure['leaf_nodes']
    spirallable_paths = mst_structure['paths']

    print(f"  Identified {len(branch_points)} branch points: {list(branch_points)}")
    print(f"  Identified {len(leaf_nodes)} leaf nodes: {list(leaf_nodes)}")
    print(f"  Identified {len(spirallable_paths)} spirallable paths/segments.")
    # for i, path in enumerate(spirallable_paths):
    #     print(f"    Path {i+1}: {' -> '.join(path)}")


    # --- Step 7: Recursive Rerouting (Bottom-Up Traversal of MST) ---
    print("\nStep 7: Performing Recursive Rerouting (Bottom-Up Traversal)...")
    # This step requires the detailed logic from the paper (Section 3, Fig 5)
    # We pass the MST, contour data, toolpath width, and the identified structure.
    final_path = perform_recursive_rerouting(
        spiral_contour_tree,
        all_contours_data,
        toolpath_width,
        mst_structure
    )

    # --- Step 8: Output ---
    print("\nStep 8: Outputting Final Path...")
    if final_path and isinstance(final_path, LineString) and not final_path.is_empty:
        print(f"  CFS path generated successfully (Length: {final_path.length:.2f}).")
        # Optional: Simplify or smooth the path (Step 9)
        # final_path = final_path.simplify(tolerance=toolpath_width / 10)
        # Pass mst_structure to the plotting function
        plot_contours_and_path(all_contours_data, spiral_contour_tree, final_path, toolpath_width, mst_structure) # Visualize result
        return final_path
    else:
        print("  CFS path generation incomplete or failed in Step 7.")
        # For debugging, visualize the contours and MST structure
        plot_contours_and_mst(all_contours_data, spiral_contour_tree, toolpath_width, mst_structure)
        return None


# --------------------------------------------------------------------------
# Geometric Helper Functions
# --------------------------------------------------------------------------

def get_coords(geom) -> List[Tuple[float, float]]:
    """Get coordinates from Point, LineString, Polygon exterior/interiors."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Point):
        return list(geom.coords)
    elif isinstance(geom, LineString):
        return list(geom.coords)
    elif isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        # Note: Ignoring interior coordinates for path generation for now
        return coords
    elif isinstance(geom, MultiPolygon):
        # Handle MultiPolygon - maybe return coords of the largest polygon?
        largest_poly = max(geom.geoms, key=lambda p: p.area)
        return list(largest_poly.exterior.coords)
    return []

def find_point_index(line: LineString, point: Point) -> int:
    """Find the index of the vertex on the line closest to the point."""
    min_dist = float('inf')
    closest_idx = -1
    if line is None or line.is_empty or point is None or point.is_empty:
        return -1
    line_coords = list(line.coords)
    if not line_coords: return -1

    for i, coord in enumerate(line_coords):
        dist = point.distance(Point(coord))
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx

def get_line_segment_coords(line: LineString, p_start: Point, p_end: Point) -> List[Tuple[float, float]]:
    """
    Get coordinates of a LineString segment between two points projected onto the line.
    Uses shapely.substring for potentially better accuracy, especially on rings.
    """
    if line is None or line.is_empty or p_start is None or p_start.is_empty or p_end is None or p_end.is_empty:
        return []
    if not isinstance(line, LineString):
        print(f"Warning: get_line_segment_coords expected LineString, got {type(line)}")
        return []

    try:
        start_dist = line.project(p_start)
        end_dist = line.project(p_end)

        # Handle potential floating point inaccuracies near ends
        tolerance = 1e-9
        if abs(start_dist - end_dist) < tolerance:
             # Start and end points project to the same location
             return [get_coords(line.interpolate(start_dist))]

        if line.is_ring:
            if abs(start_dist - end_dist) > line.length - tolerance:
                 # Points are effectively the same but distances are at opposite ends
                 return [get_coords(line.interpolate(start_dist))]

            if start_dist <= end_dist:
                 sub = substring(line, start_dist, end_dist)
                 return get_coords(sub)
            else: # Wraps around the ring
                 sub1 = substring(line, start_dist, line.length)
                 sub2 = substring(line, 0, end_dist)
                 # Combine coords, avoiding duplicate point if possible
                 coords1 = get_coords(sub1)
                 coords2 = get_coords(sub2)
                 if coords1 and coords2 and Point(coords1[-1]).distance(Point(coords2[0])) < tolerance:
                     return coords1 + coords2[1:]
                 else:
                     return coords1 + coords2
        else: # Simple line
             # Ensure start_dist <= end_dist for substring
             if start_dist > end_dist:
                  start_dist, end_dist = end_dist, start_dist
             sub = substring(line, start_dist, end_dist)
             return get_coords(sub)

    except Exception as e:
         print(f"Error using substring (start={start_dist:.2f}, end={end_dist:.2f}, len={line.length:.2f}): {e}. Returning empty list.")
         # Fallback: return just the start and end points projected onto the line
         # return [get_coords(line.interpolate(start_dist)), get_coords(line.interpolate(end_dist))]
         return []

def find_link_point(source_point: Point, source_contour: Polygon, target_contour: Polygon) -> Optional[Point]:
    """
    Finds the point on target_contour's exterior nearest to source_point.
    Approximates I(p) or O(p) by finding the nearest point on the adjacent contour.
    """
    if not source_point or not source_contour or not target_contour or \
       source_contour.is_empty or target_contour.is_empty:
        return None
    try:
        # Find the point on the target exterior closest to the source point
        nearest_geom = nearest_points(source_point, target_contour.exterior)
        if nearest_geom and len(nearest_geom) == 2:
            return nearest_geom[1] # Return the point on the target contour
        else:
            # Fallback if nearest_points fails
            print(f"Warning: Could not find nearest point from {source_point.wkt[:30]} to target contour. Using target centroid projection.")
            target_centroid_proj = target_contour.exterior.interpolate(target_contour.exterior.project(target_contour.centroid))
            nearest_geom_fallback = nearest_points(source_point, target_centroid_proj)
            return nearest_geom_fallback[1] if nearest_geom_fallback else None

    except Exception as e:
        print(f"Error finding link point: {e}")
        return None


def find_adjacent_point_on_contour(point: Point, contour_line: LineString, distance: float, forward: bool = True) -> Optional[Point]:
    """
    Finds a point on the contour_line at a specified distance along the line from the projection of 'point'.
    Approximates B(p) (backward, forward=False) or N(p) (forward, forward=True).
    Distance 'delta' should be small relative to contour length.
    """
    if not point or not contour_line or contour_line.is_empty:
        return None
    try:
        current_dist = contour_line.project(point)
        target_dist = current_dist + (distance if forward else -distance)

        # Handle wrapping for closed rings
        line_len = contour_line.length
        if contour_line.is_ring:
            target_dist = target_dist % line_len
            # Adjust for negative results from modulo if needed, although % in Python handles it well.
            # if target_dist < 0: target_dist += line_len
        else:
            # Clamp to line ends if not a ring
            target_dist = max(0, min(line_len, target_dist))

        result_point = contour_line.interpolate(target_dist)
        if result_point is None or result_point.is_empty:
             print(f"Error finding adjacent point on contour: Interpolation failed for target_dist={target_dist:.4f} on line length {contour_line.length:.4f}")
             return None
        return result_point
    except Exception as e:
        # Add more context to the error message
        line_info = f"line length {contour_line.length:.4f}" if contour_line else "invalid line"
        point_info = f"point {point.wkt[:30]}" if point else "invalid point"
        print(f"Error finding adjacent point on contour ({point_info}, {line_info}, dist={distance}, fwd={forward}): {e}")
        # Optionally re-raise or provide more debug info
        # import traceback
        # traceback.print_exc()
        return None

# --------------------------------------------------------------------------
# Helper Functions for CFS Algorithm Steps
# --------------------------------------------------------------------------

def identify_mst_structure(mst: MST, root_node_id: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes the MST to find leaf nodes, branch points, and spirallable paths.

    Args:
        mst (MST): The Minimum Spanning Tree graph.
        root_node_id (str): The ID of the root node (boundary contour).

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing 'leaf_nodes' (set),
                                   'branch_points' (set), and 'paths' (list of lists of node IDs),
                                   or None if the MST is empty or invalid.
    """
    if not mst or mst.number_of_nodes() == 0:
        print("Error: MST is empty or invalid.")
        return None

    degrees = dict(mst.degree())
    nodes = set(mst.nodes())

    # Identify leaves (degree 1) and branch points (degree > 2)
    # The root node is treated specially - it's not a leaf even if degree is 1.
    leaf_nodes = {node for node, degree in degrees.items() if degree == 1 and node != root_node_id}
    branch_nodes = {node for node, degree in degrees.items() if degree > 2}
    # Include root node as a potential branch point if its degree is > 1 (or > 0 if it's the only node?)
    # The root acts as the start/end point.
    if degrees.get(root_node_id, 0) > 1:
         branch_nodes.add(root_node_id)
    elif mst.number_of_nodes() > 1 and degrees.get(root_node_id, 0) == 1:
         # If root has degree 1 and it's not the only node, it acts like a path end.
         pass # Not technically a branch, but the path terminates/starts here.

    # Find paths (sequences of degree-2 nodes) between leaves and branch points
    paths = []
    visited_edges = set()

    # Start traversal from leaf nodes
    for start_node in leaf_nodes:
        if start_node not in nodes: continue # Should not happen
        current_path = [start_node]
        prev_node = None
        curr_node = start_node

        while True:
            neighbors = list(mst.neighbors(curr_node))
            # Find the next node that isn't the previous one
            next_node = None
            for neighbor in neighbors:
                if neighbor != prev_node:
                    edge = tuple(sorted((curr_node, neighbor)))
                    if edge not in visited_edges:
                        next_node = neighbor
                        break
                    else:
                         # Edge already part of another path, stop here
                         # This can happen if we start from a branch point later
                         pass

            if next_node is None: # Reached end of traversal (maybe back at start or visited edge)
                break

            edge = tuple(sorted((curr_node, next_node)))
            if edge in visited_edges: # Should be caught above, but double check
                 print(f"Warning: Edge {edge} already visited when traversing from {start_node}")
                 break
            visited_edges.add(edge)
            current_path.append(next_node)

            # If the next node is a branch point or the root, the path ends
            if next_node in branch_nodes or next_node == root_node_id:
                break

            # If the next node is somehow a leaf (shouldn't happen in MST unless 2 nodes total)
            if next_node in leaf_nodes:
                 print(f"Warning: Path from leaf {start_node} encountered another leaf {next_node} unexpectedly.")
                 break # Path ends

            # Continue traversal
            prev_node = curr_node
            curr_node = next_node
            # Safety break for unexpected cycles or issues
            if len(current_path) > mst.number_of_nodes() * 2:
                 print(f"Error: Path traversal from {start_node} seems stuck in a loop. Aborting path finding.")
                 return None # Indicate error

        if len(current_path) > 1:
            paths.append(current_path)

    # Start traversal from branch nodes to find paths connecting them
    for start_node in branch_nodes:
        if start_node not in nodes: continue
        for neighbor in mst.neighbors(start_node):
            edge = tuple(sorted((start_node, neighbor)))
            if edge in visited_edges:
                continue # This path segment already covered (likely starting from a leaf)

            # Start a new path from this neighbor
            current_path = [start_node, neighbor]
            visited_edges.add(edge)
            prev_node = start_node
            curr_node = neighbor

            while True:
                # If the current node is a branch point or leaf, path ends here
                if curr_node in branch_nodes or curr_node in leaf_nodes:
                    break

                neighbors = list(mst.neighbors(curr_node))
                next_node = None
                for next_cand in neighbors:
                    if next_cand != prev_node:
                        next_node = next_cand
                        break

                if next_node is None: # Should not happen in a connected MST path unless it's the end
                    print(f"Warning: Path traversal from branch {start_node} via {neighbor} ended unexpectedly at {curr_node}.")
                    break

                edge = tuple(sorted((curr_node, next_node)))
                if edge in visited_edges:
                    print(f"Warning: Edge {edge} already visited when traversing from branch {start_node} via {neighbor}.")
                    break # Avoid reusing edges
                visited_edges.add(edge)
                current_path.append(next_node)

                # Check if the new node ends the path
                if next_node in branch_nodes or next_node in leaf_nodes:
                    break

                # Continue traversal
                prev_node = curr_node
                curr_node = next_node
                # Safety break
                if len(current_path) > mst.number_of_nodes() * 2:
                     print(f"Error: Path traversal from branch {start_node} seems stuck in a loop. Aborting path finding.")
                     return None # Indicate error

            if len(current_path) > 1: # Should always be > 1 here
                 paths.append(current_path)


    # Validation: Check if all edges are covered
    if len(visited_edges) != mst.number_of_edges():
        print(f"Warning: Path identification covered {len(visited_edges)} edges, but MST has {mst.number_of_edges()}. Some edges might be missed.")
        # This might happen if the root has degree 1 and is the only branch point.

    return {
        'leaf_nodes': leaf_nodes,
        'branch_points': branch_nodes,
        'paths': paths,
        'root_node_id': root_node_id
    }


def perform_recursive_rerouting(mst: MST,
                                contours_data: List[ContourData],
                                toolpath_width: float,
                                mst_structure: Dict[str, Any]) -> Optional[LineString]:
    """
    Performs the recursive rerouting process based on the MST structure.
    Initiates the bottom-up traversal.

    Args:
        mst (MST): The Minimum Spanning Tree.
        contours_data (List[ContourData]): List containing data for each contour node.
        toolpath_width (float): The toolpath width 'w'.
        mst_structure (Dict[str, Any]): Analysis result from identify_mst_structure.

    Returns:
        Optional[LineString]: The final continuous toolpath, or None if failed.
    """
    print("  Starting Step 7: Recursive Rerouting...")
    root_node_id = mst_structure['root_node_id']
    contours_map = {cd['id']: cd for cd in contours_data}
    # processed_paths: Dict[Tuple[str, str], Optional[LineString]] = {} # No longer needed with recursive return

    # Define the recursive helper function
    def _process_node(node_id: str, parent_id: Optional[str]) -> Optional[LineString]:
        """
        Recursively processes nodes bottom-up.
        Generates/merges the path segment leading towards the parent_id.
        Returns the generated LineString segment or None.
        """
        print(f"    Processing node: {node_id} (called from parent: {parent_id})")
        neighbors = list(mst.neighbors(node_id))

        # Process children first (nodes other than the parent)
        children = [neighbor for neighbor in neighbors if neighbor != parent_id]
        child_segments: Dict[str, Optional[LineString]] = {}
        for child_id in children:
            # Recursive call returns the path segment generated from the child towards node_id
            child_segments[child_id] = _process_node(child_id, node_id)

        # --- All children processed, now process this node ---
        print(f"    Node {node_id}: All children processed. Merging/Generating path towards parent {parent_id}.")
        # We have the segments from children in child_segments.

        # Determine action based on node type and context
        node_degree = mst.degree(node_id)
        is_leaf = node_id in mst_structure['leaf_nodes']
        is_branch = node_id in mst_structure['branch_points']
        is_root = node_id == root_node_id

        # Path segment to be generated/merged for the edge connecting node_id to parent_id
        outgoing_segment: Optional[LineString] = None

        if is_leaf:
            # Generate spiral for the single path segment connecting leaf to its parent
            print(f"      Node {node_id} is a leaf. Generating initial spiral segment towards {parent_id}.")
            # The 'path' is just [node_id, parent_id]
            # --- Leaf Node ---
            print(f"      Node {node_id} is a leaf. Generating Fermat spiral towards {parent_id}.")
            if not parent_id: # Should not happen if MST is valid and > 1 node
                 print("      Error: Leaf node has no parent.")
                 return None
            path_nodes = [node_id, parent_id] # Order: inner to outer

            # Determine Pin and Pout on the parent contour (outermost for this segment)
            poly_leaf = contours_map[node_id]['polygon']
            poly_parent = contours_map[parent_id]['polygon']
            if poly_leaf.is_empty or poly_parent.is_empty:
                 print("      Error: Empty polygon for leaf or parent.")
                 return None

            # Pout: Connection point towards parent's parent (if exists) or arbitrary point
            # For simplicity, let's use the nearest point on parent to leaf as both Pin and Pout initially.
            # This assumes the spiral starts and ends at the connection point.
            # A better approach might need context from parent's processing.
            nearest = nearest_points(poly_parent.exterior, poly_leaf.exterior)
            if not nearest or len(nearest) != 2:
                 print("      Error: Cannot find nearest points between leaf and parent.")
                 return None
            p_on_parent = nearest[0]
            pin = p_on_parent # Start on parent contour near leaf
            pout = p_on_parent # Exit on parent contour near leaf

            # Call the Fermat spiral generation function
            outgoing_segment = generate_fermat_spiral_segment(path_nodes, contours_map, toolpath_width, pin, pout)
            if outgoing_segment:
                 print(f"        -> Generated Fermat segment for leaf {node_id} (Length: {outgoing_segment.length:.2f})")
            else:
                     print(f"        -> Failed to generate segment for leaf {node_id}")

        elif is_branch:
            # --- Branch Node ---
            print(f"      Node {node_id} is a branch point. Merging {len(child_segments)} segments.")
            # Pass the dictionary of segments received from children recursive calls
            # Also pass the MST to find connection points
            merged_branch_segment = merge_paths_at_branch(node_id, parent_id, child_segments, contours_map, toolpath_width, mst)

            if not merged_branch_segment:
                 print(f"        -> Failed to merge segments at branch {node_id}")
                 return None # Critical failure

            print(f"        -> Merged segments on branch contour {node_id} (Length: {merged_branch_segment.length:.2f})")

            # Now, need to generate the segment from the branch node towards the parent
            if parent_id:
                 print(f"        -> Generating Fermat segment from branch {node_id} towards parent {parent_id}")
                 path_nodes = [node_id, parent_id] # Inner (branch) to outer (parent)

                 # Pin: End point of the merged segment on the branch contour
                 pin_on_parent = Point(merged_branch_segment.coords[-1]) # This is the exit point on branch contour
                 # Pout: Connection point on parent contour towards parent's parent (needs context or approximation)
                 # Let's approximate Pout as the nearest point on parent to branch
                 poly_branch = contours_map[node_id]['polygon']
                 poly_parent = contours_map[parent_id]['polygon']
                 nearest = nearest_points(poly_parent.exterior, poly_branch.exterior)
                 if not nearest or len(nearest) != 2:
                      print("        Error: Cannot find nearest points between branch and parent.")
                      return None # Cannot determine Pout
                 pout_on_parent = nearest[0] # Point on parent contour

                 # Generate the connecting Fermat spiral segment
                 connecting_segment = generate_fermat_spiral_segment(path_nodes, contours_map, toolpath_width, pin_on_parent, pout_on_parent)

                 if connecting_segment:
                      # Combine the path on the branch with the connecting segment
                      coords_branch = get_coords(merged_branch_segment)
                      coords_connect = get_coords(connecting_segment)
                      # Check connection - start of connecting should match end of branch segment
                      if coords_branch and coords_connect and Point(coords_branch[-1]).distance(Point(coords_connect[0])) < 1e-3:
                           combined_coords = coords_branch + coords_connect[1:]
                           outgoing_segment = LineString(combined_coords)
                           print(f"        -> Combined branch and connecting segment (Total Length: {outgoing_segment.length:.2f})")
                      else:
                           print(f"        -> Warning: Mismatch connecting branch merge to parent segment. Appending.")
                           combined_coords = coords_branch + coords_connect
                           outgoing_segment = LineString(combined_coords)
                 else:
                      print(f"        -> Failed to generate connecting segment from branch {node_id} to parent {parent_id}. Returning only merged branch path.")
                      outgoing_segment = merged_branch_segment # Return only the path on the branch
            else:
                 # Branch is the root node, the merged path is the final path (or part of it)
                 print(f"        -> Branch node {node_id} is root. Returning merged path.")
                 outgoing_segment = merged_branch_segment

            if not outgoing_segment:
                 print(f"        -> Failed to merge segments at branch {node_id}")

        else: # Degree 2 node (part of a path)
            print(f"      Node {node_id} is a path node (degree 2).")
            # Should have exactly one child (neighbor that is not parent)
            child_id = children[0] if children else None
            incoming_segment = child_segments.get(child_id) if child_id else None

            # --- Path Node (Degree 2) ---
            if not children:
                 print(f"      Error: Degree-2 node {node_id} has no child in MST traversal.")
                 return None
            child_id = children[0]
            incoming_segment = child_segments.get(child_id)

            if not incoming_segment:
                 # This case shouldn't happen if leaf/child processing was successful
                 print(f"      Error: No incoming segment from child {child_id} at degree-2 node {node_id}.")
                 return None # Cannot proceed without incoming path

            if not parent_id:
                 # This node is the root, and has degree 2? Should only happen if MST is just two nodes.
                 print(f"      Node {node_id} is root with degree 2. Returning incoming segment from {child_id}.")
                 outgoing_segment = incoming_segment
            else:
                 # Extend the incoming path towards the parent using Fermat spiral logic
                 print(f"        -> Extending Fermat segment from child {child_id} via {node_id} towards parent {parent_id}.")
                 path_nodes = [node_id, parent_id] # Inner (current) to outer (parent)

                 # Pin: End point of the incoming segment (should be on node_id contour)
                 pin_on_parent = Point(incoming_segment.coords[-1])
                 # Pout: Connection point on parent contour towards parent's parent (approximate)
                 poly_current = contours_map[node_id]['polygon']
                 poly_parent = contours_map[parent_id]['polygon']
                 nearest = nearest_points(poly_parent.exterior, poly_current.exterior)
                 if not nearest or len(nearest) != 2:
                      print("        Error: Cannot find nearest points between node and parent.")
                      return None # Cannot determine Pout
                 pout_on_parent = nearest[0] # Point on parent contour

                 # Generate the connecting Fermat spiral segment
                 connecting_segment = generate_fermat_spiral_segment(path_nodes, contours_map, toolpath_width, pin_on_parent, pout_on_parent)

                 if connecting_segment:
                      # Combine the incoming segment with the connecting segment
                      coords_incoming = get_coords(incoming_segment)
                      coords_connect = get_coords(connecting_segment)
                      # Check connection - start of connecting should match end of incoming
                      if coords_incoming and coords_connect and Point(coords_incoming[-1]).distance(Point(coords_connect[0])) < 1e-3:
                           combined_coords = coords_incoming + coords_connect[1:]
                           outgoing_segment = LineString(combined_coords)
                           print(f"        -> Extended path segment (New Length: {outgoing_segment.length:.2f})")
                      else:
                           print(f"        -> Warning: Mismatch connecting incoming segment to extension at {node_id}. Appending.")
                           combined_coords = coords_incoming + coords_connect
                           outgoing_segment = LineString(combined_coords)
                 else:
                      print(f"        -> Failed to generate extension segment from {node_id} to {parent_id}. Returning only incoming segment.")
                      outgoing_segment = incoming_segment # Return only the path up to this node

            if not outgoing_segment: # Catchall if something failed above
                          print(f"        -> Failed to generate new segment.")

        # Return the segment generated/merged for the connection towards the parent
        return outgoing_segment


    # --- Start the recursive processing ---
    # Handle the case of a single node (only the boundary)
    if mst.number_of_nodes() == 1 and root_node_id in contours_map:
        print("  Only boundary contour exists. No fill path needed/possible.")
        return None
    elif mst.number_of_nodes() == 0:
         print("  MST is empty. Cannot generate path.")
         return None

    # Start recursion from the root. The function processes children first.
    print(f"  Initiating recursive processing from root: {root_node_id}")
    # The final path is the result returned by processing the root node
    final_path = _process_node(root_node_id, None)

    if final_path and isinstance(final_path, LineString) and not final_path.is_empty:
        print(f"  Recursive rerouting completed. Final path length: {final_path.length:.2f}")
        return final_path
    else:
        print("  Recursive rerouting did not produce a valid final path.")
        return None


def generate_fermat_spiral_segment(path_nodes: PathSegment,
                                   contours_map: Dict[str, ContourData],
                                   toolpath_width: float,
                                   pin: Point, # Start point on outermost contour of this segment
                                   pout: Point) -> Optional[LineString]: # Exit point on outermost contour
    """
    Generates a Fermat spiral path segment for a sequence of contours (a path in the MST).
    Implements the rerouting logic from Fig 5 of the paper using approximations.

    Args:
        path_nodes (PathSegment): List of contour node IDs forming the path (e.g., [leaf, ..., parent]).
                                  MUST be ordered from innermost (leaf) to outermost (parent).
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.
        pin (Point): The designated start point on the outermost contour (path_nodes[-1]).
        pout (Point): The designated exit point on the outermost contour (path_nodes[-1]).

    Returns:
        Optional[LineString]: The generated Fermat spiral segment, or None if failed.
    """
    print(f"  Generating Fermat spiral for path: {' -> '.join(path_nodes)}")
    print(f"    Pin: {pin.wkt[:30]}, Pout: {pout.wkt[:30]} on {path_nodes[-1]}")

    if len(path_nodes) < 1:
        print("    Error: Path needs at least one node.")
        return None
    if not pin or not pout:
        print("    Error: Valid Pin and Pout are required.")
        return None

    # --- Precompute contour geometries and links ---
    polygons = [contours_map[node_id]['polygon'] for node_id in path_nodes]
    exteriors = [poly.exterior for poly in polygons if poly and not poly.is_empty]
    if len(exteriors) != len(path_nodes):
        print("    Error: Could not retrieve valid exterior for all path nodes.")
        return None

    num_contours = len(exteriors)
    innermost_contour_idx = 0
    outermost_contour_idx = num_contours - 1

    # --- Rerouting Parameters ---
    # Delta (δ) for B(p)/N(p) - small distance along contour
    # Needs careful tuning - maybe related to toolpath width or contour resolution
    delta = toolpath_width / 4.0
    max_iterations = num_contours * 5 # Safety break for rerouting loops

    # --- Initialize Path ---
    final_coords = []
    visited_segments = set() # To track which parts of contours are used in inward vs outward

    # --- Start at Pin on the outermost contour ---
    current_contour_idx = outermost_contour_idx
    current_point = pin
    final_coords.append(get_coords(current_point)[0])
    print(f"    Starting at Pin: {current_point.wkt[:30]} on contour {current_contour_idx}")

    # --- Inward Phase ---
    print("    --- Inward Phase ---")
    iteration = 0
    prev_target_point_on_outer = None # Store the target point from the previous (outer) contour iteration
    while current_contour_idx >= innermost_contour_idx and iteration < max_iterations:
        iteration += 1
        current_contour = exteriors[current_contour_idx]
        print(f"      Iter {iteration}: On contour {current_contour_idx} at {current_point.wkt[:30]}")

        # 1. Determine the target point for this contour segment
        target_point = None
        if current_contour_idx == outermost_contour_idx:
            # On the first (outermost) contour, travel towards B(pout)
            target_point = find_adjacent_point_on_contour(pout, current_contour, delta, forward=False) # B(pout)
            if not target_point:
                 print(f"      Error: Could not calculate B(pout) on contour {current_contour_idx}. Aborting.")
                 return None
            print(f"        Target: B(pout) = {target_point.wkt[:30]}")
            # Store this target point for the next iteration's calculation
            # prev_target_point_on_outer = target_point # Moved outside if/else
        else:
            # On inner contours, travel towards B(I(B(prev_target)))
            # Use the target point stored from the *previous* (outer) contour iteration
            if not prev_target_point_on_outer:
                 print(f"      Error: Missing previous target point for inner contour calculation. Aborting.")
                 return None

            # Find I(prev_target_point_on_outer) - link from outer target to current contour
            linked_point_on_current = find_link_point(prev_target_point_on_outer, polygons[current_contour_idx+1], polygons[current_contour_idx])
            if not linked_point_on_current:
                 print(f"      Error: Could not find inward link I(prev_target) to contour {current_contour_idx}. Aborting.")
                 return None

            # Find B(linked_point_on_current) - this is the target for this contour segment
            target_point = find_adjacent_point_on_contour(linked_point_on_current, current_contour, delta, forward=False)
            if not target_point:
                 print(f"      Error: Could not calculate B(I(prev_target)) on contour {current_contour_idx}. Aborting.")
                 return None
            print(f"        Target: B(I(prev_target)) = {target_point.wkt[:30]}")


        # Check if target_point calculation failed in either branch
        if not target_point:
             print(f"      Error: Failed to determine target_point on contour {current_contour_idx} during inward phase. Aborting.")
             return None

        # Store the calculated target_point for the *next* iteration (if any)
        prev_target_point_on_outer = target_point

        # 2. Generate segment on current contour from current_point to target_point
        segment_coords = get_line_segment_coords(current_contour, current_point, target_point)
        if not segment_coords or len(segment_coords) < 2:
            print(f"      Warning: Could not generate segment on contour {current_contour_idx}. Adding jump.")
            # Add jump if segment fails
            if Point(final_coords[-1]).distance(target_point) > 1e-3:
                 final_coords.extend(get_coords(target_point))
        else:
            # Add segment coords, avoiding duplicates
            if Point(final_coords[-1]).distance(Point(segment_coords[0])) < 1e-3:
                 final_coords.extend(segment_coords[1:])
            else:
                 final_coords.extend(segment_coords)
            # Mark this segment as visited (optional, for outward phase)
            # visited_segments.add((current_contour_idx, tuple(segment_coords)))

        current_point = Point(final_coords[-1]) # Update current point
        print(f"        Added segment on contour {current_contour_idx}, length {LineString(segment_coords).length:.2f}. Current point: {current_point.wkt[:30]}")

        # 3. Reroute Inward (if not already at the innermost contour)
        if current_contour_idx > innermost_contour_idx:
            # Find inward link I(target_point) from the target point on this contour
            inner_contour = exteriors[current_contour_idx - 1]
            reroute_point = find_link_point(target_point, polygons[current_contour_idx], polygons[current_contour_idx-1])
            if not reroute_point:
                 print(f"      Error: Could not find inward link I(target) for rerouting from contour {current_contour_idx}. Aborting.")
                 return None

            # Add connection from current_point (should be target_point) to reroute_point
            if current_point.distance(reroute_point) > 1e-3:
                 final_coords.extend(get_coords(reroute_point))
            print(f"        Rerouted inward to contour {current_contour_idx - 1} at {reroute_point.wkt[:30]}")
            current_point = reroute_point
            current_contour_idx -= 1
        else:
            # Reached innermost contour, end inward phase
            print(f"      Reached innermost contour {current_contour_idx}. Ending inward phase.")
            break # Exit while loop

    if iteration >= max_iterations:
         print("    Error: Max iterations reached during inward phase. Aborting.")
         return None

    # --- Center Turn ---
    # The path currently ends at current_point on the innermost contour.
    # The paper implies a turn here. For simplicity, we can just start the outward phase.
    center_point = current_point
    print(f"    --- Center Turn at {center_point.wkt[:30]} ---")


    # --- Outward Phase ---
    print("    --- Outward Phase ---")
    # Start from center_point on innermost_contour_idx
    iteration = 0
    prev_target_point_on_inner = None # Store target from previous (inner) contour iteration
    while current_contour_idx <= outermost_contour_idx and iteration < max_iterations:
        iteration += 1
        current_contour = exteriors[current_contour_idx]
        print(f"      Iter {iteration}: On contour {current_contour_idx} at {current_point.wkt[:30]}")

        # 1. Determine the target point for this contour segment
        target_point = None
        if current_contour_idx == outermost_contour_idx:
            # On the outermost contour, travel towards Pout
            target_point = pout
            print(f"        Target: Pout = {target_point.wkt[:30]}")
            # Store this target point for the next iteration's calculation
            # prev_target_point_on_inner = target_point # Moved outside if/else
        else:
            # On inner contours, travel towards O(prev_target)
            # Use the target point stored from the *previous* (inner) contour iteration
            if not prev_target_point_on_inner:
                 print(f"      Error: Missing previous target point for outward phase calculation. Aborting.")
                 return None

            # Find O(prev_target_point_on_inner) - link from inner target to current contour
            linked_point_on_current = find_link_point(prev_target_point_on_inner, polygons[current_contour_idx], polygons[current_contour_idx+1])
            if not linked_point_on_current:
                 print(f"      Error: Could not find outward link O(prev_target) to contour {current_contour_idx}. Aborting.")
                 return None
            target_point = linked_point_on_current
            print(f"        Target: O(prev_target) = {target_point.wkt[:30]}")


        # Check if target_point calculation failed in either branch
        if not target_point:
             print(f"      Error: Failed to determine target_point on contour {current_contour_idx} during outward phase. Aborting.")
             return None

        # Store the calculated target_point for the *next* iteration (if any)
        prev_target_point_on_inner = target_point

        # 2. Generate segment on current contour from current_point to target_point
        # TODO: This segment should ideally *avoid* parts visited during the inward phase.
        # This requires tracking visited segments accurately, which is complex.
        # Simplification: Generate the full segment for now.
        segment_coords = get_line_segment_coords(current_contour, current_point, target_point)
        if not segment_coords or len(segment_coords) < 2:
            print(f"      Warning: Could not generate outward segment on contour {current_contour_idx}. Adding jump.")
            if Point(final_coords[-1]).distance(target_point) > 1e-3:
                 final_coords.extend(get_coords(target_point))
        else:
            # Add segment coords, avoiding duplicates
            if Point(final_coords[-1]).distance(Point(segment_coords[0])) < 1e-3:
                 final_coords.extend(segment_coords[1:])
            else:
                 final_coords.extend(segment_coords)

        current_point = Point(final_coords[-1]) # Update current point
        print(f"        Added outward segment on contour {current_contour_idx}, length {LineString(segment_coords).length:.2f}. Current point: {current_point.wkt[:30]}")

        # 3. Reroute Outward (if not already at the outermost contour)
        if current_contour_idx < outermost_contour_idx:
            # The target_point calculated in step 1 *is* the reroute point on the next outer contour.
            reroute_point = target_point
            # Add connection from current_point (should be target_point) to reroute_point? No, they are the same.
            # We just need to update the index and current_point.
            print(f"        Rerouted outward to contour {current_contour_idx + 1} at {reroute_point.wkt[:30]}")
            current_point = reroute_point
            current_contour_idx += 1
        else:
            # Reached outermost contour, end outward phase (should be at Pout)
            print(f"      Reached outermost contour {current_contour_idx}. Ending outward phase.")
            # Final check: are we close to Pout?
            if current_point.distance(pout) > toolpath_width: # Use toolpath_width as tolerance
                 print(f"      Warning: Final point {current_point.wkt[:30]} is far from Pout {pout.wkt[:30]}. Distance: {current_point.distance(pout):.3f}")
                 # Add a final jump to Pout
                 final_coords.extend(get_coords(pout))
            break # Exit while loop

    if iteration >= max_iterations:
         print("    Error: Max iterations reached during outward phase. Aborting.")
         return None

    # --- Final Path Construction ---
    if len(final_coords) >= 2:
        # Clean up potential duplicate consecutive points
        cleaned_coords = [final_coords[0]]
        for pt in final_coords[1:]:
            # Use a slightly larger tolerance for cleaning
            if Point(cleaned_coords[-1]).distance(Point(pt)) > 1e-6:
                cleaned_coords.append(pt)

        if len(cleaned_coords) >= 2:
             print(f"  Successfully generated Fermat spiral segment. Points: {len(cleaned_coords)}")
             return LineString(cleaned_coords)
        else:
             print("    Error: Not enough unique points generated for Fermat spiral segment.")
             return None
    else:
        print("    Error: Not enough points generated for Fermat spiral segment.")
        return None


def merge_paths_at_branch(branch_node_id: str,
                          parent_node_id: Optional[str], # The node towards which the merged path should exit
                          child_segments: Dict[str, Optional[LineString]], # Keyed by child node ID
                          contours_map: Dict[str, ContourData],
                          toolpath_width: float,
                          mst: MST) -> Optional[LineString]: # Added MST to find connection points
    """
    Merges multiple incoming path segments at a branch point contour by connecting
    endpoints along the branch contour's exterior.

    Args:
        branch_node_id (str): The ID of the branch contour node.
        parent_node_id (Optional[str]): ID of the parent node in the MST (exit direction).
        child_segments (Dict[str, Optional[LineString]]): Dict mapping child node ID to the
                                                          path segment arriving from that child.
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.
        mst (MST): The MST graph to determine connection points.

    Returns:
        Optional[LineString]: The merged path segment leading towards the parent, or None if failed.
    """
    print(f"  Merging {len(child_segments)} segments at branch node {branch_node_id} -> parent {parent_node_id}")

    branch_contour_data = contours_map.get(branch_node_id)
    if not branch_contour_data:
        print(f"    Error: Branch contour data not found for {branch_node_id}")
        return None
    poly_branch = branch_contour_data['polygon']
    if poly_branch.is_empty:
        print(f"    Error: Branch polygon {branch_node_id} is empty.")
        return None
    branch_exterior = poly_branch.exterior

    # --- Determine Connection Points on Branch Contour ---
    connection_points = {} # Map child_id/parent_id to Point on branch_exterior

    # 1. Entry points from children
    valid_incoming = {}
    for child_id, segment in child_segments.items():
        if segment and isinstance(segment, LineString) and not segment.is_empty:
            child_end_point = Point(segment.coords[-1])
            # Find nearest point on branch exterior to child's endpoint
            child_contour_data = contours_map.get(child_id)
            if child_contour_data:
                 poly_child = child_contour_data['polygon']
                 # Use nearest points between contours for connection point
                 nearest = nearest_points(branch_exterior, poly_child.exterior)
                 if nearest and len(nearest) == 2:
                      entry_point = nearest[0] # Point on branch contour
                      connection_points[child_id] = entry_point
                      valid_incoming[child_id] = {'segment': segment, 'entry_point': entry_point}
                      print(f"      Entry point for child {child_id}: {entry_point.wkt[:30]}")
                 else:
                      print(f"      Warning: Could not find nearest points for child {child_id}. Using projection of child endpoint.")
                      entry_point = branch_exterior.interpolate(branch_exterior.project(child_end_point))
                      connection_points[child_id] = entry_point
                      valid_incoming[child_id] = {'segment': segment, 'entry_point': entry_point}
            else:
                 print(f"      Warning: Missing contour data for child {child_id}. Skipping.")
        else:
             print(f"    Warning: Invalid or missing segment from child {child_id}")

    if not valid_incoming:
        print("    Error: No valid incoming segments to merge.")
        return None

    # 2. Exit point towards parent
    exit_point = None
    if parent_node_id:
        parent_contour_data = contours_map.get(parent_node_id)
        if parent_contour_data:
            poly_parent = parent_contour_data['polygon']
            if not poly_parent.is_empty:
                nearest = nearest_points(branch_exterior, poly_parent.exterior)
                if nearest and len(nearest) == 2:
                    exit_point = nearest[0] # Point on branch contour closest to parent
                    connection_points[parent_node_id] = exit_point
                    print(f"      Exit point for parent {parent_node_id}: {exit_point.wkt[:30]}")
                else:
                     print(f"      Warning: Could not find nearest points for parent {parent_node_id}. Using centroid projection.")
                     exit_point = branch_exterior.interpolate(branch_exterior.project(poly_parent.centroid))
                     connection_points[parent_node_id] = exit_point
            else:
                 print(f"    Warning: Parent polygon {parent_node_id} is empty.")
        else:
             print(f"    Warning: Parent contour data not found for {parent_node_id}")
    if not exit_point:
         # If no parent (e.g., root node is the branch), we still need to connect children.
         # The path effectively ends after connecting the children.
         print(f"    Branch node {branch_node_id} is the root or has no parent. Path will end here.")
         # Use the entry point of the "last" child processed as the nominal end? Or just return the combined path?


    # --- Combine Paths by Connecting Along Branch Exterior ---
    # Order matters here. We need a consistent traversal order.
    # Let's order connection points along the branch exterior line.
    connection_order = list(valid_incoming.keys()) # Start with children
    if exit_point:
         connection_order.append(parent_node_id) # Add parent (exit) if exists

    # Calculate projected distance for sorting
    point_distances = {node_id: branch_exterior.project(connection_points[node_id]) for node_id in connection_order}

    # Sort node IDs based on their projected distance along the branch exterior
    # We need a starting point. Let's pick the first child arbitrarily.
    start_node = list(valid_incoming.keys())[0]
    sorted_nodes = sorted(connection_order, key=lambda node_id: (point_distances[node_id] - point_distances[start_node] + branch_exterior.length) % branch_exterior.length)

    print(f"      Connection order on branch: {sorted_nodes}")

    all_merged_coords = []
    last_node_on_branch = None

    for i, current_node_id in enumerate(sorted_nodes):
        current_connection_point = connection_points[current_node_id]

        if current_node_id in valid_incoming: # This is an incoming child segment
             print(f"        Processing incoming child: {current_node_id}")
             segment_data = valid_incoming[current_node_id]
             incoming_segment = segment_data['segment']
             entry_point = segment_data['entry_point'] # Point on branch where child connects

             # 1. Add the incoming segment itself
             segment_coords = get_coords(incoming_segment)
             if not all_merged_coords:
                 all_merged_coords.extend(segment_coords)
                 print(f"          Added initial segment from {current_node_id} (len {len(segment_coords)})")
             else:
                 # 2. Connect previous point on branch to this entry point along branch exterior
                 prev_connection_point = connection_points[last_node_on_branch]
                 print(f"          Connecting {last_node_on_branch} ({prev_connection_point.wkt[:30]}) to {current_node_id} ({entry_point.wkt[:30]}) along branch")
                 connecting_segment_coords = get_line_segment_coords(branch_exterior, prev_connection_point, entry_point)
                 if connecting_segment_coords and len(connecting_segment_coords) >= 2:
                      # Append connecting segment, avoiding duplicates
                      if Point(all_merged_coords[-1]).distance(Point(connecting_segment_coords[0])) < 1e-3:
                           all_merged_coords.extend(connecting_segment_coords[1:])
                      else:
                           all_merged_coords.extend(connecting_segment_coords)
                 else:
                      print(f"          Warning: Failed to get connecting segment on branch. Adding jump.")
                      if Point(all_merged_coords[-1]).distance(entry_point) > 1e-3:
                           all_merged_coords.extend(get_coords(entry_point))

                 # 3. Append the incoming child segment (reversed?) - No, child segment leads *to* entry point.
                 # We need to ensure the last point of the combined path matches the entry point.
                 # The incoming segment should already be part of the path leading *up to* this branch.
                 # We are constructing the path *from* this branch onwards.
                 # Let's rethink: We receive paths ending at entry_point. We need to connect them via branch_exterior
                 # and then generate the path towards the parent.

                 # --- Revised Logic ---
                 # We need to append the incoming segments in the correct order,
                 # connected by segments along the branch exterior.

                 # If this is the first segment being added:
                 if last_node_on_branch is None:
                      all_merged_coords.extend(segment_coords)
                 else:
                      # Connect previous entry point to current entry point along branch exterior
                      prev_entry_point = connection_points[last_node_on_branch]
                      current_entry_point = entry_point
                      connecting_coords = get_line_segment_coords(branch_exterior, prev_entry_point, current_entry_point)
                      if connecting_coords and len(connecting_coords) >= 2:
                           # Check if the start of connecting segment matches end of previous path
                           if Point(all_merged_coords[-1]).distance(Point(connecting_coords[0])) < 1e-3:
                                all_merged_coords.extend(connecting_coords[1:])
                           else:
                                print(f"          Warning: Mismatch connecting branch segment. Appending.")
                                all_merged_coords.extend(connecting_coords)
                      else:
                           print(f"          Warning: Failed connecting segment on branch. Jumping.")
                           if Point(all_merged_coords[-1]).distance(current_entry_point) > 1e-3:
                                all_merged_coords.extend(get_coords(current_entry_point))

                      # Now, append the actual incoming segment (it should end at current_entry_point)
                      # This seems wrong - we are double-backing.

                      # --- Let's try again: Build the path segment that will be *returned* upwards ---
                      # This segment starts at the entry point of the first child processed,
                      # connects along the branch exterior to other children's entry points,
                      # and finally connects to the exit point towards the parent.
                      # The incoming child segments themselves are *not* part of this returned segment.

                      # If this is the first node in the sorted list:
                      if i == 0:
                           # Start the path at its connection point
                           all_merged_coords.extend(get_coords(current_connection_point))
                      else:
                           # Connect previous connection point to current one along branch exterior
                           prev_connection_point = connection_points[sorted_nodes[i-1]]
                           connecting_coords = get_line_segment_coords(branch_exterior, prev_connection_point, current_connection_point)
                           if connecting_coords and len(connecting_coords) >= 2:
                                if Point(all_merged_coords[-1]).distance(Point(connecting_coords[0])) < 1e-3:
                                     all_merged_coords.extend(connecting_coords[1:])
                                else:
                                     print(f"          Warning: Mismatch connecting branch segment {i}. Appending.")
                                     all_merged_coords.extend(connecting_coords)
                           else:
                                print(f"          Warning: Failed connecting segment {i} on branch. Jumping.")
                                if Point(all_merged_coords[-1]).distance(current_connection_point) > 1e-3:
                                     all_merged_coords.extend(get_coords(current_connection_point))


        elif current_node_id == parent_node_id: # This is the exit point
             print(f"        Processing exit point: {parent_node_id}")
             # Connect the last processed node's point to the exit point along branch exterior
             prev_connection_point = connection_points[last_node_on_branch]
             print(f"          Connecting {last_node_on_branch} ({prev_connection_point.wkt[:30]}) to exit ({exit_point.wkt[:30]}) along branch")
             connecting_segment_coords = get_line_segment_coords(branch_exterior, prev_connection_point, exit_point)
             if connecting_segment_coords and len(connecting_segment_coords) >= 2:
                  if Point(all_merged_coords[-1]).distance(Point(connecting_segment_coords[0])) < 1e-3:
                       all_merged_coords.extend(connecting_segment_coords[1:])
                  else:
                       all_merged_coords.extend(connecting_segment_coords)
             else:
                  print(f"          Warning: Failed to get connecting segment to exit. Adding jump.")
                  if Point(all_merged_coords[-1]).distance(exit_point) > 1e-3:
                       all_merged_coords.extend(get_coords(exit_point))
             # Path segment generation ends here.

        last_node_on_branch = current_node_id


    # --- Final Path Construction ---
    if len(all_merged_coords) >= 2:
        # Clean up potential duplicate consecutive points
        cleaned_coords = [all_merged_coords[0]]
        for pt in all_merged_coords[1:]:
            if Point(cleaned_coords[-1]).distance(Point(pt)) > 1e-6:
                cleaned_coords.append(pt)

        if len(cleaned_coords) >= 2:
             print(f"    Generated merged segment on branch {branch_node_id}. Points: {len(cleaned_coords)}")
             return LineString(cleaned_coords)
        else:
             print("    Error: Not enough unique points generated for merged branch segment.")
             return None
    else:
        print("    Error: Not enough points generated for merged branch segment.")
        # This might happen if there's only one child and no parent (root branch)
        # In this case, maybe return the child segment directly?
        if len(valid_incoming) == 1 and not parent_node_id:
             print("    Returning single child segment as branch is root.")
             return list(valid_incoming.values())[0]['segment']
        return None


# --------------------------------------------------------------------------
# Visualization Helpers
# --------------------------------------------------------------------------

def plot_contours_and_mst(contours_data: List[ContourData],
                          mst: Optional[MST],
                          toolpath_width: float,
                          mst_structure: Optional[Dict[str, Any]]):
    """ Plots contours and the MST, highlighting paths and special nodes if structure is provided. """
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    num_levels = max(cd['i'] for cd in contours_data) if contours_data else 1
    contours_map = {cd['id']: cd for cd in contours_data} # Helper map

    # Plot contours
    for cd in contours_data:
        poly = cd['polygon']
        level = cd['i']
        color = cmap(level / max(num_levels, 1))
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=1, label=f"c_{level}_{cd['j']}" if level > 0 else "Boundary (c_0_0)")
        for interior in poly.interiors:
            x_int, y_int = interior.xy
            ax.plot(x_int, y_int, color=color, linestyle='--', linewidth=1)
        # Plot centroid for node reference
        centroid = poly.centroid
        centroid = poly.centroid
        # Default node style
        marker = '.'
        markersize = 4
        markercolor = 'black'
        # Highlight special nodes if structure provided
        if mst_structure:
            if cd['id'] == mst_structure['root_node_id']:
                marker = 'H' # Hexagon for root
                markersize = 8
                markercolor = 'purple'
            elif cd['id'] in mst_structure['branch_points']:
                marker = 's' # Square for branch
                markersize = 7
                markercolor = 'red'
            elif cd['id'] in mst_structure['leaf_nodes']:
                marker = '^' # Triangle for leaf
                markersize = 7
                markercolor = 'green'

        ax.plot(centroid.x, centroid.y, marker=marker, color=markercolor, markersize=markersize, linestyle='None')
        ax.text(centroid.x + 0.1, centroid.y + 0.1, cd['id'], fontsize=8, color='black', ha='left', va='bottom')


    # Plot MST edges
    if mst:
        plotted_edges = set() # Keep track of plotted edges to avoid duplicates
        # Plot path edges first if structure is available
        if mst_structure:
            path_cmap = plt.get_cmap('cool') # Use a different colormap for paths
            num_paths = len(mst_structure['paths'])
            for i, path in enumerate(mst_structure['paths']):
                path_color = path_cmap(i / max(1, num_paths))
                for k in range(len(path) - 1):
                    u, v = path[k], path[k+1]
                    edge = tuple(sorted((u, v)))
                    if edge not in plotted_edges:
                        node_u_data = contours_map.get(u)
                        node_v_data = contours_map.get(v)
                        if node_u_data and node_v_data:
                            centroid_u = node_u_data['polygon'].centroid
                            centroid_v = node_v_data['polygon'].centroid
                            ax.plot([centroid_u.x, centroid_v.x], [centroid_u.y, centroid_v.y],
                                    color=path_color, linewidth=1.5, alpha=0.9, linestyle='-')
                            plotted_edges.add(edge)

        # Plot any remaining MST edges (shouldn't be any if paths cover all, but just in case)
        for u, v, data in mst.edges(data=True):
             edge = tuple(sorted((u, v)))
             if edge not in plotted_edges:
                 node_u_data = contours_map.get(u)
                 node_v_data = contours_map.get(v)
                 if node_u_data and node_v_data:
                     centroid_u = node_u_data['polygon'].centroid
                     centroid_v = node_v_data['polygon'].centroid
                     # Plot these with a default style (e.g., dashed grey)
                     ax.plot([centroid_u.x, centroid_v.x], [centroid_u.y, centroid_v.y],
                             color='grey', linewidth=0.8, alpha=0.7, linestyle=':')
                     plotted_edges.add(edge)


    ax.set_aspect('equal', adjustable='box')
    title = f"Iso-Contours and MST (w={toolpath_width})"
    if mst_structure:
        title += " - Structure Highlighted"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # plt.legend() # Legend might get too crowded
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show(block=False) # Use non-blocking show for potentially multiple plots


def plot_contours_and_path(contours_data: List[ContourData],
                           mst: Optional[MST],
                           final_path: Optional[LineString],
                           toolpath_width: float,
                           mst_structure: Optional[Dict[str, Any]]): # Add mst_structure parameter
    """ Plots contours, the MST structure (if provided), and the final generated path. """
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    num_levels = max(cd['i'] for cd in contours_data) if contours_data else 1
    contours_map = {cd['id']: cd for cd in contours_data} # Helper map

    # Plot contours (lightly)
    for cd in contours_data:
        poly = cd['polygon']
        level = cd['i']
        color = cmap(level / max(num_levels, 1))
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=0.5, alpha=0.6) # Lighter contours
        for interior in poly.interiors:
            x_int, y_int = interior.xy
            ax.plot(x_int, y_int, color=color, linestyle='--', linewidth=0.5, alpha=0.6)

    # Plot MST edges (optional, can be noisy)
    # if mst:
    #     for u, v, data in mst.edges(data=True):
    #         node_u_data = next((cd for cd in contours_data if cd['id'] == u), None)
    #         node_v_data = next((cd for cd in contours_data if cd['id'] == v), None)
    #         if node_u_data and node_v_data:
    #             poly_u = node_u_data['polygon']
    #             poly_v = node_v_data['polygon']
    #             centroid_u = poly_u.centroid
    #             centroid_v = poly_v.centroid
    #             ax.plot([centroid_u.x, centroid_v.x], [centroid_u.y, centroid_v.y], 'r-', linewidth=0.5, alpha=0.4)

    # Plot the final path
    if final_path and isinstance(final_path, LineString) and not final_path.is_empty:
        x_path, y_path = final_path.xy
        ax.plot(x_path, y_path, 'b-', linewidth=1.2, label='Generated Path', zorder=10) # Blue, solid line, ensure it's on top

    ax.set_aspect('equal', adjustable='box')
    title = f"Contours, MST Structure, and Generated Path (w={toolpath_width})"
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    if final_path:
        ax.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show(block=False) # Use non-blocking show


# --------------------------------------------------------------------------
# Example Usage (for testing within this file)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Create a sample polygon (e.g., a rounded rectangle)
    # sample_polygon = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]).buffer(1, join_style=2).buffer(-1, join_style=2) # Simple rectangle
    # sample_polygon = Polygon([(0,0), (10,0), (10,10), (0,10)]).buffer(2).buffer(-1) # Rounded square
    sample_polygon = Polygon([(0,0), (20,0), (15,10), (5,10)]).buffer(1.5, join_style=2) # Trapezoid-like shape
    
    # More complex shape - circle with an off-center hole
    #outer = Polygon([(0,0), (10,0), (10,10), (0,10)]).buffer(5) # Outer circle approx
    #inner = Polygon([(3,3), (7,3), (7,7), (3,7)]).buffer(1) # Inner hole approx
    #sample_polygon = outer.difference(inner)


    print("Testing CFS Fill Generation...")
    # Use a slightly larger toolpath width for fewer contours in testing
    test_toolpath_width = 0.2
    cfs_path = generate_cfs_fill(sample_polygon, toolpath_width=test_toolpath_width)

    if cfs_path:
        print("\nCFS Path generated (see plot).")
        # Visualization is now handled within generate_cfs_fill or by plot_contours_and_path
        # Keep the script running to see the plot
        plt.show() # Add a blocking show() call at the end if needed
    else:
        print("\nCFS Path generation did not complete (see plot for contours/MST).")
        # Keep the script running to see the plot
        plt.show() # Add a blocking show() call at the end if needed
