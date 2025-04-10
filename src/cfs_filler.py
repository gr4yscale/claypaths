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
            # Generate spiral for the single path segment connecting leaf to its parent
            print(f"      Node {node_id} is a leaf. Generating initial spiral segment towards {parent_id}.")
            path_nodes = [node_id, parent_id] if parent_id else [node_id]
            if len(path_nodes) > 1:
                 # Call the (simplified) spiral generation function
                 outgoing_segment = generate_spiral_segment(path_nodes, contours_map, toolpath_width)
                 if outgoing_segment:
                     print(f"        -> Generated segment for leaf {node_id} (Length: {outgoing_segment.length:.2f})")
                 else:
                     print(f"        -> Failed to generate segment for leaf {node_id}")

        elif is_branch:
            # Merge incoming paths from children and generate outgoing path towards parent
            print(f"      Node {node_id} is a branch point. Merging {len(child_segments)} segments.")
            # Pass the dictionary of segments received from children recursive calls
            outgoing_segment = merge_paths_at_branch(node_id, parent_id, child_segments, contours_map, toolpath_width)
            if outgoing_segment:
                 print(f"        -> Merged segments at branch {node_id} (Result Length: {outgoing_segment.length:.2f})")
            else:
                 print(f"        -> Failed to merge segments at branch {node_id}")

        else: # Degree 2 node (part of a path)
            print(f"      Node {node_id} is a path node (degree 2).")
            # Should have exactly one child (neighbor that is not parent)
            child_id = children[0] if children else None
            incoming_segment = child_segments.get(child_id) if child_id else None

            if incoming_segment:
                # --- Extend the incoming path ---
                # --- Extend the incoming path ---
                # The endpoint of the incoming segment is where the extension should start.
                print(f"        -> Extending segment from child {child_id} towards parent {parent_id}.")
                start_override = Point(incoming_segment.coords[-1]) # Endpoint of incoming path
                print(f"        -> Using start override: {start_override.wkt[:30]}")

                path_nodes_extension = [node_id, parent_id] if parent_id else [node_id]
                if len(path_nodes_extension) > 1:
                    # Call generate_spiral_segment WITH the override
                    extension_segment = generate_spiral_segment(
                        path_nodes_extension,
                        contours_map,
                        toolpath_width,
                        start_point_override=start_override # Pass the override
                    )
                    if extension_segment:
                        # Combine incoming_segment and extension_segment
                        # Simplification: Assume they connect correctly for now.
                        coords_incoming = get_coords(incoming_segment)
                        coords_extension = get_coords(extension_segment)
                        # Check if connection point matches (approximately)
                        if coords_incoming and coords_extension and Point(coords_incoming[-1]).distance(Point(coords_extension[0])) < 1e-3:
                             combined_coords = coords_incoming + coords_extension[1:]
                        else:
                             # If endpoints don't match, just append (will create a jump)
                             print(f"        -> Warning: Endpoints don't match for path extension at {node_id}. Appending segments.")
                             combined_coords = coords_incoming + coords_extension

                        if len(combined_coords) >= 2:
                             outgoing_segment = LineString(combined_coords)
                             print(f"        -> Extended path segment (New Length: {outgoing_segment.length:.2f})")
                        else:
                             print(f"        -> Failed to combine segments for extension.")
                             outgoing_segment = incoming_segment # Fallback to just incoming
                    else:
                         print(f"        -> Failed to generate extension segment. Passing incoming segment.")
                         outgoing_segment = incoming_segment # Pass through if extension fails
                else:
                     outgoing_segment = incoming_segment # No parent, just return incoming
            else:
                # No incoming segment (e.g., child failed or is leaf), generate new segment if possible
                # No incoming segment (e.g., child failed or is leaf), generate new segment if possible
                # This case shouldn't happen for a degree-2 node in a properly formed MST originating from a leaf,
                # unless the leaf segment generation failed.
                print(f"        -> Warning: No incoming segment from child {child_id} at degree-2 node {node_id}. Generating new segment towards {parent_id}.")
                path_nodes = [node_id, parent_id] if parent_id else [node_id]
                if len(path_nodes) > 1:
                     # No incoming segment, so no override is possible/needed.
                     outgoing_segment = generate_spiral_segment(path_nodes, contours_map, toolpath_width) # Call without override
                     if outgoing_segment:
                          print(f"        -> Generated new segment directly (Length: {outgoing_segment.length:.2f})")
                     else:
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


def generate_spiral_segment(path_nodes: PathSegment,
                            contours_map: Dict[str, ContourData],
                            toolpath_width: float,
                            start_point_override: Optional[Point] = None) -> Optional[LineString]:
    """
    Generates a path segment connecting a sequence of contours (a path in the MST).
    (Simplified Implementation: Connects nearest points and traces contour exteriors).

    Args:
        path_nodes (PathSegment): List of contour node IDs forming the path (e.g., [leaf, ..., parent]).
                                  Assumed to be ordered from inner to outer contour generally.
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.
        start_point_override (Optional[Point]): If provided, forces the segment on the first
                                                 contour (`path_nodes[0]`) to start at this exact point.

    Args:
        path_nodes (PathSegment): List of contour node IDs forming the path (e.g., [leaf, ..., parent]).
                                  Assumed to be ordered from inner to outer contour generally.
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.
        # start_point_override removed

    Returns:
        Optional[LineString]: The generated path segment, or None if failed.
    """
    print(f"  Generating segment for path: {' -> '.join(path_nodes)}")
    if len(path_nodes) < 2:
        print("    Path needs at least two nodes to generate a segment.")
        return None

    all_segment_coords = []
    last_connection_point_on_outer = None

    # Iterate through pairs of adjacent contours in the path
    for i in range(len(path_nodes) - 1):
        node_id_inner = path_nodes[i]
        node_id_outer = path_nodes[i+1]

        contour_inner_data = contours_map.get(node_id_inner)
        contour_outer_data = contours_map.get(node_id_outer)

        if not contour_inner_data or not contour_outer_data:
            print(f"    Error: Contour data not found for {node_id_inner} or {node_id_outer}")
            return None

        poly_inner = contour_inner_data['polygon']
        poly_outer = contour_outer_data['polygon']

        if poly_inner.is_empty or poly_outer.is_empty:
             print(f"    Error: Empty polygon for {node_id_inner} or {node_id_outer}")
             return None

        # Find nearest points between the exteriors (simplified connection points)
        try:
            nearest = nearest_points(poly_inner.exterior, poly_outer.exterior)
            if not nearest or len(nearest) != 2:
                 print(f"    Error: Could not find nearest points between {node_id_inner} and {node_id_outer}")
                 # Fallback: use centroids?
                 p_inner = poly_inner.centroid
                 p_outer = poly_outer.centroid
                 # return None
            else:
                 p_inner, p_outer = nearest[0], nearest[1]
        except Exception as e:
             print(f"    Error finding nearest points between {node_id_inner} and {node_id_outer}: {e}")
             # Fallback: use centroids
             p_inner = poly_inner.centroid
             p_outer = poly_outer.centroid
             # return None


        # --- Add segment on the inner contour (poly_inner, node_id_inner) ---
        start_point_this_contour = None
        if i == 0 and start_point_override:
             # Use the override for the very first contour of this path segment call
             start_point_this_contour = start_point_override
             print(f"    Using override start point on {node_id_inner}: {start_point_this_contour.wkt[:30]}")
        elif i > 0:
             # Use the connection point from the previous outer contour
             start_point_this_contour = last_connection_point_on_outer # From previous iteration's p_outer
             print(f"    Connecting from previous point on {node_id_inner}: {start_point_this_contour.wkt[:30]}")
        # Else (i == 0 and no override): This is the leaf/innermost case, handled below

        if start_point_this_contour:
            # Generate segment from start_point_this_contour to p_inner (nearest point towards outer)
            print(f"      -> Generating segment on {node_id_inner} from specified start to nearest point {p_inner.wkt[:30]}")
            segment_coords_inner = get_line_segment_coords(poly_inner.exterior, start_point_this_contour, p_inner)
            if segment_coords_inner:
                # Avoid duplicate points if segments connect
                if all_segment_coords and Point(all_segment_coords[-1]).distance(Point(segment_coords_inner[0])) < 1e-3:
                     all_segment_coords.extend(segment_coords_inner[1:])
                elif not all_segment_coords and start_point_override and start_point_override.distance(Point(segment_coords_inner[0])) > 1e-3:
                     # If this is the very start and the segment doesn't start exactly at the override, add the override point first
                     print(f"      -> Prepending override point to segment on {node_id_inner}")
                     all_segment_coords.extend(get_coords(start_point_override))
                     if Point(all_segment_coords[-1]).distance(Point(segment_coords_inner[0])) < 1e-3:
                          all_segment_coords.extend(segment_coords_inner[1:])
                     else:
                          all_segment_coords.extend(segment_coords_inner)
                else:
                     all_segment_coords.extend(segment_coords_inner)
            else:
                 print(f"    Warning: Could not get segment on inner contour {node_id_inner} from specified start. Adding jump.")
                 # Add jump if segment fails: ensure start point is added, then add p_inner
                 if not all_segment_coords or Point(all_segment_coords[-1]).distance(start_point_this_contour) > 1e-3:
                      all_segment_coords.extend(get_coords(start_point_this_contour))
                 if Point(all_segment_coords[-1]).distance(p_inner) > 1e-3:
                      all_segment_coords.extend(get_coords(p_inner))

        elif i == 0: # No override provided, and it's the first contour (leaf case)
            # Simplification: Trace the *entire* inner contour exterior starting near p_inner.
            print(f"    Tracing entire innermost contour {node_id_inner} (no override) starting near {p_inner.wkt[:30]}.")
            inner_coords = get_coords(poly_inner.exterior)
            if inner_coords:
                 start_idx = find_point_index(poly_inner.exterior, p_inner)
                 if start_idx == -1: start_idx = 0 # Fallback if point not found
                 # Rotate coords to start near p_inner and ensure it's closed
                 rotated_coords = inner_coords[start_idx:-1] + inner_coords[:start_idx+1]
                 # Ensure the start point is added if not already present
                 if not all_segment_coords:
                      all_segment_coords.extend(rotated_coords)
                 elif Point(all_segment_coords[-1]).distance(Point(rotated_coords[0])) < 1e-3:
                      all_segment_coords.extend(rotated_coords[1:])
                 else:
                      all_segment_coords.extend(rotated_coords)
            else:
                 print(f"    Warning: Could not get coordinates for inner contour {node_id_inner}")


        # --- Add connecting line between contours ---
        # Connect p_inner to p_outer
        print(f"    Adding connection from {node_id_inner} ({p_inner.wkt[:30]}) to {node_id_outer} ({p_outer.wkt[:30]})")
        # Avoid duplicate points
        if not all_segment_coords or Point(all_segment_coords[-1]).distance(p_inner) > 1e-3:
             all_segment_coords.extend(get_coords(p_inner))
        if Point(all_segment_coords[-1]).distance(p_outer) > 1e-3:
             all_segment_coords.extend(get_coords(p_outer))

        # Update the connection point for the next iteration
        last_connection_point_on_outer = p_outer


    # After loop: Add the final segment on the outermost contour of this path
    # Connect from last_connection_point_on_outer to ... where? The path ends here for this segment.
    # Let's just leave the path ending at last_connection_point_on_outer.
    # The connection to the next part of the overall path (e.g. at a branch or parent path node)
    # will be handled by the calling function (_process_node or merge_paths_at_branch).

    if len(all_segment_coords) >= 2:
        # Clean up potential duplicate consecutive points
        cleaned_coords = [all_segment_coords[0]]
        for pt in all_segment_coords[1:]:
            if Point(cleaned_coords[-1]).distance(Point(pt)) > 1e-6:
                cleaned_coords.append(pt)

        if len(cleaned_coords) >= 2:
             return LineString(cleaned_coords)
        else:
             print("    Error: Not enough unique points generated for segment.")
             return None
    else:
        print("    Error: Not enough points generated for segment.")
        return None


def merge_paths_at_branch(branch_node_id: str,
                          parent_node_id: Optional[str], # The node towards which the merged path should exit
                          child_segments: Dict[str, Optional[LineString]], # Keyed by child node ID
                          contours_map: Dict[str, ContourData],
                          toolpath_width: float) -> Optional[LineString]:
    """
    Merges multiple incoming path segments at a branch point contour.
    (Simplified Implementation: Connects endpoints via branch centroid).

    Args:
        branch_node_id (str): The ID of the branch contour node.
        parent_node_id (Optional[str]): ID of the parent node in the MST (exit direction).
        child_segments (Dict[str, Optional[LineString]]): Dict mapping child node ID to the
                                                          path segment arriving from that child.
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.

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
    branch_centroid = poly_branch.centroid

    # Identify valid incoming segments and their endpoints
    valid_incoming = {}
    for child_id, segment in child_segments.items():
        if segment and isinstance(segment, LineString) and not segment.is_empty:
            # Assume the segment ends near the branch node
            valid_incoming[child_id] = {'segment': segment, 'end_point': Point(segment.coords[-1])}
        else:
             print(f"    Warning: Invalid or missing segment from child {child_id}")

    if not valid_incoming:
        print("    Error: No valid incoming segments to merge.")
        # If there's a parent, maybe generate a path just from centroid to parent connection?
        return None

    # Determine the exit point towards the parent (if exists)
    exit_point = None
    if parent_node_id:
        parent_contour_data = contours_map.get(parent_node_id)
        if parent_contour_data:
            poly_parent = parent_contour_data['polygon']
            if not poly_parent.is_empty:
                # Find nearest point on branch contour to parent contour (simplified exit point)
                nearest = nearest_points(poly_branch.exterior, poly_parent.exterior)
                if nearest and len(nearest) == 2:
                    exit_point = nearest[0] # Point on branch contour closest to parent
                else:
                     # Fallback: point on branch contour closest to parent centroid
                     exit_point = nearest_points(poly_branch.exterior, poly_parent.centroid)[0]
            else:
                 print(f"    Warning: Parent polygon {parent_node_id} is empty.")
        else:
             print(f"    Warning: Parent contour data not found for {parent_node_id}")
    if not exit_point:
         # If no parent or parent invalid, use centroid as nominal exit? Or endpoint of one child?
         print(f"    Warning: Could not determine exit point towards parent {parent_node_id}. Using centroid.")
         exit_point = branch_centroid


    # --- Combine paths (Simplified: connect all via centroid) ---
    # Order of merging might matter for efficiency but not for this simple connection.
    # We will connect: incoming_end -> centroid -> next_incoming_end -> centroid -> ... -> exit_point
    all_merged_coords = []
    last_point = None

    # Process incoming segments one by one
    processed_children = set()
    current_child_id = list(valid_incoming.keys())[0] # Start with the first child

    while len(processed_children) < len(valid_incoming):
         if current_child_id in processed_children:
             # Find next unprocessed child (shouldn't happen if logic is right)
             found_next = False
             for cid in valid_incoming:
                 if cid not in processed_children:
                     current_child_id = cid
                     found_next = True
                     break
             if not found_next: break # All processed

         data = valid_incoming[current_child_id]
         segment = data['segment']
         end_point = data['end_point'] # End point of the incoming segment

         # Add the incoming segment coords
         segment_coords = get_coords(segment)
         if not all_merged_coords:
             all_merged_coords.extend(segment_coords)
         else:
             # Try to connect smoothly, otherwise just append
             if Point(all_merged_coords[-1]).distance(Point(segment_coords[0])) < 1e-3:
                 all_merged_coords.extend(segment_coords[1:])
             else:
                 # This case implies the previous connection (via centroid) didn't end
                 # where this segment starts. This indicates flaw in centroid connection logic.
                 # For now, just append.
                 print(f"    Warning: Mismatch connecting segments via centroid at branch {branch_node_id}.")
                 all_merged_coords.extend(segment_coords)

         last_point = Point(all_merged_coords[-1]) # Should be close to end_point

         # Add connection from end_point to centroid
         if last_point.distance(branch_centroid) > 1e-6:
              all_merged_coords.extend(get_coords(branch_centroid))
         last_point = branch_centroid
         processed_children.add(current_child_id)

         # Find the next child to connect to (simplification: just take the next unprocessed)
         next_child_id = None
         for cid in valid_incoming:
             if cid not in processed_children:
                 next_child_id = cid
                 break

         if next_child_id:
              # Add connection from centroid to the start/end of the next child's segment
              next_end_point = valid_incoming[next_child_id]['end_point']
              # Connect centroid to that end point
              if last_point.distance(next_end_point) > 1e-6:
                   all_merged_coords.extend(get_coords(next_end_point))
              last_point = next_end_point
              current_child_id = next_child_id
         # else: all children processed, loop will end.


    # Finally, connect the last point (centroid) to the exit point
    if last_point and exit_point and last_point.distance(exit_point) > 1e-6:
         all_merged_coords.extend(get_coords(exit_point))

    if len(all_merged_coords) >= 2:
        # Clean up potential duplicate consecutive points
        cleaned_coords = [all_merged_coords[0]]
        for pt in all_merged_coords[1:]:
            if Point(cleaned_coords[-1]).distance(Point(pt)) > 1e-6:
                cleaned_coords.append(pt)

        if len(cleaned_coords) >= 2:
             return LineString(cleaned_coords)
        else:
             print("    Error: Not enough unique points generated for merged segment.")
             return None
    else:
        print("    Error: Not enough points generated for merged segment.")
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
    test_toolpath_width = 1.0
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
