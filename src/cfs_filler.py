from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, nearest_points
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

        else:
             print("  Computing MST on the full graph.")
             spiral_contour_tree = nx.minimum_spanning_tree(connectivity_graph, weight='weight')

        print(f"  MST computed with {spiral_contour_tree.number_of_nodes()} nodes and {spiral_contour_tree.number_of_edges()} edges.")

    except Exception as e:
        print(f"Error computing MST: {e}")
        # Consider visualization or further debugging here
        # plot_graph(connectivity_graph, "Connectivity Graph")
        return None


    # --- Step 6: Identify Spirallable Regions & Branch Points ---
    print("\nStep 6: Identifying Spirallable Regions & Branch Points...")
    mst_structure = identify_mst_structure(spiral_contour_tree, boundary_node['id'])
    if not mst_structure:
        print("Error: Failed to analyze MST structure.")
        plot_contours_and_mst(all_contours_data, spiral_contour_tree, toolpath_width)
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
        plot_contours_and_path(all_contours_data, spiral_contour_tree, final_path, toolpath_width) # Visualize result
        return final_path
    else:
        print("  CFS path generation incomplete or failed in Step 7.")
        # For debugging, visualize the contours and MST
        plot_contours_and_mst(all_contours_data, spiral_contour_tree, toolpath_width)
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
    (Currently a placeholder implementation).

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
    paths = mst_structure['paths']
    branch_points = mst_structure['branch_points']
    contours_map = {cd['id']: cd for cd in contours_data}

    # This requires a bottom-up traversal strategy (from leaves/inner contours towards root)
    # We need to process spirallable paths and merge them at branch points.

    # Placeholder: Just connect centroids of the first path found as a dummy output
    if paths:
        first_path_nodes = paths[0]
        path_coords = []
        for node_id in first_path_nodes:
            contour_data = contours_map.get(node_id)
            if contour_data:
                path_coords.append(contour_data['polygon'].centroid.coords[0])

        if len(path_coords) >= 2:
            print("  (Placeholder: Returning LineString connecting centroids of the first identified path)")
            return LineString(path_coords)
        else:
             print("  (Placeholder: Not enough points in the first path to create a LineString)")
             return None
    else:
        print("  No spirallable paths found to generate even a placeholder path.")
        # Handle the case of a single contour (just the boundary)
        if len(contours_map) == 1 and root_node_id in contours_map:
             print("  Only boundary contour exists. No fill path needed/possible.")
             # Returning the exterior might be an option, but CFS is for filling.
             # return contours_map[root_node_id]['polygon'].exterior
             return None # No fill path
        return None

    # --- Actual Implementation Sketch ---
    # 1. Data Structure: Need to store generated path segments (LineStrings) associated with MST edges or nodes.
    # 2. Traversal Order: Determine a processing order, likely starting from paths connected to leaves.
    #    - Could use a topological sort if viewed as a directed tree from root, then reverse?
    #    - Or, recursive function starting from root, processing children first.
    # 3. Process Spirallable Paths:
    #    - For each path in mst_structure['paths']:
    #        - Call `generate_spiral_segment(path_nodes, contours_map, toolpath_width)`
    #        - This function needs the detailed Fermat spiral logic (inward/outward links, B(p), N(p), rerouting points).
    #        - It should return a LineString representing the spiral for that segment, connecting the entry/exit points defined by MST connections.
    # 4. Process Branch Points:
    #    - When the traversal reaches a branch point where all incoming child paths have been processed:
    #        - Call `merge_paths_at_branch(branch_node_id, incoming_paths, contours_map, toolpath_width)`
    #        - This function needs logic to connect the ends of the incoming spiral segments smoothly within the contour of the branch node. Connection points depend on the MST edges (connecting segments O).
    # 5. Final Path: The result of processing the root node should be the complete path.

    # --- Placeholder Return ---
    # return None # Replace with actual result


def generate_spiral_segment(path_nodes: PathSegment,
                            contours_map: Dict[str, ContourData],
                            toolpath_width: float) -> Optional[LineString]:
    """
    Generates the Fermat spiral segment for a sequence of contours (a path in the MST).
    (Placeholder - Requires detailed implementation based on Section 3 of the paper).

    Args:
        path_nodes (PathSegment): List of contour node IDs forming the path.
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.

    Returns:
        Optional[LineString]: The generated spiral segment, or None if failed.
    """
    print(f"  (Placeholder: Generate spiral segment for path: {' -> '.join(path_nodes)})")
    # TODO: Implement Fermat spiral generation logic here.
    # - Determine inward/outward links based on path direction.
    # - Find rerouting points B(p), N(p) on adjacent contours.
    # - Connect segments according to Figure 5 in the paper.
    # - Needs robust geometric calculations (intersections, projections, etc.).
    return None


def merge_paths_at_branch(branch_node_id: str,
                          incoming_segments: Dict[str, LineString], # Keyed by child node ID
                          contours_map: Dict[str, ContourData],
                          toolpath_width: float) -> Optional[LineString]:
    """
    Merges multiple incoming spiral path segments at a branch point contour.
    (Placeholder - Requires detailed implementation).

    Args:
        branch_node_id (str): The ID of the branch contour node.
        incoming_segments (Dict[str, LineString]): Dictionary mapping the child node ID
                                                   (from which the segment originates)
                                                   to the LineString segment itself.
        contours_map (Dict[str, ContourData]): Map of node IDs to contour data.
        toolpath_width (float): Toolpath width 'w'.

    Returns:
        Optional[LineString]: The merged path segment, or None if failed.
    """
    print(f"  (Placeholder: Merge {len(incoming_segments)} segments at branch node {branch_node_id})")
    # TODO: Implement path merging logic here.
    # - Identify connection points on the branch contour based on MST edges (connecting segments O).
    # - Connect the endpoints of the incoming_segments smoothly.
    # - May involve generating short connecting paths within the branch contour polygon.
    return None


# --------------------------------------------------------------------------
# Visualization Helpers
# --------------------------------------------------------------------------

def plot_contours_and_mst(contours_data: List[ContourData], mst: Optional[MST], toolpath_width: float):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    num_levels = max(cd['i'] for cd in contours_data) if contours_data else 1

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
        ax.plot(centroid.x, centroid.y, 'o', color=color, markersize=3)
        ax.text(centroid.x, centroid.y, cd['id'], fontsize=8, color='black')


    # Plot MST edges
    if mst:
        for u, v, data in mst.edges(data=True):
            node_u_data = next((cd for cd in contours_data if cd['id'] == u), None)
            node_v_data = next((cd for cd in contours_data if cd['id'] == v), None)
            if node_u_data and node_v_data:
                poly_u = node_u_data['polygon']
                poly_v = node_v_data['polygon']
                # Draw line between centroids
                centroid_u = poly_u.centroid
                centroid_v = poly_v.centroid
                ax.plot([centroid_u.x, centroid_v.x], [centroid_u.y, centroid_v.y], 'r-', linewidth=0.8, alpha=0.7)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Iso-Contours and MST (w={toolpath_width})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # plt.legend() # Legend might get too crowded
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show(block=False) # Use non-blocking show for potentially multiple plots


def plot_contours_and_path(contours_data: List[ContourData], mst: Optional[MST], final_path: Optional[LineString], toolpath_width: float):
    """ Plots contours, optionally the MST, and the final generated path. """
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    num_levels = max(cd['i'] for cd in contours_data) if contours_data else 1

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
        ax.plot(x_path, y_path, 'b-', linewidth=1.0, label='Generated Path') # Blue, solid line

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Contours and Generated Path (w={toolpath_width})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
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
    # sample_polygon = Polygon([(0,0), (20,0), (15,10), (5,10)]).buffer(1.5, join_style=2) # Trapezoid-like shape
    
    # More complex shape - circle with an off-center hole
    outer = Polygon([(0,0), (10,0), (10,10), (0,10)]).buffer(5) # Outer circle approx
    inner = Polygon([(3,3), (7,3), (7,7), (3,7)]).buffer(1) # Inner hole approx
    sample_polygon = outer.difference(inner)


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
