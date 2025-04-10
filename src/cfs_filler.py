from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import networkx as nx
import math

def generate_cfs_fill(region: Polygon, toolpath_width: float = 0.4):
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
    # Traverse MST, calculate node degrees.
    # Degree <= 2: Part of spirallable region path.
    # Degree > 2: Branch point.
    degrees = dict(spiral_contour_tree.degree())
    branch_points = [node for node, degree in degrees.items() if degree > 2]
    leaf_nodes = [node for node, degree in degrees.items() if degree == 1] # Exclude root if it's a leaf? Root c_0_0 degree can be 1.

    print(f"  Identified {len(branch_points)} branch points: {branch_points}")
    print(f"  Identified {len(leaf_nodes)} leaf nodes (potential spiral ends): {leaf_nodes}")
    # Further analysis needed to group nodes into spirallable paths between branch/leaf nodes.


    # --- Step 7: Recursive Rerouting (Bottom-Up Traversal of MST) ---
    print("\nStep 7: Recursive Rerouting (Bottom-Up Traversal)...")
    # This is the core logic involving Fermat spiral generation (Section 3/Figure 5)
    # and merging at branch points. Requires detailed implementation.
    # Placeholder for now.
    final_path = None # Placeholder for the resulting LineString
    print("  (Placeholder for Rerouting Logic)")


    # --- Step 8: Output ---
    print("\nStep 8: Outputting Final Path...")
    if final_path:
        print(f"  CFS path generated successfully.")
        # Optional: Simplify or smooth the path (Step 9)
        return final_path
    else:
        print("  CFS path generation incomplete or failed.")
        # For debugging, let's visualize the contours and MST
        plot_contours_and_mst(all_contours_data, spiral_contour_tree, toolpath_width)
        return None


# Helper function for visualization (optional, requires matplotlib)
def plot_contours_and_mst(contours_data, mst, toolpath_width):
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
    plt.show()


# Example Usage (for testing within this file)
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
    cfs_path = generate_cfs_fill(sample_polygon, toolpath_width=0.8)

    if cfs_path:
        # Visualize the final path if generated
        fig, ax = plt.subplots(figsize=(8, 8))
        x_orig, y_orig = sample_polygon.exterior.xy
        ax.plot(x_orig, y_orig, 'k--', label='Original Boundary')
        for interior in sample_polygon.interiors:
             x_int, y_int = interior.xy
             ax.plot(x_int, y_int, 'k:')

        x_path, y_path = cfs_path.xy
        ax.plot(x_path, y_path, 'b-', label='CFS Path')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Generated CFS Path (Placeholder)")
        ax.legend()
        plt.show()
    else:
        print("CFS Path generation did not complete.")
