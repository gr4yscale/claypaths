import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import networkx as nx
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def generate_cfs_fill(region, toolpath_width, visualize_steps=True, debug=True):
    """
    Generate a Continuous Fermat Spiral (CFS) fill for a 2D region.
    
    Args:
        region (Polygon): The 2D region to fill
        toolpath_width (float): The desired spacing between adjacent path segments
        visualize_steps (bool): Whether to visualize intermediate steps
        debug (bool): Whether to print detailed debugging information
        
    Returns:
        tuple: (final_path, contours, mst) - The continuous toolpath, contours and MST
    """
    if not isinstance(region, Polygon) or not region.is_valid:
        logger.error("Input must be a valid Polygon")
        return None, None, None
    
    # Step 1: Generate iso-contours (offset curves)
    logger.info("Step 5: Generating iso-contours...")
    contours = generate_iso_contours(region, toolpath_width)
    if not contours:
        logger.error("Failed to generate iso-contours")
        return None, None, None
    
    if debug:
        logger.info(f"Generated {len(contours)} contour levels")
        for level in contours:
            logger.info(f"  Level {level}: {len(contours[level])} contours")
    
    # Visualize Step 5: Iso-contours
    if visualize_steps:
        visualize_iso_contours(region, contours)
    
    # Step 2: Build connectivity graph and MST
    logger.info("Step 6: Building spiral contour tree...")
    graph, mst = build_spiral_contour_tree(contours)
    if not mst:
        logger.error("Failed to build spiral-contour tree")
        return None, None, None
    
    if debug:
        logger.info(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        logger.info(f"MST has {len(mst.nodes())} nodes and {len(mst.edges())} edges")
        logger.info("MST Nodes:")
        for node in sorted(mst.nodes()):
            logger.info(f"  {node} - Level: {node[0]}, Index: {node[1]}")
        logger.info("MST Edges:")
        for edge in mst.edges(data=True):
            logger.info(f"  {edge[0]} -> {edge[1]}, Weight: {edge[2].get('weight', 'N/A')}")
    
    # Visualize Step 6: Spiral contour tree
    if visualize_steps:
        visualize_spiral_contour_tree(region, contours, graph, mst)
    
    # Step 3: Perform recursive rerouting to generate the final path
    logger.info("Step 7: Performing recursive rerouting...")
    final_path = perform_recursive_rerouting(contours, mst, toolpath_width)
    
    if final_path:
        logger.info(f"Generated continuous path with {len(final_path.coords)} points")
        
        # Visualize Step 7: Recursive rerouting
        if visualize_steps:
            visualize_recursive_rerouting(region, contours, mst, final_path)
    else:
        logger.error("Failed to generate continuous path through recursive rerouting")
    
    return final_path, contours, mst

def generate_iso_contours(region, toolpath_width):
    """
    Generate a set of iso-contours (offset curves) for the region.
    
    Args:
        region (Polygon): The 2D region
        toolpath_width (float): The spacing between contours
        
    Returns:
        dict: A dictionary mapping contour indices to lists of Polygons
    """
    contours = {}
    current_offset = 0
    level = 1
    
    # Start with the original boundary
    contours[level] = [region]
    
    while True:
        current_offset += toolpath_width
        level += 1
        
        # Generate the next offset (negative buffer)
        offset_polygons = []
        for poly in contours[level-1]:
            # Create a negative buffer (inward offset)
            offset = poly.buffer(-toolpath_width, join_style=1)
            
            # Skip if the offset is empty or invalid
            if offset.is_empty:
                continue
                
            # Handle potential MultiPolygon results
            if isinstance(offset, MultiPolygon):
                offset_polygons.extend(list(offset.geoms))
            else:
                offset_polygons.append(offset)
        
        # If no valid offsets were generated, we're done
        if not offset_polygons:
            break
            
        contours[level] = offset_polygons
    
    logger.info(f"Generated {level} contour levels")
    return contours

def build_spiral_contour_tree(contours, debug=True):
    """
    Build a connectivity graph and minimum spanning tree for the contours.
    
    Args:
        contours (dict): Dictionary mapping contour levels to lists of Polygons
        debug (bool): Whether to print detailed debugging information
        
    Returns:
        tuple: (connectivity graph, minimum spanning tree)
    """
    # Create a graph where nodes are contours
    G = nx.Graph()
    
    if debug:
        logger.info("Building connectivity graph...")
    
    # Add nodes for each contour
    for level in contours:
        for j, contour in enumerate(contours[level]):
            node_id = (level, j)
            area = contour.area
            G.add_node(node_id, contour=contour, level=level, area=area)
            if debug and j == 0:
                logger.info(f"  Added nodes for level {level} contours")
    
    if debug:
        logger.info(f"Added {len(G.nodes())} nodes to the graph")
    
    # Add edges between contours at adjacent levels
    edge_count = 0
    for level in range(1, len(contours)):
        if debug:
            logger.info(f"  Processing connections between levels {level} and {level+1}")
        
        level_edge_count = 0
        for j, contour_outer in enumerate(contours[level]):
            for k, contour_inner in enumerate(contours[level+1]):
                # Check if inner contour is contained within outer contour
                if contour_outer.contains(contour_inner):
                    # Find the connecting segment (points on outer closer to inner)
                    connecting_segment = find_connecting_segment(contour_outer, contour_inner)
                    
                    if connecting_segment:
                        # Use the length of the connecting segment as the edge weight
                        weight = connecting_segment.length
                        G.add_edge((level, j), (level+1, k), weight=weight, 
                                  connecting_segment=connecting_segment)
                        edge_count += 1
                        level_edge_count += 1
                        
                        if debug and level_edge_count <= 3:  # Limit logging to first few edges
                            logger.info(f"    Added edge ({level},{j}) -> ({level+1},{k}) with weight {weight:.2f}")
        
        if debug:
            logger.info(f"    Added {level_edge_count} edges between levels {level} and {level+1}")
    
    if debug:
        logger.info(f"Added {edge_count} edges to the graph")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            logger.warning(f"Found {len(isolated_nodes)} isolated nodes in the graph")
            for node in isolated_nodes[:5]:  # Show first 5
                logger.warning(f"  Isolated node: {node}")
            if len(isolated_nodes) > 5:
                logger.warning(f"  ... and {len(isolated_nodes) - 5} more")
    
    # Compute the minimum spanning tree starting from the root (1,0)
    try:
        # Check if the graph is connected
        if not nx.is_connected(G):
            if debug:
                components = list(nx.connected_components(G))
                logger.warning(f"Graph is not connected. Found {len(components)} connected components")
                for i, component in enumerate(components[:3]):  # Show first 3
                    logger.warning(f"  Component {i+1}: {len(component)} nodes")
                if len(components) > 3:
                    logger.warning(f"  ... and {len(components) - 3} more components")
            
            largest_cc = max(nx.connected_components(G), key=len)
            G_connected = G.subgraph(largest_cc).copy()
            
            if debug:
                logger.info(f"Using largest connected component with {len(G_connected.nodes())} nodes")
        else:
            G_connected = G
            if debug:
                logger.info("Graph is connected")
        
        # Compute MST
        if debug:
            logger.info("Computing minimum spanning tree...")
        
        mst = nx.minimum_spanning_tree(G_connected, weight='weight')
        
        if debug:
            logger.info(f"MST has {len(mst.nodes())} nodes and {len(mst.edges())} edges")
            
            # Find the root node
            root_candidates = [(1, 0)]  # Typical root node
            root_node = None
            for node in root_candidates:
                if node in mst:
                    root_node = node
                    break
            
            if not root_node and mst.nodes():
                # If typical root not found, use the node with the lowest level
                root_node = min(mst.nodes(), key=lambda x: x[0])
            
            if root_node:
                logger.info(f"Root node identified as {root_node}")
                
                # Analyze tree structure
                leaf_nodes = [node for node in mst.nodes() if mst.degree(node) == 1 and node != root_node]
                branch_nodes = [node for node in mst.nodes() if mst.degree(node) > 2]
                
                logger.info(f"Tree structure: {len(leaf_nodes)} leaf nodes, {len(branch_nodes)} branch nodes")
                logger.info(f"Tree depth: {nx.eccentricity(mst, v=root_node)}")
                
                # Identify spirallable regions
                spirallable_regions = identify_spirallable_regions(mst)
                logger.info(f"Identified {len(spirallable_regions)} spirallable regions")
                
                # Log some details about the largest spirallable regions
                if spirallable_regions:
                    sorted_regions = sorted(spirallable_regions, key=len, reverse=True)
                    for i, region in enumerate(sorted_regions[:3]):  # Show top 3
                        logger.info(f"  Spirallable region {i+1}: {len(region)} nodes")
                    if len(sorted_regions) > 3:
                        logger.info(f"  ... and {len(sorted_regions) - 3} more regions")
        
        return G, mst
    except Exception as e:
        logger.error(f"Error building MST: {e}")
        return G, None

def find_connecting_segment(outer_contour, inner_contour):
    """
    Find the connecting segment on the outer contour that is closest to the inner contour.
    
    Args:
        outer_contour (Polygon): The outer contour
        inner_contour (Polygon): The inner contour
        
    Returns:
        LineString: The connecting segment
    """
    # Simplification: Use the shortest distance between contours
    # A more sophisticated approach would use Voronoi diagrams
    
    # Sample points along the outer contour
    num_samples = 100
    outer_points = [outer_contour.exterior.interpolate(i/num_samples, normalized=True)
                   for i in range(num_samples)]
    
    # Find the point on the outer contour closest to the inner contour
    min_dist = float('inf')
    closest_point = None
    
    for point in outer_points:
        dist = point.distance(inner_contour)
        if dist < min_dist:
            min_dist = dist
            closest_point = point
    
    if closest_point:
        # Create a small segment around the closest point
        t = outer_contour.exterior.project(closest_point, normalized=True)
        t1 = max(0, t - 0.05)
        t2 = min(1, t + 0.05)
        p1 = outer_contour.exterior.interpolate(t1, normalized=True)
        p2 = outer_contour.exterior.interpolate(t2, normalized=True)
        return LineString([p1, closest_point, p2])
    
    return None

def find_link_point(source_point, source_contour, target_contour):
    """
    Find the link point on the target contour from the source point.
    This implements the inward/outward link concept from the paper.
    
    Args:
        source_point (Point): The source point
        source_contour (Polygon): The contour containing the source point
        target_contour (Polygon): The target contour
        
    Returns:
        Point: The link point on the target contour
    """
    # Find the closest point on the target contour to the source point
    target_exterior = target_contour.exterior
    
    # Project the source point onto the target contour
    distance = target_exterior.project(source_point)
    link_point = target_exterior.interpolate(distance)
    
    return link_point

def get_line_segment_coords(contour, start_point, end_point, clockwise=True):
    """
    Get coordinates for a line segment along a contour from start to end point.
    
    Args:
        contour (Polygon or LineString): The contour
        start_point (Point): The start point on the contour
        end_point (Point): The end point on the contour
        clockwise (bool): Direction to travel along the contour
        
    Returns:
        list: List of coordinate tuples for the segment
    """
    # Check if contour is valid
    if contour is None:
        print(f"Error: Contour is None")
        return None
    
    # Get the exterior LineString from the contour
    if hasattr(contour, 'exterior'):
        # If contour is a Polygon
        exterior = contour.exterior
    elif isinstance(contour, LineString):
        # If contour is already a LineString
        exterior = contour
    else:
        print(f"Error: Unsupported contour type: {type(contour)}")
        return None
    
    # Project points onto the contour
    start_dist = exterior.project(start_point)
    end_dist = exterior.project(end_point)
    
    # Get the total length of the contour
    total_length = exterior.length
    
    # Determine the segment direction and length
    if clockwise:
        if end_dist < start_dist:
            end_dist += total_length
        segment_length = end_dist - start_dist
    else:
        if start_dist < end_dist:
            start_dist += total_length
        segment_length = start_dist - end_dist
    
    # Sample points along the segment
    num_samples = max(2, int(segment_length / 0.1))  # At least 2 points, otherwise ~0.1 spacing
    
    coords = []
    for i in range(num_samples):
        t = i / (num_samples - 1)
        if clockwise:
            dist = (start_dist + t * segment_length) % total_length
        else:
            dist = (start_dist - t * segment_length) % total_length
        
        point = exterior.interpolate(dist)
        coords.append((point.x, point.y))
    
    return coords

def get_coords(point):
    """Convert a Point to a list of coordinate tuples."""
    return [(point.x, point.y)]

def generate_fermat_spiral_segment(polygons, exteriors, innermost_contour_idx, outermost_contour_idx, 
                                  pin, pout, center_point, toolpath_width):
    """
    Generate a Fermat spiral segment for a spirallable region.
    
    Args:
        polygons: List of polygons for each contour
        exteriors: List of exterior LineStrings for each contour
        innermost_contour_idx: Index of the innermost contour
        outermost_contour_idx: Index of the outermost contour
        pin: Entry point on the outermost contour
        pout: Exit point on the outermost contour
        center_point: Center point of the innermost contour
        toolpath_width: Width between toolpaths
        
    Returns:
        LineString: The Fermat spiral segment
    """
    # Initialize variables
    current_contour_idx = outermost_contour_idx
    current_point = pin
    final_coords = get_coords(current_point)
    max_iterations = 1000  # Safety limit
    
    # --- Inward Phase ---
    print("    --- Inward Phase ---")
    iteration = 0
    while current_contour_idx >= innermost_contour_idx and iteration < max_iterations:
        iteration += 1
        # Get the current contour - make sure we're accessing it correctly
        if current_contour_idx in exteriors:
            if isinstance(exteriors[current_contour_idx], dict):
                # If exteriors is a nested dictionary, we need the first item
                if len(exteriors[current_contour_idx]) > 0:
                    current_contour = exteriors[current_contour_idx][0]
                else:
                    print(f"      Error: No contours found at level {current_contour_idx}")
                    return None
            else:
                current_contour = exteriors[current_contour_idx]
        else:
            print(f"      Error: Contour index {current_contour_idx} not found in exteriors")
            return None
        
        # 1. Determine the target point for this contour segment
        if current_contour_idx == innermost_contour_idx:
            # On the innermost contour, travel towards the center
            target_point = center_point
            print(f"        Target: center_point = {target_point.wkt[:30]}")
        else:
            # On outer contours, travel towards I(current_point)
            # Find I(current_point) - link from current point to inner contour
            target_point = find_link_point(current_point, polygons[current_contour_idx], 
                                          polygons[current_contour_idx-1])
            if not target_point:
                print(f"      Error: Could not find inward link I(current) to contour {current_contour_idx-1}. Aborting.")
                return None
            print(f"        Target: I(current) = {target_point.wkt[:30]}")
        
        # 2. Generate segment on current contour from current_point to target_point
        segment_coords = get_line_segment_coords(current_contour, current_point, target_point)
        if not segment_coords or len(segment_coords) < 2:
            print(f"      Warning: Could not generate segment on contour {current_contour_idx}. Adding jump.")
            # Add jump if segment fails
            if Point(final_coords[-1]).distance(target_point) > 1e-3:
                 final_coords.extend(get_coords(target_point))
        else:
            # Add segment coords, avoiding duplicates
            if Point(final_coords[-1]).distance(Point(segment_coords[0])) > 1e-3:
                final_coords.extend(segment_coords)
            else:
                final_coords.extend(segment_coords[1:])
        
        # 3. Reroute Inward (if not already at the innermost contour)
        if current_contour_idx > innermost_contour_idx:
            # Jump to the inner contour at the target point
            inner_point = target_point  # We already calculated this as I(current_point)
            
            # Add the jump coordinates
            if Point(final_coords[-1]).distance(inner_point) > 1e-3:
                final_coords.extend(get_coords(inner_point))
            
            print(f"        Rerouted inward to contour {current_contour_idx - 1} at {inner_point.wkt[:30]}")
            current_point = inner_point
            current_contour_idx -= 1
        else:
            # We've reached the innermost contour
            current_point = target_point
            print(f"        Reached innermost contour center at {current_point.wkt[:30]}")
    
    # --- Outward Phase ---
    print("    --- Outward Phase ---")
    # Start from center_point on innermost_contour_idx
    iteration = 0
    prev_target_point_on_inner = None # Store target from previous (inner) contour iteration
    while current_contour_idx <= outermost_contour_idx and iteration < max_iterations:
        iteration += 1
        # Get the current contour - make sure we're accessing it correctly
        if current_contour_idx in exteriors:
            if isinstance(exteriors[current_contour_idx], dict):
                # If exteriors is a nested dictionary, we need the first item
                if len(exteriors[current_contour_idx]) > 0:
                    current_contour = exteriors[current_contour_idx][0]
                else:
                    print(f"      Error: No contours found at level {current_contour_idx}")
                    return None
            else:
                current_contour = exteriors[current_contour_idx]
        else:
            print(f"      Error: Contour index {current_contour_idx} not found in exteriors")
            return None
        
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
        
        # Simplification: Generate the full segment for now.
        segment_coords = get_line_segment_coords(current_contour, current_point, target_point)
        if not segment_coords or len(segment_coords) < 2:
            print(f"      Warning: Could not generate outward segment on contour {current_contour_idx}. Adding jump.")
            if Point(final_coords[-1]).distance(target_point) > 1e-3:
                 final_coords.extend(get_coords(target_point))
        else:
            # Add segment coords, avoiding duplicates
            if Point(final_coords[-1]).distance(Point(segment_coords[0])) > 1e-3:
                final_coords.extend(segment_coords)
            else:
                final_coords.extend(segment_coords[1:])
        
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
            # We've reached the outermost contour and should be at pout
            current_point = target_point
            print(f"        Reached exit point at {current_point.wkt[:30]}")
    
    # Create the final LineString from the coordinates
    if len(final_coords) < 2:
        print("      Error: Not enough points to create a valid path")
        return None
    
    return LineString(final_coords)

def identify_spirallable_regions(mst):
    """
    Identify spirallable regions in the MST.
    
    Args:
        mst: The minimum spanning tree
        
    Returns:
        list: List of spirallable regions, each as a list of node IDs
    """
    # Nodes with degree <= 2 are part of spirallable regions
    # Nodes with degree > 2 are branch points
    
    spirallable_regions = []
    branch_points = []
    
    for node in mst.nodes():
        if mst.degree(node) > 2:
            branch_points.append(node)
    
    # If no branch points, the entire tree is one spirallable region
    if not branch_points:
        spirallable_regions.append(list(mst.nodes()))
        return spirallable_regions
    
    # Remove branch points to get connected components (spirallable regions)
    mst_copy = mst.copy()
    mst_copy.remove_nodes_from(branch_points)
    
    # Each connected component is a spirallable region
    for component in nx.connected_components(mst_copy):
        spirallable_regions.append(list(component))
    
    return spirallable_regions

def perform_recursive_rerouting(contours, mst, toolpath_width):
    """
    Perform recursive rerouting to generate the final continuous path.
    
    Args:
        contours: Dictionary of contours
        mst: The minimum spanning tree
        toolpath_width: Width between toolpaths
        
    Returns:
        LineString: The final continuous path
    """
    # Extract polygons and their exteriors for easier access
    polygons = {}
    exteriors = {}
    
    print(f"Processing {len(contours)} contour levels")
    for level in contours:
        polygons[level] = {}
        exteriors[level] = {}
        print(f"  Level {level}: {len(contours[level])} contours")
        for j, contour in enumerate(contours[level]):
            polygons[level][j] = contour
            exteriors[level][j] = contour.exterior
    
    # Check if the MST is empty
    if len(list(mst.nodes())) == 0:
        print("Error: Empty MST provided")
        return None
        
    # Find the root node (typically (1,0) for the outermost contour)
    root_node = None
    for node in mst.nodes():
        level, idx = node
        if level == 1 and idx == 0:
            root_node = node
            break
    
    if not root_node:
        # If no (1,0) node, use the node with the lowest level
        try:
            root_node = min(mst.nodes(), key=lambda x: x[0])
        except ValueError:
            # This should not happen due to the earlier check, but just in case
            print("Error: Could not find a root node in the MST")
            return None
    
    # Create a directed graph for the traversal
    directed_mst = nx.DiGraph()
    
    # Add edges with direction from root to leaves
    def add_directed_edges(node, parent=None):
        for neighbor in mst.neighbors(node):
            if neighbor != parent:
                directed_mst.add_edge(node, neighbor)
                add_directed_edges(neighbor, node)
    
    add_directed_edges(root_node)
    
    # Assign unique IDs to nodes for tracking
    node_to_id = {node: i for i, node in enumerate(directed_mst.nodes())}
    id_to_node = {i: node for node, i in node_to_id.items()}
    
    # Dictionary to store processed paths for each node
    processed_paths = {}
    
    # Process nodes bottom-up (leaves to root)
    def _process_node(node_id, parent_id, depth=0):
        # Prevent excessive recursion
        if depth > 250:  # Set a reasonable limit
            print(f"Warning: Maximum recursion depth reached ({depth}). Terminating branch.")
            return None
            
        node = id_to_node[node_id]
        level, idx = node
        
        # Get children of this node
        children = [node_to_id[child] for child in directed_mst.successors(node)]
        
        # If leaf node, create a more comprehensive path for the innermost contour
        if not children:
            # For leaf nodes, we'll create a spiral-like path that fills the contour
            contour = polygons[level][idx]
            exterior = exteriors[level][idx]
            
            # Get the centroid as a reference point
            centroid = contour.centroid
            
            # Create a denser sampling of points along the contour
            num_samples = max(150, int(exterior.length / toolpath_width))
            points = []
            
            # Generate points along the entire contour
            for i in range(num_samples):
                t = i / num_samples
                point = exterior.interpolate(t, normalized=True)
                points.append((point.x, point.y))
            
            # Add the first point again to close the loop
            points.append(points[0])
            
            # Create a zigzag pattern from the boundary toward the center
            zigzag_points = []
            num_inward_steps = 15  # Number of steps toward the center
            
            for i in range(0, len(points) - 1, 2):  # Skip every other point for efficiency
                # Add the boundary point
                zigzag_points.append(points[i])
                
                # Add points moving toward the center
                for step in range(1, num_inward_steps + 1):
                    # Interpolate between boundary point and centroid
                    t = step / (num_inward_steps + 1)
                    x = points[i][0] * (1 - t) + centroid.x * t
                    y = points[i][1] * (1 - t) + centroid.y * t
                    
                    # Only add the point if it's inside the contour
                    if contour.contains(Point(x, y)):
                        zigzag_points.append((x, y))
                    else:
                        break
                
                # If we have a next point, add it and move back to the boundary
                if i + 1 < len(points):
                    # Add points moving from center to the next boundary point
                    for step in range(num_inward_steps, 0, -1):
                        t = step / (num_inward_steps + 1)
                        x = points[i+1][0] * (1 - t) + centroid.x * t
                        y = points[i+1][1] * (1 - t) + centroid.y * t
                        
                        # Only add the point if it's inside the contour
                        if contour.contains(Point(x, y)):
                            zigzag_points.append((x, y))
                        
                    # Add the next boundary point
                    zigzag_points.append(points[i+1])
            
            # Create the path from the zigzag points
            if len(zigzag_points) < 2:
                # Fallback to simple contour if zigzag fails
                path = LineString(points)
            else:
                path = LineString(zigzag_points)
            
            processed_paths[node_id] = path
            return path
        
        # For non-leaf nodes, process children first
        child_paths = []
        for child_id in children:
            child_path = _process_node(child_id, node_id, depth+1)
            if child_path:
                child_paths.append((child_id, child_path))
        
        # If this is a spirallable region (node with 1 child), generate a denser Fermat spiral
        if len(children) == 1:
            child_id, child_path = child_paths[0]
            child_node = id_to_node[child_id]
            child_level, child_idx = child_node
            
            # Determine innermost and outermost contours
            innermost_contour_idx = max(level, child_level)
            outermost_contour_idx = min(level, child_level)
            
            # Get the contours
            innermost_contour = polygons[innermost_contour_idx][0 if innermost_contour_idx == level else child_idx]
            outermost_contour = polygons[outermost_contour_idx][0 if outermost_contour_idx == level else child_idx]
            
            # Calculate center point of innermost contour
            center_point = Point(innermost_contour.centroid)
            
            # Choose multiple entry/exit points on the outermost contour for better coverage
            num_spiral_segments = max(4, int(outermost_contour.exterior.length / (toolpath_width * 10)))
            
            # Create multiple spiral segments and combine them
            all_spiral_coords = []
            
            for i in range(num_spiral_segments):
                t1 = i / num_spiral_segments
                t2 = (i + 0.1) / num_spiral_segments  # Close to t1 for better connectivity
                
                pin = outermost_contour.exterior.interpolate(t1, normalized=True)
                pout = outermost_contour.exterior.interpolate(t2, normalized=True)
                
                # Generate the Fermat spiral segment
                spiral_segment = generate_fermat_spiral_segment(
                    polygons, exteriors, innermost_contour_idx, outermost_contour_idx,
                    pin, pout, center_point, toolpath_width
                )
                
                if spiral_segment:
                    # Add the coordinates to our collection
                    segment_coords = list(spiral_segment.coords)
                    
                    # Connect to previous segment if needed
                    if all_spiral_coords and segment_coords:
                        # Add a connecting line if the segments aren't already connected
                        if not np.allclose(all_spiral_coords[-1], segment_coords[0]):
                            # Create a direct line between the segments
                            all_spiral_coords.append(all_spiral_coords[-1])
                            all_spiral_coords.append(segment_coords[0])
                    
                    all_spiral_coords.extend(segment_coords)
            
            # Create the final spiral path
            if len(all_spiral_coords) < 2:
                # Fallback to original method if the multi-segment approach fails
                spiral_path = generate_fermat_spiral_segment(
                    polygons, exteriors, innermost_contour_idx, outermost_contour_idx,
                    outermost_contour.exterior.interpolate(0.0, normalized=True),
                    outermost_contour.exterior.interpolate(0.1, normalized=True),
                    center_point, toolpath_width
                )
            else:
                spiral_path = LineString(all_spiral_coords)
            
            if not spiral_path:
                return None
            
            processed_paths[node_id] = spiral_path
            return spiral_path
        
        # For branch nodes (nodes with multiple children), create a more comprehensive fill
        # Get the contour for this node
        contour = polygons[level][idx]
        exterior = exteriors[level][idx]
        centroid = contour.centroid
        
        # Create a grid of points inside the contour for better filling
        minx, miny, maxx, maxy = contour.bounds
        grid_spacing = toolpath_width * 0.6  # Slightly denser than toolpath width
        
        # Calculate number of points in each dimension
        nx = max(5, int((maxx - minx) / grid_spacing))
        ny = max(5, int((maxy - miny) / grid_spacing))
        
        # Create the grid points
        grid_points = []
        for i in range(nx):
            for j in range(ny):
                x = minx + i * (maxx - minx) / (nx - 1)
                y = miny + j * (maxy - miny) / (ny - 1)
                point = Point(x, y)
                
                # Only include points inside the contour
                if contour.contains(point):
                    grid_points.append(point)
        
        # Create connection points on the contour for each child
        connection_points = []
        for i, (child_id, _) in enumerate(child_paths):
            t = i / len(child_paths)  # Distribute evenly
            point = exterior.interpolate(t, normalized=True)
            connection_points.append(point)
        
        # Create a path that visits all grid points in a zigzag pattern
        # and connects to all children
        zigzag_coords = []
        
        # Start with the first connection point
        if connection_points:
            zigzag_coords.append((connection_points[0].x, connection_points[0].y))
        
        # Sort grid points by distance from centroid for a spiral-like effect
        grid_points.sort(key=lambda p: p.distance(centroid))
        
        # Add grid points in sorted order
        for point in grid_points:
            zigzag_coords.append((point.x, point.y))
        
        # Add remaining connection points
        for i in range(1, len(connection_points)):
            zigzag_coords.append((connection_points[i].x, connection_points[i].y))
        
        # Create segments connecting the children with the zigzag path
        segments = []
        if len(zigzag_coords) >= 2:
            segments.append(LineString(zigzag_coords))
        
        # Also add segments along the contour between connection points
        for i in range(len(connection_points)):
            start_point = connection_points[i]
            end_point = connection_points[(i+1) % len(connection_points)]
            
            # Generate segment along the contour
            coords = get_line_segment_coords(contour, start_point, end_point)
            if coords and len(coords) >= 2:
                segments.append(LineString(coords))
        
        # Connect all segments and child paths
        all_paths = []
        for i, (child_id, child_path) in enumerate(child_paths):
            all_paths.append(child_path)
            if i < len(segments):
                all_paths.append(segments[i])
        
        # Merge all paths into one
        merged_coords = []
        for path in all_paths:
            path_coords = list(path.coords)
            if merged_coords and np.allclose(merged_coords[-1], path_coords[0]):
                merged_coords.extend(path_coords[1:])
            else:
                merged_coords.extend(path_coords)
        
        if len(merged_coords) < 2:
            return None
        
        final_path = LineString(merged_coords)
        processed_paths[node_id] = final_path
        return final_path
    
    # Start processing from the root
    try:
        root_node_id = node_to_id[root_node]
        print(f"Starting recursive rerouting from root node: {root_node}")
        final_path = _process_node(root_node_id, None, 0)
        
        return final_path
    except Exception as e:
        print(f"Error during recursive rerouting: {e}")
        return None

def visualize_iso_contours(region, contours, title="Iso-Contours (Step 5)"):
    """
    Visualize the iso-contours (Step 5 of the algorithm).
    
    Args:
        region (Polygon): The original region
        contours (dict): The generated iso-contours
        title (str): Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the original region
    x, y = region.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Region Boundary')
    
    # Plot holes if any
    for interior in region.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot contours with different colors for each level
    colors = plt.cm.viridis(np.linspace(0, 1, len(contours) + 1))
    for i, level in enumerate(sorted(contours.keys())):
        for j, contour in enumerate(contours[level]):
            x, y = contour.exterior.xy
            ax.plot(x, y, '-', color=colors[i], linewidth=1.5, 
                   label=f'Level {level}' if j == 0 else "")
            
            # Add level and index labels
            centroid = contour.centroid
            ax.text(centroid.x, centroid.y, f"({level},{j})", 
                   ha='center', va='center', fontsize=8)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Create a custom legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.tight_layout()
    plt.show()

def visualize_spiral_contour_tree(region, contours, graph, mst, title="Spiral Contour Tree (Step 6)", save_path=None):
    """
    Visualize the spiral contour tree (Step 6 of the algorithm).
    
    Args:
        region (Polygon): The original region
        contours (dict): The generated iso-contours
        graph (nx.Graph): The connectivity graph
        mst (nx.Graph): The minimum spanning tree
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot the original region
    x, y = region.exterior.xy
    ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.5, label='Region Boundary')
    
    # Plot holes if any
    for interior in region.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.5)
    
    # Plot contours with different colors for each level
    colors = plt.cm.viridis(np.linspace(0, 1, len(contours) + 1))
    for i, level in enumerate(sorted(contours.keys())):
        for j, contour in enumerate(contours[level]):
            x, y = contour.exterior.xy
            ax.plot(x, y, '-', color=colors[i], linewidth=1.0, alpha=0.4,
                   label=f'Level {level}' if j == 0 else "")
    
    # Create a position dictionary for the graph nodes
    pos = {}
    node_sizes = {}
    for node in graph.nodes():
        level, idx = node
        if level in contours and idx < len(contours[level]):
            contour = contours[level][idx]
            pos[node] = (contour.centroid.x, contour.centroid.y)
            # Size nodes based on contour area
            node_sizes[node] = max(50, min(500, contour.area * 0.1))
    
    # Plot all edges in the graph as light gray with weights
    for u, v, data in graph.edges(data=True):
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   'gray', linestyle=':', linewidth=0.5, alpha=0.5)
            
            # Add weight label to graph edges
            if 'weight' in data:
                # Position the label at the middle of the edge
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(mid_x, mid_y, f"{data['weight']:.1f}", 
                       color='gray', fontsize=7, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Plot MST edges as bold blue with weights and directions
    for u, v, data in mst.edges(data=True):
        if u in pos and v in pos:
            # Draw the edge
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   'blue', linestyle='-', linewidth=2)
            
            # Add an arrow to show parent-child relationship
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            dx = (pos[v][0] - pos[u][0]) * 0.4
            dy = (pos[v][1] - pos[u][1]) * 0.4
            ax.arrow(mid_x - dx/2, mid_y - dy/2, dx, dy, 
                    head_width=0.3, head_length=0.5, fc='blue', ec='blue', 
                    length_includes_head=True)
            
            # Add weight label to MST edges
            if 'weight' in data:
                ax.text(mid_x, mid_y + 0.5, f"{data['weight']:.1f}", 
                       color='blue', fontsize=9, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, pad=2))
    
    # Plot nodes with different sizes and colors
    mst_nodes = list(mst.nodes())
    
    # Find the root node (typically the one with the lowest level)
    root_node = None
    for node in mst_nodes:
        if node[0] == 1 and node[1] == 0:  # Level 1, index 0 is typically the root
            root_node = node
            break
    if not root_node and mst_nodes:
        root_node = min(mst_nodes, key=lambda x: x[0])
    
    # Plot non-MST nodes
    for node in graph.nodes():
        if node in pos and node not in mst:
            ax.plot(pos[node][0], pos[node][1], 'go', 
                   markersize=np.sqrt(node_sizes.get(node, 100)/np.pi))
    
    # Plot MST nodes
    for node in mst_nodes:
        if node in pos:
            if node == root_node:
                # Root node in purple
                ax.plot(pos[node][0], pos[node][1], 'mo', 
                       markersize=np.sqrt(node_sizes.get(node, 150)/np.pi))
            else:
                # Other MST nodes in red
                ax.plot(pos[node][0], pos[node][1], 'ro', 
                       markersize=np.sqrt(node_sizes.get(node, 100)/np.pi))
    
    # Add node labels with more information
    for node in graph.nodes():
        if node in pos:
            level, idx = node
            # Get node degree
            degree = graph.degree(node)
            mst_degree = mst.degree(node) if node in mst else 0
            
            # Create label with node ID and degree info
            label = f"({level},{idx})"
            if node in mst:
                label += f"\nD:{mst_degree}"
                
                # Add special marker for leaf nodes (degree 1 in MST)
                if mst_degree == 1 and node != root_node:
                    ax.text(pos[node][0], pos[node][1] - 0.8, "LEAF", 
                           color='red', fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='yellow', alpha=0.7, pad=1))
                
                # Add special marker for branch nodes (degree > 2 in MST)
                if mst_degree > 2:
                    ax.text(pos[node][0], pos[node][1] - 0.8, "BRANCH", 
                           color='blue', fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='cyan', alpha=0.7, pad=1))
            
            ax.text(pos[node][0], pos[node][1], label, 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Mark the root node
    if root_node and root_node in pos:
        ax.text(pos[root_node][0], pos[root_node][1] - 0.8, "ROOT", 
               color='purple', fontsize=10, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.9, pad=2))
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    
    # Add detailed legend
    legend_elements = [
        plt.Line2D([0], [0], color='k', lw=1.5, alpha=0.5, label='Region Boundary'),
        plt.Line2D([0], [0], color=colors[0], lw=1, alpha=0.4, label='Contour Level 1'),
        plt.Line2D([0], [0], color=colors[-1], lw=1, alpha=0.4, label=f'Contour Level {len(contours)}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=10, label='Root Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='MST Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Other Nodes'),
        plt.Line2D([0], [0], color='blue', lw=2, label='MST Edges'),
        plt.Line2D([0], [0], color='gray', lw=0.5, linestyle=':', label='Graph Edges')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Add statistics text box
    stats_text = (
        f"Region Area: {region.area:.1f}\n"
        f"Contour Levels: {len(contours)}\n"
        f"Total Contours: {sum(len(contours[level]) for level in contours)}\n"
        f"Graph Nodes: {len(graph.nodes())}\n"
        f"Graph Edges: {len(graph.edges())}\n"
        f"MST Nodes: {len(mst.nodes())}\n"
        f"MST Edges: {len(mst.edges())}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def visualize_recursive_rerouting(region, contours, mst, final_path, title="Recursive Rerouting (Step 7)"):
    """
    Visualize the recursive rerouting result (Step 7 of the algorithm).
    
    Args:
        region (Polygon): The original region
        contours (dict): The generated iso-contours
        mst (nx.Graph): The minimum spanning tree
        final_path (LineString): The final continuous path
        title (str): Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the original region
    x, y = region.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Region Boundary')
    
    # Plot contours with light colors
    for level in contours:
        for contour in contours[level]:
            x, y = contour.exterior.xy
            ax.plot(x, y, 'g--', linewidth=0.5, alpha=0.3)
    
    # Plot the final path with a color gradient to show direction
    if final_path:
        points = np.array(final_path.coords)
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        
        # Create a line collection for better visualization of direction
        from matplotlib.collections import LineCollection
        
        # Create a colormap that transitions from green to red
        cmap = plt.cm.winter
        colors = np.linspace(0, 1, len(segments))
        
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), linewidth=2)
        lc.set_array(colors)
        ax.add_collection(lc)
        
        # Mark start and end points
        ax.plot(points[0, 0], points[0, 1], 'go', markersize=8, label='Start')
        ax.plot(points[-1, 0], points[-1, 1], 'ro', markersize=8, label='End')
        
        # Add arrows to show direction
        arrow_indices = np.linspace(0, len(points) - 2, min(20, len(points) - 1)).astype(int)
        for i in arrow_indices:
            p1, p2 = points[i], points[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                ax.arrow(p1[0], p1[1], dx * 0.8, dy * 0.8, 
                        head_width=0.3, head_length=0.5, fc='blue', ec='blue', 
                        length_includes_head=True, alpha=0.7)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Add a colorbar to show the direction of the path
    if final_path:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Path Direction')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Start', 'End'])
    
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def visualize_cfs_fill(region, toolpath=None, toolpath_width=None, contours=None, mst=None, save_path=None):
    """
    Visualize the CFS fill path.
    
    Args:
        region (Polygon): The original region
        toolpath (LineString, optional): The generated toolpath
        toolpath_width (float, optional): The toolpath width for generating contours
        contours (dict, optional): Pre-computed contours to visualize
        mst (nx.Graph, optional): The minimum spanning tree to visualize
        save_path (str, optional): Path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the original region
    x, y = region.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Region Boundary')
    
    # Plot holes if any
    for interior in region.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot contours if provided
    if contours:
        colors = plt.cm.viridis(np.linspace(0, 1, len(contours) + 1))
        for i, level in enumerate(sorted(contours.keys())):
            for contour in contours[level]:
                x, y = contour.exterior.xy
                ax.plot(x, y, '-', color=colors[i], linewidth=0.8, alpha=0.5,
                       label=f'Level {level}' if i == 0 else "")
                
                # Add level labels to some contours
                if i % 2 == 0:  # Only label every other level to avoid clutter
                    centroid = contour.centroid
                    ax.text(centroid.x, centroid.y, f"{level}", 
                           ha='center', va='center', fontsize=8, alpha=0.7)
    
    # Plot MST if provided
    if mst and contours:
        # Create a position dictionary for the graph nodes
        pos = {}
        for node in mst.nodes():
            level, idx = node
            if level in contours and idx < len(contours[level]):
                contour = contours[level][idx]
                pos[node] = (contour.centroid.x, contour.centroid.y)
        
        # Plot MST edges
        for u, v in mst.edges():
            if u in pos and v in pos:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                       'blue', linestyle='-', linewidth=1, alpha=0.7)
        
        # Plot MST nodes
        for node in mst.nodes():
            if node in pos:
                ax.plot(pos[node][0], pos[node][1], 'ro', markersize=4, alpha=0.7)
    
    # Plot the toolpath
    if toolpath:
        x, y = toolpath.xy
        ax.plot(x, y, 'r-', linewidth=1.5, label='CFS Toolpath')
        
        # Mark start and end points
        ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
        ax.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
        
        # Add arrows to show direction
        arrow_indices = np.linspace(0, len(x) - 2, min(20, len(x) - 1)).astype(int)
        for i in arrow_indices:
            dx, dy = x[i+1] - x[i], y[i+1] - y[i]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                ax.arrow(x[i], y[i], dx * 0.8, dy * 0.8, 
                        head_width=0.2, head_length=0.3, fc='blue', ec='blue', 
                        length_includes_head=True, alpha=0.7)
    
    ax.set_aspect('equal')
    ax.set_title('Continuous Fermat Spiral (CFS) Fill', fontsize=14)
    
    # Create a custom legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    # Add statistics text box
    stats_text = f"Region Area: {region.area:.1f}"
    if contours:
        stats_text += f"\nContour Levels: {len(contours)}"
        stats_text += f"\nTotal Contours: {sum(len(contours[level]) for level in contours)}"
    if mst:
        stats_text += f"\nMST Nodes: {len(mst.nodes())}"
        stats_text += f"\nMST Edges: {len(mst.edges())}"
    if toolpath:
        stats_text += f"\nToolpath Length: {toolpath.length:.1f}"
        stats_text += f"\nToolpath Points: {len(toolpath.coords)}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
def analyze_spiral_contour_tree(contours, mst):
    """
    Analyze the spiral contour tree and print detailed information.
    
    Args:
        contours (dict): Dictionary of contours
        mst (nx.Graph): The minimum spanning tree
    """
    if not mst:
        logger.error("No MST provided for analysis")
        return
    
    logger.info("\n=== Spiral Contour Tree Analysis ===")
    
    # Basic statistics
    logger.info(f"MST has {len(mst.nodes())} nodes and {len(mst.edges())} edges")
    
    # Find the root node (typically the one with the lowest level)
    root_node = None
    for node in mst.nodes():
        if node[0] == 1 and node[1] == 0:  # Level 1, index 0 is typically the root
            root_node = node
            break
    if not root_node and mst.nodes():
        root_node = min(mst.nodes(), key=lambda x: x[0])
    
    if root_node:
        logger.info(f"Root node: {root_node}")
        
        # Calculate tree depth
        try:
            tree_depth = nx.eccentricity(mst, v=root_node)
            logger.info(f"Tree depth from root: {tree_depth}")
        except:
            logger.warning("Could not calculate tree depth")
        
        # Identify node types
        leaf_nodes = [node for node in mst.nodes() if mst.degree(node) == 1 and node != root_node]
        branch_nodes = [node for node in mst.nodes() if mst.degree(node) > 2]
        chain_nodes = [node for node in mst.nodes() if mst.degree(node) == 2 or (node == root_node and mst.degree(node) == 1)]
        
        logger.info(f"Node types:")
        logger.info(f"  Root node: {root_node} (degree: {mst.degree(root_node)})")
        logger.info(f"  Leaf nodes: {len(leaf_nodes)} ({', '.join(str(node) for node in leaf_nodes[:5])}{'...' if len(leaf_nodes) > 5 else ''})")
        logger.info(f"  Branch nodes: {len(branch_nodes)} ({', '.join(str(node) for node in branch_nodes[:5])}{'...' if len(branch_nodes) > 5 else ''})")
        logger.info(f"  Chain nodes: {len(chain_nodes)}")
        
        # Analyze levels
        levels = {}
        for node in mst.nodes():
            level = node[0]
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        logger.info(f"Level distribution:")
        for level in sorted(levels.keys()):
            logger.info(f"  Level {level}: {len(levels[level])} nodes")
        
        # Identify spirallable regions
        spirallable_regions = identify_spirallable_regions(mst)
        logger.info(f"Spirallable regions: {len(spirallable_regions)}")
        
        # Analyze spirallable regions
        if spirallable_regions:
            sorted_regions = sorted(spirallable_regions, key=len, reverse=True)
            for i, region in enumerate(sorted_regions[:5]):  # Show top 5
                min_level = min(node[0] for node in region)
                max_level = max(node[0] for node in region)
                logger.info(f"  Region {i+1}: {len(region)} nodes, levels {min_level}-{max_level}")
                
                # Check if this is a simple chain
                is_chain = all(sum(1 for n in mst.neighbors(node) if n in region) <= 2 for node in region)
                logger.info(f"    Is chain: {is_chain}")
                
                # List some nodes in this region
                logger.info(f"    Sample nodes: {', '.join(str(node) for node in region[:5])}{'...' if len(region) > 5 else ''}")
            
            if len(sorted_regions) > 5:
                logger.info(f"  ... and {len(sorted_regions) - 5} more regions")
    else:
        logger.warning("Could not identify root node in MST")
    
    logger.info("=== End of Analysis ===\n")
