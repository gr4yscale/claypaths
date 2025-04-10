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

def generate_cfs_fill(region, toolpath_width):
    """
    Generate a Continuous Fermat Spiral (CFS) fill for a 2D region.
    
    Args:
        region (Polygon): The 2D region to fill
        toolpath_width (float): The desired spacing between adjacent path segments
        
    Returns:
        LineString: The continuous toolpath as a LineString
    """
    if not isinstance(region, Polygon) or not region.is_valid:
        logger.error("Input must be a valid Polygon")
        return None
    
    # Step 1: Generate iso-contours (offset curves)
    contours = generate_iso_contours(region, toolpath_width)
    if not contours:
        logger.error("Failed to generate iso-contours")
        return None
    
    # Step 2: Build connectivity graph and MST
    graph, mst = build_spiral_contour_tree(contours)
    if not mst:
        logger.error("Failed to build spiral-contour tree")
        return None
    
    # Step 3: Perform recursive rerouting to generate the final path
    final_path = perform_recursive_rerouting(contours, mst, toolpath_width)
    
    return final_path

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

def build_spiral_contour_tree(contours):
    """
    Build a connectivity graph and minimum spanning tree for the contours.
    
    Args:
        contours (dict): Dictionary mapping contour levels to lists of Polygons
        
    Returns:
        tuple: (connectivity graph, minimum spanning tree)
    """
    # Create a graph where nodes are contours
    G = nx.Graph()
    
    # Add nodes for each contour
    for level in contours:
        for j, contour in enumerate(contours[level]):
            node_id = (level, j)
            G.add_node(node_id, contour=contour, level=level)
    
    # Add edges between contours at adjacent levels
    for level in range(1, len(contours)):
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
    
    # Compute the minimum spanning tree starting from the root (1,0)
    try:
        # Ensure the graph is connected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Compute MST
        mst = nx.minimum_spanning_tree(G, weight='weight')
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
        if depth > 100:  # Set a reasonable limit
            print(f"Warning: Maximum recursion depth reached ({depth}). Terminating branch.")
            return None
            
        node = id_to_node[node_id]
        level, idx = node
        
        # Get children of this node
        children = [node_to_id[child] for child in directed_mst.successors(node)]
        
        # If leaf node, create a simple path along its contour
        if not children:
            # For leaf nodes, we'll create a simple path along part of the contour
            contour = polygons[level][idx]
            exterior = exteriors[level][idx]
            
            # Choose arbitrary start/end points for the leaf
            t1, t2 = 0.0, 0.5  # Arbitrary parameters
            start_point = exterior.interpolate(t1, normalized=True)
            end_point = exterior.interpolate(t2, normalized=True)
            
            # Generate a simple path along the contour
            coords = get_line_segment_coords(contour, start_point, end_point)
            if not coords or len(coords) < 2:
                return None
            
            path = LineString(coords)
            processed_paths[node_id] = path
            return path
        
        # For non-leaf nodes, process children first
        child_paths = []
        for child_id in children:
            child_path = _process_node(child_id, node_id, depth+1)
            if child_path:
                child_paths.append((child_id, child_path))
        
        # If this is a spirallable region (node with 1 child), generate a Fermat spiral
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
            
            # Choose entry/exit points on the outermost contour
            # For simplicity, we'll use arbitrary points
            t1, t2 = 0.0, 0.1  # Close to each other for better connectivity
            pin = outermost_contour.exterior.interpolate(t1, normalized=True)
            pout = outermost_contour.exterior.interpolate(t2, normalized=True)
            
            # Generate the Fermat spiral
            spiral_path = generate_fermat_spiral_segment(
                polygons, exteriors, innermost_contour_idx, outermost_contour_idx,
                pin, pout, center_point, toolpath_width
            )
            
            if not spiral_path:
                return None
            
            processed_paths[node_id] = spiral_path
            return spiral_path
        
        # For branch nodes (nodes with multiple children), connect the child paths
        # This is a simplified approach - in a full implementation, you would need
        # more sophisticated logic to connect the paths optimally
        
        # Get the contour for this node
        contour = polygons[level][idx]
        exterior = exteriors[level][idx]
        
        # Create connection points on the contour for each child
        connection_points = []
        for i, (child_id, _) in enumerate(child_paths):
            t = i / len(child_paths)  # Distribute evenly
            point = exterior.interpolate(t, normalized=True)
            connection_points.append(point)
        
        # Create segments connecting the children
        segments = []
        for i in range(len(child_paths)):
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

def visualize_cfs_fill(region, toolpath, toolpath_width=None, contours=None):
    """
    Visualize the CFS fill path.
    
    Args:
        region (Polygon): The original region
        toolpath (LineString): The generated toolpath
        toolpath_width (float, optional): The toolpath width for generating contours
        contours (dict, optional): Pre-computed contours to visualize
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the original region
    x, y = region.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Region Boundary')
    
    # Plot holes if any
    for interior in region.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot contours if provided
    if contours:
        for level in contours:
            for contour in contours[level]:
                x, y = contour.exterior.xy
                ax.plot(x, y, 'g--', linewidth=0.5, alpha=0.5)
    
    # Plot the toolpath
    if toolpath:
        x, y = toolpath.xy
        ax.plot(x, y, 'r-', linewidth=1, label='CFS Toolpath')
        
        # Mark start and end points
        ax.plot(x[0], y[0], 'go', markersize=6, label='Start')
        ax.plot(x[-1], y[-1], 'ro', markersize=6, label='End')
    
    ax.set_aspect('equal')
    ax.set_title('Continuous Fermat Spiral (CFS) Fill')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
