import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import nearest_points, unary_union
from scipy.spatial import Voronoi
from src.config import get_config

def generate_iso_contours(polygon, toolpath_width):
    """
    Generate iso-contours by successive inward offsets.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to offset
        toolpath_width (float): Distance between contours
        
    Returns:
        list: List of contours, each as a tuple (exterior, interiors)
    """
    contours = []
    current_polygon = polygon
    min_area = toolpath_width * toolpath_width * 4  # Minimum area threshold
    
    while current_polygon.area > min_area:
        # Get exterior and interiors
        exterior = list(current_polygon.exterior.coords)
        interiors = [list(interior.coords) for interior in current_polygon.interiors]
        contours.append((exterior, interiors))
        
        # Generate next inward offset
        next_polygon = current_polygon.buffer(-toolpath_width, join_style=2)
        
        # Handle MultiPolygon results
        if isinstance(next_polygon, MultiPolygon):
            # Take the largest polygon
            largest = max(next_polygon.geoms, key=lambda p: p.area)
            next_polygon = largest
        
        if next_polygon.is_empty or not isinstance(next_polygon, Polygon):
            break
            
        current_polygon = next_polygon
    
    return contours

def find_connecting_segment(outer_contour, inner_contour, toolpath_width):
    """
    Find the connecting segment between two contours using Voronoi diagrams.
    
    Args:
        outer_contour (list): Outer contour coordinates
        inner_contour (list): Inner contour coordinates
        toolpath_width (float): Toolpath width for proximity threshold
        
    Returns:
        list: List of points forming the connecting segment
    """
    # Combine points from both contours
    points = np.array(outer_contour + inner_contour)
    
    # Compute Voronoi diagram
    vor = Voronoi(points)
    
    # Find ridge points between outer and inner contours
    connecting_segment = []
    for ridge in vor.ridge_points:
        if (ridge[0] < len(outer_contour) and ridge[1] >= len(outer_contour)) or \
           (ridge[1] < len(outer_contour) and ridge[0] >= len(outer_contour)):
            p1 = points[ridge[0]]
            p2 = points[ridge[1]]
            if np.linalg.norm(p1 - p2) < toolpath_width * 1.5:
                connecting_segment.extend([p1, p2])
    
    return connecting_segment

def build_connectivity_graph(contours, toolpath_width):
    """
    Build a connectivity graph between contours.
    
    Args:
        contours (list): List of contours from generate_iso_contours()
        toolpath_width (float): Toolpath width for proximity calculations
        
    Returns:
        networkx.Graph: Connectivity graph between contours
    """
    G = nx.Graph()
    
    # Add nodes for each contour
    for i, (exterior, interiors) in enumerate(contours):
        G.add_node(i, exterior=exterior, interiors=interiors)
    
    # Add edges between adjacent contours
    for i in range(len(contours) - 1):
        outer_contour = contours[i][0]
        inner_contour = contours[i+1][0]
        
        # Find connecting segments using Voronoi diagram
        connecting_segment = find_connecting_segment(outer_contour, inner_contour, toolpath_width)
        
        if connecting_segment:
            # Add edge with weight equal to segment length
            length = LineString(connecting_segment).length
            G.add_edge(i, i+1, weight=length, segment=connecting_segment)
    
    return G

def construct_spiral_contour_tree(connectivity_graph):
    """
    Construct the spiral-contour tree using Minimum Spanning Tree (MST).
    
    Args:
        connectivity_graph (networkx.Graph): Connectivity graph from build_connectivity_graph()
        
    Returns:
        networkx.Graph: Spiral-contour tree
    """
    # Compute MST starting from the outermost contour (node 0)
    return nx.minimum_spanning_tree(connectivity_graph)

def identify_spirallable_regions(spiral_tree):
    """
    Identify spirallable regions and branch points in the spiral-contour tree.
    
    Args:
        spiral_tree (networkx.Graph): Spiral-contour tree
        
    Returns:
        tuple: (spirallable_regions, branch_points)
    """
    spirallable_regions = []
    branch_points = []
    
    # Traverse the tree to find paths and branch points
    for node in spiral_tree.nodes:
        if spiral_tree.degree(node) <= 2:
            spirallable_regions.append(node)
        else:
            branch_points.append(node)
    
    return spirallable_regions, branch_points

def generate_simple_spiral(spiral_tree, toolpath_width):
    """
    Generate a simple spiral path as a placeholder for the full algorithm.
    
    Args:
        spiral_tree (networkx.Graph): Spiral-contour tree
        toolpath_width (float): Toolpath width
        
    Returns:
        list: List of points representing a simple spiral path
    """
    path = []
    
    # Traverse the tree in depth-first order
    for node in nx.dfs_preorder_nodes(spiral_tree, source=0):
        exterior = spiral_tree.nodes[node]['exterior']
        path.extend(exterior)
        
        # Add connection to next contour
        if node < len(spiral_tree) - 1:
            next_node = node + 1
            if spiral_tree.has_edge(node, next_node):
                segment = spiral_tree.edges[node, next_node]['segment']
                path.extend(segment)
    
    return path

def perform_recursive_rerouting(spiral_tree, spirallable_regions, branch_points, toolpath_width):
    """
    Perform recursive rerouting to generate the continuous spiral path.
    
    Args:
        spiral_tree (networkx.Graph): Spiral-contour tree
        spirallable_regions (list): List of spirallable region nodes
        branch_points (list): List of branch point nodes
        toolpath_width (float): Toolpath width
        
    Returns:
        list: List of points representing the continuous toolpath
    """
    # TODO: Implement the recursive rerouting logic
    # This is the most complex part of the algorithm and requires
    # careful implementation of the inward/outward links and
    # branch point merging logic described in the paper
    
    # For now, return a simple spiral path as a placeholder
    return generate_simple_spiral(spiral_tree, toolpath_width)

def generate_continuous_fill(polygon, toolpath_width=1.0, prev_end_point=None):
    """
    Generate a continuous fill pattern using the Continuous Fermat Spiral (CFS) algorithm.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple): The end point of the previous layer's path (x, y)
        
    Returns:
        list: List of points representing the continuous toolpath
    """
    # Validate input polygon
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        print("ERROR: Invalid polygon for region fill (not a polygon or empty)")
        return []
    
    if not polygon.is_valid:
        print("WARNING: Polygon is not valid, attempting to fix...")
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            print("ERROR: Failed to fix invalid polygon")
            return []
    
    # Step 1: Generate iso-contours
    iso_contours = generate_iso_contours(polygon, toolpath_width)
    if not iso_contours:
        print("ERROR: Failed to generate iso-contours")
        return []
    
    # Step 2: Build connectivity graph
    connectivity_graph = build_connectivity_graph(iso_contours, toolpath_width)
    if not connectivity_graph:
        print("ERROR: Failed to build connectivity graph")
        return []
    
    # Step 3: Construct spiral-contour tree
    spiral_tree = construct_spiral_contour_tree(connectivity_graph)
    if not spiral_tree:
        print("ERROR: Failed to construct spiral-contour tree")
        return []
    
    # Step 4: Identify spirallable regions and branch points
    spirallable_regions, branch_points = identify_spirallable_regions(spiral_tree)
    
    # Step 5: Perform recursive rerouting
    toolpath = perform_recursive_rerouting(spiral_tree, spirallable_regions, branch_points, toolpath_width)
    
    return toolpath

def generate_contour_fill(polygon, toolpath_width, prev_end_point=None):
    """
    Generate a continuous fill pattern using inward contours.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple, optional): The end point of the previous layer's path (x, y)
        
    Returns:
        list: List of points representing the continuous toolpath
    """
    if not polygon.is_valid:
        print("Invalid polygon for contour fill")
        return []
    
    # Get the configuration to check the polygonization method
    from src.config import get_config
    config = get_config()
    polygonization_method = config.get('polygonization_method', 'standard')
    
    print(f"  Using polygonization method: {polygonization_method}")
    
    # Start with the original polygon boundary
    current_polygon = polygon
    contours = []
    
    # Generate inward contours until the polygon becomes too small
    min_area = toolpath_width * toolpath_width * 4  # Minimum area threshold
    print(f"  Minimum area threshold: {min_area:.4f} sq units")
    
    contour_count = 0
    
    while current_polygon.area > min_area:
        # Add the current contour
        contours.append(current_polygon.exterior)
        contour_count += 1
        
        # Generate the next inward contour
        # Adjust buffer distance based on polygonization method
        buffer_distance = -toolpath_width
        
        # For small_buffer and large_buffer methods, we need to be more careful
        # with the buffer operation to avoid invalid geometries
        if polygonization_method in ['small_buffer', 'large_buffer']:
            # Use a more conservative buffer distance for these methods
            # to prevent issues with the already buffered polygons
            buffer_distance = -max(toolpath_width * 0.9, 0.1)
            
            # For very small polygons, be even more conservative
            if current_polygon.area < min_area * 2:
                buffer_distance = -max(toolpath_width * 0.8, 0.05)
        
        print(f"  Contour {contour_count}: Area={current_polygon.area:.4f}, Buffer={buffer_distance:.4f}")
        
        try:
            next_polygon = current_polygon.buffer(buffer_distance, join_style=2)
            
            # Check if the buffer operation resulted in a valid polygon
            if next_polygon.is_empty:
                print(f"  Stopping: Buffer resulted in empty polygon")
                break
                
            # Handle potential MultiPolygon results
            if next_polygon.geom_type == 'MultiPolygon':
                print(f"  Buffer resulted in MultiPolygon with {len(next_polygon.geoms)} parts")
                # Take the largest polygon from the multipolygon
                largest_area = 0
                largest_poly = None
                for poly in next_polygon.geoms:
                    if poly.area > largest_area:
                        largest_area = poly.area
                        largest_poly = poly
                
                if largest_poly is None or largest_poly.area < min_area:
                    print(f"  Stopping: Largest polygon too small ({largest_area if largest_poly else 0:.4f} < {min_area:.4f})")
                    break
                    
                next_polygon = largest_poly
                print(f"  Selected largest polygon with area {largest_area:.4f}")
            elif next_polygon.geom_type != 'Polygon':
                print(f"  Stopping: Buffer resulted in {next_polygon.geom_type} instead of Polygon")
                break
                
            current_polygon = next_polygon
        except Exception as e:
            print(f"  Error during buffer operation: {e}")
            break
    
    # Connect the contours to form a continuous spiral path
    # We don't optimize for previous end point here as the toolpath optimizer will handle that
    path = connect_contours(contours, toolpath_width)
    
    return path

def connect_contours(contours, toolpath_width, prev_end_point=None):
    """
    Connect contours to form a continuous spiral path.
    
    Args:
        contours (list): List of LinearRings representing contours
        toolpath_width (float): Width of the toolpath
        prev_end_point (tuple, optional): The end point of the previous layer's path (x, y)
        
    Returns:
        list: List of points representing the continuous path
    """
    if not contours:
        print("  No contours to connect")
        return []
    
    print(f"  Connecting {len(contours)} contours to form continuous path")
    path = []
    
    # Start with the outermost contour
    outer_contour_coords = list(contours[0].coords)[:-1]  # Exclude the last point as it's the same as the first
    print(f"  Outer contour has {len(outer_contour_coords)} points")
    
    # Add the outer contour points to the path
    for x, y in outer_contour_coords:
        path.append((x, y))
    
    # Connect to inner contours
    for i in range(1, len(contours)):
        # Find the closest point on the next contour to the current position
        current_point = Point(path[-1])
        next_contour = contours[i]
        
        # Sample points along the contour to find the closest
        next_coords = list(next_contour.coords)[:-1]  # Exclude the last duplicate point
        print(f"  Contour {i} has {len(next_coords)} points")
        
        # Find the closest point
        min_dist = float('inf')
        closest_idx = 0
        
        for j, point in enumerate(next_coords):
            dist = current_point.distance(Point(point))
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        
        print(f"  Connecting to contour {i} at point {closest_idx} (distance: {min_dist:.4f})")
        
        # Create a smooth connection to the next contour
        connection_points = create_smooth_connection(
            path[-1], 
            next_coords[closest_idx], 
            toolpath_width
        )
        
        print(f"  Added {len(connection_points)} connection points")
        
        for point in connection_points:
            path.append(point)
        
        # Add the next contour, starting from the closest point and wrapping around
        for j in range(len(next_coords)):
            idx = (closest_idx + j) % len(next_coords)
            path.append(next_coords[idx])
    
    return path

def create_smooth_connection(start_point, end_point, toolpath_width):
    """
    Create a smooth connection between two points.
    
    Args:
        start_point (tuple): Starting point (x, y)
        end_point (tuple): Ending point (x, y)
        toolpath_width (float): Width of the toolpath
        
    Returns:
        list: List of points forming a smooth connection
    """
    # Calculate direction vector
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    # If points are very close, just return a direct line
    if distance < toolpath_width:
        return [end_point]
    
    # Create a smooth curve using a few intermediate points
    num_points = max(3, int(distance / toolpath_width))
    
    # Use a simple curve approximation
    connection_points = []
    
    for i in range(1, num_points):
        t = i / num_points
        # Simple quadratic curve
        x = start_point[0] + t * dx
        y = start_point[1] + t * dy
        connection_points.append((x, y))
    
    return connection_points

def visualize_fill_path(polygon, path, title="Continuous Fill Path"):
    """
    Visualize the polygon and the fill path.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon
        path (list): List of points representing the path
        title (str): Title for the plot
    """
    # Check if visualization is enabled
    from src.config import get_config
    config = get_config()
    if not config.get('visualize_fill_paths', True):
        print("Fill path visualization disabled in config")
        return
        
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the polygon
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon Boundary')
    
    # Plot holes if any
    for interior in polygon.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'b-', linewidth=2)
    
    # Plot the fill path
    if path:
        path_x, path_y = zip(*path)
        
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
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
