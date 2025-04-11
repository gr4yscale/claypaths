import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

def generate_continuous_fill(polygon, toolpath_width=1.0):
    """
    Generate a continuous fill pattern for a polygon.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill
        toolpath_width (float): Width of the toolpath
        
    Returns:
        list: List of points representing the continuous toolpath
    """
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        print("Invalid polygon for region fill")
        return []
    
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Calculate the number of parallel lines needed
    num_lines = int((maxy - miny) / toolpath_width) + 1
    
    # Generate parallel horizontal lines
    lines = []
    for i in range(num_lines):
        y = miny + i * toolpath_width
        if y <= maxy:
            # Alternate direction for zigzag pattern
            if i % 2 == 0:
                line = LineString([(minx - 1, y), (maxx + 1, y)])
            else:
                line = LineString([(maxx + 1, y), (minx - 1, y)])
            lines.append(line)
    
    # Intersect lines with the polygon
    toolpath_segments = []
    for line in lines:
        if polygon.intersects(line):
            intersection = polygon.intersection(line)
            if not intersection.is_empty:
                if intersection.geom_type == 'MultiLineString':
                    for segment in intersection.geoms:
                        toolpath_segments.append(segment)
                elif intersection.geom_type == 'LineString':
                    toolpath_segments.append(intersection)
    
    # Connect segments to form a continuous path
    continuous_path = connect_segments(toolpath_segments, toolpath_width)
    
    return continuous_path

def connect_segments(segments, toolpath_width):
    """
    Connect line segments to form a continuous path.
    
    Args:
        segments (list): List of LineString segments
        toolpath_width (float): Width of the toolpath
        
    Returns:
        list: List of points representing the continuous path
    """
    if not segments:
        return []
    
    # Sort segments by y-coordinate (assuming horizontal segments)
    segments.sort(key=lambda s: s.coords[0][1])
    
    # Initialize the continuous path with the first segment
    path = []
    for x, y in segments[0].coords:
        path.append((x, y))
    
    # Connect remaining segments
    current_point = Point(path[-1])
    remaining_segments = segments[1:]
    
    while remaining_segments:
        # Find the closest segment
        closest_idx = -1
        closest_dist = float('inf')
        closest_reverse = False
        
        for i, segment in enumerate(remaining_segments):
            # Check distance to start point
            start_point = Point(segment.coords[0])
            end_point = Point(segment.coords[-1])
            
            dist_to_start = current_point.distance(start_point)
            dist_to_end = current_point.distance(end_point)
            
            if dist_to_start < closest_dist:
                closest_idx = i
                closest_dist = dist_to_start
                closest_reverse = False
            
            if dist_to_end < closest_dist:
                closest_idx = i
                closest_dist = dist_to_end
                closest_reverse = True
        
        # Add connecting segment if needed
        if closest_dist > toolpath_width * 0.1:  # Small threshold to avoid duplicate points
            next_segment = remaining_segments[closest_idx]
            if closest_reverse:
                next_point = Point(next_segment.coords[-1])
            else:
                next_point = Point(next_segment.coords[0])
            
            # Add a vertical connecting line
            path.append((current_point.x, next_point.y))
        
        # Add the closest segment to the path
        next_segment = remaining_segments.pop(closest_idx)
        coords = list(next_segment.coords)
        
        if closest_reverse:
            coords.reverse()
        
        # Add points from the segment (skip first if it's too close to avoid duplicates)
        start_idx = 1 if closest_dist <= toolpath_width * 0.1 else 0
        for x, y in coords[start_idx:]:
            path.append((x, y))
        
        # Update current point
        current_point = Point(path[-1])
    
    return path

def visualize_fill_path(polygon, path, title="Continuous Fill Path"):
    """
    Visualize the polygon and the fill path.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon
        path (list): List of points representing the path
        title (str): Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the polygon
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    
    # Plot holes if any
    for interior in polygon.interiors:
        x, y = interior.xy
        ax.plot(x, y, 'b-', linewidth=2)
    
    # Plot the fill path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r-', linewidth=1, label='Fill Path')
        
        # Mark start and end points
        ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
