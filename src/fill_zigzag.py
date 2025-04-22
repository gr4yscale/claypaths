import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

def generate_zigzag_fill(polygon, toolpath_width, angle=45):
    """
    Generates a direction-parallel zigzag fill pattern for a polygon.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill.
        toolpath_width (float): Width of the toolpath (spacing between lines).
        angle (float): Angle of the fill lines in degrees (0 is horizontal).

    Returns:
        list[list[tuple[float, float]]]: A list of paths, where each path is a list of 
                                         points representing a continuous zigzag segment.
                                         Multiple paths are returned if holes break continuity.
    """
    if not polygon.is_valid or polygon.is_empty:
        print("ERROR: Invalid or empty polygon for zigzag fill.")
        return []

    print(f"  Generating zigzag fill with angle {angle} degrees and spacing {toolpath_width:.3f}")

    # Rotate the polygon so the fill lines are horizontal
    center = polygon.centroid
    rotated_polygon = rotate(polygon, -angle, origin=center)
    
    minx, miny, maxx, maxy = rotated_polygon.bounds
    
    # Generate horizontal lines
    lines = []
    y = miny + toolpath_width / 2.0
    while y < maxy:
        # Extend lines slightly beyond bounds to ensure intersection
        line = LineString([(minx - toolpath_width, y), (maxx + toolpath_width, y)])
        lines.append(line)
        y += toolpath_width

    if not lines:
        print("  No fill lines generated (polygon might be too small).")
        return []

    # Intersect lines with the rotated polygon's area
    intersected_segments = []
    for line in lines:
        # Intersect the infinite horizontal line with the polygon's area
        intersection = rotated_polygon.intersection(line)
        
        if intersection.is_empty:
            continue
            
        # Collect the resulting line segments that are inside the polygon
        if isinstance(intersection, LineString):
            # Ensure segment has non-zero length
            if intersection.length > 1e-6:
                intersected_segments.append(list(intersection.coords))
        elif isinstance(intersection, MultiLineString):
            for segment in intersection.geoms:
                 # Ensure segment has non-zero length
                if segment.length > 1e-6:
                    intersected_segments.append(list(segment.coords))
        # Ignore Point or other geometry types resulting from intersection

    if not intersected_segments:
        print("  No intersections found between fill lines and polygon.")
        return []

    # Sort segments primarily by Y, then by min X
    # Sort segments primarily by their average Y coordinate
    intersected_segments.sort(key=lambda seg: (seg[0][1] + seg[-1][1]) / 2)

    # Connect segments intelligently
    final_paths = []
    if not intersected_segments:
        return []

    current_path = []
    # Use squared distance for efficiency
    # Allow connection slightly larger than toolpath_width for diagonal moves
    connection_threshold_sq = (toolpath_width * 1.5) ** 2 

    # Start with the first segment
    # Decide initial direction arbitrarily (e.g., min X to max X)
    first_segment = sorted(intersected_segments[0], key=lambda p: p[0])
    current_path.extend(first_segment)
    
    remaining_segments = intersected_segments[1:]

    while remaining_segments:
        last_point = current_path[-1]
        
        # Find the closest segment endpoint among remaining segments
        best_match = None
        min_dist_sq = float('inf')
        best_segment_index = -1
        
        for i, segment in enumerate(remaining_segments):
            p1, p2 = segment[0], segment[-1] # Endpoints of the candidate segment
            
            # Calculate squared distances from last_point to both endpoints
            dist_sq1 = (last_point[0] - p1[0])**2 + (last_point[1] - p1[1])**2
            dist_sq2 = (last_point[0] - p2[0])**2 + (last_point[1] - p2[1])**2
            
            # Check if the closer endpoint is within the threshold
            if dist_sq1 < connection_threshold_sq and dist_sq1 < min_dist_sq:
                min_dist_sq = dist_sq1
                # Connect to p1, segment goes from p1 to p2
                best_match = (p1, p2) 
                best_segment_index = i
                
            if dist_sq2 < connection_threshold_sq and dist_sq2 < min_dist_sq:
                min_dist_sq = dist_sq2
                # Connect to p2, segment goes from p2 to p1
                best_match = (p2, p1)
                best_segment_index = i

        if best_match:
            # Found a connectable segment
            entry_point, exit_point = best_match
            
            # Add the connection (entry point might be same as last_point, that's ok)
            # Only add if it's truly different to avoid duplicate points if connection is perfect
            # if entry_point != last_point: # This check might be too strict, let optimizer handle duplicates
            current_path.append(entry_point)
            current_path.append(exit_point)
            
            # Remove the used segment from remaining_segments
            remaining_segments.pop(best_segment_index)
        else:
            # Cannot connect further, finalize current path
            if current_path:
                final_paths.append(current_path)
            
            # Start a new path with the next available segment
            if remaining_segments:
                 # Decide initial direction arbitrarily (e.g., min X to max X)
                next_segment = sorted(remaining_segments[0], key=lambda p: p[0])
                current_path = list(next_segment)
                remaining_segments.pop(0)
            else:
                 # No more segments left
                 current_path = [] # Ensure loop terminates

    # Add the last constructed path
    if current_path:
        final_paths.append(current_path)

    if not final_paths:
        print("  No paths generated after attempting connections.")
        return []

    # Rotate all generated paths back
    rotated_final_paths = []
    origin_np = np.array([center.x, center.y])
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    for path in final_paths:
        rotated_path = []
        for point in path:
            point_np = np.array(point)
            # Translate point to origin, rotate, translate back
            rotated_point = np.dot(rotation_matrix, point_np - origin_np) + origin_np
            rotated_path.append(tuple(rotated_point))
        if rotated_path:
            rotated_final_paths.append(rotated_path)

    total_points = sum(len(p) for p in rotated_final_paths)
    print(f"  Generated {len(rotated_final_paths)} continuous zigzag path(s) with {total_points} total points.")
    return rotated_final_paths
