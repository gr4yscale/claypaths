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
    intersected_segments.sort(key=lambda seg: ( (seg[0][1] + seg[-1][1]) / 2, min(p[0] for p in seg) ))

    # Connect segments, breaking path if connection distance is too large (heuristic for holes)
    final_paths = []
    current_path = []
    connection_threshold_sq = (2 * toolpath_width) ** 2 # Use squared distance

    for i, segment in enumerate(intersected_segments):
        # Ensure segment points are ordered by X coordinate initially
        segment.sort(key=lambda p: p[0])
        
        # Determine segment direction based on index (alternating)
        if i % 2 == 0:  # Even index: left to right
            ordered_segment = segment
        else:  # Odd index: right to left
            ordered_segment = segment[::-1]

        if not current_path:
            # Start a new path
            current_path.extend(ordered_segment)
        else:
            # Check connection validity
            prev_end_point = current_path[-1]
            current_start_point = ordered_segment[0]
            
            # Calculate squared distance for connection
            dx = prev_end_point[0] - current_start_point[0]
            dy = prev_end_point[1] - current_start_point[1]
            dist_sq = dx*dx + dy*dy

            if dist_sq < connection_threshold_sq:
                # Connect segment to current path
                current_path.append(current_start_point) # Add connection point
                current_path.extend(ordered_segment[1:]) # Add rest of segment points
            else:
                # Connection distance too large, finalize current path and start new one
                if current_path:
                    final_paths.append(current_path)
                current_path = list(ordered_segment) # Start new path

    # Add the last path if it exists
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
