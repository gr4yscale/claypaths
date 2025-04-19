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
        list: List of points representing the continuous zigzag toolpath.
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

    # Intersect lines with the rotated polygon
    intersected_segments = []
    for line in lines:
        intersection = rotated_polygon.intersection(line)
        if intersection.is_empty:
            continue
        if isinstance(intersection, LineString):
            intersected_segments.append(list(intersection.coords))
        elif isinstance(intersection, MultiLineString):
            for seg in intersection.geoms:
                intersected_segments.append(list(seg.coords))

    if not intersected_segments:
        print("  No intersections found between fill lines and polygon.")
        return []

    # Sort segments by their average Y coordinate, then by min X
    intersected_segments.sort(key=lambda seg: ( (seg[0][1] + seg[-1][1]) / 2, min(p[0] for p in seg) ))

    # Connect segments in a zigzag pattern
    zigzag_path = []
    for i, segment in enumerate(intersected_segments):
        # Ensure segment points are ordered by X coordinate
        segment.sort(key=lambda p: p[0])
        
        start_point = segment[0]
        end_point = segment[-1]

        if i % 2 == 0:  # Even index: left to right
            current_segment_points = segment
        else:  # Odd index: right to left
            current_segment_points = segment[::-1]

        # Connect to the previous segment's end point if not the first segment
        if zigzag_path:
            # Add the connection point (start of the current segment in zigzag order)
            zigzag_path.append(current_segment_points[0]) 
            
        # Add the points of the current segment
        zigzag_path.extend(current_segment_points)

    if not zigzag_path:
        print("  Zigzag path is empty after connecting segments.")
        return []

    # Rotate the path back
    final_path_points = []
    origin_np = np.array([center.x, center.y])
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    for point in zigzag_path:
        point_np = np.array(point)
        # Translate point to origin, rotate, translate back
        rotated_point = np.dot(rotation_matrix, point_np - origin_np) + origin_np
        final_path_points.append(tuple(rotated_point))

    print(f"  Generated zigzag path with {len(final_path_points)} points.")
    return final_path_points
