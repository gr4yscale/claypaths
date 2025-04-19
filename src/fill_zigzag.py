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
                                         points representing an individual fill segment.
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

    # Sort segments primarily by their Y coordinate for consistent processing order
    intersected_segments.sort(key=lambda seg: (seg[0][1] + seg[-1][1]) / 2)

    # Rotate each segment back individually
    final_paths = []
    origin_np = np.array([center.x, center.y])
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    for segment in intersected_segments:
        # Ensure segment points are ordered by X coordinate before rotation
        segment.sort(key=lambda p: p[0]) 
        
        rotated_path = []
        for point in segment:
            point_np = np.array(point)
            # Translate point to origin, rotate, translate back
            rotated_point = np.dot(rotation_matrix, point_np - origin_np) + origin_np
            rotated_path.append(tuple(rotated_point))
        
        if rotated_path:
            final_paths.append(rotated_path)

    print(f"  Generated {len(final_paths)} separate zigzag segments.")
    return final_paths
