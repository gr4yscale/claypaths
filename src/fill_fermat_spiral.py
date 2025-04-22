import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import math

def generate_fermat_spiral_fill(polygon, toolpath_width):
    """
    Generates a continuous fill path based on a Fermat spiral within the polygon.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill.
        toolpath_width (float): The width of the toolpath (distance between spiral arms).

    Returns:
        list[tuple[float, float]]: A list of points representing the spiral path,
                                   clipped to the polygon boundary. Returns an empty
                                   list if generation fails or the polygon is unsuitable.
    """
    if not isinstance(polygon, Polygon) or polygon.is_empty or not polygon.is_valid:
        print("ERROR: Invalid or empty polygon provided for Fermat spiral fill.")
        return []

    # --- Basic Fermat Spiral Generation ---
    # r = a * sqrt(theta)
    # x = r * cos(theta)
    # y = r * sin(theta)

    # Determine spiral parameters
    # 'a' controls the spacing. We want the distance between arms to be toolpath_width.
    # For large theta, the distance between arms approaches pi*a.
    # So, toolpath_width = pi * a => a = toolpath_width / pi
    a = toolpath_width / np.pi

    # Determine the maximum radius needed to cover the polygon
    # Use the distance from the centroid to the furthest point on the exterior boundary
    centroid = polygon.centroid
    max_dist = 0
    if polygon.exterior:
        for coord in polygon.exterior.coords:
            dist = centroid.distance(Point(coord))
            max_dist = max(max_dist, dist)

    if max_dist <= 0:
        print("WARN: Polygon has zero or negative maximum extent for spiral.")
        return []

    # Calculate the maximum theta needed: max_r = a * sqrt(max_theta)
    # max_theta = (max_r / a)^2
    max_theta = (max_dist / a) ** 2

    # Increase max_theta slightly to ensure full coverage potential before clipping
    max_theta *= 1.2 # Add a 20% buffer

    # Generate spiral points
    # Choose a suitable number of points based on desired resolution/smoothness
    # More points for larger theta ranges or smaller toolpath widths
    num_points = int(max_theta * 10 / (2 * np.pi)) # Heuristic: ~10 points per revolution
    num_points = max(100, num_points) # Ensure a minimum number of points

    theta = np.linspace(0, max_theta, num_points)
    r = a * np.sqrt(theta)
    x = r * np.cos(theta) + centroid.x
    y = r * np.sin(theta) + centroid.y

    spiral_points = list(zip(x, y))

    if len(spiral_points) < 2:
        print("WARN: Not enough points generated for the spiral path.")
        return []

    # --- Clipping the Spiral to the Polygon ---
    try:
        spiral_line = LineString(spiral_points)

        # Intersect the spiral line with the polygon
        clipped_spiral = spiral_line.intersection(polygon)

        # The intersection might result in MultiLineString if the spiral goes in and out
        final_path_points = []
        if clipped_spiral.is_empty:
            print("WARN: Fermat spiral does not intersect the polygon.")
            return []
        elif clipped_spiral.geom_type == 'LineString':
            final_path_points = list(clipped_spiral.coords)
        elif clipped_spiral.geom_type == 'MultiLineString':
            # For now, just take the longest segment.
            # TODO: A more sophisticated approach might be needed to connect segments
            #       or handle multiple disjoint fill areas if the polygon shape requires it.
            longest_line = max(clipped_spiral.geoms, key=lambda line: line.length)
            final_path_points = list(longest_line.coords)
            print(f"WARN: Clipped spiral resulted in MultiLineString. Using longest segment ({len(final_path_points)} points).")
        else:
            print(f"WARN: Unexpected geometry type after clipping: {clipped_spiral.geom_type}")
            return []

    except Exception as e:
        print(f"ERROR: Exception during spiral clipping: {e}")
        return []

    if not final_path_points:
         print("WARN: No path points after clipping.")
         return []

    print(f"Successfully generated Fermat spiral fill path with {len(final_path_points)} points.")
    return final_path_points
