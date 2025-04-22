import numpy as np
from shapely.geometry import Point, Polygon, LineString
import math

# --- Hilbert Curve Generation Logic ---

def _int_to_Hilbert(x, y, order):
    """Convert 2D coordinates to 1D Hilbert curve distance."""
    n = 2**order
    h = 0
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        h += s * s * ((3 * rx) ^ ry)
        x, y = _rotate(s, x, y, rx, ry)
        s //= 2
    return h

def _rotate(n, x, y, rx, ry):
    """Rotate/flip a quadrant appropriately."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # Swap x and y
        x, y = y, x
    return x, y

def _Hilbert_to_int(h, order):
    """Convert 1D Hilbert curve distance back to 2D coordinates (integer grid)."""
    n = 2**order
    x = y = 0
    t = h
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rotate(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

# --- Main Fill Function ---

def generate_hilbert_fill(polygon, toolpath_width):
    """
    Generates a fill path for a polygon using a Hilbert curve ordering approach.

    Creates a grid within the polygon's bounds, filters points inside the polygon,
    sorts these points based on their Hilbert curve distance, and connects them.

    Args:
        polygon (Polygon): The polygon to fill.
        toolpath_width (float): The desired spacing between path segments.

    Returns:
        list[tuple[float, float]]: A single list of points representing the fill path,
                                   or an empty list if generation fails.
    """
    if not isinstance(polygon, Polygon) or polygon.is_empty or not polygon.is_valid:
        print("  Hilbert Fill: Invalid or empty polygon provided.")
        return []

    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny

    if width <= 0 or height <= 0:
        print("  Hilbert Fill: Polygon has zero width or height.")
        return []

    # Determine grid size based on toolpath width
    # Aim for grid cells roughly toolpath_width wide
    grid_res_x = max(1, int(np.ceil(width / toolpath_width)))
    grid_res_y = max(1, int(np.ceil(height / toolpath_width)))

    # Determine Hilbert curve order needed to cover the grid
    # We need 2^order >= max(grid_res_x, grid_res_y)
    order = math.ceil(math.log2(max(grid_res_x, grid_res_y)))
    n_hilbert = 2**order # Size of the Hilbert grid (n x n)

    print(f"  Hilbert Fill: Bounds=({minx:.2f},{miny:.2f})->({maxx:.2f},{maxy:.2f}), "
          f"GridRes=({grid_res_x},{grid_res_y}), Order={order}, HilbertN={n_hilbert}")

    # Generate grid points scaled to the Hilbert curve's integer grid
    grid_points_hilbert = []
    for i in range(grid_res_x + 1): # +1 to include boundary
        for j in range(grid_res_y + 1):
            # Map grid indices (i, j) to Hilbert integer coordinates (hx, hy)
            # Scale i to range [0, n_hilbert-1] and j to range [0, n_hilbert-1]
            hx = int(round((i / grid_res_x) * (n_hilbert - 1)))
            hy = int(round((j / grid_res_y) * (n_hilbert - 1)))

            # Map grid indices (i, j) back to real-world coordinates (px, py)
            px = minx + (i / grid_res_x) * width
            py = miny + (j / grid_res_y) * height
            point = Point(px, py)

            # Check if the real-world point is inside the polygon
            # Use a small buffer for contains check to be more inclusive near boundaries
            if polygon.buffer(1e-9).contains(point):
                # Calculate Hilbert value for the integer coordinates
                h_value = _int_to_Hilbert(hx, hy, order)
                grid_points_hilbert.append({'point': (px, py), 'h_value': h_value, 'hx': hx, 'hy': hy})

    if not grid_points_hilbert:
        print("  Hilbert Fill: No grid points found inside the polygon.")
        return []

    # Sort points based on Hilbert value
    grid_points_hilbert.sort(key=lambda p: p['h_value'])

    # Create the final path
    hilbert_path = [p['point'] for p in grid_points_hilbert]

    # Optional: Simplify the path slightly if needed (e.g., remove collinear points)
    # This might be necessary if the grid resolution is very high
    # For now, return the direct path through sorted grid points

    print(f"  Hilbert Fill: Generated path with {len(hilbert_path)} points.")
    return hilbert_path

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Example polygon (e.g., a square)
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    tool_width = 1.0

    path = generate_hilbert_fill(poly, tool_width)

    if path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = poly.exterior.xy
        ax.plot(x, y, 'b-', label='Polygon')

        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r.-', label='Hilbert Path')

        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        plt.show()
