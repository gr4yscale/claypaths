import math
import copy
import os
import sys
from typing import List, Tuple, Optional, Dict, Any

# Add project root to sys.path to allow importing project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pyclipper
from shapely.geometry import Polygon as ShapelyPolygon, LineString as ShapelyLineString, Point as ShapelyPoint
from shapely.strtree import STRtree
import numpy as np
try:
    import cv2 # For image processing
    from scipy.ndimage import gaussian_filter # For Gaussian blur
    cv2_available = True
except ImportError:
    cv2_available = False
    print("Warning: OpenCV (cv2) or SciPy not found. Rasterization optimization will be skipped.")
    # Define dummy gaussian_filter if scipy is missing but cv2 might be present (less likely scenario)
    if 'gaussian_filter' not in locals():
        def gaussian_filter(img, sigma):
            print("Warning: SciPy not found, cannot apply Gaussian blur.")
            return img


# Import project modules
from src.stl_loader import load_stl
from src.slicer import slice_mesh
from src.config import load_config, get_config

# Define a scaling factor for Pyclipper (uses integers)
CLIPPER_SCALE = 10000.0

#-----------------------------------------------------------------------------
# 1. Data Structures
#-----------------------------------------------------------------------------

class Point:
    """Represents a 2D point."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        # Use tolerance for floating point comparison
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __repr__(self):
        return f"Point({self.x:.3f}, {self.y:.3f})"

    def distance_to(self, other: 'Point') -> float:
        """Calculates Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Segment:
    """Represents a line segment defined by two points."""
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def length(self) -> float:
        """Calculates the length of the segment."""
        return self.p1.distance_to(self.p2)

    def __repr__(self):
        return f"Segment({self.p1}, {self.p2})"

    def point_projection(self, p: Point) -> Point:
        """Projects a point onto the infinite line defined by the segment."""
        ap = (p.x - self.p1.x, p.y - self.p1.y)
        ab = (self.p2.x - self.p1.x, self.p2.y - self.p1.y)
        ab2 = ab[0]**2 + ab[1]**2
        if ab2 == 0: # Segment has zero length
            return self.p1
        ap_dot_ab = ap[0] * ab[0] + ap[1] * ab[1]
        t = ap_dot_ab / ab2
        # Clamp t to [0, 1] to project onto the segment itself, not the infinite line
        # t = max(0, min(1, t))
        # Note: Algorithm 2 seems to project onto the infinite line based on context
        proj_x = self.p1.x + t * ab[0]
        proj_y = self.p1.y + t * ab[1]
        return Point(proj_x, proj_y)

    def distance_to_point(self, p: Point) -> float:
         """Calculates the shortest distance from a point to the line segment."""
         proj = self.point_projection(p)
         # Check if projection is within the segment bounds
         ap = (p.x - self.p1.x, p.y - self.p1.y)
         ab = (self.p2.x - self.p1.x, self.p2.y - self.p1.y)
         ab2 = ab[0]**2 + ab[1]**2
         if ab2 == 0: return p.distance_to(self.p1)
         ap_dot_ab = ap[0] * ab[0] + ap[1] * ab[1]
         t = ap_dot_ab / ab2

         if t < 0.0:
             closest_point = self.p1
         elif t > 1.0:
             closest_point = self.p2
         else:
             closest_point = proj # Projection lies on the segment

         return p.distance_to(closest_point)


class Contour:
    """Represents a closed contour (polygon)."""
    def __init__(self, points: List[Point]):
        if not points:
            raise ValueError("Contour must have points.")
        # Ensure contour is closed if it isn't already
        if points[0] != points[-1]:
            points.append(points[0])
        self.points = points
        self.level: int = -1 # Level assigned by re-leveling
        self.type: Optional[int] = None # 0 for outer, >0 for hole ID, None for intermediate offsets
        self.is_hole: bool = False # Flag if it originated from a hole
        self.original_level: int = -1 # Level assigned by the offsetting algorithm
        self.breakpoints: List[Tuple[Point, Point, Point, Point]] = [] # Stores (p1, p2, p1_proj, p2_proj)
        self.connecting_segments: List[Segment] = [] # Segments added between breakpoints

    def __repr__(self):
        type_str = f"Hole({self.type})" if self.is_hole else f"Outer({self.type})"
        return f"Contour(OrigLevel={self.original_level}, NewLevel={self.level}, Type={type_str}, Points={len(self.points)})"

    def get_segments(self) -> List[Segment]:
        """Returns the list of segments forming the contour."""
        return [Segment(self.points[i], self.points[i+1]) for i in range(len(self.points) - 1)]

    def length(self) -> float:
        """Calculates the total length (perimeter) of the contour."""
        return sum(seg.length() for seg in self.get_segments())

    def get_point_at_dist(self, distance: float) -> Tuple[Optional[Point], Optional[Segment], int]:
        """Finds the point and segment at a given distance along the contour from the start."""
        accumulated_length = 0.0
        segments = self.get_segments()
        for i, seg in enumerate(segments):
            seg_len = seg.length()
            if accumulated_length + seg_len >= distance - 1e-9: # Tolerance for float comparison
                remaining_dist = distance - accumulated_length
                if seg_len == 0: return seg.p1, seg, i # Handle zero-length segment
                ratio = remaining_dist / seg_len
                # Interpolate point
                x = seg.p1.x + ratio * (seg.p2.x - seg.p1.x)
                y = seg.p1.y + ratio * (seg.p2.y - seg.p1.y)
                return Point(x, y), seg, i
            accumulated_length += seg_len
        # Distance exceeds contour length (shouldn't happen for closed loop if dist < length)
        return None, None, -1

class SubPath:
    """Represents an open path derived from breaking a contour."""
    def __init__(self, points: List[Point]):
        self.points = points

    def __repr__(self):
        return f"SubPath(Points={len(self.points)})"

#-----------------------------------------------------------------------------
# 2. Geometric Helper Functions (Basic Implementation)
#    -> Recommend using Shapely for robust geometry operations
#-----------------------------------------------------------------------------

def find_closest_segment_to_point(point: Point, target_contour: Contour) -> Tuple[Optional[Segment], float, int]:
    """
    Finds the segment in target_contour closest to the given point using Shapely for robust distance calculation.

    Args:
        point: The query point.
        target_contour: The contour to search within.

    Returns:
        A tuple containing:
        - The closest Segment object (or None if not found).
        - The minimum distance calculated by Shapely.
        - The index of the closest segment in the contour's segment list.
    """
    if not target_contour.points or len(target_contour.points) < 2:
        return None, float('inf'), -1

    # Use Shapely for robust distance calculation
    try:
        shapely_point = ShapelyPoint(point.x, point.y)
        # Create LineString from contour points (excluding duplicate end point)
        contour_coords = [(p.x, p.y) for p in target_contour.points[:-1]]
        # Ensure at least two points for LineString
        if len(contour_coords) < 2:
             # Handle degenerate case (e.g., contour is just one point repeated)
             if contour_coords:
                 dist = point.distance_to(target_contour.points[0])
                 segments = target_contour.get_segments()
                 return segments[0] if segments else None, dist, 0 if segments else -1
             else:
                 return None, float('inf'), -1

        shapely_contour_line = ShapelyLineString(contour_coords)
        min_dist_shapely = shapely_point.distance(shapely_contour_line)
    except Exception as e:
        print(f"Error creating Shapely objects or calculating distance: {e}")
        # Fallback to simple iteration if Shapely fails
        min_dist_shapely = float('inf')


    # Find the original Segment object that corresponds to this minimum distance
    closest_seg: Optional[Segment] = None
    closest_seg_index: int = -1
    min_segment_dist_diff = float('inf')
    segments = target_contour.get_segments()

    if not segments: # Should not happen if points exist, but check anyway
        return None, min_dist_shapely, -1

    for i, seg in enumerate(segments):
        # Use the segment's own distance calculation method
        dist_to_segment = seg.distance_to_point(point)

        # Find the segment whose distance is closest to Shapely's minimum distance
        diff = abs(dist_to_segment - min_dist_shapely)
        if diff < min_segment_dist_diff:
            min_segment_dist_diff = diff
            closest_seg = seg
            closest_seg_index = i

    # Optional: Add a tolerance check if needed, but finding the minimum difference should work
    # if min_segment_dist_diff > 1e-6: # Arbitrary tolerance
    #     print(f"Warning: Large difference ({min_segment_dist_diff:.3e}) between Shapely distance and segment distance.")

    return closest_seg, min_dist_shapely, closest_seg_index


#-----------------------------------------------------------------------------
# Helper functions for conversions
#-----------------------------------------------------------------------------

def shapely_polygon_to_clipper(polygon: ShapelyPolygon) -> List[List[Tuple[int, int]]]:
    """Converts a Shapely Polygon to Pyclipper path format (scaled integers)."""
    exterior = [(int(x * CLIPPER_SCALE), int(y * CLIPPER_SCALE)) for x, y in polygon.exterior.coords]
    interiors = []
    for interior in polygon.interiors:
        interiors.append([(int(x * CLIPPER_SCALE), int(y * CLIPPER_SCALE)) for x, y in interior.coords])
    # Pyclipper expects exterior first, then interiors
    return [exterior] + interiors

def clipper_path_to_contour(path: List[Tuple[int, int]]) -> Contour:
    """Converts a Pyclipper path (scaled integers) back to a Contour object."""
    points = [Point(x / CLIPPER_SCALE, y / CLIPPER_SCALE) for x, y in path]
    # Ensure the contour is closed for the Contour class logic
    if not points or len(points) < 3:
        return Contour([]) # Return empty if not enough points
    if points[0] != points[-1]:
        points.append(points[0])
    return Contour(points)

def clipper_paths_to_contours(paths: List[List[Tuple[int, int]]]) -> List[Contour]:
    """Converts multiple Pyclipper paths to a list of Contour objects."""
    return [clipper_path_to_contour(path) for path in paths if len(path) >= 3]


#-----------------------------------------------------------------------------
# 3. Algorithm 1: Re-level the offset contours
#-----------------------------------------------------------------------------

def re_level_contours(offset_results: List[List[Contour]], initial_outer: List[Contour], initial_holes: List[Contour]) -> List[List[Contour]]:
    """
    Re-levels contours generated by an offsetting algorithm.
    Assumes offset_results[0] contains the initial contours (outer and holes).
    Levels increase inwards.

    Args:
        offset_results: List where each inner list contains contours from one offset iteration.
                        offset_results[0] = initial contours.
                        offset_results[1] = first offset, etc.
        initial_outer: The list of original outer contours.
        initial_holes: The list of original hole contours.

    Returns:
        A list of lists of contours, re-leveled and typed.
        result[0] is the outermost level (original contours), result[-1] is the innermost.
    """
    if not offset_results:
        return []

    num_offset_levels = len(offset_results)
    all_contours_flat: List[Contour] = []

    # --- Step 1: Assign initial types and levels ---
    # The offset_results already contain contours generated level by level.
    # We need to assign the 'type' (outer=0, hole=1, hole=2...) based on the initial contours.
    # And assign the 'original_level' based on the offset iteration.

    all_contours_flat: List[Contour] = []
    initial_contours_level_0 = offset_results[0] # The first list contains the initial shapes

    # Identify initial outer and holes (assuming simple case: one outer, rest are holes)
    # A more robust way would involve checking polygon containment or winding order.
    initial_outer_contours = []
    initial_hole_contours = []
    if initial_contours_level_0:
        # Assume the largest area polygon is the outer one (simplification)
        areas = [(c.length(), c) for c in initial_contours_level_0] # Use length as proxy for now
        areas.sort(key=lambda x: x[0], reverse=True)
        initial_outer_contours.append(areas[0][1])
        initial_hole_contours.extend([a[1] for a in areas[1:]])

    num_hole = 0
    for i, contour in enumerate(initial_contours_level_0):
        contour.original_level = 0
        is_initial_hole = contour in initial_hole_contours
        if is_initial_hole:
            contour.is_hole = True
            contour.type = num_hole + 1
            num_hole += 1
        else: # Assume outer
            contour.is_hole = False
            contour.type = 0
        all_contours_flat.append(contour)

    # --- Step 2: Assign types and original levels to subsequent offset contours ---
    # Use proximity/containment to link contours between levels and propagate type.
    for level_idx in range(1, num_offset_levels):
        prev_level_contours = offset_results[level_idx - 1]
        current_level_contours = offset_results[level_idx]

        for contour in current_level_contours:
            contour.original_level = level_idx
            min_dist = float('inf')
            parent_type = None
            parent_is_hole = False

            # Calculate centroid of current contour
            if not contour.points: continue
            centroid_x = sum(p.x for p in contour.points) / len(contour.points)
            centroid_y = sum(p.y for p in contour.points) / len(contour.points)
            current_centroid = Point(centroid_x, centroid_y)

            # Find the closest contour in the *previous* level
            for prev_contour in prev_level_contours:
                if not prev_contour.points: continue
                prev_centroid_x = sum(p.x for p in prev_contour.points) / len(prev_contour.points)
                prev_centroid_y = sum(p.y for p in prev_contour.points) / len(prev_contour.points)
                prev_centroid = Point(prev_centroid_x, prev_centroid_y)
                dist = current_centroid.distance_to(prev_centroid)

                if dist < min_dist:
                    min_dist = dist
                    parent_type = prev_contour.type
                    parent_is_hole = prev_contour.is_hole

            contour.type = parent_type
            contour.is_hole = parent_is_hole
            all_contours_flat.append(contour)

    # --- Step 3: Re-level based on type and original level ---
    max_levels_by_type: Dict[int, int] = {}
    for contour in all_contours_flat:
        if contour.type is not None:
            max_levels_by_type[contour.type] = max(max_levels_by_type.get(contour.type, -1), contour.original_level)

    final_max_level = 0
    for contour in all_contours_flat:
        if contour.type is not None:
            max_level_for_this_type = max_levels_by_type[contour.type]
            # New level increases inwards: level = max_original_level_for_type - original_level
            contour.level = max_level_for_this_type - contour.original_level
            final_max_level = max(final_max_level, contour.level)
        else:
            contour.level = -1 # Should not happen if typing worked

    # --- Step 4: Group contours by the new level ---
    releveled_contours_grouped: List[List[Contour]] = [[] for _ in range(final_max_level + 1)]
    for contour in all_contours_flat:
        if contour.level != -1:
            # Ensure list is large enough (can happen with sparse levels)
            while len(releveled_contours_grouped) <= contour.level:
                releveled_contours_grouped.append([])
            releveled_contours_grouped[contour.level].append(contour)

    # Filter out empty level lists that might have been created
    releveled_contours_grouped = [level for level in releveled_contours_grouped if level]

    return releveled_contours_grouped


#-----------------------------------------------------------------------------
# 4. Algorithm 2: Finding breakpoints
#-----------------------------------------------------------------------------

def find_breakpoints(leveled_contours: List[List[Contour]], line_spacing: float, layer_index: int, n_layer_period: int) -> List[List[Contour]]:
    """
    Identifies breakpoints to connect adjacent contours. Modifies contours in place.

    Args:
        leveled_contours: Contours grouped by level (output of re_level_contours).
        line_spacing: The distance 'd' between offset contours.
        layer_index: The current layer number (0-based).
        n_layer_period: How often to change the breakpoint selection strategy (every 'n' layers).

    Returns:
        The input leveled_contours list, with breakpoint information added to each contour.
    """
    num_levels = len(leveled_contours)
    contours_with_breaks = copy.deepcopy(leveled_contours) # Work on a copy

    # --- Build Spatial Index ONCE for ALL Segments ---
    all_segments_data = []
    for level_idx in range(num_levels):
        for contour_k in contours_with_breaks[level_idx]:
             if not contour_k.points or len(contour_k.points) < 3: continue
             # Store level index on the contour object itself for easy access later
             contour_k.level = level_idx # Ensure level is set correctly here if not done before
             for seg_idx, seg in enumerate(contour_k.get_segments()):
                 # Create Shapely LineString for indexing
                 shapely_line = ShapelyLineString([(seg.p1.x, seg.p1.y), (seg.p2.x, seg.p2.y)])
                 # Store tuple: (geometry, original_segment_object, original_contour_object)
                 all_segments_data.append((shapely_line, seg, contour_k))

    # Build tree only from geometries
    all_geometries_for_index = [item[0] for item in all_segments_data]
    if not all_geometries_for_index:
        print("Warning: No segments found in any level to build spatial index.")
        return contours_with_breaks # Return early if no segments exist
    tree = STRtree(all_geometries_for_index)
    print(f"Built STRtree with {len(all_geometries_for_index)} total segments.")
    # --- End Spatial Index Build ---

    # Iterate from inner levels outwards (highest level index to 0)
    # This direction is crucial for the logic of connecting inwards.
    for i in range(num_levels - 1, -1, -1):
        current_level_contours = contours_with_breaks[i]

        # No need to check target_level_start_index >= num_levels here,
        # the filtering logic inside the query handles it.

        for j, contour in enumerate(current_level_contours):
            # Ensure contour is valid and has non-negligible length
            if not contour.points or len(contour.points) < 3:
                # print(f"Debug: Skipping invalid contour {j} in level {i}")
                continue
            contour_len = contour.length()
            if contour_len < 1e-6:
                # print(f"Debug: Skipping zero-length contour {j} in level {i}")
                continue

            # --- Step 1: Find candidate breakpoint p1 ---
            # Vary starting point based on layer index and contour index to distribute breakpoints
            # Use modulo arithmetic to cycle through starting positions
            start_offset_ratio = ((layer_index + j) % n_layer_period) / n_layer_period
            start_dist = start_offset_ratio * contour_len

            p1, seg_p1, seg_p1_idx = contour.get_point_at_dist(start_dist)
            if p1 is None or seg_p1 is None:
                print(f"Warning: Could not find p1 at dist {start_dist:.2f} on contour {j} level {i}")
                continue # Should not happen on valid closed contour

            # --- Step 2 & 3: Find p2 at distance 'line_spacing' from p1 ---
            # Find p2 by walking 'line_spacing' distance from p1 along the contour
            p2_dist = (start_dist + line_spacing) % contour_len # Wrap around contour
            p2, seg_p2, seg_p2_idx = contour.get_point_at_dist(p2_dist)
            if p2 is None or seg_p2 is None:
                print(f"Warning: Could not find p2 at dist {p2_dist:.2f} on contour {j} level {i}")
                continue

            # Ensure p1 and p2 are distinct points
            if p1.distance_to(p2) < 1e-6:
                # print(f"Debug: p1 and p2 are too close on contour {j} level {i}. Skipping breakpoint.")
                continue

            # --- Ambiguity in Original Step 2 ---
            # The original pseudo-code mentions checking if the segment *containing p1*
            # is shorter than line_spacing and potentially advancing p1.
            # This is ambiguous and might lead to complex logic. We proceed assuming
            # valid p1 and p2 have been found, regardless of the length of seg_p1 or seg_p2.
            # A check like `if seg_p1.length() < line_spacing:` could be added, but the
            # corrective action (e.g., how far to advance p1) is unclear.

            # --- Step 4, 5, 6: Find closest valid target segment using Spatial Index ---
            shapely_p1 = ShapelyPoint(p1.x, p1.y)
            search_radius = line_spacing * 3.0 # Search radius around p1
            query_geom = shapely_p1.buffer(search_radius)
            nearby_indices = tree.query(query_geom) # Indices in geometries_for_index

            valid_candidates = []
            for index in nearby_indices:
                # Retrieve data associated with the indexed geometry from the global list
                shapely_line, seg_sl, cand_target_contour = all_segments_data[index]

                # Filter: Must be in a subsequent level (level > i) and match type
                if cand_target_contour.level > i and cand_target_contour.type == contour.type:
                    # Calculate exact distance using Shapely
                    dist_sl = shapely_p1.distance(shapely_line)
                    valid_candidates.append({
                        'distance': dist_sl,
                        'level': cand_target_contour.level, # Level of the target contour
                        'segment': seg_sl,
                        'contour': cand_target_contour
                    })

            # Sort candidates: first by level (ascending), then by distance (ascending)
            valid_candidates.sort(key=lambda c: (c['level'], c['distance']))

            # Find the best match according to original logic's criteria
            closest_target_seg: Optional[Segment] = None
            target_contour_k: Optional[Contour] = None
            min_dist_to_target = float('inf')
            found_target = False

            for candidate in valid_candidates:
                # Check distance tolerance (must be reasonably close to line_spacing)
                if candidate['distance'] <= line_spacing * 1.5:
                    closest_target_seg = candidate['segment']
                    target_contour_k = candidate['contour']
                    min_dist_to_target = candidate['distance']
                    found_target = True
                    # print(f"Debug: Found target seg via spatial index for L{i} C{j} -> L{target_contour_k.level} Dist: {min_dist_to_target:.2f}")
                    break # Found the best one (lowest level, then lowest distance within tolerance)

            if not found_target:
                 # print(f"Warning: Could not find suitable target segment via spatial index for point {p1} originating from contour {j} level {i}")
                 continue # Could not find a suitable target segment

            # --- Step 7: Find projected point p1_proj of p1 onto the target segment sl ---
            # Use the original Segment object's projection method
            p1_proj = closest_target_seg.point_projection(p1)

            # --- Step 8: Find p2_proj (Simplified Approach) ---
            # Project p2 onto the *chosen* target contour (target_contour_k).
            # Use the efficient find_closest_segment_to_point restricted to this contour.
            seg_p2_target, dist_p2_target, _ = find_closest_segment_to_point(p2, target_contour_k)
            if seg_p2_target is None:
                # This should ideally not happen if target_contour_k is valid
                print(f"Warning: Could not find target segment for point {p2} near chosen target contour L{target_contour_k.level}")
                continue
            p2_proj = seg_p2_target.point_projection(p2)

            # Ensure projections are valid
            if p1_proj is None or p2_proj is None:
                 print(f"Warning: Failed to calculate projections for breakpoint on contour {j} level {i}")
                 continue

            # --- Step 9: Store breakpoint information and connecting segments ---
            # Store the four key points defining the break and connection.
            contour.breakpoints.append((p1, p2, p1_proj, p2_proj))
            # Store the conceptual connecting segments (will be added as subpaths later)
            contour.connecting_segments.append(Segment(p1, p1_proj))
            contour.connecting_segments.append(Segment(p2, p2_proj))
            # Note: Breakpoints are stored on the contour they originate from (level i).
            # The sub-path formation logic will use this information.

    return contours_with_breaks


#-----------------------------------------------------------------------------
# 5. Algorithm 3: Forming sub-paths
#-----------------------------------------------------------------------------

def form_sub_paths(contours_with_breaks: List[List[Contour]]) -> List[SubPath]:
    """
    Forms open sub-paths by breaking contours at the identified breakpoints.

    Args:
        contours_with_breaks: Contours with breakpoint information added.

    Returns:
        A list of SubPath objects.
    """
    list_sub_paths: List[SubPath] = []
    processed_contours = copy.deepcopy(contours_with_breaks) # Work on a copy

    for i in range(len(processed_contours)):
        for j in range(len(processed_contours[i])):
            contour = processed_contours[i][j]
            if not contour.points or len(contour.points) < 3: continue

            # Get original contour points (excluding connecting segments for now)
            # Need to handle the breaks introduced by connecting segments.
            # The breakpoints p1, p2 on *this* contour mark the interruptions.

            # Create a list of all points, including breakpoints p1 and p2
            all_points_on_contour = contour.points[:-1] # Exclude duplicate end point
            break_indices = set() # Indices where breaks occur

            # Find indices of p1 and p2 points on the contour
            # This requires inserting p1/p2 into the point list if they aren't vertices
            # Or, more simply, track the segments interrupted by breaks.

            # Simpler approach: Identify segments that contain p1 or p2 from breakpoints
            interrupted_segment_indices = set()
            break_points_on_this_contour = set() # Store (x,y) tuples

            for bp_info in contour.breakpoints:
                p1, p2, _, _ = bp_info
                break_points_on_this_contour.add((p1.x, p1.y))
                break_points_on_this_contour.add((p2.x, p2.y))

                # Find which segments p1 and p2 lie on (or are close to)
                for k, seg in enumerate(contour.get_segments()):
                    # Check if p1 or p2 is very close to this segment's endpoints or interior
                    if seg.distance_to_point(p1) < 1e-6:
                         interrupted_segment_indices.add(k)
                    if seg.distance_to_point(p2) < 1e-6:
                         interrupted_segment_indices.add(k)


            # Traverse the contour point by point, creating subpaths between breaks
            current_sub_path_points: List[Point] = []
            if not all_points_on_contour: continue

            start_index = 0
            num_points = len(all_points_on_contour)

            for k in range(num_points + 1): # Iterate one extra time to handle wrap-around
                current_idx = k % num_points
                current_point = all_points_on_contour[current_idx]
                current_segment_idx = (current_idx - 1 + num_points) % num_points # Segment ending at current_point

                # Add point to current subpath
                if not current_sub_path_points or current_point != current_sub_path_points[-1]:
                     current_sub_path_points.append(current_point)

                # Check if the segment *ending* at this point was a break segment
                # Or if the point itself is a breakpoint
                is_break_point = (current_point.x, current_point.y) in break_points_on_this_contour
                is_break_segment = current_segment_idx in interrupted_segment_indices

                # If we hit a break point/segment AND the subpath is not empty, end the current subpath
                if (is_break_point or is_break_segment) and len(current_sub_path_points) > 1:
                    # Check if the *next* point is also a breakpoint (can happen if p1/p2 are vertices)
                    # Avoid creating tiny subpaths if breaks are adjacent
                    next_idx = (current_idx + 1) % num_points
                    next_point = all_points_on_contour[next_idx]
                    next_point_is_break = (next_point.x, next_point.y) in break_points_on_this_contour
                    next_segment_idx = current_idx
                    next_segment_is_break = next_segment_idx in interrupted_segment_indices

                    # Finalize subpath
                    list_sub_paths.append(SubPath(current_sub_path_points))
                    # Start new subpath, potentially starting with the current break point
                    current_sub_path_points = [current_point]


            # Add any remaining points after the loop
            if len(current_sub_path_points) > 1:
                 # Check if it closes on itself and matches the first subpath start
                 first_subpath_start = list_sub_paths[0].points[0] if list_sub_paths else None
                 last_subpath_end = current_sub_path_points[-1] if current_sub_path_points else None
                 if first_subpath_start and last_subpath_end and first_subpath_start == last_subpath_end:
                      # Merge with first subpath
                      list_sub_paths[0].points = current_sub_path_points + list_sub_paths[0].points[1:]
                 else:
                      list_sub_paths.append(SubPath(current_sub_path_points))


    # Add the connecting segments themselves as subpaths
    all_connecting_segments: List[Segment] = []
    for level in processed_contours:
        for contour in level:
            all_connecting_segments.extend(contour.connecting_segments)

    for seg in all_connecting_segments:
        # Ensure connecting segments are not zero length
        if not math.isclose(seg.p1.x, seg.p2.x) or not math.isclose(seg.p1.y, seg.p2.y):
             list_sub_paths.append(SubPath([seg.p1, seg.p2]))


    return list_sub_paths


#-----------------------------------------------------------------------------
# 6. Generating continuous path (Connecting Sub-paths) - Improved with Spatial Index
#-----------------------------------------------------------------------------

def connect_sub_paths(sub_paths: List[SubPath]) -> List[Point]:
    """
    Connects sub-paths into a single continuous global toolpath using a spatial index
    for potentially faster endpoint matching.

    Args:
        sub_paths: A list of SubPath objects.

    Returns:
        A list of Points representing the final toolpath.
    """
    if not sub_paths:
        return []

    num_paths = len(sub_paths)
    # Filter out empty paths initially and map original index to filtered index
    valid_paths = [(i, path) for i, path in enumerate(sub_paths) if path.points]
    if not valid_paths:
        return []

    num_valid_paths = len(valid_paths)
    original_indices = [item[0] for item in valid_paths]
    filtered_sub_paths = [item[1] for item in valid_paths]

    # --- Build Spatial Index for Endpoints ---
    endpoints_data = []
    shapely_points = []
    for i in range(num_valid_paths):
        path = filtered_sub_paths[i]
        start_pt = path.points[0]
        end_pt = path.points[-1]
        shapely_start = ShapelyPoint(start_pt.x, start_pt.y)
        shapely_end = ShapelyPoint(end_pt.x, end_pt.y)
        # Store: (shapely_geom, index_in_filtered_list, is_start_point)
        endpoints_data.append((shapely_start, i, True))
        endpoints_data.append((shapely_end, i, False))
        shapely_points.append(shapely_start)
        shapely_points.append(shapely_end)

    if not shapely_points: return [] # Should not happen if valid_paths is not empty
    tree = STRtree(shapely_points)
    # --- End Spatial Index Build ---

    used_indices = set() # Stores indices *from the filtered_sub_paths list*
    global_path: List[Point] = []

    # Start with the first valid path (index 0 in filtered list)
    start_idx_filtered = 0
    current_path_idx_filtered = start_idx_filtered
    current_path = filtered_sub_paths[current_path_idx_filtered]
    global_path.extend(current_path.points)
    used_indices.add(current_path_idx_filtered)
    num_remaining = num_valid_paths - 1

    # Tolerance for comparing floating point coordinates
    # Should be slightly larger than 1/CLIPPER_SCALE to account for float errors.
    CONNECT_TOLERANCE = 1e-3 # 0.001 mm (1 micrometer) - A more reasonable value

    while num_remaining > 0:
        current_end_point = global_path[-1]
        shapely_current_end = ShapelyPoint(current_end_point.x, current_end_point.y)
        # Query slightly larger than tolerance to be safe
        query_geom = shapely_current_end.buffer(CONNECT_TOLERANCE * 1.1)

        # Query the STRtree for nearby endpoint indices (indices into shapely_points list)
        nearby_tree_indices = tree.query(query_geom)

        best_match_filtered_idx = -1
        min_dist = CONNECT_TOLERANCE
        reverse_needed = False
        found_match = False

        # Process nearby points found by STRtree
        candidate_endpoints = []
        for tree_idx in nearby_tree_indices:
             # Retrieve associated data using the index from the tree query
             geom, path_idx_filtered, is_start = endpoints_data[tree_idx]

             # Check if the path corresponding to this endpoint is already used
             if path_idx_filtered not in used_indices:
                 # Calculate exact distance
                 dist = current_end_point.distance_to(Point(geom.x, geom.y))
                 # Check if within tolerance
                 if dist < min_dist:
                     # Store potential candidate info
                     candidate_endpoints.append({
                         'dist': dist,
                         'path_idx_filtered': path_idx_filtered,
                         'is_start': is_start
                     })

        # If candidates were found, sort them by distance and pick the best
        if candidate_endpoints:
             candidate_endpoints.sort(key=lambda x: x['dist'])
             best_candidate = candidate_endpoints[0]
             best_match_filtered_idx = best_candidate['path_idx_filtered']
             # Reverse if we matched an end point (is_start is False)
             reverse_needed = not best_candidate['is_start']
             found_match = True

        # --- Append matched path or trigger fallback ---
        if found_match:
            # Append the path found via spatial index
            matched_sub_path = filtered_sub_paths[best_match_filtered_idx]
            points_to_add = matched_sub_path.points
            used_indices.add(best_match_filtered_idx)

            if reverse_needed:
                global_path.extend(reversed(points_to_add[:-1])) # Skip duplicate end
            else:
                global_path.extend(points_to_add[1:]) # Skip duplicate start
        else:
            # Fallback: No suitable connection found via spatial index within tolerance.
            # Use spatial index again to find the NEAREST unused endpoint.
            print(f"Warning: Could not find connection within tolerance for endpoint {current_end_point}.")
            print(f"Remaining paths: {num_remaining}. Performing fallback: Finding nearest unused endpoint via STRtree.")

            # Query for k nearest neighbors (adjust k if needed)
            k_nearest = 10
            # Ensure k is not larger than the total number of points in the tree
            k_actual = min(k_nearest, len(shapely_points))

            # Use query_nearest to find the SINGLE closest point in the tree (Shapely 1.x compatible)
            closest_fallback_filtered_idx = -1
            min_fallback_dist = float('inf')
            fallback_reverse_needed = False
            found_spatial_fallback = False

            try:
                # query_nearest in Shapely 1.x returns the geometry object directly
                nearest_geom = tree.query_nearest(shapely_current_end)

                if nearest_geom:
                    # Find the index of this geometry in our original list
                    tree_idx = -1
                    for idx, pt in enumerate(shapely_points):
                         # Use coordinate comparison as object identity might differ
                         if math.isclose(pt.x, nearest_geom.x) and math.isclose(pt.y, nearest_geom.y):
                             tree_idx = idx
                             break

                    if tree_idx != -1:
                        # Retrieve associated data using the found index
                        geom, path_idx_filtered, is_start = endpoints_data[tree_idx]

                        # Check if the path corresponding to this endpoint is already used
                        if path_idx_filtered not in used_indices:
                            # This is the closest unused endpoint
                            closest_fallback_filtered_idx = path_idx_filtered
                            min_fallback_dist = current_end_point.distance_to(Point(geom.x, geom.y))
                            fallback_reverse_needed = not is_start # Reverse if it's an end point
                            found_spatial_fallback = True
                        else:
                            print("Fallback: Nearest endpoint found by STRtree is already used. Reverting to linear scan.")
                    else:
                         print("Fallback: Could not find nearest geometry index. Reverting to linear scan.")
                else:
                    print("Fallback: STRtree query_nearest returned no result. Reverting to linear scan.")

            except Exception as e:
                 print(f"Error during STRtree.query_nearest (Shapely 1.x attempt): {e}. Reverting to linear scan.")
                 found_spatial_fallback = False # Ensure linear scan runs

            # --- Execute connection based on spatial fallback or linear scan ---
            if found_spatial_fallback:
                # Append the closest path found by the spatial fallback search
                print(f"Fallback (Spatial): Jumping {min_fallback_dist:.3f} units to the {'end' if fallback_reverse_needed else 'start'} of filtered path index {closest_fallback_filtered_idx}.")
                matched_sub_path = filtered_sub_paths[closest_fallback_filtered_idx]
                points_to_add = matched_sub_path.points
                used_indices.add(closest_fallback_filtered_idx)

                # When jumping, include the first point of the jumped-to path
                if fallback_reverse_needed:
                    global_path.extend(reversed(points_to_add))
                else:
                    global_path.extend(points_to_add)
            else:
                # --- Linear Scan Fallback (Original Method) ---
                # Execute if spatial fallback failed or found an already used path.
                print("Executing linear scan fallback...")
                closest_linear_fallback_idx = -1
                min_linear_fallback_dist = float('inf')
                linear_fallback_reverse_needed = False
                found_linear_fallback = False

                # Iterate through the filtered list indices
                for i in range(num_valid_paths):
                    if i not in used_indices:
                        fallback_path = filtered_sub_paths[i]
                        fb_start_point = fallback_path.points[0]
                        fb_end_point = fallback_path.points[-1]

                        dist_to_fb_start = current_end_point.distance_to(fb_start_point)
                        dist_to_fb_end = current_end_point.distance_to(fb_end_point)

                        if dist_to_fb_start < min_linear_fallback_dist:
                            min_linear_fallback_dist = dist_to_fb_start
                            closest_linear_fallback_idx = i
                            linear_fallback_reverse_needed = False
                            found_linear_fallback = True

                        if dist_to_fb_end < min_linear_fallback_dist:
                            min_linear_fallback_dist = dist_to_fb_end
                            closest_linear_fallback_idx = i
                            linear_fallback_reverse_needed = True
                            found_linear_fallback = True

                if found_linear_fallback:
                    # Append the closest path found by linear search
                    print(f"Fallback (Linear): Jumping {min_linear_fallback_dist:.3f} units to the {'end' if linear_fallback_reverse_needed else 'start'} of filtered path index {closest_linear_fallback_idx}.")
                    matched_sub_path = filtered_sub_paths[closest_linear_fallback_idx]
                    points_to_add = matched_sub_path.points
                    used_indices.add(closest_linear_fallback_idx)

                    # When jumping, include the first point of the jumped-to path
                    if linear_fallback_reverse_needed:
                        global_path.extend(reversed(points_to_add))
                    else:
                        global_path.extend(points_to_add)
                else:
                    # If even the linear scan didn't find an unused path
                    print("Error: Fallback failed completely. Could not find any unused path via linear scan.")
                    break # Exit loop

        # Decrement remaining count
        num_remaining = num_valid_paths - len(used_indices)

    return global_path


#-----------------------------------------------------------------------------
# 7. Path Optimization via Rasterization (Optional Post-processing)
#-----------------------------------------------------------------------------

def optimize_path_via_rasterization(
    path: List[Point],
    line_spacing: float,
    resolution: float = 0.1, # mm per pixel
    gaussian_sigma_factor: float = 1.5 # Sigma relative to line spacing in pixels
) -> List[Point]:
    """
    Optimizes a toolpath using rasterization, Gaussian blur, and skeletonization.

    Args:
        path: The input toolpath as a list of Points.
        line_spacing: The characteristic width of the toolpath (used for blur sigma).
        resolution: The size of each pixel in millimeters.
        gaussian_sigma_factor: Multiplier for line_spacing to determine Gaussian sigma.

    Returns:
        A new, potentially optimized, toolpath as a list of Points.
    """
    if not cv2_available:
        print("Raster Optimization: Skipping because OpenCV (cv2) or SciPy is not installed.")
        return path

    if not path or len(path) < 2:
        print("Raster Optimization: Path too short, returning original.")
        return path

    print(f"Raster Optimization: Starting with resolution {resolution} mm/pixel.")

    # 1. Determine bounds and image size
    min_x = min(p.x for p in path)
    max_x = max(p.x for p in path)
    min_y = min(p.y for p in path)
    max_y = max(p.y for p in path)

    padding = line_spacing * 3 # Add padding around the path
    world_min_x = min_x - padding
    world_min_y = min_y - padding
    world_max_x = max_x + padding
    world_max_y = max_y + padding

    width_mm = world_max_x - world_min_x
    height_mm = world_max_y - world_min_y

    img_width = int(np.ceil(width_mm / resolution))
    img_height = int(np.ceil(height_mm / resolution))

    if img_width <= 0 or img_height <= 0 or img_width * img_height > 50_000_000: # Safety limit
        print(f"Raster Optimization: Image size too large or invalid ({img_width}x{img_height}). Skipping.")
        return path

    print(f"Raster Optimization: Image size {img_width}x{img_height}.")

    # 2. World-to-Image Transformation
    def world_to_img(wx, wy):
        ix = int((wx - world_min_x) / resolution)
        iy = int((wy - world_min_y) / resolution)
        # Clamp coordinates to be within image bounds
        ix = max(0, min(img_width - 1, ix))
        iy = max(0, min(img_height - 1, iy))
        return ix, iy

    # 3. Image-to-World Transformation
    def img_to_world(ix, iy):
        wx = world_min_x + (ix + 0.5) * resolution # Use pixel center
        wy = world_min_y + (iy + 0.5) * resolution
        return wx, wy

    # 4. Rasterize the path
    image = np.zeros((img_height, img_width), dtype=np.uint8)
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        ix1, iy1 = world_to_img(p1.x, p1.y)
        ix2, iy2 = world_to_img(p2.x, p2.y)
        # OpenCV uses (x, y) coordinates, which correspond to (col, row)
        # Image shape is (rows, cols) = (height, width)
        cv2.line(image, (ix1, iy1), (ix2, iy2), 255, thickness=1) # Draw white line

    # 5. Apply Gaussian Blur
    sigma_pixels = (line_spacing / resolution) * gaussian_sigma_factor
    blurred_image = gaussian_filter(image.astype(float), sigma=sigma_pixels)
    print(f"Raster Optimization: Applied Gaussian blur with sigma={sigma_pixels:.2f} pixels.")

    # 6. Threshold the blurred image
    # Use a threshold slightly above zero to capture the blurred area
    threshold_value = np.max(blurred_image) * 0.1 # Example: 10% of max value
    _, thresholded_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)
    thresholded_image = thresholded_image.astype(np.uint8)

    # 7. Skeletonize
    # Use cv2.ximgproc.thinning (requires opencv-contrib-python)
    # Alternatively, use skimage.morphology.skeletonize
    try:
        # Note: THINNING_ZHANGSUEN expects white foreground on black background
        skeleton = cv2.ximgproc.thinning(thresholded_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        print("Raster Optimization: Performed skeletonization.")
    except AttributeError:
        print("Raster Optimization: cv2.ximgproc.thinning not available (install opencv-contrib-python?). Skipping skeletonization.")
        # Fallback: Use the thresholded image directly (less ideal)
        skeleton = thresholded_image
    except Exception as e:
        print(f"Raster Optimization: Error during skeletonization: {e}. Skipping.")
        skeleton = thresholded_image


    # 8. Find contours of the skeleton
    # Find external contours of the skeleton pixels
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Get all points

    if not contours:
        print("Raster Optimization: No contours found after skeletonization. Returning original path.")
        return path

    # 9. Select the longest contour and convert back to world coordinates
    longest_contour = max(contours, key=cv2.contourArea) # Use area as proxy for length here
    optimized_path_pixels = longest_contour.reshape(-1, 2) # Reshape to list of [ix, iy]

    optimized_path: List[Point] = []
    for ix, iy in optimized_path_pixels:
        wx, wy = img_to_world(ix, iy)
        # Avoid adding duplicate consecutive points
        if not optimized_path or not math.isclose(optimized_path[-1].x, wx) or not math.isclose(optimized_path[-1].y, wy):
            optimized_path.append(Point(wx, wy))

    print(f"Raster Optimization: Extracted path with {len(optimized_path)} points.")

    # Optional: Simplify the resulting path (e.g., Ramer-Douglas-Peucker)
    # ...

    return optimized_path


#-----------------------------------------------------------------------------
# Main Execution Logic
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- 0. Configuration and Setup ---
    load_config(os.path.join(project_root, 'config.yaml'))
    config = get_config()
    line_spacing = config.get('toolpath_width', 0.4) # Use toolpath_width as line spacing
    layer_height = config.get('layer_height', 1.0)
    layer_to_process_idx = config.get('layer_to_process_for_algo3', 2) # Choose a layer index
    n_layers_period = config.get('breakpoint_period', 5) # How often breakpoint strategy changes

    # Select STL file
    #stl_file_path = os.path.join(project_root, "models", "mine", "blob-with-slots.stl")


    #stl_file_path = os.path.join(project_root, "models", "cuboid.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")
    #stl_file_path = os.path.join(project_root, "models", "hollow-cuboid.stl")
    #stl_file_path = os.path.join(project_root, "models", "cuboid-with-holes.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "hex.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "gear.stl")



    stl_file_path = os.path.join("models", "wrench.stl")
    #stl_file_path = os.path.join("models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join("models", "cuboid-with-holes.stl")

    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")
    #stl_file_path = os.path.join("models", "u-shape.stl")
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")



    # testing (stlparts)
    #stl_file_path = os.path.join("models", "test", "hollow-cuboid.stl") #kinda works
    #stl_file_path = os.path.join("models", "test", "5cm-cube-with-80-diameter-hole.stl")
    #stl_file_path = os.path.join("models", "test", "hollow-cylinder.stl")
    #stl_file_path = os.path.join("models", "test", "hollow-cylinder-with-floor.stl")
    #stl_file_path = os.path.join("models", "test", "hollow-stadium.stl")
    #stl_file_path = os.path.join("models", "test", "mountainbike-cable-holder.stl")
    #stl_file_path = os.path.join("models", "test", "ring.stl")
    #stl_file_path = os.path.join("models", "test", "truncated-cone.stl")
    #stl_file_path = os.path.join("models", "test", "truncated-cone-with-hole.stl")
    
    # testing (mine, freecad)
    #stl_file_path = os.path.join("models", "mine", "hex.stl")

    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")
    
    #confirmed working, simple models
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "cuboid.stl")
    #stl_file_path = os.path.join("models", "extruded-rounded-rectangle.stl")
    #stl_file_path = os.path.join("models", "right-triangular-prism.stl")
    #stl_file_path = os.path.join("models", "stack-of-cuboids.stl")
    #stl_file_path = os.path.join("models", "stack-of-cylinders.stl")









    
    print(f"Processing STL: {os.path.basename(stl_file_path)}")
    print(f"Using Line Spacing (Toolpath Width): {line_spacing} mm")
    print(f"Processing Layer Index: {layer_to_process_idx}")

    # --- 1. Load STL and Slice ---
    stl_mesh = load_stl(stl_file_path)
    if stl_mesh is None:
        sys.exit("Failed to load STL file.")

    layers_shapely = slice_mesh(stl_mesh, layer_height)
    if not layers_shapely:
        sys.exit("Slicing resulted in no layers.")

    num_layers_generated = len(layers_shapely)
    if layer_to_process_idx >= num_layers_generated:
        print(f"Warning: Requested layer index {layer_to_process_idx} is out of bounds (0-{num_layers_generated-1}).")
        layer_to_process_idx = num_layers_generated - 1
        print(f"Processing last available layer index instead: {layer_to_process_idx}")

    # Select the specific layer (list of Shapely Polygons)
    selected_layer_polygons = layers_shapely[layer_to_process_idx]
    if not selected_layer_polygons:
        sys.exit(f"Layer {layer_to_process_idx} contains no polygons.")

    # For simplicity, assume we process the first polygon in the selected layer
    # TODO: Handle multiple polygons per layer if necessary
    if len(selected_layer_polygons) > 1:
        print(f"Warning: Layer {layer_to_process_idx} has multiple polygons. Processing only the first one.")
    shapely_polygon = selected_layer_polygons[0]

    print(f"\n--- Original Polygon (Layer {layer_to_process_idx}) ---")
    print(f"Area: {shapely_polygon.area:.2f}, Length: {shapely_polygon.length:.2f}")
    print(f"Has {len(shapely_polygon.interiors)} holes.")


    # --- 2. Perform Offsetting using Pyclipper ---
    print("\n--- Generating Offsets using Pyclipper ---")
    offset_results: List[List[Contour]] = []
    line_spacing_scaled = int(line_spacing * CLIPPER_SCALE)

    # Initial contours (level 0)
    clipper_subject = shapely_polygon_to_clipper(shapely_polygon)
    initial_contours = clipper_paths_to_contours(clipper_subject)
    if not initial_contours:
        sys.exit("Failed to convert initial Shapely polygon to Contour objects.")
    offset_results.append(initial_contours)
    print(f"Level 0: {len(initial_contours)} contours")

    # Iteratively generate inward offsets
    current_clipper_paths = clipper_subject
    level = 1
    while True:
        pco = pyclipper.PyclipperOffset()
        # Use JT_ROUND for smoother corners, adjust sensitivity if needed
        pco.AddPaths(current_clipper_paths, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # Negative delta for inward offset
        offset_paths_scaled = pco.Execute(-line_spacing_scaled)

        if not offset_paths_scaled:
            print("No further offsets possible.")
            break # Stop if no more offsets generated

        level_contours = clipper_paths_to_contours(offset_paths_scaled)
        if not level_contours:
             print(f"Offsetting produced invalid contours at level {level}. Stopping.")
             break

        print(f"Level {level}: {len(level_contours)} contours generated")
        offset_results.append(level_contours)
        current_clipper_paths = offset_paths_scaled # Use result as input for next offset
        level += 1

        # Safety break
        if level > 50:
             print("Warning: Exceeded maximum offset levels (50). Stopping.")
             break


    # --- 3. Re-level Contours (Algorithm 1) ---
    if not offset_results:
         sys.exit("No offset contours were generated.")

    # Pass only the initial outer/hole contours for typing reference if needed by the algorithm
    # The current re_level assumes offset_results[0] IS the initial set.
    print("\n--- Re-leveling Contours ---")
    leveled_contours = re_level_contours(offset_results, [], []) # Pass empty lists as initial shapes are derived within
    print(f"Re-leveled into {len(leveled_contours)} final levels.")
    # for i, level_list in enumerate(leveled_contours):
    #     print(f"  Final Level {i}: {[str(c) for c in level_list]}")


    # --- 4. Find Breakpoints (Algorithm 2) ---
    print("\n--- Finding Breakpoints ---")
    contours_with_breaks = find_breakpoints(leveled_contours, line_spacing, layer_to_process_idx, n_layers_period)
    bp_count = sum(len(c.breakpoints) for level in contours_with_breaks for c in level)
    print(f"Total breakpoints found: {bp_count}")


    # --- 5. Form Sub-paths (Algorithm 3) ---
    print("\n--- Forming Sub-paths ---")
    sub_paths = form_sub_paths(contours_with_breaks)
    print(f"Total sub-paths created: {len(sub_paths)}")


    # --- 6. Connect Sub-paths ---
    print("\n--- Connecting Sub-paths ---")
    final_toolpath = connect_sub_paths(sub_paths)
    print(f"Total points in connected toolpath: {len(final_toolpath)}")

    # --- 6b. Optional Raster Optimization ---
    if config.get('optimize_via_rasterization', False): # Add this flag to config.yaml if desired
        print("\n--- Optimizing Path via Rasterization ---")
        raster_resolution = config.get('raster_resolution', 0.05) # e.g., 50 microns per pixel
        raster_sigma_factor = config.get('raster_sigma_factor', 1.0)
        optimized_toolpath = optimize_path_via_rasterization(
            final_toolpath,
            line_spacing,
            resolution=raster_resolution,
            gaussian_sigma_factor=raster_sigma_factor
        )
        print(f"Total points after raster optimization: {len(optimized_toolpath)}")
        # Decide whether to use the optimized path for visualization/output
        path_to_visualize = optimized_toolpath
    else:
        path_to_visualize = final_toolpath


    # --- 7. Optional: Visualize ---
    if config.get('visualize_algo3_results', True):
        print("\n--- Visualizing Results ---")
        try:
            import matplotlib.pyplot as plt
            import numpy as np # Import numpy for linspace

            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

            # Plot original Shapely polygon
            orig_x, orig_y = shapely_polygon.exterior.xy
            ax.plot(orig_x, orig_y, 'k-', label=f'Original Polygon (Layer {layer_to_process_idx})', linewidth=1.5)
            for interior in shapely_polygon.interiors:
                int_x, int_y = interior.xy
                ax.plot(int_x, int_y, 'k-', linewidth=1.5)

            # Plot generated offset contours (from offset_results for clarity)
            colors = plt.cm.viridis(np.linspace(0, 1, len(offset_results)))
            for i, level_list in enumerate(offset_results):
                 for contour in level_list:
                     if contour.points:
                         x = [p.x for p in contour.points]
                         y = [p.y for p in contour.points]
                         ax.plot(x, y, color=colors[i], linestyle='--', linewidth=0.8, label=f'Offset Level {i}' if 'Offset' not in plt.gca().get_legend_handles_labels()[1] else "")


            # Plot final toolpath (potentially optimized)
            if path_to_visualize:
                tp_x = [p.x for p in path_to_visualize]
                tp_y = [p.y for p in path_to_visualize]
                ax.plot(tp_x, tp_y, 'b-', marker='.', markersize=2, linewidth=1.0, label='Final Toolpath')
                ax.plot(tp_x[0], tp_y[0], 'go', markersize=6, label='Start') # Mark start
                ax.plot(tp_x[-1], tp_y[-1], 'ro', markersize=6, label='End')   # Mark end

            # Plot breakpoints and connecting segments (optional, can be noisy)
            # for level_list in contours_with_breaks:
            #     for contour in level_list:
            #         for p1, p2, p1_proj, p2_proj in contour.breakpoints:
            #             ax.plot([p1.x, p1_proj.x], [p1.y, p1_proj.y], 'g:', linewidth=0.7)
            #             ax.plot([p2.x, p2_proj.x], [p2.y, p2_proj.y], 'm:', linewidth=0.7)
            #             ax.plot(p1.x, p1.y, 'gx', markersize=4)
            #             ax.plot(p2.x, p2.y, 'mx', markersize=4)


            plt.title(f"Algo3 Toolpath Generation - Layer {layer_to_process_idx} ({os.path.basename(stl_file_path)})")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            # # Get unique labels for legend (Removed)
            # handles, labels = plt.gca().get_legend_handles_labels()
            # by_label = dict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys(), fontsize='small')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nInstall matplotlib to visualize the results: pip install matplotlib")
        except Exception as e:
            print(f"\nError during visualization: {e}")

    print("\nProcessing finished.")

