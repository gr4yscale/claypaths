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
    """Finds the segment in target_contour closest to the given point."""
    min_dist = float('inf')
    closest_seg = None
    closest_seg_index = -1
    segments = target_contour.get_segments()
    for i, seg in enumerate(segments):
        dist = seg.distance_to_point(point)
        if dist < min_dist:
            min_dist = dist
            closest_seg = seg
            closest_seg_index = i
    return closest_seg, min_dist, closest_seg_index

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

    # Iterate from inner levels outwards (highest level index to 0)
    for i in range(num_levels - 1, -1, -1):
        current_level_contours = contours_with_breaks[i]
        # Determine target level for projection (usually the next level inwards)
        # For the innermost level (i == num_levels - 1), there's no inner contour to connect to.
        # For the outermost level (i == 0), there's no outer contour to connect to using this logic.
        target_level_index = i + 1 # Project onto the next level inwards

        if target_level_index >= num_levels:
            continue # Innermost level, nothing to connect inwards to

        target_level_contours = contours_with_breaks[target_level_index]
        if not target_level_contours:
             continue # Skip if the target level is empty

        for j, contour in enumerate(current_level_contours):
            if not contour.points or len(contour.points) < 3: continue # Skip invalid contours

            contour_len = contour.length()
            if contour_len < 1e-6: continue # Skip zero-length contours

            # --- Step 1: Find a candidate segment for breakpoint p1 ---
            # Vary starting point based on layer index to avoid repetition
            start_offset_ratio = (layer_index % n_layer_period) / n_layer_period
            start_dist = start_offset_ratio * contour_len

            p1, seg_p1, seg_p1_idx = contour.get_point_at_dist(start_dist)
            if p1 is None or seg_p1 is None: continue # Should not happen on closed contour

            # --- Step 2: If the candidate segment is shorter than linespacing ---
            # The pseudo-code suggests moving to the *next* segment if the *selected* one
            # is too short. This seems slightly odd. A more logical interpretation might be
            # to ensure the segment *containing p2* is long enough, or that p1 and p2
            # don't fall on the same very short segment.
            # Let's stick to the pseudo-code: check seg_p1's length.
            # We also need p2 first to check the segment containing p2.

            # --- Step 3: Find p2 in contour with distance p1p2 = linespacing ---
            # Find p2 by walking 'line_spacing' distance from p1 along the contour
            p2_dist = (start_dist + line_spacing) % contour_len # Wrap around contour
            p2, seg_p2, seg_p2_idx = contour.get_point_at_dist(p2_dist)
            if p2 is None or seg_p2 is None: continue

            # Re-check based on Step 2 interpretation: If seg_p1 is too short, advance p1?
            # This part is ambiguous. Let's assume for now we proceed if p1, p2 are found.
            # A check like `if seg_p1.length() < line_spacing:` might be added here
            # to advance p1 if needed, but the logic for advancement isn't fully specified.

            # --- Step 4 & 5: Find closest segment in the target level ---
            # Find the contour in the target level that is 'closest' to p1.
            # Closeness can be defined in various ways (centroid, point-to-polygon).
            # We'll find the segment across *all* target contours closest to p1.
            closest_target_seg: Optional[Segment] = None
            min_dist_to_target = float('inf')
            target_contour_k: Optional[Contour] = None # The contour containing the closest segment
            p1_proj: Optional[Point] = None

            found_target = False
            current_target_level_idx = target_level_index
            while not found_target and current_target_level_idx < num_levels:
                potential_targets = contours_with_breaks[current_target_level_idx]
                for k, cand_target_contour in enumerate(potential_targets):
                     # Check if contours belong to the same family (outer/hole)
                     if cand_target_contour.type == contour.type:
                         seg_sl, dist_sl, _ = find_closest_segment_to_point(p1, cand_target_contour)
                         if seg_sl is not None and dist_sl < min_dist_to_target:
                             min_dist_to_target = dist_sl
                             closest_target_seg = seg_sl
                             target_contour_k = cand_target_contour

                # --- Step 6: Check if closest segment distance is acceptable ---
                # The pseudo-code checks `dist > linespacing`. This implies we want
                # the connection distance to be *less* than or equal to the line spacing.
                # If the closest segment is *too far* away, we might look at the next level.
                if target_contour_k is not None and min_dist_to_target <= line_spacing * 1.5: # Allow some tolerance
                    found_target = True
                    break
                else:
                    # --- Step 6 (continued): Go back to step 5 (look in next level) ---
                    current_target_level_idx += 1
                    min_dist_to_target = float('inf') # Reset for next level search
                    target_contour_k = None


            if not found_target or closest_target_seg is None or target_contour_k is None:
                 print(f"Warning: Could not find suitable target segment for point {p1} in contour {j} level {i}")
                 continue # Could not find a suitable target segment

            # --- Step 7: Find projected point p1' of p1 in sl ---
            p1_proj = closest_target_seg.point_projection(p1)

            # --- Step 8: Find p2' in target_contour_k with p1'p2' = linespacing ---
            # This requires finding a point p2' on target_contour_k such that the
            # distance between p1' (which might not be on the contour) and p2' (on the contour)
            # is exactly line_spacing. This is non-trivial.
            # A simpler interpretation, often used in practice, is to project p2 onto
            # the same target contour k and find p2_proj. The connecting segments
            # will then be p1-p1_proj and p2-p2_proj. Let's use this simpler approach.

            seg_p2_target, dist_p2_target, _ = find_closest_segment_to_point(p2, target_contour_k)
            if seg_p2_target is None:
                print(f"Warning: Could not find target segment for point {p2} near contour {k} level {target_level_index}")
                continue
            p2_proj = seg_p2_target.point_projection(p2)

            # --- Step 9: Add connecting segments ---
            # Store breakpoints and connecting segments conceptually
            contour.breakpoints.append((p1, p2, p1_proj, p2_proj))
            contour.connecting_segments.append(Segment(p1, p1_proj))
            contour.connecting_segments.append(Segment(p2, p2_proj))
            # Also add to the target contour for forming subpaths later?
            # The pseudo code implies modification only on All_contours[i][j]
            # Let's store breaks on the contour they originate from (level i)

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
# 6. Generating continuous path (Connecting Sub-paths)
#-----------------------------------------------------------------------------

def connect_sub_paths(sub_paths: List[SubPath]) -> List[Point]:
    """
    Connects sub-paths into a single continuous global toolpath.

    Args:
        sub_paths: A list of SubPath objects.

    Returns:
        A list of Points representing the final toolpath.
    """
    if not sub_paths:
        return []

    remaining_sub_paths = sub_paths[:] # Work on a copy
    global_path: List[Point] = []

    # Start with the first sub-path
    current_sub_path = remaining_sub_paths.pop(0)
    global_path.extend(current_sub_path.points)

    while remaining_sub_paths:
        found_next = False
        current_end_point = global_path[-1]

        for i, next_sub_path in enumerate(remaining_sub_paths):
            next_start_point = next_sub_path.points[0]
            next_end_point = next_sub_path.points[-1]

            # Check if current end connects to next start
            if current_end_point == next_start_point:
                global_path.extend(next_sub_path.points[1:]) # Add points, skip duplicate start
                remaining_sub_paths.pop(i)
                found_next = True
                break
            # Check if current end connects to next end (requires reversing next path)
            elif current_end_point == next_end_point:
                global_path.extend(reversed(next_sub_path.points[:-1])) # Add reversed points, skip duplicate end
                remaining_sub_paths.pop(i)
                found_next = True
                break

        if not found_next:
            # If no direct connection found, this indicates an issue:
            # 1. Disconnected graph of paths (problem in breakpoint logic?)
            # 2. Floating point inaccuracies preventing endpoint match
            # 3. Algorithm needs a more sophisticated search (e.g., nearest endpoint)
            print(f"Warning: Could not find connection for endpoint {current_end_point}.")
            print(f"Remaining paths: {len(remaining_sub_paths)}")
            # As a fallback, just append the next available path (will cause a jump)
            if remaining_sub_paths:
                 print("Performing fallback: Jumping to next available path.")
                 next_sub_path = remaining_sub_paths.pop(0)
                 global_path.extend(next_sub_path.points)
            else:
                 break # No more paths

    return global_path


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
    # stl_file_path = os.path.join(project_root, "models", "cuboid.stl")
    # stl_file_path = os.path.join(project_root, "models", "hollow-cuboid.stl")
    #stl_file_path = os.path.join(project_root, "models", "cuboid-with-holes.stl")
    stl_file_path = os.path.join(project_root, "models", "mine", "hex-with-hex-hole.stl")

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
    print(f"Total points in final toolpath: {len(final_toolpath)}")


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


            # Plot final toolpath
            if final_toolpath:
                tp_x = [p.x for p in final_toolpath]
                tp_y = [p.y for p in final_toolpath]
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
            # Get unique labels for legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize='small')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nInstall matplotlib to visualize the results: pip install matplotlib")
        except Exception as e:
            print(f"\nError during visualization: {e}")

    print("\nProcessing finished.")

