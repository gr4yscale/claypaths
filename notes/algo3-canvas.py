import math
import copy
import os
import sys
from typing import List, Tuple, Optional, Dict, Any, Set

# Add project root to sys.path to allow importing project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pyclipper
from shapely.geometry import Polygon as ShapelyPolygon, LineString as ShapelyLineString, Point as ShapelyPoint, MultiLineString, Polygon
from shapely.strtree import STRtree
import numpy as np
try:
    import cv2 # For image processing
    from scipy.ndimage import distance_transform_edt # For distance transform (alternative)
    cv2_available = True
except ImportError:
    cv2_available = False
    print("Warning: OpenCV (cv2) or SciPy not found. Rasterization approach will not work.")


# Import project modules
from src.stl_loader import load_stl
from src.slicer import slice_mesh
from src.config import load_config, get_config

# Define a scaling factor for Pyclipper (uses integers)
CLIPPER_SCALE = 10000.0

#-----------------------------------------------------------------------------
# 1. Data Structures (Using Shapely Point)
#-----------------------------------------------------------------------------

# Custom Point class removed, using shapely.geometry.Point aliased as ShapelyPoint

# Tolerance for point equality checks
POINT_EQUALITY_TOLERANCE = 1e-9

class Segment:
    """Represents a line segment defined by two Shapely Points."""
    def __init__(self, p1: ShapelyPoint, p2: ShapelyPoint):
        self.p1 = p1
        self.p2 = p2
        self._line = ShapelyLineString([p1, p2]) # Cache LineString representation

    def length(self) -> float:
        """Calculates the length of the segment."""
        # return self.p1.distance(self.p2) # Direct distance
        return self._line.length # Use cached LineString length

    def __repr__(self):
        # Format Shapely points for representation
        p1_repr = f"ShapelyPoint({self.p1.x:.3f}, {self.p1.y:.3f})"
        p2_repr = f"ShapelyPoint({self.p2.x:.3f}, {self.p2.y:.3f})"
        return f"Segment({p1_repr}, {p2_repr})"

    def point_projection(self, p: ShapelyPoint) -> ShapelyPoint:
        """Projects a point onto the infinite line defined by the segment using Shapely."""
        # Project point p onto the LineString representing the segment
        projected_point = self._line.interpolate(self._line.project(p))
        return projected_point

    def distance_to_point(self, p: ShapelyPoint) -> float:
         """Calculates the shortest distance from a point to the line segment using Shapely."""
         return p.distance(self._line)


class Contour:
    """Represents a closed contour using Shapely Points."""
    def __init__(self, points: List[ShapelyPoint]):
        if not points:
            raise ValueError("Contour must have points.")
        # Ensure contour is closed if it isn't already
        # Use distance check with tolerance for Shapely points
        if points[0].distance(points[-1]) > POINT_EQUALITY_TOLERANCE:
            points.append(points[0])
        self.points = points
        self._line = ShapelyLineString(points) # Cache LineString representation
        self.level: int = -1 # Level assigned by re-leveling
        self.type: Optional[int] = None # 0 for outer, >0 for hole ID, None for intermediate offsets
        self.is_hole: bool = False # Flag if it originated from a hole
        self.original_level: int = -1 # Level assigned by the offsetting algorithm
        self.breakpoints: List[Tuple[ShapelyPoint, ShapelyPoint, ShapelyPoint, ShapelyPoint]] = [] # Stores (p1, p2, p1_proj, p2_proj)
        self.connecting_segments: List[Segment] = [] # Segments added between breakpoints

    def __repr__(self):
        type_str = f"Hole({self.type})" if self.is_hole else f"Outer({self.type})"
        return f"Contour(OrigLevel={self.original_level}, NewLevel={self.level}, Type={type_str}, Points={len(self.points)})"

    def get_segments(self) -> List[Segment]:
        """Returns the list of segments forming the contour."""
        # Ensure points list is valid for creating segments
        if len(self.points) < 2:
            return []
        return [Segment(self.points[i], self.points[i+1]) for i in range(len(self.points) - 1)]

    def length(self) -> float:
        """Calculates the total length (perimeter) of the contour using Shapely."""
        return self._line.length

    def get_point_at_dist(self, distance: float) -> Tuple[Optional[ShapelyPoint], Optional[Segment], int]:
        """Finds the point and segment at a given distance along the contour using Shapely."""
        if distance < 0 or distance > self.length() + POINT_EQUALITY_TOLERANCE:
             print(f"Warning: Requested distance {distance:.3f} is outside contour length {self.length():.3f}")
             # Clamp distance to valid range
             distance = max(0, min(self.length(), distance))
             # return None, None, -1 # Or clamp/wrap distance? Let's interpolate clamped value.

        # Use Shapely's interpolate method
        interpolated_point = self._line.interpolate(distance)

        # Find which original segment this point lies on or is closest to its end
        segments = self.get_segments()
        min_dist_to_segment_end = float('inf')
        containing_segment_idx = -1
        containing_segment = None

        accumulated_length = 0.0
        for i, seg in enumerate(segments):
            seg_len = seg.length()
            # Check if the distance falls within this segment's range along the contour
            if accumulated_length <= distance <= accumulated_length + seg_len + POINT_EQUALITY_TOLERANCE:
                 containing_segment = seg
                 containing_segment_idx = i
                 break
            accumulated_length += seg_len
        else:
             # If distance is very close to total length, it's on the last segment
             if segments and math.isclose(distance, self.length()):
                 containing_segment = segments[-1]
                 containing_segment_idx = len(segments) - 1

        if containing_segment is None:
             print(f"Warning: Could not reliably determine segment for distance {distance:.3f}")
             # Fallback: find closest segment geometrically (less accurate for 'along contour')
             min_geom_dist = float('inf')
             for i, seg in enumerate(segments):
                 d = interpolated_point.distance(seg._line)
                 if d < min_geom_dist:
                     min_geom_dist = d
                     containing_segment = seg
                     containing_segment_idx = i


        return interpolated_point, containing_segment, containing_segment_idx

class SubPath:
    """Represents an open path using Shapely Points."""
    def __init__(self, points: List[ShapelyPoint]):
        self.points = points

    def __repr__(self):
        start_repr = f"({self.points[0].x:.1f},{self.points[0].y:.1f})" if self.points else "N/A"
        end_repr = f"({self.points[-1].x:.1f},{self.points[-1].y:.1f})" if self.points else "N/A"
        return f"SubPath(Points={len(self.points)}, Start={start_repr}, End={end_repr})"

#-----------------------------------------------------------------------------
# 2. Geometric Helper Functions (Basic Implementation)
#    -> Recommend using Shapely for robust geometry operations (Done)
#-----------------------------------------------------------------------------

def find_closest_segment_to_point(point: ShapelyPoint, target_contour: Contour) -> Tuple[Optional[Segment], float, int]:
    """
    Finds the segment in target_contour closest to the given Shapely point using Shapely for robust distance calculation.

    Args:
        point: The query ShapelyPoint.
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
    # The target_contour._line is already a Shapely LineString
    try:
        # Ensure the contour's internal LineString is valid
        if not target_contour._line or target_contour._line.is_empty:
             # Attempt to recreate if necessary
             if len(target_contour.points) >= 2:
                 target_contour._line = ShapelyLineString(target_contour.points)
             else:
                 # Handle degenerate case
                 if target_contour.points:
                     dist = point.distance(target_contour.points[0])
                     segments = target_contour.get_segments()
                     return segments[0] if segments else None, dist, 0 if segments else -1
                 else:
                     return None, float('inf'), -1

        min_dist_shapely = point.distance(target_contour._line)
    except Exception as e:
        print(f"Error calculating distance with Shapely: {e}")
        # Fallback calculation needed? Or rely on segment iteration below?
        min_dist_shapely = float('inf')


    # Find the original Segment object that corresponds to this minimum distance
    closest_seg: Optional[Segment] = None
    closest_seg_index: int = -1
    min_segment_dist_diff = float('inf')
    segments = target_contour.get_segments()

    if not segments: # Should not happen if points exist, but check anyway
        return None, min_dist_shapely, -1

    for i, seg in enumerate(segments):
        # Use the segment's Shapely-based distance calculation method
        dist_to_segment = seg.distance_to_point(point) # This now uses Shapely internally

        # Find the segment whose distance is closest to Shapely's minimum distance to the whole contour
        # This helps identify the specific segment even if the point is near a vertex
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
    # Use ShapelyPoint directly
    points = [ShapelyPoint(x / CLIPPER_SCALE, y / CLIPPER_SCALE) for x, y in path]
    # Ensure the contour is closed for the Contour class logic
    if not points or len(points) < 3:
        return Contour([]) # Return empty if not enough points
    if points[0] != points[-1]:
        points.append(points[0])
    # Return Contour object initialized with Shapely Points
    return Contour([ShapelyPoint(p.x, p.y) for p in points])

def clipper_paths_to_contours(paths: List[List[Tuple[int, int]]]) -> List[Contour]:
    """Converts multiple Pyclipper paths to a list of Contour objects (using Shapely Points)."""
    contours = []
    for path in paths:
        if len(path) >= 3: # Need at least 3 points for a valid polygon/contour
            points = [ShapelyPoint(x / CLIPPER_SCALE, y / CLIPPER_SCALE) for x, y in path]
            # Ensure closure for Contour class logic
            if points[0].distance(points[-1]) > POINT_EQUALITY_TOLERANCE:
                 points.append(points[0])
            contours.append(Contour(points))
    return contours


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

            # Calculate centroid of current contour using Shapely
            if not contour._line or contour._line.is_empty: continue
            current_centroid = contour._line.centroid # Use Shapely's centroid

            # Find the closest contour in the *previous* level
            for prev_contour in prev_level_contours:
                if not prev_contour._line or prev_contour._line.is_empty: continue
                prev_centroid = prev_contour._line.centroid # Use Shapely's centroid
                dist = current_centroid.distance(prev_centroid)

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
# 4. (Removed) Algorithm 2: Finding breakpoints (Geometric approach)
#-----------------------------------------------------------------------------

def create_sub_paths_via_rasterization(
    leveled_contours: List[List[Contour]],
    line_spacing: float,
    resolution: float = 0.1 # mm per pixel
) -> List[SubPath]:
    """
    Creates sub-paths by finding breakpoints between contours using image processing.
    
    Args:
        leveled_contours: Contours grouped by level (output of re_level_contours).
        line_spacing: The characteristic width/distance between contours.
        resolution: The size of each pixel in millimeters (for image-based processing).

    Returns:
        A list of SubPath objects representing the connected contour segments.
    """
    if not cv2_available:
        print("Image Processing: Skipping because OpenCV (cv2) or SciPy is not installed.")
        return []

    all_contours = [c for level in leveled_contours for c in level if c.points and len(c.points) >= 2]
    if not all_contours:
        print("Image Processing: No valid contours provided.")
        return []

    print(f"Image Processing: Starting with {len(all_contours)} contours.")
    
    # Flatten contours by level for processing
    contours_by_level = {}
    for level_idx, level_list in enumerate(leveled_contours):
        contours_by_level[level_idx] = [c for c in level_list if c.points and len(c.points) >= 2]
    
    # Find breakpoints between adjacent levels
    all_breakpoints = []
    
    # Process each pair of adjacent levels
    for level_idx in range(len(leveled_contours) - 1):
        current_level = level_idx
        next_level = level_idx + 1
        
        current_contours = contours_by_level.get(current_level, [])
        next_contours = contours_by_level.get(next_level, [])
        
        if not current_contours or not next_contours:
            continue
            
        print(f"Finding breakpoints between level {current_level} and {next_level}")
        
        # For each contour in current level, find closest points to contours in next level
        for i, current_contour in enumerate(current_contours):
            current_line = current_contour._line
            if not current_line or current_line.is_empty:
                continue
                
            for j, next_contour in enumerate(next_contours):
                next_line = next_contour._line
                if not next_line or next_line.is_empty:
                    continue
                
                # Find closest points between the two contours
                # Use distance matrix approach for better performance
                current_points = np.array([(p.x, p.y) for p in current_contour.points])
                next_points = np.array([(p.x, p.y) for p in next_contour.points])
                
                # Calculate pairwise distances between all points
                if len(current_points) > 0 and len(next_points) > 0:
                    # Use vectorized operations for speed
                    distances = np.sqrt(((current_points[:, np.newaxis, :] - next_points[np.newaxis, :, :]) ** 2).sum(axis=2))
                    
                    # Find minimum distance and corresponding indices
                    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    min_dist = distances[min_dist_idx]
                    
                    # Only create breakpoints if contours are close enough
                    if min_dist < line_spacing * 1.5:
                        current_idx, next_idx = min_dist_idx
                        
                        # Get the actual points
                        p1 = current_contour.points[current_idx]
                        p2 = next_contour.points[next_idx]
                        
                        # Store breakpoint information
                        all_breakpoints.append((current_level, i, current_idx, next_level, j, next_idx))
                        
                        # Store breakpoint in contour objects for visualization
                        current_contour.breakpoints.append((p1, p2, p1, p2))
                        
                        print(f"  Found breakpoint: Level {current_level}[{i}] to Level {next_level}[{j}]")
    
    # Create sub-paths from contours and breakpoints
    sub_paths = []
    
    # First, create a sub-path for each contour segment
    for level_idx, level_contours in contours_by_level.items():
        for contour_idx, contour in enumerate(level_contours):
            # Check if this contour has breakpoints
            contour_breakpoints = [(bp_idx, bp) for bp_idx, bp in enumerate(all_breakpoints) 
                                  if (bp[0] == level_idx and bp[1] == contour_idx) or 
                                     (bp[3] == level_idx and bp[4] == contour_idx)]
            
            if not contour_breakpoints:
                # No breakpoints, add the entire contour as a sub-path
                sub_paths.append(SubPath(contour.points))
                continue
            
            # Sort breakpoints by position along the contour
            contour_breakpoints.sort(key=lambda x: x[1][2] if x[1][0] == level_idx else x[1][5])
            
            # Create sub-paths between breakpoints
            prev_idx = 0
            for _, bp in contour_breakpoints:
                if bp[0] == level_idx:  # Current contour is the "from" contour
                    bp_idx = bp[2]
                else:  # Current contour is the "to" contour
                    bp_idx = bp[5]
                
                # Create sub-path from prev_idx to bp_idx
                if bp_idx > prev_idx:
                    segment_points = contour.points[prev_idx:bp_idx+1]
                    if len(segment_points) >= 2:
                        sub_paths.append(SubPath(segment_points))
                
                prev_idx = bp_idx
            
            # Add final segment if needed
            if prev_idx < len(contour.points) - 1:
                segment_points = contour.points[prev_idx:]
                if len(segment_points) >= 2:
                    sub_paths.append(SubPath(segment_points))
    
    # Now create connecting sub-paths between breakpoints
    for bp in all_breakpoints:
        current_level, current_contour_idx, current_point_idx, next_level, next_contour_idx, next_point_idx = bp
        
        # Get the actual contours
        current_contour = contours_by_level[current_level][current_contour_idx]
        next_contour = contours_by_level[next_level][next_contour_idx]
        
        # Get the actual points
        p1 = current_contour.points[current_point_idx]
        p2 = next_contour.points[next_point_idx]
        
        # Create a connecting sub-path
        connecting_path = SubPath([p1, p2])
        sub_paths.append(connecting_path)
    
    print(f"Created {len(sub_paths)} sub-paths from contours and breakpoints.")
    return sub_paths


#-----------------------------------------------------------------------------
# 6. Generating continuous path (Connecting Sub-paths) - Linear Scan Method
#-----------------------------------------------------------------------------

def connect_sub_paths(sub_paths: List[SubPath]) -> List[ShapelyPoint]:
    """
    Connects sub-paths (containing Shapely Points) into a single continuous global toolpath
    using a linear scan approach with tolerance checking.

    Args:
        sub_paths: A list of SubPath objects.

    Returns:
        A list of Shapely Points representing the final toolpath.
    """
    if not sub_paths:
        return []

    # Filter out empty paths
    remaining_sub_paths = [p for p in sub_paths if p.points]
    if not remaining_sub_paths:
        return []

    global_path: List[ShapelyPoint] = []

    # Start with the first valid sub-path
    current_sub_path = remaining_sub_paths.pop(0)
    global_path.extend(current_sub_path.points)

    # Tolerance for comparing floating point coordinates
    # Should be slightly larger than 1/CLIPPER_SCALE to account for float errors.
    #CONNECT_TOLERANCE = 1e-2 # A previously used reasonable value

    CONNECT_TOLERANCE = 0.1 # Use the last set value, adjust if needed

    while remaining_sub_paths:
        current_end_point = global_path[-1]
        best_match_idx = -1
        min_dist = CONNECT_TOLERANCE # Only connect if distance is within tolerance
        reverse_needed = False
        found_next = False

        # --- Linear Scan for Connection ---
        # Search all remaining paths for the best connection within tolerance
        for i, next_sub_path in enumerate(remaining_sub_paths):
            # next_sub_path guaranteed to have points due to initial filtering

            next_start_point = next_sub_path.points[0]
            next_end_point = next_sub_path.points[-1]

            # Use Shapely distance
            dist_to_start = current_end_point.distance(next_start_point)
            dist_to_end = current_end_point.distance(next_end_point)

            # Check connection to the start of the next path
            if dist_to_start < min_dist:
                min_dist = dist_to_start
                best_match_idx = i
                reverse_needed = False
                found_next = True # Found a potential match within tolerance

            # Check connection to the end of the next path (requires reversal)
            # Only consider if it's a better match than the start connection found so far
            if dist_to_end < min_dist:
                min_dist = dist_to_end
                best_match_idx = i
                reverse_needed = True
                found_next = True # Found a potential match within tolerance
        # --- End Linear Scan ---

        # --- Append matched path or trigger fallback ---
        if found_next and best_match_idx != -1:
            # Append the path found within tolerance
            matched_sub_path = remaining_sub_paths.pop(best_match_idx)
            points_to_add = matched_sub_path.points

            if reverse_needed:
                # Add reversed points, skip the duplicate endpoint (which is now the first element)
                global_path.extend(reversed(points_to_add[:-1]))
            else:
                # Add points, skip the duplicate start point (which is the first element)
                global_path.extend(points_to_add[1:])
        else:
            # --- Fallback: No connection found within tolerance ---
            # Find the geometrically closest endpoint among all remaining paths.
            print(f"Warning: Could not find connection within tolerance {CONNECT_TOLERANCE} for endpoint {current_end_point}.")
            print(f"Remaining paths: {len(remaining_sub_paths)}. Performing fallback: Finding closest jump.")

            closest_fallback_idx = -1
            min_fallback_dist = float('inf')
            fallback_reverse_needed = False
            found_fallback_candidate = False

            # Iterate through remaining paths to find the absolute closest endpoint
            for i, fallback_path in enumerate(remaining_sub_paths):
                fb_start_point = fallback_path.points[0] # ShapelyPoint
                fb_end_point = fallback_path.points[-1] # ShapelyPoint

                # Use Shapely distance
                dist_to_fb_start = current_end_point.distance(fb_start_point)
                dist_to_fb_end = current_end_point.distance(fb_end_point)

                if dist_to_fb_start < min_fallback_dist:
                    min_fallback_dist = dist_to_fb_start
                    closest_fallback_idx = i
                    fallback_reverse_needed = False
                    found_fallback_candidate = True

                if dist_to_fb_end < min_fallback_dist:
                    min_fallback_dist = dist_to_fb_end
                    closest_fallback_idx = i
                    fallback_reverse_needed = True
                    found_fallback_candidate = True

            if found_fallback_candidate:
                # Append the closest path found by fallback search
                print(f"Fallback: Jumping {min_fallback_dist:.3f} units to the {'end' if fallback_reverse_needed else 'start'} of path index {closest_fallback_idx} in remaining list.")
                matched_sub_path = remaining_sub_paths.pop(closest_fallback_idx)
                points_to_add = matched_sub_path.points

                # When jumping, include the first point of the jumped-to path
                if fallback_reverse_needed:
                    global_path.extend(reversed(points_to_add))
                else:
                    global_path.extend(points_to_add)
            else:
                # Should not happen if remaining_sub_paths is not empty
                print("Error: Fallback failed completely. Could not find any remaining path.")
                break # Exit loop

    return global_path


#-----------------------------------------------------------------------------
# 7. Path Resampling (Uniform Segment Length)
#-----------------------------------------------------------------------------

def resample_path_uniformly(path: List[ShapelyPoint], segment_length: float) -> List[ShapelyPoint]:
    """
    Resamples a path to have approximately uniform segment lengths.

    Args:
        path: The input path as a list of Shapely Points.
        segment_length: The desired length of each segment.

    Returns:
        A resampled path as a list of Shapely Points.
    """
    if not path or len(path) < 2 or segment_length <= 1e-6:
        return path # Cannot resample

    # Create LineString and attempt to clean self-intersections/invalid geometry
    try:
        line = ShapelyLineString(path)
        # buffer(0) can fix invalid geometries like self-intersections
        cleaned_geom = line.buffer(0)

        if cleaned_geom.is_empty:
             print("Resampling: buffer(0) resulted in empty geometry. Returning original path.")
             return path
        elif isinstance(cleaned_geom, ShapelyLineString):
            line = cleaned_geom
            print("Resampling: Applied buffer(0) cleanup, result is LineString.")
        elif isinstance(cleaned_geom, MultiLineString):
             # If buffer(0) results in multiple lines, pick the longest one
             print("Resampling: buffer(0) resulted in MultiLineString, selecting longest.")
             line = max(cleaned_geom.geoms, key=lambda l: l.length)
        elif isinstance(cleaned_geom, ShapelyPolygon): # Use alias
             # If it becomes a polygon (e.g., input was closed loop), use its exterior
             print("Resampling: buffer(0) resulted in Polygon, using exterior.")
             line = cleaned_geom.exterior
             if not isinstance(line, ShapelyLineString): # Use alias
                  print("Resampling: Polygon exterior is not LineString. Using original path.")
                  line = ShapelyLineString(path) # Fallback
        else:
             print(f"Resampling: buffer(0) resulted in unexpected geometry type ({type(cleaned_geom)}). Using original line.")
             # Fallback to original line if cleanup fails or returns something weird
             line = ShapelyLineString(path)

    except Exception as e:
        print(f"Resampling: Error during buffer(0) cleanup: {e}. Using original line.")
        line = ShapelyLineString(path)


    total_length = line.length
    if total_length < segment_length:
        print("Resampling: Path length shorter than segment length after cleanup.")
        return path # Path is shorter than desired segment length

    print(f"Resampling path (length {total_length:.2f}) with target segment length {segment_length:.3f}...")

    num_segments = math.ceil(total_length / segment_length)
    resampled_path: List[ShapelyPoint] = []

    for i in range(num_segments + 1):
        distance = min(i * segment_length, total_length) # Ensure we don't exceed total length
        point = line.interpolate(distance)
        # Avoid adding duplicate points if interpolation yields the same point
        if not resampled_path or point.distance(resampled_path[-1]) > POINT_EQUALITY_TOLERANCE:
            resampled_path.append(point)

    # Ensure the very last point of the original path is included if not already captured
    if resampled_path and path[-1].distance(resampled_path[-1]) > POINT_EQUALITY_TOLERANCE:
         # Check if the last interpolated point is very close to the end
         if total_length - (num_segments * segment_length) > POINT_EQUALITY_TOLERANCE:
              resampled_path.append(path[-1])


    print(f"Resampled path has {len(resampled_path)} points.")
    return resampled_path


#-----------------------------------------------------------------------------
# 8. Visualization Functions
#-----------------------------------------------------------------------------

def visualize_final_toolpath(
    shapely_polygon: ShapelyPolygon,
    offset_results: List[List[Contour]],
    path_to_visualize: List[ShapelyPoint],
    layer_to_process_idx: int,
    stl_file_path: str
):
    """Visualizes the original polygon, offsets, and the final toolpath."""
    print("\n--- Visualizing Final Toolpath ---")
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
                 # Use the cached LineString for plotting coordinates
                 if contour._line and not contour._line.is_empty:
                     x, y = contour._line.xy
                     ax.plot(x, y, color=colors[i], linestyle='--', linewidth=0.8, label=f'Offset Level {i}' if 'Offset' not in plt.gca().get_legend_handles_labels()[1] else "")


        # Plot final toolpath (potentially resampled)
        if path_to_visualize:
            # Extract coordinates directly from Shapely Points
            tp_x = [p.x for p in path_to_visualize]
            tp_y = [p.y for p in path_to_visualize]
            ax.plot(tp_x, tp_y, 'b-', marker='.', markersize=2, linewidth=1.0, label='Final Toolpath')
            ax.plot(tp_x[0], tp_y[0], 'go', markersize=6, label='Start') # Mark start
            ax.plot(tp_x[-1], tp_y[-1], 'ro', markersize=6, label='End')   # Mark end

        # Plot breakpoints and connecting segments (optional, can be noisy)
        # for level_list in contours_with_breaks:
        #     for contour in level_list:
        #         # p1, p2, p1_proj, p2_proj are ShapelyPoints
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


def visualize_resampled_path(
    resampled_path: List[ShapelyPoint],
    segment_length: float,
    layer_to_process_idx: int,
    stl_file_path: str
):
    """Visualizes the resampled path with points marked."""
    print("\n--- Visualizing Resampled Path ---")
    if not resampled_path:
        print("No resampled path to visualize.")
        return
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        # Plot resampled path
        tp_x = [p.x for p in resampled_path]
        tp_y = [p.y for p in resampled_path]
        ax.plot(tp_x, tp_y, 'r-', linewidth=0.8, label='Resampled Path')
        ax.plot(tp_x, tp_y, 'r.', markersize=4, label='Resampled Points') # Mark points
        ax.plot(tp_x[0], tp_y[0], 'go', markersize=8, label='Start') # Mark start
        ax.plot(tp_x[-1], tp_y[-1], 'mo', markersize=8, label='End')   # Mark end

        plt.title(f"Resampled Toolpath (Seg Len ~{segment_length:.3f}mm) - Layer {layer_to_process_idx} ({os.path.basename(stl_file_path)})")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.legend(fontsize='small')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nInstall matplotlib to visualize the results: pip install matplotlib")
    except Exception as e:
        print(f"\nError during visualization: {e}")


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


    stl_file_path = os.path.join(project_root, "models", "mine", "gear.stl")
    #stl_file_path = os.path.join("models", "wrench.stl")
    #stl_file_path = os.path.join("models", "cuboid-with-holes.stl")
    #stl_file_path = os.path.join("models", "u-shape.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "hex.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")

    #stl_file_path = os.path.join(project_root, "models", "cuboid.stl")
    #stl_file_path = os.path.join(project_root, "models", "hollow-cuboid.stl")
    #stl_file_path = os.path.join(project_root, "models", "cuboid-with-holes.stl")



    #stl_file_path = os.path.join("models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")

    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")

    #stl_file_path = os.path.join("models", "extruded-rounded-rectangle.stl")



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

    
    #confirmed working, simple models
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")
    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "cuboid.stl")
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


    # --- 4/5. Create Sub-paths via Rasterization (Replaces Algo 2 & 3) ---
    print("\n--- Creating Sub-paths via Rasterization ---")
    raster_resolution = config.get('raster_resolution', 0.05) # Default if not in config
    sub_paths = create_sub_paths_via_rasterization(
        leveled_contours,
        line_spacing,
        resolution=raster_resolution
        # Removed raster_line_thickness argument
    )
    if not sub_paths:
         sys.exit("Failed to create sub-paths using rasterization.")
    print(f"Total sub-paths created: {len(sub_paths)}")


    # --- 6. Connect Sub-paths (Using Linear Scan Method) ---
    # The sub-paths from rasterization should ideally form a connected graph
    print("\n--- Connecting Sub-paths ---")
    final_toolpath = connect_sub_paths(sub_paths) # Use the existing linear scan connector
    print(f"Total points in connected toolpath: {len(final_toolpath)}")

    # --- 6b. Optional Path Resampling ---
    resampled_toolpath = None # Initialize
    if config.get('enable_resampling', False):
        segment_length = config.get('resampling_segment_length', 0.5)
        resampled_toolpath = resample_path_uniformly(final_toolpath, segment_length=segment_length)
        path_to_visualize = resampled_toolpath # Visualize the resampled path in the main plot
    else:
        # Path to visualize is the result of the connection if resampling is disabled
        path_to_visualize = final_toolpath


    # --- 7. Optional: Visualize Main Results ---
    if config.get('visualize_algo3_results', True):
         visualize_final_toolpath(
              shapely_polygon,
              offset_results,
              path_to_visualize,
              layer_to_process_idx,
              stl_file_path
         )

    # --- 8. Optional: Visualize Resampled Path Separately ---
    if resampled_toolpath and config.get('visualize_resampled_path', True): # Add new config flag if needed
         segment_length = config.get('resampling_segment_length', 0.5) # Get length again for title
         visualize_resampled_path(
              resampled_toolpath,
              segment_length,
              layer_to_process_idx,
              stl_file_path
         )


    print("\nProcessing finished.")

