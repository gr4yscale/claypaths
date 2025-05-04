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
    from skimage.morphology import skeletonize, medial_axis # For skeletonization
    from skimage import measure # For contour finding in skeleton
    cv2_available = True
except ImportError:
    cv2_available = False
    print("Warning: OpenCV (cv2), SciPy, or scikit-image not found. Advanced image processing will not work.")


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
    skeleton_data = None  # Class attribute to store skeleton data for visualization
    
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

def create_sub_paths(
    leveled_contours: List[List[Contour]],
    line_spacing: float,
    resolution: float = 0.1 # mm per pixel
) -> List[SubPath]:
    """
    Creates sub-paths by finding optimal breakpoints between contours.
    This is a placeholder implementation that simply creates one sub-path per contour.
    
    Args:
        leveled_contours: Contours grouped by level (output of re_level_contours).
        line_spacing: The characteristic width/distance between contours.
        resolution: The size of each pixel in millimeters for rasterization.

    Returns:
        A list of SubPath objects representing the connected contour segments.
    """
    print("Using placeholder sub-path creation method")
    
    # Flatten contours by level for processing
    contours_by_level = {}
    for level_idx, level_list in enumerate(leveled_contours):
        contours_by_level[level_idx] = [c for c in level_list if c.points and len(c.points) >= 2]
    
    all_contours = [c for level in leveled_contours for c in level if c.points and len(c.points) >= 2]
    if not all_contours:
        print("No valid contours provided.")
        return []

    print(f"Starting with {len(all_contours)} contours.")
    
    # Create a simple sub-path for each contour
    sub_paths = []
    for level_idx, level_contours in contours_by_level.items():
        for contour_idx, contour in enumerate(level_contours):
            if contour.points and len(contour.points) >= 2:
                sub_paths.append(SubPath(contour.points))
    
    print(f"Created {len(sub_paths)} sub-paths from contours.")
    
    # Create a wrapper object to hold both the sub_paths and skeleton data
    class SubPathsWithSkeletonData:
        def __init__(self, paths, skeleton_data):
            self.paths = paths
            self.skeleton_data = skeleton_data
            self.junction_points = []
    
        def __len__(self):
            return len(self.paths)
    
        def __getitem__(self, idx):
            return self.paths[idx]
    
        def __iter__(self):
            return iter(self.paths)
    
    # Return the wrapper object with sub_paths and empty skeleton data
    return SubPathsWithSkeletonData(
        sub_paths,
        {
            'contours_by_level': contours_by_level,
            'all_breakpoints': []
        }
    )


#-----------------------------------------------------------------------------
# 6. Generating continuous path (Connecting Sub-paths) - Linear Scan Method
#-----------------------------------------------------------------------------

def connect_sub_paths(sub_paths: List[SubPath]) -> List[ShapelyPoint]:
    """
    Connects sub-paths into a single continuous global toolpath using breakpoints
    from skeleton analysis to create an optimal path.

    Args:
        sub_paths: A list of SubPath objects or a SubPathsWithSkeletonData object.

    Returns:
        A list of Shapely Points representing the final toolpath.
    """
    if not sub_paths:
        return []

    # Check if we have skeleton data available
    skeleton_data = None
    
    if hasattr(sub_paths, 'skeleton_data'):
        skeleton_data = sub_paths.skeleton_data
        sub_paths_list = sub_paths.paths
        
        # Store skeleton data in the SubPath class for visualization
        SubPath.skeleton_data = skeleton_data
    else:
        sub_paths_list = sub_paths

    # Filter out empty paths
    remaining_sub_paths = [p for p in sub_paths_list if p.points and len(p.points) >= 2]
    if not remaining_sub_paths:
        return []

    # Check for breakpoints in the skeleton data
    all_breakpoints = []
    if skeleton_data and 'all_breakpoints' in skeleton_data:
        all_breakpoints = skeleton_data.get('all_breakpoints', [])
        print(f"Found {len(all_breakpoints)} breakpoints in skeleton data")
        
        if all_breakpoints:
            contours_by_level = skeleton_data.get('contours_by_level', {})
            if contours_by_level:
                print("Using breakpoints for path connection")
                return connect_using_breakpoints(remaining_sub_paths, all_breakpoints, contours_by_level)
    
    print("No breakpoints available, using distance-based connection")
    return connect_using_distance(remaining_sub_paths)

def connect_using_breakpoints(sub_paths: List[SubPath], all_breakpoints: list, contours_by_level: dict) -> List[ShapelyPoint]:
    """
    Placeholder for connecting sub-paths using breakpoints.
    Since we've removed the breakpoint finding code, this just falls back to distance-based connection.
    
    Args:
        sub_paths: List of SubPath objects
        all_breakpoints: List of breakpoints (empty in this implementation)
        contours_by_level: Dictionary mapping level indices to lists of contours
        
    Returns:
        A list of Shapely Points representing the connected path
    """
    print("No breakpoints available, falling back to distance-based connection")
    return connect_using_distance(sub_paths)

def connect_using_distance(sub_paths: List[SubPath], start_point=None) -> List[ShapelyPoint]:
    """
    Connects sub-paths based on distance between endpoints.
    
    Args:
        sub_paths: List of SubPath objects
        start_point: Optional starting point
        
    Returns:
        A list of Shapely Points representing the connected path
    """
    if not sub_paths:
        return []
        
    # Filter out empty paths
    remaining_sub_paths = [p for p in sub_paths if p.points and len(p.points) >= 2]
    if not remaining_sub_paths:
        return []

    global_path: List[ShapelyPoint] = []

    # Start with the first valid sub-path or use the provided start point
    if start_point is None:
        current_sub_path = remaining_sub_paths.pop(0)
        global_path.extend(current_sub_path.points)
        current_end_point = global_path[-1]
    else:
        global_path.append(start_point)
        current_end_point = start_point

    # Tolerance for comparing floating point coordinates
    CONNECT_TOLERANCE = 0.1  # Use the last set value, adjust if needed

    while remaining_sub_paths:
        best_match_idx = -1
        min_connection_cost = float('inf')
        reverse_needed = False
        found_next = False

        # Find the closest path
        for i, next_sub_path in enumerate(remaining_sub_paths):
            next_start_point = next_sub_path.points[0]
            next_end_point = next_sub_path.points[-1]

            # Use Shapely distance
            dist_to_start = current_end_point.distance(next_start_point)
            dist_to_end = current_end_point.distance(next_end_point)

            # Only consider connections within a reasonable distance
            max_connection_dist = CONNECT_TOLERANCE * 10  # Allow longer jumps but penalize them
            
            # Check connection to the start of the next path
            if dist_to_start < max_connection_dist:
                connection_cost = dist_to_start
                
                if connection_cost < min_connection_cost:
                    min_connection_cost = connection_cost
                    best_match_idx = i
                    reverse_needed = False
                    found_next = True
            
            # Check connection to the end of the next path (requires reversal)
            if dist_to_end < max_connection_dist:
                connection_cost = dist_to_end
                
                if connection_cost < min_connection_cost:
                    min_connection_cost = connection_cost
                    best_match_idx = i
                    reverse_needed = True
                    found_next = True

        # Append matched path or trigger fallback
        if found_next and best_match_idx != -1:
            # Append the path found with lowest connection cost
            matched_sub_path = remaining_sub_paths.pop(best_match_idx)
            points_to_add = matched_sub_path.points

            if reverse_needed:
                # Add reversed points, skip the duplicate endpoint
                global_path.extend(reversed(points_to_add[:-1]))
            else:
                # Add points, skip the duplicate start point
                global_path.extend(points_to_add[1:])
                
            current_end_point = global_path[-1]
        else:
            # Fallback: No connection found within reasonable cost
            print(f"Warning: Could not find connection with reasonable cost for endpoint {current_end_point}.")
            print(f"Remaining paths: {len(remaining_sub_paths)}. Performing fallback: Finding least-worst connection.")

            # Find the path with the lowest overall connection cost
            best_fallback_idx = -1
            min_fallback_cost = float('inf')
            fallback_reverse_needed = False

            for i, fallback_path in enumerate(remaining_sub_paths):
                fb_start_point = fallback_path.points[0]
                fb_end_point = fallback_path.points[-1]

                # Calculate costs for both connection options
                start_dist = current_end_point.distance(fb_start_point)
                end_dist = current_end_point.distance(fb_end_point)
                
                # Simply use distances as costs
                start_cost = start_dist
                end_cost = end_dist
                
                # Choose the better of the two options
                if start_cost < end_cost and start_cost < min_fallback_cost:
                    min_fallback_cost = start_cost
                    best_fallback_idx = i
                    fallback_reverse_needed = False
                elif end_cost < min_fallback_cost:
                    min_fallback_cost = end_cost
                    best_fallback_idx = i
                    fallback_reverse_needed = True

            if best_fallback_idx != -1:
                # Append the path with lowest overall connection cost
                print(f"Fallback: Jumping to path index {best_fallback_idx} with connection cost {min_fallback_cost:.3f}.")
                matched_sub_path = remaining_sub_paths.pop(best_fallback_idx)
                points_to_add = matched_sub_path.points

                # When jumping, include the first point of the jumped-to path
                if fallback_reverse_needed:
                    global_path.extend(reversed(points_to_add))
                else:
                    global_path.extend(points_to_add)
                    
                current_end_point = global_path[-1]
            else:
                # Should not happen if remaining_sub_paths is not empty
                print("Error: Fallback failed completely. Could not find any remaining path.")
                break  # Exit loop

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

    # Create a clean LineString from the path
    try:
        line = ShapelyLineString(path)
        if line.is_empty:
            print("Resampling: Empty LineString. Returning original path.")
            return path
    except Exception as e:
        print(f"Resampling: Error creating LineString: {e}. Returning original path.")
        return path

    total_length = line.length
    if total_length < segment_length:
        print("Resampling: Path length shorter than segment length.")
        return path # Path is shorter than desired segment length

    print(f"Resampling path (length {total_length:.2f}) with target segment length {segment_length:.3f}...")

    # Calculate number of segments needed
    num_segments = max(1, int(total_length / segment_length))
    
    # Create a new path with evenly spaced points
    resampled_path: List[ShapelyPoint] = []
    
    # Always include the first point
    resampled_path.append(ShapelyPoint(path[0].x, path[0].y))
    
    # Add evenly spaced points along the path
    for i in range(1, num_segments):
        # Calculate distance along the path for this point
        distance = (i / num_segments) * total_length
        
        # Get point at this distance
        point = line.interpolate(distance)
        
        # Add to resampled path if it's not a duplicate
        if point.distance(resampled_path[-1]) > POINT_EQUALITY_TOLERANCE:
            resampled_path.append(point)
    
    # Always include the last point of the original path
    last_point = ShapelyPoint(path[-1].x, path[-1].y)
    if last_point.distance(resampled_path[-1]) > POINT_EQUALITY_TOLERANCE:
        resampled_path.append(last_point)

    print(f"Resampled path has {len(resampled_path)} points (from original {len(path)} points).")
    
    # Verify segment lengths
    total_new_length = 0
    segment_lengths = []
    for i in range(len(resampled_path) - 1):
        seg_len = resampled_path[i].distance(resampled_path[i+1])
        segment_lengths.append(seg_len)
        total_new_length += seg_len
    
    avg_segment = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0
    print(f"Average segment length: {avg_segment:.3f} mm (target: {segment_length:.3f} mm)")
    
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
            
            # Plot the path as a continuous line
            ax.plot(tp_x, tp_y, 'b-', linewidth=1.0, label='Final Toolpath')
            
            # Plot each point as a dot
            if config.get('enable_resampling', False):
                ax.scatter(tp_x, tp_y, color='cyan', s=10, marker='.', label='Resampled Points')
            
            # Mark start and end points
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
        
        # Plot the path as a continuous line
        ax.plot(tp_x, tp_y, 'r-', linewidth=0.8, label='Resampled Path')
        
        # Plot each point as a dot
        ax.scatter(tp_x, tp_y, color='blue', s=15, marker='.', label='Resampled Points')
        
        # Mark start and end points
        ax.plot(tp_x[0], tp_y[0], 'go', markersize=8, label='Start')
        ax.plot(tp_x[-1], tp_y[-1], 'mo', markersize=8, label='End')

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


    #stl_file_path = os.path.join(project_root, "models", "mine", "hex.stl")
    #stl_file_path = os.path.join("models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join("models", "extruded-rounded-rectangle.stl")
    #stl_file_path = os.path.join("models", "extruded-polygon.stl")

    stl_file_path = os.path.join(project_root, "models", "mine", "gear.stl")
    #stl_file_path = os.path.join("models", "wrench.stl")

    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "u-shape.stl")

    #stl_file_path = os.path.join("models", "cuboid-with-holes.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "hex.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")

    #stl_file_path = os.path.join(project_root, "models", "cuboid.stl")
    #stl_file_path = os.path.join(project_root, "models", "hollow-cuboid.stl")
    #stl_file_path = os.path.join(project_root, "models", "cuboid-with-holes.stl")


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
    sub_paths = create_sub_paths(
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
    
    # Debug check for breakpoints
    if hasattr(sub_paths, 'skeleton_data') and 'all_breakpoints' in sub_paths.skeleton_data:
        print(f"DEBUG: Main function - found {len(sub_paths.skeleton_data['all_breakpoints'])} breakpoints")
    else:
        print("DEBUG: Main function - no breakpoints found in data")
    
    # Extract the actual sub_paths from the wrapper object if needed
    sub_paths_list = sub_paths.paths if hasattr(sub_paths, 'paths') else sub_paths
    final_toolpath = connect_sub_paths(sub_paths) # Pass the whole object to preserve junction points
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
         
    # --- 7b. Optional: Visualize Contours ---
    # Simplified visualization without breakpoints
    if config.get('visualize_breakpoints', True) and hasattr(sub_paths, 'skeleton_data') and sub_paths.skeleton_data:
        try:
            import matplotlib.pyplot as plt
            
            # Unpack the data
            contours_by_level = sub_paths.skeleton_data.get('contours_by_level', {})
            
            if contours_by_level:
                plt.figure(figsize=(10, 10))
                
                # Plot contours
                for level_idx, contours in contours_by_level.items():
                    for contour in contours:
                        if contour._line and not contour._line.is_empty:
                            x, y = contour._line.xy
                            plt.plot(x, y, '-', linewidth=1, alpha=0.7)
                
                plt.title(f"Contours by Level - Layer {layer_to_process_idx}")
                plt.axis('equal')
                plt.tight_layout()
                plt.show()
            else:
                print("Contour data not available for visualization")
        except Exception as e:
            print(f"Error visualizing contours: {e}")

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

