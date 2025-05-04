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

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

def create_sub_paths(
    leveled_contours: List[List[Contour]],
    line_spacing: float,
) -> List[SubPath]:
    """
    Creates sub-paths by finding optimal breakpoints between contours.
    Identifies breakpoints in each contour that will be used to connect one contour to another.
    Uses a spatial index to speed up finding nearby contours.
    Uses parallel processing for better performance.
    
    Args:
        leveled_contours: Contours grouped by level (output of re_level_contours).
        line_spacing: The characteristic width/distance between contours.

    Returns:
        A list of SubPath objects representing the connected contour segments.
    """
    print("Finding breakpoints between contours...")
    
    # Flatten contours by level for processing
    contours_by_level = {}
    for level_idx, level_list in enumerate(leveled_contours):
        contours_by_level[level_idx] = [c for c in level_list if c.points and len(c.points) >= 2]
    
    all_contours = [c for level in leveled_contours for c in level if c.points and len(c.points) >= 2]
    if not all_contours:
        print("No valid contours provided.")
        return []

    print(f"Starting with {len(all_contours)} contours.")
    
    # Get the breakpoint period from config
    config = get_config()
    n_layers_period = config.get('breakpoint_period', 5)  # Default to 5 if not specified
    max_breakpoint_distance = config.get('max_breakpoint_distance', 2.0)  # Maximum distance to consider
    spatial_index_buffer = config.get('spatial_index_buffer', 1.5)  # Buffer multiplier for spatial index
    
    # Find breakpoints between adjacent contour levels
    all_breakpoints = []
    
    # Prepare a distance cache to avoid recalculating distances
    distance_cache = {}
    
    # Process each level (except the last one)
    for level_idx in range(len(contours_by_level) - 1):
        current_level_contours = contours_by_level[level_idx]
        next_level_contours = contours_by_level[level_idx + 1]
        
        # Skip if either level has no contours
        if not current_level_contours or not next_level_contours:
            continue
            
        print(f"Finding breakpoints between level {level_idx} and {level_idx + 1}")
        
        # Create spatial index for next level contours to speed up proximity queries
        next_level_linestrings = [c._line for c in next_level_contours if c._line and not c._line.is_empty]
        if not next_level_linestrings:
            continue
            
        # Build spatial index
        spatial_index = STRtree(next_level_linestrings)
        
        # Pre-filter current contours that are too far from any next level contour
        # Create a union of all next level contours with a buffer
        next_level_union = None
        try:
            from shapely.ops import unary_union
            buffered_next_level = [ls.buffer(max_breakpoint_distance * spatial_index_buffer) 
                                  for ls in next_level_linestrings]
            next_level_union = unary_union(buffered_next_level)
        except Exception as e:
            print(f"Warning: Could not create union of next level contours: {e}")
        
        # Filter current contours that intersect with the next level union
        filtered_current_contours = []
        if next_level_union:
            filtered_current_contours = [c for c in current_level_contours 
                                        if c._line and not c._line.is_empty and c._line.intersects(next_level_union)]
        else:
            filtered_current_contours = current_level_contours
            
        print(f"Filtered from {len(current_level_contours)} to {len(filtered_current_contours)} current contours")
        
        # Prepare contour pairs for parallel processing
        contour_pairs = []
        for current_contour in filtered_current_contours:
            if not current_contour._line or current_contour._line.is_empty:
                continue
                
            # Create a buffer around the current contour to find potential matches
            search_buffer = current_contour._line.buffer(max_breakpoint_distance * spatial_index_buffer)
            
            # Query the spatial index to find nearby contours
            potential_matches_idx = spatial_index.query(search_buffer)
            
            # Add valid pairs to the processing list
            for idx in potential_matches_idx:
                next_contour = next_level_contours[idx]
                if next_contour._line and not next_contour._line.is_empty:
                    # Quick distance check before adding to processing list
                    pair_key = (id(current_contour), id(next_contour))
                    if pair_key not in distance_cache:
                        distance_cache[pair_key] = current_contour._line.distance(next_contour._line)
                    
                    if distance_cache[pair_key] <= max_breakpoint_distance:
                        contour_pairs.append((current_contour, next_contour))
        
        print(f"Processing {len(contour_pairs)} contour pairs")
        
        # Process contour pairs in parallel if there are enough pairs
        level_breakpoints = []
        if len(contour_pairs) > 1:  # Only use parallel processing if there are enough pairs
            try:
                # Use ThreadPoolExecutor for I/O bound tasks or when using Shapely objects
                # that might not be picklable for ProcessPoolExecutor
                with ThreadPoolExecutor(max_workers=min(8, multiprocessing.cpu_count())) as executor:
                    # Create a partial function with fixed parameters
                    find_breakpoints_partial = partial(
                        process_contour_pair,
                        line_spacing=line_spacing,
                        n_layers_period=n_layers_period,
                        max_breakpoint_distance=max_breakpoint_distance
                    )
                    
                    # Process all pairs in parallel
                    results = list(executor.map(find_breakpoints_partial, contour_pairs))
                    
                    # Collect all breakpoints
                    for result in results:
                        if result:  # result is a list of breakpoints for one contour pair
                            level_breakpoints.extend(result)
            except Exception as e:
                print(f"Warning: Parallel processing failed: {e}. Falling back to sequential processing.")
                # Fall back to sequential processing
                for current_contour, next_contour in contour_pairs:
                    breakpoints = find_breakpoints_between_contours(
                        current_contour, 
                        next_contour, 
                        line_spacing,
                        n_layers_period,
                        max_breakpoint_distance
                    )
                    if breakpoints:
                        level_breakpoints.extend(breakpoints)
        else:
            # Sequential processing for small number of pairs
            for current_contour, next_contour in contour_pairs:
                breakpoints = find_breakpoints_between_contours(
                    current_contour, 
                    next_contour, 
                    line_spacing,
                    n_layers_period,
                    max_breakpoint_distance
                )
                if breakpoints:
                    level_breakpoints.extend(breakpoints)
        
        # Assign breakpoints to contours and create connecting segments
        for bp in level_breakpoints:
            p1, p2, p1_proj, p2_proj = bp
            
            # Find which contour this breakpoint belongs to
            for current_contour in filtered_current_contours:
                # Check if p1 or p2 is on this contour
                on_contour = False
                for point in current_contour.points:
                    if point.distance(p1) < POINT_EQUALITY_TOLERANCE or point.distance(p2) < POINT_EQUALITY_TOLERANCE:
                        on_contour = True
                        break
                
                if on_contour:
                    current_contour.breakpoints.append(bp)
                    current_contour.connecting_segments.append(Segment(p1, p1_proj))
                    current_contour.connecting_segments.append(Segment(p2, p2_proj))
                    break
        
        # Add level breakpoints to all breakpoints
        all_breakpoints.extend(level_breakpoints)
    
    print(f"Found {len(all_breakpoints)} breakpoints across all contours")
    
    # Create sub-paths from contours, incorporating breakpoints
    sub_paths = []
    for level_idx, level_contours in contours_by_level.items():
        for contour_idx, contour in enumerate(level_contours):
            if contour.points and len(contour.points) >= 2:
                # Create a sub-path from the contour points
                sub_paths.append(SubPath(contour.points))
                
                # Create additional sub-paths for connecting segments if they exist
                for segment in contour.connecting_segments:
                    sub_paths.append(SubPath([segment.p1, segment.p2]))
    
    print(f"Created {len(sub_paths)} sub-paths from contours and connections")
    
    # Return the sub_paths along with the contours and breakpoints
    return {
        'paths': sub_paths,
        'contours_by_level': contours_by_level,
        'all_breakpoints': all_breakpoints
    }

def process_contour_pair(pair, line_spacing, n_layers_period, max_breakpoint_distance):
    """
    Process a pair of contours to find breakpoints between them.
    This function is designed to be used with parallel processing.
    
    Args:
        pair: A tuple of (current_contour, next_contour)
        line_spacing: The spacing between contours
        n_layers_period: How often to place breakpoints
        max_breakpoint_distance: Maximum distance to consider for breakpoints
        
    Returns:
        List of breakpoint tuples (p1, p2, p1_proj, p2_proj)
    """
    current_contour, next_contour = pair
    return find_breakpoints_between_contours(
        current_contour, 
        next_contour, 
        line_spacing, 
        n_layers_period,
        max_breakpoint_distance
    )

def find_breakpoints_between_contours(
    current_contour: Contour,
    next_contour: Contour,
    line_spacing: float,
    n_layers_period: int,
    max_breakpoint_distance: float = None
) -> List[Tuple[ShapelyPoint, ShapelyPoint, ShapelyPoint, ShapelyPoint]]:
    """
    Finds breakpoints between two contours.
    Uses Shapely's distance calculations for efficiency.
    
    Args:
        current_contour: The current contour
        next_contour: The neighboring contour
        line_spacing: The spacing between contours
        n_layers_period: How often to place breakpoints
        max_breakpoint_distance: Maximum distance to consider for breakpoints
        
    Returns:
        List of breakpoint tuples (p1, p2, p1_proj, p2_proj)
    """
    breakpoints = []
    
    # Skip if either contour is invalid
    if not current_contour.points or len(current_contour.points) < 2:
        return breakpoints
    if not next_contour.points or len(next_contour.points) < 2:
        return breakpoints
    
    # Skip if either LineString is invalid
    if not current_contour._line or current_contour._line.is_empty:
        return breakpoints
    if not next_contour._line or next_contour._line.is_empty:
        return breakpoints
    
    # Check if contours are close enough to consider breakpoints
    min_distance = current_contour._line.distance(next_contour._line)
    
    if max_breakpoint_distance is None:
        config = get_config()
        max_breakpoint_distance = config.get('max_breakpoint_distance', 2.0)
    
    if min_distance > max_breakpoint_distance:
        return breakpoints  # Contours are too far apart
    
    # Get the total length of the current contour
    total_length = current_contour.length()
    
    # Calculate how many breakpoints to create based on contour length and n_layers_period
    # We want approximately one breakpoint every (line_spacing * n_layers_period) distance
    target_spacing = line_spacing * n_layers_period
    
    # Adaptive number of breakpoints based on contour length and proximity
    # Fewer breakpoints for distant contours, more for close ones
    proximity_factor = 1.0 - (min_distance / max_breakpoint_distance)
    adjusted_num_breakpoints = max(1, int((total_length / target_spacing) * (0.5 + proximity_factor)))
    num_breakpoints = min(adjusted_num_breakpoints, 10)  # Cap at 10 breakpoints per contour pair
    
    # For each breakpoint position - use vectorized operations where possible
    points_at_distances = []
    
    # Pre-calculate all points at evenly spaced distances
    for i in range(num_breakpoints):
        # Calculate the position along the contour for p1
        p1_distance = (i / num_breakpoints) * total_length
        
        # Get the point at this distance and its segment
        p1, p1_segment, p1_segment_idx = current_contour.get_point_at_dist(p1_distance)
        
        if p1 is None or p1_segment is None:
            continue
            
        # Calculate the position for p2, which is line_spacing distance from p1 along the contour
        p2_distance = p1_distance + line_spacing
        if p2_distance > total_length:
            p2_distance = p2_distance - total_length  # Wrap around
            
        # Get the point at this distance and its segment
        p2, p2_segment, p2_segment_idx = current_contour.get_point_at_dist(p2_distance)
        
        if p2 is None or p2_segment is None:
            continue
            
        points_at_distances.append((p1, p2, p1_segment, p2_segment))
    
    # Skip if no valid points were found
    if not points_at_distances:
        return breakpoints
    
    # Create a list of all points to find projections for
    all_points = []
    for p1, p2, _, _ in points_at_distances:
        all_points.append(p1)
        all_points.append(p2)
    
    # Find the closest segments for all points at once using a spatial index
    # This is more efficient than finding them one by one
    next_contour_segments = next_contour.get_segments()
    segment_linestrings = [seg._line for seg in next_contour_segments if seg._line and not seg._line.is_empty]
    
    if not segment_linestrings:
        return breakpoints
    
    # Create a spatial index for the segments
    segment_index = STRtree(segment_linestrings)
    
    # Find projections for all points
    point_to_projection = {}
    for point in all_points:
        # Find the closest segment using the spatial index
        nearest_idx = segment_index.nearest(point)
        if nearest_idx is not None:
            nearest_segment = next_contour_segments[nearest_idx]
            # Project the point onto the segment
            projected_point = nearest_segment.point_projection(point)
            point_to_projection[point] = projected_point
    
    # Create breakpoints using the projections
    for p1, p2, _, _ in points_at_distances:
        if p1 in point_to_projection and p2 in point_to_projection:
            p1_proj = point_to_projection[p1]
            p2_proj = point_to_projection[p2]
            
            # Check if the projections are valid
            if p1_proj and p2_proj:
                # Check if the distance is within the maximum allowed distance
                if p1.distance(p1_proj) <= max_breakpoint_distance and p2.distance(p2_proj) <= max_breakpoint_distance:
                    # Add the breakpoint
                    breakpoints.append((p1, p2, p1_proj, p2_proj))
    
    return breakpoints


#-----------------------------------------------------------------------------
# 6. Generating continuous path (Connecting Sub-paths) - Linear Scan Method
#-----------------------------------------------------------------------------

def connect_subpaths_to_breakpoints(sub_paths_data: Dict) -> Tuple[List[SubPath], Dict]:
    """
    First step of the connection process: Connect subpaths to their breakpoints.
    This prepares the data for the final global path creation.
    Uses caching and early filtering for better performance.

    Args:
        sub_paths_data: A dictionary containing paths, contours_by_level, and all_breakpoints.

    Returns:
        A tuple containing:
        - List of processed SubPath objects with breakpoint connections
        - Dictionary with connection metadata for the global path creation
    """
    # Enable debug visualization if configured
    config = get_config()
    debug_visualization = config.get('debug_connect_subpaths', False)
    if not sub_paths_data:
        return [], {}

    # Extract data from the dictionary
    sub_paths_list = sub_paths_data.get('paths', [])
    all_breakpoints = sub_paths_data.get('all_breakpoints', [])
    contours_by_level = sub_paths_data.get('contours_by_level', {})

    # Quick check if we have enough data to proceed
    if not all_breakpoints or not contours_by_level:
        # Filter out empty paths and return early
        remaining_sub_paths = [p for p in sub_paths_list if p.points and len(p.points) >= 2]
        print("No valid contour or breakpoint data available for connection")
        return remaining_sub_paths, {}

    # Filter out empty paths
    remaining_sub_paths = [p for p in sub_paths_list if p.points and len(p.points) >= 2]
    if not remaining_sub_paths:
        return [], {}
    
    print(f"Found {len(all_breakpoints)} breakpoints in skeleton data")
    
    if all_breakpoints and contours_by_level:
        print("Processing breakpoints for path connection")
        processed_paths, connection_metadata = process_breakpoint_connections(
            remaining_sub_paths, 
            all_breakpoints, 
            contours_by_level
        )
        # Debug visualization if enabled
        if debug_visualization:
            visualize_connect_subpaths_debug(processed_paths, connection_metadata, "With Breakpoint Connections")
            
        return processed_paths, connection_metadata
    
    print("No breakpoints available for connection")
    
    # Debug visualization if enabled
    if debug_visualization:
        visualize_connect_subpaths_debug(remaining_sub_paths, {}, "No Breakpoints Available")
        
    return remaining_sub_paths, {}

def connect_all_subpaths_to_create_global_path(processed_paths: List[SubPath], connection_metadata: Dict) -> List[ShapelyPoint]:
    """
    Second step of the connection process: Create a single continuous global path
    from the processed subpaths and connection metadata.

    Args:
        processed_paths: List of SubPath objects with breakpoint connections
        connection_metadata: Dictionary with connection information from the first step

    Returns:
        A list of Shapely Points representing the final continuous toolpath
    """
    if not processed_paths:
        return []
        
    # If we have connection metadata, use it to create an optimized path
    if connection_metadata and 'breakpoint_connections' in connection_metadata:
        print("Creating global path using breakpoint connections")
        return create_global_path_with_breakpoints(processed_paths, connection_metadata)
    
    # Fallback to distance-based connection if no metadata available
    print("No connection metadata available, using distance-based connection")
    return connect_using_distance(processed_paths)

def connect_sub_paths(sub_paths: List[SubPath]) -> List[ShapelyPoint]:
    """
    Legacy function that combines both steps of the connection process.
    For backward compatibility.

    Args:
        sub_paths: A list of SubPath objects or a SubPathsWithSkeletonData object.

    Returns:
        A list of Shapely Points representing the final toolpath.
    """
    processed_paths, connection_metadata = connect_subpaths_to_breakpoints(sub_paths)
    return connect_all_subpaths_to_create_global_path(processed_paths, connection_metadata)

def process_breakpoint_connections(sub_paths: List[SubPath], all_breakpoints: list, contours_by_level: dict) -> Tuple[List[SubPath], Dict]:
    """
    Processes subpaths and their breakpoint connections to prepare for global path creation.
    Uses spatial indexing and caching for better performance.
    
    Args:
        sub_paths: List of SubPath objects
        all_breakpoints: List of breakpoints (p1, p2, p1_proj, p2_proj)
        contours_by_level: Dictionary mapping level indices to lists of contours
        
    Returns:
        Tuple containing:
        - List of processed SubPath objects
        - Dictionary with connection metadata
    """
    if not all_breakpoints:
        print("No breakpoints available for processing")
        return sub_paths, {}
        
    print(f"Processing {len(all_breakpoints)} breakpoints for connections")
    
    # Step 1: Create a map of contours to their sub-paths using a more efficient approach
    contour_to_subpath = {}
    
    # Create a dictionary of subpath endpoints for faster lookup
    subpath_endpoints = {}
    for i, sub_path in enumerate(sub_paths):
        # Skip connecting segments (they're handled separately)
        if len(sub_path.points) < 3:  # Connecting segments typically have just 2 points
            continue
        
        # Store the first and last point of each subpath for quick lookup
        key = (len(sub_path.points), 
               (sub_path.points[0].x, sub_path.points[0].y), 
               (sub_path.points[-1].x, sub_path.points[-1].y))
        subpath_endpoints[key] = i
    
    # Match contours to subpaths using the endpoint dictionary
    for level_idx, contours in contours_by_level.items():
        for contour in contours:
            if len(contour.points) < 3:
                continue
                
            # Create a key for this contour
            key = (len(contour.points), 
                   (contour.points[0].x, contour.points[0].y), 
                   (contour.points[-1].x, contour.points[-1].y))
            
            # Check if we have a matching subpath
            if key in subpath_endpoints:
                contour_to_subpath[contour] = subpath_endpoints[key]
    
    # Step 2: Create a map of breakpoints to their connecting segments using spatial indexing
    breakpoint_connections = {}
    
    # Identify connecting segments (those with only 2 points)
    connecting_segments = [(i, sub_path) for i, sub_path in enumerate(sub_paths) if len(sub_path.points) == 2]
    
    # Create spatial index for segment endpoints
    from collections import defaultdict
    endpoint_to_segments = defaultdict(list)
    
    # Round coordinates to reduce floating point comparison issues
    precision = int(-math.log10(POINT_EQUALITY_TOLERANCE))
    
    for seg_idx, segment in connecting_segments:
        seg_p1, seg_p2 = segment.points
        # Round coordinates to handle floating point precision
        p1_key = (round(seg_p1.x, precision), round(seg_p1.y, precision))
        p2_key = (round(seg_p2.x, precision), round(seg_p2.y, precision))
        
        endpoint_to_segments[p1_key].append((seg_idx, 0))  # 0 indicates first point
        endpoint_to_segments[p2_key].append((seg_idx, 1))  # 1 indicates second point
    
    # Map breakpoints to their connecting segments using the spatial index
    for bp_idx, bp in enumerate(all_breakpoints):
        p1, p2, p1_proj, p2_proj = bp
        
        # Create rounded keys for each breakpoint point
        p1_key = (round(p1.x, precision), round(p1.y, precision))
        p2_key = (round(p2.x, precision), round(p2.y, precision))
        p1_proj_key = (round(p1_proj.x, precision), round(p1_proj.y, precision))
        p2_proj_key = (round(p2_proj.x, precision), round(p2_proj.y, precision))
        
        # Check all possible connections using the spatial index
        connected_segments = set()
        
        # Check p1 to p1_proj connections
        for p1_seg_idx, p1_point_idx in endpoint_to_segments.get(p1_key, []):
            for p1_proj_seg_idx, p1_proj_point_idx in endpoint_to_segments.get(p1_proj_key, []):
                if p1_seg_idx == p1_proj_seg_idx:
                    connected_segments.add(p1_seg_idx)
        
        # Check p2 to p2_proj connections
        for p2_seg_idx, p2_point_idx in endpoint_to_segments.get(p2_key, []):
            for p2_proj_seg_idx, p2_proj_point_idx in endpoint_to_segments.get(p2_proj_key, []):
                if p2_seg_idx == p2_proj_seg_idx:
                    connected_segments.add(p2_seg_idx)
        
        # Store the connections
        if connected_segments:
            breakpoint_connections[bp_idx] = list(connected_segments)
    
    # Step 3: Create connection metadata for each contour using a more efficient approach
    contour_breakpoints = {}
    
    # Create a spatial index for contour points
    contour_point_index = {}
    for level_idx, contours in contours_by_level.items():
        for contour in contours:
            for i, point in enumerate(contour.points):
                point_key = (round(point.x, precision), round(point.y, precision))
                if point_key not in contour_point_index:
                    contour_point_index[point_key] = []
                contour_point_index[point_key].append((contour, i))
    
    # Process each breakpoint to find source and target contours
    for bp_idx, bp in enumerate(all_breakpoints):
        p1, p2, p1_proj, p2_proj = bp
        
        # Create rounded keys for breakpoint points
        p1_key = (round(p1.x, precision), round(p1.y, precision))
        p2_key = (round(p2.x, precision), round(p2.y, precision))
        p1_proj_key = (round(p1_proj.x, precision), round(p1_proj.y, precision))
        p2_proj_key = (round(p2_proj.x, precision), round(p2_proj.y, precision))
        
        # Find source contour (contains p1/p2)
        source_contour = None
        position = -1
        
        # Check p1 first
        for contour, idx in contour_point_index.get(p1_key, []):
            source_contour = contour
            position = idx
            break
            
        # If not found, check p2
        if not source_contour:
            for contour, idx in contour_point_index.get(p2_key, []):
                source_contour = contour
                position = idx
                break
        
        # Find target contour (contains p1_proj/p2_proj)
        target_contour = None
        
        # Check p1_proj first
        for contour, _ in contour_point_index.get(p1_proj_key, []):
            target_contour = contour
            break
            
        # If not found, check p2_proj
        if not target_contour:
            for contour, _ in contour_point_index.get(p2_proj_key, []):
                target_contour = contour
                break
        
        # Store the connection if both contours were found
        if source_contour and target_contour:
            if source_contour not in contour_breakpoints:
                contour_breakpoints[source_contour] = []
            
            contour_breakpoints[source_contour].append({
                'bp_idx': bp_idx,
                'target_contour': target_contour,
                'position': position,
                'breakpoint': bp
            })
    
    # Sort breakpoints by position along each contour
    for contour in contour_breakpoints:
        contour_breakpoints[contour].sort(key=lambda x: x['position'])
    
    # Create the connection metadata
    connection_metadata = {
        'contour_to_subpath': contour_to_subpath,
        'breakpoint_connections': breakpoint_connections,
        'contour_breakpoints': contour_breakpoints,
        'all_breakpoints': all_breakpoints,
        'contours_by_level': contours_by_level
    }
    
    return sub_paths, connection_metadata

def create_global_path_with_breakpoints(sub_paths: List[SubPath], connection_metadata: Dict) -> List[ShapelyPoint]:
    """
    Creates a global continuous path using the processed breakpoint connections.
    
    Args:
        sub_paths: List of SubPath objects
        connection_metadata: Dictionary with connection information
        
    Returns:
        A list of Shapely Points representing the connected path
    """
    if not connection_metadata:
        print("No connection metadata available, falling back to distance-based connection")
        return connect_using_distance(sub_paths)
    
    # Extract metadata
    contour_to_subpath = connection_metadata.get('contour_to_subpath', {})
    breakpoint_connections = connection_metadata.get('breakpoint_connections', {})
    contour_breakpoints = connection_metadata.get('contour_breakpoints', {})
    contours_by_level = connection_metadata.get('contours_by_level', {})
    
    if not contour_to_subpath or not contours_by_level:
        print("Incomplete connection metadata, falling back to distance-based connection")
        return connect_using_distance(sub_paths)
    
    print("Creating global path using breakpoint connections")
    
    # Initialize the global path
    global_path = []
    visited_contours = set()
    visited_segments = set()
    
    # Start with the first contour in level 0
    if 0 in contours_by_level and contours_by_level[0]:
        start_contour = contours_by_level[0][0]
        current_contour = start_contour
        current_level = 0
    else:
        print("No level 0 contours found, falling back to distance-based")
        return connect_using_distance(sub_paths)
    
    # Main traversal loop - follow contours in counterclockwise direction
    while len(visited_contours) < sum(len(contours) for contours in contours_by_level.values()):
        # If we haven't visited this contour yet, add its points to the path
        if current_contour not in visited_contours:
            if current_contour in contour_to_subpath:
                subpath_idx = contour_to_subpath[current_contour]
                subpath = sub_paths[subpath_idx]
                
                # Add points in counterclockwise direction
                if len(global_path) == 0:
                    # First contour, add all points
                    global_path.extend(subpath.points)
                else:
                    # Connect to previous point
                    last_point = global_path[-1]
                    
                    # Find the closest endpoint of the contour
                    start_dist = last_point.distance(subpath.points[0])
                    end_dist = last_point.distance(subpath.points[-1])
                    
                    if start_dist < end_dist:
                        # Add points in original order, skipping first to avoid duplication
                        global_path.extend(subpath.points[1:])
                    else:
                        # Add points in reverse order, skipping last to avoid duplication
                        global_path.extend(reversed(subpath.points[:-1]))
                
                visited_contours.add(current_contour)
            
            # Check if this contour has breakpoints to follow
            if current_contour in contour_breakpoints and contour_breakpoints[current_contour]:
                # Get the sorted breakpoints for this contour
                sorted_bps = contour_breakpoints[current_contour]
                
                # Try to connect using each breakpoint
                for bp_info in sorted_bps:
                    bp_idx = bp_info['bp_idx']
                    target_contour = bp_info['target_contour']
                    bp = bp_info['breakpoint']
                    
                    # Skip if target contour already visited
                    if target_contour in visited_contours:
                        continue
                    
                    # Skip if we've already used this breakpoint
                    if bp_idx in breakpoint_connections:
                        # Find the connecting segments for this breakpoint
                        for seg_idx in breakpoint_connections[bp_idx]:
                            if seg_idx not in visited_segments:
                                # Add this connecting segment
                                segment = sub_paths[seg_idx]
                                
                                # Check if we need to reverse the segment
                                last_point = global_path[-1]
                                if last_point.distance(segment.points[0]) > last_point.distance(segment.points[-1]):
                                    global_path.extend(reversed(segment.points))
                                else:
                                    global_path.extend(segment.points[1:])  # Skip first point to avoid duplication
                                
                                visited_segments.add(seg_idx)
                                
                                # Move to the target contour
                                if target_contour not in visited_contours:
                                    # Find the level of the target contour
                                    for level_idx, contours in contours_by_level.items():
                                        if target_contour in contours:
                                            current_contour = target_contour
                                            current_level = level_idx
                                            break
                                    break
        
        # If we couldn't find a connection or have visited this contour, move to the next unvisited contour
        if current_contour in visited_contours:
            found_next = False
            
            # Try to find an unvisited contour in the current level first
            if current_level in contours_by_level:
                for next_contour in contours_by_level[current_level]:
                    if next_contour not in visited_contours:
                        current_contour = next_contour
                        found_next = True
                        break
            
            # If we couldn't find an unvisited contour in the current level,
            # try other levels
            if not found_next:
                for level_idx, contours in contours_by_level.items():
                    for next_contour in contours:
                        if next_contour not in visited_contours:
                            current_contour = next_contour
                            current_level = level_idx
                            found_next = True
                            break
                    if found_next:
                        break
            
            # If we still couldn't find an unvisited contour, we're done
            if not found_next:
                break
    
    return global_path

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


def visualize_connect_subpaths_debug(processed_paths: List[SubPath], connection_metadata: Dict, title_suffix: str = ""):
    """
    Debug visualization for the results of connect_subpaths_to_breakpoints.
    Shows the processed subpaths and their connections to breakpoints.
    
    Args:
        processed_paths: List of SubPath objects after processing
        connection_metadata: Dictionary with connection information
        title_suffix: Optional suffix for the plot title
    """
    print("\n--- Debug Visualization: connect_subpaths_to_breakpoints ---")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
        
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        # Plot all subpaths with different colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(processed_paths)))
        
        # Create a legend dictionary to track unique path types
        legend_elements = []
        
        # Plot each subpath
        for i, subpath in enumerate(processed_paths):
            if not subpath.points or len(subpath.points) < 2:
                continue
                
            # Extract coordinates
            x = [p.x for p in subpath.points]
            y = [p.y for p in subpath.points]
            
            # Determine if this is a connecting segment (2 points) or a contour
            if len(subpath.points) == 2:
                # This is likely a connecting segment
                ax.plot(x, y, '--', color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
                
                # Add to legend if not already there
                if not any(item.get_label() == 'Connecting Segment' for item in legend_elements):
                    legend_elements.append(Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='Connecting Segment'))
                
                # Mark endpoints
                ax.plot(x[0], y[0], 'o', color=colors[i % len(colors)], markersize=4)
                ax.plot(x[1], y[1], 'o', color=colors[i % len(colors)], markersize=4)
            else:
                # This is a contour
                ax.plot(x, y, '-', color=colors[i % len(colors)], linewidth=1.0)
                
                # Add to legend if not already there
                if not any(item.get_label() == 'Contour' for item in legend_elements):
                    legend_elements.append(Patch(facecolor='blue', edgecolor='black', alpha=0.5, label='Contour'))
                
                # Mark start point
                ax.plot(x[0], y[0], 'o', color='green', markersize=5)
                
                # Add subpath index as text
                mid_idx = len(x) // 2
                ax.text(x[mid_idx], y[mid_idx], f'SP{i}', fontsize=8, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # If we have connection metadata, visualize the connections
        if connection_metadata:
            # Extract data from connection metadata
            contour_to_subpath = connection_metadata.get('contour_to_subpath', {})
            breakpoint_connections = connection_metadata.get('breakpoint_connections', {})
            contour_breakpoints = connection_metadata.get('contour_breakpoints', {})
            all_breakpoints = connection_metadata.get('all_breakpoints', [])
            
            # Plot breakpoints
            if all_breakpoints:
                for bp_idx, bp in enumerate(all_breakpoints):
                    p1, p2, p1_proj, p2_proj = bp
                    
                    # Plot the breakpoint pairs
                    ax.plot(p1.x, p1.y, 'ro', markersize=5)
                    ax.plot(p2.x, p2.y, 'bo', markersize=5)
                    
                    # Plot the projected points
                    ax.plot(p1_proj.x, p1_proj.y, 'go', markersize=5)
                    ax.plot(p2_proj.x, p2_proj.y, 'mo', markersize=5)
                    
                    # Add breakpoint index as text
                    ax.text((p1.x + p2.x)/2, (p1.y + p2.y)/2, f'BP{bp_idx}', fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    
                    # Add to legend if not already there
                    if not any(item.get_label() == 'Breakpoint P1' for item in legend_elements):
                        legend_elements.append(Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Breakpoint P1'))
                    if not any(item.get_label() == 'Breakpoint P2' for item in legend_elements):
                        legend_elements.append(Patch(facecolor='blue', edgecolor='black', alpha=0.5, label='Breakpoint P2'))
                    if not any(item.get_label() == 'Projected Point' for item in legend_elements):
                        legend_elements.append(Patch(facecolor='green', edgecolor='black', alpha=0.5, label='Projected Point'))
            
            # Visualize contour-to-subpath mapping
            for contour, subpath_idx in contour_to_subpath.items():
                if contour._line and not contour._line.is_empty:
                    centroid = contour._line.centroid
                    ax.text(centroid.x, centroid.y, f'C→SP{subpath_idx}', fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black', pad=1))
            
            # Visualize breakpoint connections
            for bp_idx, segment_indices in breakpoint_connections.items():
                if bp_idx < len(all_breakpoints):
                    bp = all_breakpoints[bp_idx]
                    p1, p2, p1_proj, p2_proj = bp
                    
                    # Draw connections
                    ax.plot([p1.x, p1_proj.x], [p1.y, p1_proj.y], 'g--', linewidth=1.0, alpha=0.7)
                    ax.plot([p2.x, p2_proj.x], [p2.y, p2_proj.y], 'm--', linewidth=1.0, alpha=0.7)
                    
                    # Add text showing which segments are connected
                    mid_x1 = (p1.x + p1_proj.x) / 2
                    mid_y1 = (p1.y + p1_proj.y) / 2
                    mid_x2 = (p2.x + p2_proj.x) / 2
                    mid_y2 = (p2.y + p2_proj.y) / 2
                    
                    seg_text = ', '.join([f'S{idx}' for idx in segment_indices])
                    ax.text(mid_x1, mid_y1, f'BP{bp_idx}→{seg_text}', fontsize=7, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        plt.title(f"Connect Subpaths to Breakpoints Debug {title_suffix}")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        
        # Add the legend
        if legend_elements:
            plt.legend(handles=legend_elements, loc='best', fontsize='small')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nInstall matplotlib to visualize the debug results: pip install matplotlib")
    except Exception as e:
        print(f"\nError during debug visualization: {e}")
        import traceback
        traceback.print_exc()

def visualize_breakpoint_connections(connection_metadata: Dict, layer_to_process_idx: int):
    """
    Visualizes the breakpoint connections between contours.
    
    Args:
        connection_metadata: Dictionary with connection information
        layer_to_process_idx: The layer index being processed
    """
    print("\n--- Visualizing Breakpoint Connections ---")
    if not connection_metadata:
        print("No connection metadata available for visualization.")
        return
        
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data from connection metadata
        contour_to_subpath = connection_metadata.get('contour_to_subpath', {})
        breakpoint_connections = connection_metadata.get('breakpoint_connections', {})
        contour_breakpoints = connection_metadata.get('contour_breakpoints', {})
        all_breakpoints = connection_metadata.get('all_breakpoints', [])
        contours_by_level = connection_metadata.get('contours_by_level', {})
        
        if not contours_by_level or not all_breakpoints:
            print("Insufficient data for breakpoint visualization.")
            return
            
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        # Plot contours by level with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(contours_by_level)))
        for i, (level_idx, contours) in enumerate(contours_by_level.items()):
            for contour in contours:
                if contour._line and not contour._line.is_empty:
                    x, y = contour._line.xy
                    ax.plot(x, y, color=colors[i], linestyle='-', linewidth=1.0, 
                           label=f'Level {level_idx}' if f'Level {level_idx}' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # Plot breakpoints and their connections
        for bp_idx, bp in enumerate(all_breakpoints):
            p1, p2, p1_proj, p2_proj = bp
            
            # Plot the breakpoint pairs
            ax.plot(p1.x, p1.y, 'ro', markersize=5, label='Breakpoint P1' if 'Breakpoint P1' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax.plot(p2.x, p2.y, 'bo', markersize=5, label='Breakpoint P2' if 'Breakpoint P2' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Plot the projected points
            ax.plot(p1_proj.x, p1_proj.y, 'go', markersize=5, label='Projected P1' if 'Projected P1' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax.plot(p2_proj.x, p2_proj.y, 'mo', markersize=5, label='Projected P2' if 'Projected P2' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Plot the connections
            ax.plot([p1.x, p1_proj.x], [p1.y, p1_proj.y], 'r--', linewidth=1.0, alpha=0.7, 
                   label='P1 Connection' if 'P1 Connection' not in plt.gca().get_legend_handles_labels()[1] else "")
            ax.plot([p2.x, p2_proj.x], [p2.y, p2_proj.y], 'b--', linewidth=1.0, alpha=0.7,
                   label='P2 Connection' if 'P2 Connection' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Add breakpoint index labels
            ax.text(p1.x, p1.y, f'{bp_idx}', fontsize=8, ha='right', va='bottom')
            ax.text(p2.x, p2.y, f'{bp_idx}', fontsize=8, ha='right', va='bottom')
        
        # Plot the contour-to-contour connections from contour_breakpoints
        for source_contour, bp_infos in contour_breakpoints.items():
            for bp_info in bp_infos:
                target_contour = bp_info['target_contour']
                bp_idx = bp_info['bp_idx']
                
                # Find the centroids of source and target contours
                if source_contour._line and not source_contour._line.is_empty and target_contour._line and not target_contour._line.is_empty:
                    source_centroid = source_contour._line.centroid
                    target_centroid = target_contour._line.centroid
                    
                    # Draw a light connection between contour centroids
                    ax.plot([source_centroid.x, target_centroid.x], [source_centroid.y, target_centroid.y], 
                           'k:', linewidth=0.5, alpha=0.3)
                    
                    # Add text label with breakpoint index
                    mid_x = (source_centroid.x + target_centroid.x) / 2
                    mid_y = (source_centroid.y + target_centroid.y) / 2
                    ax.text(mid_x, mid_y, f'BP{bp_idx}', fontsize=8, ha='center', va='center', 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        plt.title(f"Breakpoint Connections - Layer {layer_to_process_idx}")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        
        # Create legend with unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='small', loc='best')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nInstall matplotlib to visualize the breakpoint connections: pip install matplotlib")
    except Exception as e:
        print(f"\nError during breakpoint visualization: {e}")
        import traceback
        traceback.print_exc()

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

    #stl_file_path = os.path.join(project_root, "models", "mine", "gear.stl")

    #stl_file_path = os.path.join("models", "wrench.stl")

    stl_file_path = os.path.join(project_root, "models", "hollow-cuboid.stl")

    #stl_file_path = os.path.join("models", "t-shape.stl")
    #stl_file_path = os.path.join("models", "u-shape.stl")

    #stl_file_path = os.path.join("models", "cuboid-with-holes.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "hex.stl")
    #stl_file_path = os.path.join("models", "mine", "polygon-c-solid.stl")

    #stl_file_path = os.path.join(project_root, "models", "cuboid.stl")
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


    # --- 4/5. Create Sub-paths with Breakpoints ---
    print("\n--- Creating Sub-paths with Breakpoints ---")
    n_layers_period = config.get('breakpoint_period', 5) # How often breakpoint strategy changes
    sub_paths = create_sub_paths(
        leveled_contours,
        line_spacing,
    )
    if not sub_paths:
         sys.exit("Failed to create sub-paths using rasterization.")
    print(f"Total sub-paths created: {len(sub_paths)}")


    # --- 6a. Connect Sub-paths to Breakpoints ---
    print("\n--- Connecting Sub-paths to Breakpoints ---")
    
    # Debug check for breakpoints
    if sub_paths and 'all_breakpoints' in sub_paths:
        print(f"DEBUG: Main function - found {len(sub_paths['all_breakpoints'])} breakpoints")
    else:
        print("DEBUG: Main function - no breakpoints found in data")
    
    # Step 1: Connect subpaths to breakpoints
    processed_paths, connection_metadata = connect_subpaths_to_breakpoints(sub_paths)
    print(f"Processed {len(processed_paths)} sub-paths with breakpoint connections")
    
    # --- 6b. Create Global Path from Connected Sub-paths ---
    print("\n--- Creating Global Path from Connected Sub-paths ---")
    
    # Step 2: Create the global path
    final_toolpath = connect_all_subpaths_to_create_global_path(processed_paths, connection_metadata)
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
         
    # --- 7b. Optional: Visualize Contours and Breakpoints ---
    if config.get('visualize_breakpoints', True) and sub_paths:
        try:
            import matplotlib.pyplot as plt
            
            # Unpack the data
            contours_by_level = sub_paths.get('contours_by_level', {})
            all_breakpoints = sub_paths.get('all_breakpoints', [])
            
            if contours_by_level:
                # Visualize contours
                plt.figure(figsize=(10, 10))
                
                # Plot contours
                for level_idx, contours in contours_by_level.items():
                    for contour in contours:
                        if contour._line and not contour._line.is_empty:
                            x, y = contour._line.xy
                            plt.plot(x, y, '-', linewidth=1, alpha=0.7)
                
                # Plot breakpoints and connections
                if all_breakpoints:
                    for p1, p2, p1_proj, p2_proj in all_breakpoints:
                        # Plot the breakpoints
                        plt.plot(p1.x, p1.y, 'ro', markersize=4)
                        plt.plot(p2.x, p2.y, 'bo', markersize=4)
                        plt.plot(p1_proj.x, p1_proj.y, 'go', markersize=4)
                        plt.plot(p2_proj.x, p2_proj.y, 'mo', markersize=4)
                        
                        # Plot the connections
                        plt.plot([p1.x, p1_proj.x], [p1.y, p1_proj.y], 'r--', linewidth=0.8, alpha=0.6)
                        plt.plot([p2.x, p2_proj.x], [p2.y, p2_proj.y], 'b--', linewidth=0.8, alpha=0.6)
                
                plt.title(f"Contours and Breakpoints - Layer {layer_to_process_idx}")
                plt.axis('equal')
                plt.tight_layout()
                plt.show()
                
                # Call the dedicated breakpoint visualization function
                visualize_breakpoint_connections(connection_metadata, layer_to_process_idx)
            else:
                print("Contour data not available for visualization")
        except Exception as e:
            print(f"Error visualizing contours and breakpoints: {e}")

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

