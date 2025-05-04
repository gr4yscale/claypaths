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

def create_sub_paths_via_rasterization(
    leveled_contours: List[List[Contour]],
    line_spacing: float,
    resolution: float = 0.1 # mm per pixel
) -> List[SubPath]:
    """
    Creates sub-paths by finding optimal breakpoints between contours using skeletonization.
    
    Args:
        leveled_contours: Contours grouped by level (output of re_level_contours).
        line_spacing: The characteristic width/distance between contours.
        resolution: The size of each pixel in millimeters for rasterization.

    Returns:
        A list of SubPath objects representing the connected contour segments.
    """
    if not cv2_available:
        print("Warning: OpenCV and scikit-image not available. Falling back to spatial index method.")
        # Implement fallback method here if needed
        return []
        
    all_contours = [c for level in leveled_contours for c in level if c.points and len(c.points) >= 2]
    if not all_contours:
        print("Skeletonization: No valid contours provided.")
        return []

    print(f"Skeletonization: Starting with {len(all_contours)} contours.")
    
    # Flatten contours by level for processing
    contours_by_level = {}
    for level_idx, level_list in enumerate(leveled_contours):
        contours_by_level[level_idx] = [c for c in level_list if c.points and len(c.points) >= 2]
    
    # Find the bounding box of all contours to determine image size
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for contour in all_contours:
        for point in contour.points:
            min_x = min(min_x, point.x)
            min_y = min(min_y, point.y)
            max_x = max(max_x, point.x)
            max_y = max(max_y, point.y)
    
    # Add padding to the bounding box
    padding = line_spacing * 2
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    
    # Calculate image dimensions based on resolution
    # Use higher resolution for skeletonization
    resolution = min(resolution, 0.05)  # Ensure fine enough resolution for skeleton
    width = int((max_x - min_x) / resolution) + 1
    height = int((max_y - min_y) / resolution) + 1
    
    # Ensure reasonable image dimensions
    if width > 5000 or height > 5000:
        scale_factor = min(5000 / width, 5000 / height)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        resolution = resolution / scale_factor
        print(f"Adjusted resolution to {resolution:.4f} mm/pixel to limit image size")
    
    print(f"Creating image of size {width}x{height} pixels at {resolution:.4f} mm/pixel")
    
    # Create transformation functions between world and image coordinates
    def world_to_image(x, y):
        img_x = int((x - min_x) / resolution)
        img_y = int((y - min_y) / resolution)
        return img_x, img_y
    
    def image_to_world(img_x, img_y):
        x = img_x * resolution + min_x
        y = img_y * resolution + min_y
        return x, y
    
    # Create separate images for each level
    level_images = {}
    for level_idx, contours in contours_by_level.items():
        img = np.zeros((height, width), dtype=np.uint8)
        
        for contour in contours:
            # Convert contour points to image coordinates
            contour_points = []
            for point in contour.points:
                img_x, img_y = world_to_image(point.x, point.y)
                contour_points.append([img_x, img_y])
            
            # Convert to numpy array for OpenCV
            contour_points = np.array(contour_points, dtype=np.int32)
            
            # Draw the contour
            cv2.drawContours(img, [contour_points], 0, 255, thickness=max(1, int(line_spacing / resolution / 2)))
        
        level_images[level_idx] = img
    
    # Create a combined image for skeletonization
    combined_img = np.zeros((height, width), dtype=np.uint8)
    for level_img in level_images.values():
        combined_img = np.maximum(combined_img, level_img)
    
    # Create skeleton of the combined image
    print("Generating skeleton of the combined contours...")
    # Apply slight blur to smooth the contours before skeletonization
    blurred_img = cv2.GaussianBlur(combined_img, (5, 5), 0)
    
    # Binarize the image
    _, binary_img = cv2.threshold(blurred_img, 127, 1, cv2.THRESH_BINARY)
    
    # Generate skeleton
    skeleton = skeletonize(binary_img)
    skeleton_img = skeleton.astype(np.uint8) * 255
    
    # Find junction points in the skeleton
    # These are points where multiple branches meet
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    # Convolve with kernel to find junction points (pixels with >2 neighbors)
    convolved = cv2.filter2D(skeleton_img, -1, kernel)
    junction_points = np.where((convolved > 11*255) & (skeleton_img > 0))
    junction_coords = list(zip(junction_points[1], junction_points[0]))  # x, y format
    
    print(f"Found {len(junction_coords)} junction points in skeleton")
    
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
        
        # Get images for current and next level
        current_img = level_images.get(current_level)
        next_img = level_images.get(next_level)
        
        if current_img is None or next_img is None:
            continue
        
        # For each junction point, find the closest points on both contour levels
        for junction_x, junction_y in junction_coords:
            junction_point_world = image_to_world(junction_x, junction_y)
            junction_point = ShapelyPoint(junction_point_world[0], junction_point_world[1])
            
            # Check if this junction is between the current and next level
            # Use full resolution distance transforms for more accurate measurement
            
            # Calculate distance transforms if not already cached
            if not hasattr(current_img, 'dist_transform'):
                current_img.dist_transform = cv2.distanceTransform(255 - current_img, cv2.DIST_L2, 5)
            if not hasattr(next_img, 'dist_transform'):
                next_img.dist_transform = cv2.distanceTransform(255 - next_img, cv2.DIST_L2, 5)
            
            # Get distances at the junction point
            if (0 <= junction_y < current_img.dist_transform.shape[0] and 
                0 <= junction_x < current_img.dist_transform.shape[1]):
                current_level_dist = current_img.dist_transform[junction_y, junction_x] * resolution
                next_level_dist = next_img.dist_transform[junction_y, junction_x] * resolution
            else:
                # Fallback to point-polygon test if out of bounds
                current_level_dist = cv2.pointPolygonTest(
                    np.array([world_to_image(p.x, p.y) for c in current_contours for p in c.points], dtype=np.int32),
                    (junction_x, junction_y),
                    True
                ) * resolution
                
                next_level_dist = cv2.pointPolygonTest(
                    np.array([world_to_image(p.x, p.y) for c in next_contours for p in c.points], dtype=np.int32),
                    (junction_x, junction_y),
                    True
                ) * resolution
            
            # Only consider junctions that are close to both levels
            max_distance = line_spacing * config.get('max_breakpoint_distance', 1.5)
            # Junction should be between the two levels, not too close to either
            min_distance = line_spacing * 0.2  # Minimum distance to ensure junction is between levels
            
            # Calculate the ratio of distances to both levels - should be balanced for good junctions
            distance_ratio = 1.0
            if abs(current_level_dist) > 0 and abs(next_level_dist) > 0:
                distance_ratio = max(abs(current_level_dist), abs(next_level_dist)) / min(abs(current_level_dist), abs(next_level_dist))
            
            # Good junction points are those that are:
            # 1. Close enough to both levels
            # 2. Not too close to either level
            # 3. Have a balanced distance ratio (not much closer to one than the other)
            if (abs(current_level_dist) < max_distance and abs(next_level_dist) < max_distance and
                abs(current_level_dist) > min_distance and abs(next_level_dist) > min_distance and
                distance_ratio < 3.0):  # Ratio threshold - adjust as needed
                # Find closest contour and point in current level
                best_current_contour_idx = -1
                best_current_point_idx = -1
                min_current_dist = float('inf')
                
                for i, current_contour in enumerate(current_contours):
                    if not current_contour.points or len(current_contour.points) < 2:
                        continue
                    
                    # Find closest point on current contour
                    current_line = current_contour._line
                    if not current_line or current_line.is_empty:
                        continue
                    
                    # Project junction point onto current contour
                    current_point_coords = current_line.interpolate(current_line.project(junction_point))
                    current_point = ShapelyPoint(current_point_coords.x, current_point_coords.y)
                    
                    # Calculate distance
                    dist = junction_point.distance(current_point)
                    
                    if dist < min_current_dist:
                        min_current_dist = dist
                        best_current_contour_idx = i
                        
                        # Find the index of the closest point in the current contour
                        best_current_point_idx = min(range(len(current_contour.points)), 
                                                  key=lambda k: current_contour.points[k].distance(current_point))
                
                # Find closest contour and point in next level
                best_next_contour_idx = -1
                best_next_point_idx = -1
                min_next_dist = float('inf')
                
                for j, next_contour in enumerate(next_contours):
                    if not next_contour.points or len(next_contour.points) < 2:
                        continue
                    
                    # Find closest point on next contour
                    next_line = next_contour._line
                    if not next_line or next_line.is_empty:
                        continue
                    
                    # Project junction point onto next contour
                    next_point_coords = next_line.interpolate(next_line.project(junction_point))
                    next_point = ShapelyPoint(next_point_coords.x, next_point_coords.y)
                    
                    # Calculate distance
                    dist = junction_point.distance(next_point)
                    
                    if dist < min_next_dist:
                        min_next_dist = dist
                        best_next_contour_idx = j
                        
                        # Find the index of the closest point in the next contour
                        best_next_point_idx = min(range(len(next_contour.points)), 
                                                key=lambda k: next_contour.points[k].distance(next_point))
                
                # If we found good connections to both levels, add as a breakpoint
                if (best_current_contour_idx != -1 and best_next_contour_idx != -1 and 
                    min_current_dist < max_distance and min_next_dist < max_distance):
                    
                    # Check if there are existing breakpoints on this contour pair
                    existing_breakpoint = False
                    min_existing_distance = float('inf')
                    
                    # Get the points we're considering adding
                    current_contour = current_contours[best_current_contour_idx]
                    next_contour = next_contours[best_next_contour_idx]
                    current_point = current_contour.points[best_current_point_idx]
                    next_point = next_contour.points[best_next_point_idx]
                    
                    # Check distance to existing breakpoints
                    for bp in all_breakpoints:
                        if (bp[0] == current_level and bp[1] == best_current_contour_idx and 
                            bp[3] == next_level and bp[4] == best_next_contour_idx):
                            
                            # Get existing breakpoint points
                            bp_current_point = current_contour.points[bp[2]]
                            bp_next_point = next_contour.points[bp[5]]
                            
                            # Calculate distance along contour
                            current_contour_length = current_contour.length()
                            dist_along_contour = abs(current_contour._line.project(current_point) - 
                                                    current_contour._line.project(bp_current_point))
                            
                            # Normalize by contour length
                            normalized_dist = dist_along_contour / current_contour_length
                            
                            if normalized_dist < config.get('min_breakpoint_spacing', 0.2):
                                existing_breakpoint = True
                                break
                            
                            min_existing_distance = min(min_existing_distance, normalized_dist)
                    
                    # Only add if not too close to existing breakpoints
                    if not existing_breakpoint:
                        # Store breakpoint information
                        all_breakpoints.append((
                            current_level, best_current_contour_idx, best_current_point_idx, 
                            next_level, best_next_contour_idx, best_next_point_idx
                        ))
                        
                        # Store breakpoint in contour objects for visualization
                        current_contour = current_contours[best_current_contour_idx]
                        next_contour = next_contours[best_next_contour_idx]
                        
                        current_contour.breakpoints.append((
                            current_contour.points[best_current_point_idx], 
                            next_contour.points[best_next_point_idx],
                            current_contour.points[best_current_point_idx], 
                            next_contour.points[best_next_point_idx]
                        ))
                        
                        print(f"  Found skeleton junction breakpoint: Level {current_level}[{best_current_contour_idx}] to Level {next_level}[{best_next_contour_idx}]")
        
        # If we didn't find enough breakpoints using skeleton junctions,
        # add some additional breakpoints using the distance transform method
        if len([bp for bp in all_breakpoints if bp[0] == current_level and bp[3] == next_level]) < len(current_contours):
            print(f"  Adding additional breakpoints using distance transform...")
            
            # Create a distance transform for the next level
            next_dist_transform = cv2.distanceTransform(255 - next_img, cv2.DIST_L2, 5)
            
            # For each contour in current level that doesn't have a breakpoint yet
            for i, current_contour in enumerate(current_contours):
                # Skip if this contour already has enough breakpoints to the next level
                existing_breakpoints = [bp for bp in all_breakpoints 
                                      if bp[0] == current_level and bp[1] == i and bp[3] == next_level]
                
                # Calculate how many breakpoints we want based on contour length and complexity
                if current_contour.points and len(current_contour.points) >= 2:
                    contour_length = current_contour.length()
                    
                    # Estimate contour complexity by comparing perimeter to area
                    # More complex shapes need more breakpoints
                    try:
                        # Create a temporary polygon to calculate area
                        contour_poly = ShapelyPolygon([(p.x, p.y) for p in current_contour.points])
                        if not contour_poly.is_empty:
                            # Calculate complexity factor (perimeter²/area)
                            # Higher value means more complex shape
                            complexity = (contour_length ** 2) / (contour_poly.area * 4 * math.pi)
                            # Adjust for very complex shapes
                            complexity = min(complexity, 5.0)  # Cap complexity factor
                        else:
                            complexity = 1.0
                    except Exception:
                        complexity = 1.0
                    
                    # Adjust target breakpoints based on length and complexity
                    # For longer or more complex contours, add more breakpoints
                    target_breakpoints = max(1, int((contour_length / (line_spacing * 5)) * complexity))
                    
                    # For higher levels (deeper in the shape), we might need more breakpoints
                    level_factor = 1.0 + (level_idx * 0.1)  # Increase by 10% per level
                    target_breakpoints = max(1, int(target_breakpoints * level_factor))
                    
                    if len(existing_breakpoints) >= target_breakpoints:
                        continue
                
                if not current_contour.points or len(current_contour.points) < 2:
                    continue
                
                # Sample points along the current contour
                # Use more sample points for better coverage
                num_samples = min(max(50, len(current_contour.points) // 3), len(current_contour.points))
                
                # If we already have some breakpoints, avoid sampling near them
                if existing_breakpoints:
                    # Get existing breakpoint indices
                    existing_indices = [bp[2] for bp in existing_breakpoints]
                    
                    # Create a mask of valid sampling positions
                    valid_positions = np.ones(len(current_contour.points), dtype=bool)
                    min_spacing = int(len(current_contour.points) * config.get('min_breakpoint_spacing', 0.2))
                    
                    for idx in existing_indices:
                        # Mark positions too close to existing breakpoints as invalid
                        start_idx = max(0, idx - min_spacing)
                        end_idx = min(len(valid_positions), idx + min_spacing + 1)
                        valid_positions[start_idx:end_idx] = False
                    
                    # Get valid indices
                    valid_indices = np.where(valid_positions)[0]
                    
                    if len(valid_indices) > 0:
                        # Sample from valid positions
                        if len(valid_indices) <= num_samples:
                            sample_indices = valid_indices
                        else:
                            # Evenly sample from valid positions
                            sample_idx = np.linspace(0, len(valid_indices)-1, num_samples, dtype=int)
                            sample_indices = valid_indices[sample_idx]
                    else:
                        # If no valid positions, use regular sampling
                        sample_indices = np.linspace(0, len(current_contour.points)-1, num_samples, dtype=int)
                else:
                    # No existing breakpoints, use regular sampling
                    sample_indices = np.linspace(0, len(current_contour.points)-1, num_samples, dtype=int)
                
                # For each sample point, find the closest point on the next level
                best_sample_idx = -1
                best_next_contour_idx = -1
                best_next_point_idx = -1
                min_connection_cost = float('inf')
                
                for sample_idx in sample_indices:
                    sample_point = current_contour.points[sample_idx]
                    img_x, img_y = world_to_image(sample_point.x, sample_point.y)
                    
                    # Ensure coordinates are within image bounds
                    if 0 <= img_x < width and 0 <= img_y < height:
                        # Get distance to next level contour at this point
                        dist_to_next = next_dist_transform[img_y, img_x] * resolution
                        
                        # Only consider points that are close enough
                        max_distance = line_spacing * config.get('max_breakpoint_distance', 1.2)
                        if dist_to_next < max_distance:
                            # Find the closest contour in the next level
                            for j, next_contour in enumerate(next_contours):
                                if not next_contour.points or len(next_contour.points) < 2:
                                    continue
                                
                                # Find closest point on next contour
                                next_line = next_contour._line
                                if not next_line or next_line.is_empty:
                                    continue
                                    
                                # Project sample point onto next contour
                                next_point_coords = next_line.interpolate(next_line.project(sample_point))
                                next_point = ShapelyPoint(next_point_coords.x, next_point_coords.y)
                                
                                # Calculate direct distance
                                direct_dist = sample_point.distance(next_point)
                                
                                # Calculate connection cost (just distance for now)
                                connection_cost = direct_dist
                                
                                if connection_cost < min_connection_cost:
                                    min_connection_cost = connection_cost
                                    best_sample_idx = sample_idx
                                    best_next_contour_idx = j
                                    
                                    # Find the index of the closest point in the next contour
                                    best_next_point_idx = min(range(len(next_contour.points)), 
                                                            key=lambda k: next_contour.points[k].distance(next_point))
                
                # If we found a good connection, add it as a breakpoint
                if best_sample_idx != -1 and best_next_contour_idx != -1:
                    # Store breakpoint information
                    all_breakpoints.append((
                        current_level, i, best_sample_idx, 
                        next_level, best_next_contour_idx, best_next_point_idx
                    ))
                    
                    # Store breakpoint in contour objects for visualization
                    current_contour.breakpoints.append((
                        current_contour.points[best_sample_idx], 
                        next_contours[best_next_contour_idx].points[best_next_point_idx],
                        current_contour.points[best_sample_idx], 
                        next_contours[best_next_contour_idx].points[best_next_point_idx]
                    ))
                    
                    print(f"  Found additional breakpoint: Level {current_level}[{i}] to Level {next_level}[{best_next_contour_idx}]")
    
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
    
    # Create a wrapper object to hold both the sub_paths and skeleton data
    class SubPathsWithSkeletonData:
        def __init__(self, paths, skeleton_data):
            self.paths = paths
            self.skeleton_data = skeleton_data
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            return self.paths[idx]
        
        def __iter__(self):
            return iter(self.paths)
    
    # Return the wrapper object with both sub_paths and skeleton data
    return SubPathsWithSkeletonData(
        sub_paths,
        {
            'skeleton_img': skeleton_img,
            'junction_coords': junction_coords,
            'all_breakpoints': all_breakpoints,
            'contours_by_level': contours_by_level,
            'world_to_image': world_to_image
        }
    )


#-----------------------------------------------------------------------------
# 6. Generating continuous path (Connecting Sub-paths) - Linear Scan Method
#-----------------------------------------------------------------------------

def connect_sub_paths(sub_paths: List[SubPath]) -> List[ShapelyPoint]:
    """
    Connects sub-paths into a single continuous global toolpath using skeleton-based
    breakpoints to create an optimal path.

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
    else:
        sub_paths_list = sub_paths

    # Filter out empty paths
    remaining_sub_paths = [p for p in sub_paths_list if p.points and len(p.points) >= 2]
    if not remaining_sub_paths:
        return []

    # If we have skeleton data with breakpoints, use it to create a connection graph
    if skeleton_data and 'all_breakpoints' in skeleton_data and skeleton_data['all_breakpoints']:
        print("Using skeleton-based breakpoints for path connection")
        return connect_using_skeleton_breakpoints(remaining_sub_paths, skeleton_data)
    else:
        print("No skeleton breakpoints available, using distance-based connection")
        return connect_using_distance(remaining_sub_paths)

def connect_using_skeleton_breakpoints(sub_paths: List[SubPath], skeleton_data: dict) -> List[ShapelyPoint]:
    """
    Connects sub-paths using the breakpoints identified by the skeleton algorithm.
    
    Args:
        sub_paths: List of SubPath objects
        skeleton_data: Dictionary containing skeleton analysis data
        
    Returns:
        A list of Shapely Points representing the connected path
    """
    global_path: List[ShapelyPoint] = []
    
    # Extract breakpoint data
    all_breakpoints = skeleton_data.get('all_breakpoints', [])
    contours_by_level = skeleton_data.get('contours_by_level', {})
    
    if not all_breakpoints or not contours_by_level:
        print("Incomplete skeleton data, falling back to distance-based connection")
        return connect_using_distance(sub_paths)
    
    print(f"Using {len(all_breakpoints)} skeleton-based breakpoints for path connection")
    
    # Create a graph representation of the sub-paths and their connections
    # Each node is a sub-path, edges are the breakpoint connections
    from collections import defaultdict
    connection_graph = defaultdict(list)
    
    # Map sub-paths to their indices for easy lookup
    sub_path_map = {}
    for i, path in enumerate(sub_paths):
        if not path.points or len(path.points) < 2:
            continue
            
        # Use the first and last points as keys for quick identification
        # Round to reduce floating point precision issues
        start_key = (round(path.points[0].x, 6), round(path.points[0].y, 6))
        end_key = (round(path.points[-1].x, 6), round(path.points[-1].y, 6))
        sub_path_map[start_key] = (i, False)  # (index, needs_reverse)
        sub_path_map[end_key] = (i, True)     # (index, needs_reverse)
    
    # Build the connection graph using breakpoints
    connections_found = 0
    
    # Group breakpoints by level pairs for better organization
    breakpoints_by_level_pair = {}
    for bp in all_breakpoints:
        current_level, current_idx, current_point_idx, next_level, next_idx, next_point_idx = bp
        level_pair = (current_level, next_level)
        if level_pair not in breakpoints_by_level_pair:
            breakpoints_by_level_pair[level_pair] = []
        breakpoints_by_level_pair[level_pair].append(bp)
    
    # Process breakpoints level by level
    for level_pair, level_breakpoints in breakpoints_by_level_pair.items():
        print(f"Processing {len(level_breakpoints)} breakpoints between levels {level_pair[0]} and {level_pair[1]}")
        
        for bp in level_breakpoints:
            current_level, current_idx, current_point_idx, next_level, next_idx, next_point_idx = bp
            
            # Get the actual contours
            if (current_level in contours_by_level and current_idx < len(contours_by_level[current_level]) and
                next_level in contours_by_level and next_idx < len(contours_by_level[next_level])):
                
                current_contour = contours_by_level[current_level][current_idx]
                next_contour = contours_by_level[next_level][next_idx]
                
                if (current_point_idx < len(current_contour.points) and
                    next_point_idx < len(next_contour.points)):
                    
                    # Get the actual points
                    p1 = current_contour.points[current_point_idx]
                    p2 = next_contour.points[next_point_idx]
                    
                    # Use a larger tolerance for endpoint matching to improve connectivity
                    endpoint_tolerance = POINT_EQUALITY_TOLERANCE * 10
                    
                    # Find the sub-paths that contain these points
                    # Round to reduce floating point precision issues
                    p1_key = (round(p1.x, 6), round(p1.y, 6))
                    p2_key = (round(p2.x, 6), round(p2.y, 6))
                    
                    # Find paths containing p1
                    p1_paths = []
                    for key, (path_idx, needs_reverse) in sub_path_map.items():
                        if (abs(key[0] - p1_key[0]) < endpoint_tolerance and 
                            abs(key[1] - p1_key[1]) < endpoint_tolerance):
                            p1_paths.append((path_idx, needs_reverse))
                    
                    # Find paths containing p2
                    p2_paths = []
                    for key, (path_idx, needs_reverse) in sub_path_map.items():
                        if (abs(key[0] - p2_key[0]) < endpoint_tolerance and 
                            abs(key[1] - p2_key[1]) < endpoint_tolerance):
                            p2_paths.append((path_idx, needs_reverse))
                    
                    # Connect all paths containing p1 to all paths containing p2
                    for path_idx1, _ in p1_paths:
                        for path_idx2, _ in p2_paths:
                            if path_idx1 != path_idx2:
                                # Calculate actual distance between path endpoints
                                if path_idx1 < len(sub_paths) and path_idx2 < len(sub_paths):
                                    path1 = sub_paths[path_idx1]
                                    path2 = sub_paths[path_idx2]
                                    
                                    if (path1.points and len(path1.points) >= 2 and 
                                        path2.points and len(path2.points) >= 2):
                                        
                                        # Calculate distances between all possible endpoint combinations
                                        distances = [
                                            (path1.points[0].distance(path2.points[0]), False, False),
                                            (path1.points[0].distance(path2.points[-1]), False, True),
                                            (path1.points[-1].distance(path2.points[0]), True, False),
                                            (path1.points[-1].distance(path2.points[-1]), True, True)
                                        ]
                                        
                                        # Use the minimum distance
                                        min_dist, _, _ = min(distances, key=lambda x: x[0])
                                        
                                        # Add to connection graph with actual endpoint distance
                                        connection_graph[path_idx1].append((path_idx2, min_dist))
                                        connection_graph[path_idx2].append((path_idx1, min_dist))
                                        connections_found += 1
    
    # If we couldn't build a good connection graph from breakpoints, fall back
    if not connection_graph:
        print("Could not build connection graph from breakpoints, falling back")
        return connect_using_distance(sub_paths)
    
    print(f"Built connection graph with {connections_found} connections between {len(connection_graph)} paths")
    
    # Find a good starting path using a more sophisticated approach
    # Prefer paths that:
    # 1. Are on the outermost level (level 0)
    # 2. Have fewer connections (endpoints)
    # 3. Are longer (more significant)
    
    best_score = float('inf')
    start_path_idx = 0
    
    for idx, path in enumerate(sub_paths):
        if not path.points or len(path.points) < 2:
            continue
            
        # Skip paths that aren't in the connection graph
        if idx not in connection_graph:
            continue
            
        # Calculate a score based on our criteria
        connections_count = len(connection_graph[idx])
        
        # Find which contour this path belongs to
        path_level = -1
        for level, contours in contours_by_level.items():
            for contour_idx, contour in enumerate(contours):
                # Check if this path's endpoints match the contour
                for point in [path.points[0], path.points[-1]]:
                    for contour_point in contour.points:
                        if point.distance(contour_point) < POINT_EQUALITY_TOLERANCE * 10:
                            path_level = level
                            break
                    if path_level != -1:
                        break
                if path_level != -1:
                    break
            if path_level != -1:
                break
        
        # Calculate path length
        path_length = 0
        for i in range(len(path.points) - 1):
            path_length += path.points[i].distance(path.points[i+1])
        
        # Score: prefer outer levels, fewer connections, and longer paths
        level_factor = path_level + 1 if path_level != -1 else 10
        connection_factor = connections_count if connections_count > 0 else 10
        length_factor = 1.0 / max(path_length, 0.1)  # Invert so longer paths have lower scores
        
        score = level_factor * connection_factor * length_factor
        
        if score < best_score:
            best_score = score
            start_path_idx = idx
    
    # Start with the selected path
    visited = set()
    current_path_idx = start_path_idx
    global_path.extend(sub_paths[current_path_idx].points)
    visited.add(current_path_idx)
    
    # Connect paths using the graph
    max_iterations = len(sub_paths) * 2  # Safety limit
    iteration = 0
    
    while len(visited) < len(sub_paths) and iteration < max_iterations:
        iteration += 1
        
        # Find the next best path to connect to
        best_next_idx = -1
        min_cost = float('inf')
        reverse_needed = False
        
        current_end_point = global_path[-1]
        
        # Check all neighbors in the connection graph
        for neighbor_idx, distance in connection_graph.get(current_path_idx, []):
            if neighbor_idx in visited:
                continue
                
            if neighbor_idx >= len(sub_paths):
                print(f"Warning: Invalid neighbor index {neighbor_idx}, max is {len(sub_paths)-1}")
                continue
                
            neighbor_path = sub_paths[neighbor_idx]
            
            if not neighbor_path.points or len(neighbor_path.points) < 2:
                continue
            
            # Check both ends of the neighbor path
            dist_to_start = current_end_point.distance(neighbor_path.points[0])
            dist_to_end = current_end_point.distance(neighbor_path.points[-1])
            
            # Use a weighted cost that considers both the graph distance and actual endpoint distance
            start_cost = dist_to_start * 0.8 + distance * 0.2
            end_cost = dist_to_end * 0.8 + distance * 0.2
            
            if start_cost < min_cost:
                min_cost = start_cost
                best_next_idx = neighbor_idx
                reverse_needed = False
                
            if end_cost < min_cost:
                min_cost = end_cost
                best_next_idx = neighbor_idx
                reverse_needed = True
        
        # If we found a connected path, add it
        if best_next_idx != -1:
            next_path = sub_paths[best_next_idx]
            points_to_add = next_path.points
            
            # Check if the connection distance is reasonable
            connection_dist = min(current_end_point.distance(points_to_add[0]), 
                                 current_end_point.distance(points_to_add[-1]))
            
            if connection_dist > config.get('max_connection_distance', 2.0) * config.get('toolpath_width', 0.4):
                print(f"Warning: Long connection distance ({connection_dist:.2f}mm) between paths")
            
            if reverse_needed:
                # Add reversed points, skip the duplicate endpoint
                global_path.extend(reversed(points_to_add[:-1]))
            else:
                # Add points, skip the duplicate start point
                global_path.extend(points_to_add[1:])
                
            current_path_idx = best_next_idx
            visited.add(current_path_idx)
        else:
            # Try to find any unvisited path with the shortest distance
            best_fallback_idx = -1
            min_fallback_dist = float('inf')
            fallback_reverse = False
            
            for i in range(len(sub_paths)):
                if i in visited or not sub_paths[i].points or len(sub_paths[i].points) < 2:
                    continue
                    
                dist_to_start = current_end_point.distance(sub_paths[i].points[0])
                dist_to_end = current_end_point.distance(sub_paths[i].points[-1])
                
                if dist_to_start < min_fallback_dist:
                    min_fallback_dist = dist_to_start
                    best_fallback_idx = i
                    fallback_reverse = False
                    
                if dist_to_end < min_fallback_dist:
                    min_fallback_dist = dist_to_end
                    best_fallback_idx = i
                    fallback_reverse = True
            
            if best_fallback_idx != -1:
                print(f"No connected path in graph, jumping to nearest path (dist: {min_fallback_dist:.2f}mm)")
                next_path = sub_paths[best_fallback_idx]
                points_to_add = next_path.points
                
                if fallback_reverse:
                    global_path.extend(reversed(points_to_add[:-1]))
                else:
                    global_path.extend(points_to_add[1:])
                    
                current_path_idx = best_fallback_idx
                visited.add(current_path_idx)
            else:
                # No more paths to connect
                print(f"No more paths to connect, visited {len(visited)} of {len(sub_paths)} paths")
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
    #kstl_file_path = os.path.join(project_root, "models", "mine", "hex-with-hex-hole.stl")
    #stl_file_path = os.path.join(project_root, "models", "mine", "gear.stl")
    stl_file_path = os.path.join("models", "wrench.stl")
    #stl_file_path = os.path.join("models", "cuboid-with-holes.stl")
    #stl_file_path = os.path.join("models", "u-shape.stl")
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
    # Extract the actual sub_paths from the wrapper object if needed
    sub_paths_list = sub_paths.paths if hasattr(sub_paths, 'paths') else sub_paths
    final_toolpath = connect_sub_paths(sub_paths_list) # Use the existing linear scan connector
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
         
    # --- 7b. Optional: Visualize Skeleton ---
    # Only attempt to visualize if the skeleton data was returned from the function
    if config.get('visualize_skeleton', True) and hasattr(sub_paths, 'skeleton_data') and sub_paths.skeleton_data:
        try:
            import matplotlib.pyplot as plt
            
            # Unpack the skeleton data
            skeleton_img = sub_paths.skeleton_data.get('skeleton_img')
            junction_coords = sub_paths.skeleton_data.get('junction_coords', [])
            all_breakpoints = sub_paths.skeleton_data.get('all_breakpoints', [])
            contours_by_level = sub_paths.skeleton_data.get('contours_by_level', {})
            world_to_image = sub_paths.skeleton_data.get('world_to_image')
            
            if skeleton_img is not None and world_to_image is not None:
                plt.figure(figsize=(10, 10))
                plt.imshow(skeleton_img, cmap='gray')
                
                # Plot junction points
                if junction_coords:
                    junction_x = [x for x, y in junction_coords]
                    junction_y = [y for x, y in junction_coords]
                    plt.scatter(junction_x, junction_y, c='red', s=30, marker='o')
                
                # Plot breakpoints
                if all_breakpoints and contours_by_level:
                    for bp in all_breakpoints:
                        current_level, current_idx, current_point_idx, next_level, next_idx, next_point_idx = bp
                        
                        if (current_level in contours_by_level and 
                            current_idx < len(contours_by_level[current_level]) and
                            next_level in contours_by_level and
                            next_idx < len(contours_by_level[next_level])):
                            
                            current_contour = contours_by_level[current_level][current_idx]
                            next_contour = contours_by_level[next_level][next_idx]
                            
                            if (current_point_idx < len(current_contour.points) and
                                next_point_idx < len(next_contour.points)):
                                
                                current_point = current_contour.points[current_point_idx]
                                next_point = next_contour.points[next_point_idx]
                                
                                current_img_x, current_img_y = world_to_image(current_point.x, current_point.y)
                                next_img_x, next_img_y = world_to_image(next_point.x, next_point.y)
                                
                                plt.plot([current_img_x, next_img_x], [current_img_y, next_img_y], 'g-', linewidth=2)
                                plt.scatter([current_img_x, next_img_x], [current_img_y, next_img_y], c='blue', s=30, marker='x')
                
                plt.title(f"Skeleton with Junction Points and Breakpoints - Layer {layer_to_process_idx}")
                plt.axis('equal')
                plt.tight_layout()
                plt.show()
            else:
                print("Skeleton image data not available for visualization")
        except Exception as e:
            print(f"Error visualizing skeleton: {e}")

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

