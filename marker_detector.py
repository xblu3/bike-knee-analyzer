"""
Color marker detection and tracking for knee angle analysis.
Supports per-marker colors and robust HSV detection including red (hue wraps).
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


# HSV ranges for supported colors.
# Red wraps around 0/180 in HSV so it gets two ranges.
COLOR_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    'red': [
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([170, 120, 70]),  np.array([180, 255, 255])),
    ],
    'orange': [
        (np.array([10, 120, 70]),   np.array([25,  255, 255])),
    ],
    'yellow': [
        (np.array([25, 120, 70]),   np.array([35,  255, 255])),
    ],
    'green': [
        (np.array([35, 80,  50]),   np.array([85,  255, 255])),
    ],
    'blue': [
        (np.array([100, 80, 50]),   np.array([130, 255, 255])),
    ],
    'purple': [
        (np.array([130, 60, 50]),   np.array([160, 255, 255])),
    ],
    'white': [
        (np.array([0,   0,  200]),  np.array([180, 40,  255])),
    ],
    'pink': [
        (np.array([160, 60, 100]),  np.array([175, 255, 255])),
    ],
}

SUPPORTED_COLORS = list(COLOR_RANGES.keys())

# BGR colors used when drawing each named marker
MARKER_DRAW_COLORS = {
    'hip':   (255, 100,   0),   # blue-ish
    'knee':  (  0, 255,   0),   # green
    'ankle': (  0,  80, 255),   # red-ish
}


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Return a binary mask for a given color name."""
    color = color.lower()
    if color not in COLOR_RANGES:
        raise ValueError(f"Unsupported color '{color}'. Choose from: {SUPPORTED_COLORS}")
    mask = None
    for lower, upper in COLOR_RANGES[color]:
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def _largest_blob_center(mask: np.ndarray,
                          min_area: int = 40,
                          max_area: int = 8000) -> Optional[Tuple[int, int]]:
    """Return (cx, cy) of the largest blob in mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area and area > best_area:
            M = cv2.moments(c)
            if M['m00'] != 0:
                best = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                best_area = area
    return best


class ColorMarkerDetector:
    """
    Detect and track three colored markers (hip, knee, ankle).

    Markers can all share one color, or each can have its own distinct color.
    When a single color is used the three blobs are separated by vertical
    position (top → hip, middle → knee, bottom → ankle).
    When per-marker colors are provided each marker is detected independently.
    """

    def __init__(self,
                 marker_color: Optional[str] = None,
                 hip_color:    Optional[str] = None,
                 knee_color:   Optional[str] = None,
                 ankle_color:  Optional[str] = None):
        """
        Args:
            marker_color: Single color for all three markers.
                          Ignored if any per-marker color is given.
            hip_color:    Color of the hip marker.
            knee_color:   Color of the knee marker.
            ankle_color:  Color of the ankle marker.

        At least one of marker_color or all three per-marker colors must be set.
        """
        if hip_color and knee_color and ankle_color:
            self.per_marker_colors = {
                'hip':   hip_color.lower(),
                'knee':  knee_color.lower(),
                'ankle': ankle_color.lower(),
            }
            self.single_color = None
        elif marker_color:
            self.single_color = marker_color.lower()
            self.per_marker_colors = None
        else:
            raise ValueError(
                "Provide either --marker-color OR all three of "
                "--hip-color / --knee-color / --ankle-color"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_marker_positions(
        self, frame: np.ndarray
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Detect hip, knee, ankle markers in frame.

        Returns dict {'hip': (x,y), 'knee': (x,y), 'ankle': (x,y)}
        or None if any marker could not be found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.per_marker_colors:
            return self._detect_per_color(hsv)
        else:
            return self._detect_single_color(hsv)

    def draw_markers(
        self,
        frame: np.ndarray,
        positions: Dict[str, Tuple[int, int]],
    ) -> np.ndarray:
        """Overlay labeled circles on detected marker positions."""
        out = frame.copy()
        for joint, (x, y) in positions.items():
            color = MARKER_DRAW_COLORS.get(joint, (255, 255, 255))
            cv2.circle(out, (x, y), 10, color, -1)
            cv2.circle(out, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(out, joint, (x + 14, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_per_color(
        self, hsv: np.ndarray
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Each marker has its own color — detect independently."""
        positions = {}
        for joint, color in self.per_marker_colors.items():
            mask   = _color_mask(hsv, color)
            center = _largest_blob_center(mask)
            if center is None:
                return None          # marker missing — bail out
            positions[joint] = center
        return positions

    def _detect_single_color(
        self, hsv: np.ndarray
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        All markers share one color.
        Find up to 3 blobs and assign top→hip, middle→knee, bottom→ankle.
        """
        mask     = _color_mask(hsv, self.single_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if 40 < area < 8000:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    blobs.append((cx, cy, area))

        if len(blobs) < 3:
            return None

        # Keep 3 largest, sort top → bottom
        blobs.sort(key=lambda b: b[2], reverse=True)
        top3 = sorted(blobs[:3], key=lambda b: b[1])   # sort by y

        return {
            'hip':   (top3[0][0], top3[0][1]),
            'knee':  (top3[1][0], top3[1][1]),
            'ankle': (top3[2][0], top3[2][1]),
        }
