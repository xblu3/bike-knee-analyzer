"""
Angle calculation utilities for knee bend analysis.
"""

import numpy as np
from typing import Tuple


def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    """
    Calculate the angle at point_b formed by points a-b-c.
    
    Uses the dot product of vectors BA and BC to compute the angle at B.
    
    Args:
        point_a: First point (hip) as [x, y]
        point_b: Vertex point (knee) as [x, y]
        point_c: Third point (ankle) as [x, y]
    
    Returns:
        Angle in degrees (0-180)
    """
    # Vector from B to A
    vec_ba = point_a - point_b
    # Vector from B to C
    vec_bc = point_c - point_b
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vec_ba, vec_bc)
    magnitude_ba = np.linalg.norm(vec_ba)
    magnitude_bc = np.linalg.norm(vec_bc)
    
    # Avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    # Clamp to [-1, 1] to avoid numerical errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_knee_angle(hip: Tuple[float, float], 
                        knee: Tuple[float, float], 
                        ankle: Tuple[float, float]) -> float:
    """
    Calculate the knee bend angle.
    
    Args:
        hip: (x, y) coordinates of hip
        knee: (x, y) coordinates of knee
        ankle: (x, y) coordinates of ankle
    
    Returns:
        Knee bend angle in degrees
    """
    hip_arr = np.array(hip)
    knee_arr = np.array(knee)
    ankle_arr = np.array(ankle)
    
    return calculate_angle(hip_arr, knee_arr, ankle_arr)
