"""
Shared utility functions for geometry, zone mapping, and safe operations.
"""
import math
import numpy as np


def pdist(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def box_center(b):
    """Center point of a bounding box [x1, y1, x2, y2]."""
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def box_bottom(b):
    """Bottom-center point of a bounding box."""
    return ((b[0] + b[2]) / 2, b[3])


def box_height(b):
    """Height of a bounding box."""
    return b[3] - b[1]


def box_width(b):
    """Width of a bounding box."""
    return b[2] - b[0]


def get_tactical_zone(x, y, is_team1, W, H):
    """
    Maps global video x,y to standard volleyball tactical zones 1-6.
    
    Zone layout (from each team's perspective):
      4 | 3 | 2   (front row — near net)
      5 | 6 | 1   (back row — near baseline)
    """
    net_x = W / 2
    
    # Broadcast cameras usually show court from middle to bottom
    court_top = H * 0.4
    court_h = H - court_top
    
    if y < court_top:
        lane = 0
    else:
        rel_y = (y - court_top) / max(1, court_h)
        lane = int(rel_y * 3)
        
    lane = max(0, min(2, lane))

    if is_team1:
        # Front row is within 15% of screen width from the net
        depth = 0 if x > (net_x - W * 0.15) else 1
        if depth == 0:
            return [4, 3, 2][lane]
        else:
            return [5, 6, 1][lane]
    else:
        depth = 0 if x < (net_x + W * 0.15) else 1
        if depth == 0:
            return [2, 3, 4][lane]
        else:
            return [1, 6, 5][lane]


def safe_div(a, b, default=0.0):
    """Safe division that returns default if divisor is zero."""
    return a / b if b != 0 else default


def clamp(val, lo, hi):
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, val))


def format_time(frames, fps):
    """Convert frame count to MM:SS string."""
    if fps <= 0:
        fps = 30
    total_s = int(frames / fps)
    mins = total_s // 60
    secs = total_s % 60
    return f"{mins:02d}:{secs:02d}"


def moving_average(data, window=5):
    """Simple moving average for smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid').tolist()
