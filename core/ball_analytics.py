"""
Ball Analytics — tracking, speed, trail, trajectory prediction.
"""
import cv2
import numpy as np
from collections import deque
from core.utils import pdist
from core.config import MAX_TRAIL, BALL_COAST_FRAMES


class BallTracker:
    """Tracks ball position, speed, trail, and side-of-court."""

    def __init__(self, fps=30):
        self.fps = fps if fps > 0 else 30
        self.trail = deque(maxlen=MAX_TRAIL)
        self.prev_pos = None
        self.speed_px_frame = 0.0
        self.speed_px_sec = 0.0
        self.speeds_history = []
        self.no_ball_frames = 0
        self.current_pos = None
        self.current_box = None
        self.side = None
        
        # ── Kalman Filter Setup ──
        # State: [x, y, dx, dy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf_initialized = False
        self.gravity = 0.8  # Pixels per frame squared proxy

    def update(self, ball_box, ball_center, net_x):
        """Update ball state with new detection (or Kalman prediction if missing)."""
        
        # 1. Kalman Prediction
        if self.kf_initialized:
            self.kf.predict()
            self.kf.statePre[3] += self.gravity  # Apply gravity to dy

        # 2. Handle missing detection
        is_predicted = False
        if ball_center is None:
            self.no_ball_frames += 1
            if self.no_ball_frames <= BALL_COAST_FRAMES and self.kf_initialized:
                # Coast using Kalman prediction
                px, py = float(self.kf.statePre[0]), float(self.kf.statePre[1])
                ball_center = (int(px), int(py))
                is_predicted = True
                
                if self.current_box is not None:
                    w = self.current_box[2] - self.current_box[0]
                    h = self.current_box[3] - self.current_box[1]
                    ball_box = [int(px - w/2), int(py - h/2), int(px + w/2), int(py + h/2)]
                else:
                    ball_box = [int(px-10), int(py-10), int(px+10), int(py+10)]
        else:
            # Real detection received
            real_center = ball_center
            self.no_ball_frames = 0
            
            # Correct Kalman Filter
            meas = np.array([[np.float32(real_center[0])], [np.float32(real_center[1])]])
            if not self.kf_initialized:
                self.kf.statePost = np.array([real_center[0], real_center[1], 0, 0], np.float32)
                self.kf_initialized = True
            else:
                self.kf.correct(meas)

        # 3. Finalize State
        if ball_center is not None:
            self.current_pos = ball_center
            self.current_box = ball_box
            self.trail.append(ball_center)

            if self.prev_pos is not None:
                self.speed_px_frame = pdist(self.prev_pos, ball_center)
                self.speed_px_sec = self.speed_px_frame * self.fps
            else:
                self.speed_px_frame = 0.0
                self.speed_px_sec = 0.0
            
            if not is_predicted:
                self.speeds_history.append(self.speed_px_sec)

            self.prev_pos = ball_center
            self.side = 'L' if ball_center[0] < net_x else 'R'
        else:
            self.current_pos = None
            self.current_box = None
            self.speed_px_frame = 0.0
            self.prev_pos = None

    @property
    def is_detected(self):
        return self.current_pos is not None

    @property
    def is_missing_long(self):
        from core.config import BALL_MISSING_END
        return self.no_ball_frames > BALL_MISSING_END

    def get_trail_points(self):
        return list(self.trail)

    def get_trajectory_prediction(self, steps=5):
        if not self.kf_initialized:
            return []
        
        temp_state = self.kf.statePost.copy()
        pred_pts = []
        for i in range(1, steps + 1):
            # Simple linear + gravity prediction for the UI visualization
            temp_state[0] += temp_state[2]
            temp_state[1] += temp_state[3]
            temp_state[3] += self.gravity
            pred_pts.append((int(temp_state[0]), int(temp_state[1])))
        return pred_pts

    def get_avg_speed(self):
        if not self.speeds_history: return 0.0
        return sum(self.speeds_history) / len(self.speeds_history)

    def get_max_speed(self):
        if not self.speeds_history: return 0.0
        return max(self.speeds_history)

    def reset(self):
        self.trail.clear()
        self.prev_pos = None
        self.speed_px_frame = 0.0
        self.speed_px_sec = 0.0
        self.speeds_history.clear()
        self.no_ball_frames = 0
        self.kf_initialized = False
        self.current_pos = None
        self.current_box = None
        self.side = None
