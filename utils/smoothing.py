from typing import List, Tuple, Optional
import numpy as np


class LandmarkSmoother:
    """
    Temporal landmark smoother using Exponential Moving Average (EMA).
    Eliminates tracking jitter and micro-shaking on finger joints.
    """

    def __init__(self, alpha: float = 0.40):
        self.alpha = alpha
        self.prev_landmarks: Optional[np.ndarray] = None

    def update(self, current_landmarks: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
        """
        Smooth a list of (x, y, z) landmarks and return pixel coordinates (x, y).
        """
        curr_arr = np.array(current_landmarks, dtype=np.float32)

        if self.prev_landmarks is None or self.prev_landmarks.shape != curr_arr.shape:
            self.prev_landmarks = curr_arr
        else:
            # Velocity-aware EMA: if movement is very large (snap gesture), alpha increases to prevent drag
            diff = np.linalg.norm(curr_arr[:, :2] - self.prev_landmarks[:, :2], axis=1)
            mean_vel = np.mean(diff)
            effective_alpha = min(0.85, self.alpha + mean_vel * 0.01)
            self.prev_landmarks = effective_alpha * curr_arr + (1.0 - effective_alpha) * self.prev_landmarks

        smoothed_xy = [(int(pt[0]), int(pt[1])) for pt in self.prev_landmarks]
        return smoothed_xy

    def reset(self):
        self.prev_landmarks = None
