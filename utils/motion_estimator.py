"""
Lightweight camera motion estimator for DomainVision parallax system.

Uses sparse optical flow on a downscaled (160x90) frame to estimate global
camera translation (dx, dy) per frame. Total budget: < 1.5ms on CPU.

Algorithm:
  1. Detect feature points on previous downscaled frame (every K frames)
  2. Lucas-Kanade optical flow to track them to current frame
  3. Median of per-point displacements = global camera translation
  4. EMA smoothing to remove jitter

Outputs (dx, dy) in pixels at full resolution (scaled up from tracking resolution).
"""
from typing import Tuple, Optional
import cv2
import numpy as np


class CameraMotionEstimator:
    """
    Lightweight sparse optical flow camera translation estimator.

    Usage:
        estimator = CameraMotionEstimator()
        dx, dy = estimator.estimate(frame_bgr)  # call each frame
        domain_layers.set_camera_offset(dx, dy)
    """

    # Downscale resolution for tracking (small = fast)
    TRACK_W = 160
    TRACK_H = 90

    def __init__(
        self,
        max_features: int = 60,
        redetect_interval: int = 8,
        smooth_alpha: float = 0.35,
        max_shift: float = 15.0,  # clamp outlier translations to this many pixels
    ):
        self.max_features = max_features
        self.redetect_interval = redetect_interval
        self.smooth_alpha = smooth_alpha
        self.max_shift = max_shift

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._frame_counter = 0

        # Smoothed translation in full-res pixels
        self._smooth_dx: float = 0.0
        self._smooth_dy: float = 0.0

        # Scale factor from track resolution to full frame
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

        # LK flow parameters
        self._lk_params = dict(
            winSize=(11, 11),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        # Feature detection parameters
        self._feat_params = dict(
            maxCorners=self.max_features,
            qualityLevel=0.04,
            minDistance=8,
            blockSize=7,
        )

    def _downscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self._scale_x = w / self.TRACK_W
        self._scale_y = h / self.TRACK_H
        small = cv2.resize(frame_bgr, (self.TRACK_W, self.TRACK_H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def estimate(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Estimate camera translation from current frame.

        Returns:
            (dx, dy) in full-resolution pixels, EMA-smoothed.
        """
        self._frame_counter += 1
        if self._frame_counter % 2 != 0 and self._prev_pts is not None:
            return self._smooth_dx, self._smooth_dy

        gray = self._downscale(frame_bgr)

        raw_dx, raw_dy = 0.0, 0.0

        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts) > 3:
            # Track features to new frame
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, **self._lk_params
            )

            if curr_pts is not None and status is not None:
                good_prev = self._prev_pts[status.ravel() == 1]
                good_curr = curr_pts[status.ravel() == 1]

                if len(good_curr) > 2:
                    good_prev_2d = good_prev.reshape(-1, 2)
                    good_curr_2d = good_curr.reshape(-1, 2)
                    # Per-point displacement
                    displacements = good_curr_2d - good_prev_2d
                    # Median displacement = global camera translation (robust to local motion)
                    raw_dx_small = float(np.median(displacements[:, 0]))
                    raw_dy_small = float(np.median(displacements[:, 1]))

                    # Scale up to full resolution
                    raw_dx = raw_dx_small * self._scale_x
                    raw_dy = raw_dy_small * self._scale_y

                    # Clamp extreme outliers (sudden large motions cause artifacts)
                    raw_dx = float(np.clip(raw_dx, -self.max_shift, self.max_shift))
                    raw_dy = float(np.clip(raw_dy, -self.max_shift, self.max_shift))

                    # Update tracked points to current good points
                    self._prev_pts = good_curr.reshape(-1, 1, 2)

        # EMA smoothing
        self._smooth_dx = (1.0 - self.smooth_alpha) * self._smooth_dx + self.smooth_alpha * raw_dx
        self._smooth_dy = (1.0 - self.smooth_alpha) * self._smooth_dy + self.smooth_alpha * raw_dy

        # Re-detect feature points periodically or when count drops
        need_redetect = (
            self._frame_counter % self.redetect_interval == 0
            or self._prev_pts is None
            or len(self._prev_pts) < self.max_features // 3
        )
        if need_redetect:
            pts = cv2.goodFeaturesToTrack(gray, **self._feat_params)
            self._prev_pts = pts

        self._prev_gray = gray
        return self._smooth_dx, self._smooth_dy

    def reset(self):
        """Reset estimator state (e.g. when domain collapses)."""
        self._prev_gray = None
        self._prev_pts = None
        self._smooth_dx = 0.0
        self._smooth_dy = 0.0
        self._frame_counter = 0
