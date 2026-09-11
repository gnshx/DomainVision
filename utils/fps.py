import time
from typing import Tuple, Optional
import cv2
import numpy as np


class PerformanceProfiler:
    """
    Real-time performance latency and FPS telemetry monitor.
    Measures tracking latency, rendering latency, and total frame rate.
    """

    def __init__(self, smoothing_window: int = 15):
        self.window = smoothing_window
        self.t_start = time.time()
        self.t_track_start = 0.0
        self.track_ms = 0.0
        self.render_ms = 0.0
        self.fps = 30.0
        self.frame_times = []
        self.track_times = []
        self.render_times = []
        self.last_frame_t = time.time()

    def start_frame(self):
        self.t_start = time.time()

    def start_tracking(self):
        self.t_track_start = time.time()

    def end_tracking(self):
        now = time.time()
        dur_ms = (now - self.t_track_start) * 1000.0
        self.track_times.append(dur_ms)
        if len(self.track_times) > self.window:
            self.track_times.pop(0)
        self.track_ms = float(np.mean(self.track_times))

    def end_rendering(self):
        now = time.time()
        dur_ms = (now - self.t_start) * 1000.0 - self.track_ms
        self.render_times.append(max(0.1, dur_ms))
        if len(self.render_times) > self.window:
            self.render_times.pop(0)
        self.render_ms = float(np.mean(self.render_times))

        # Total FPS
        dt = now - self.last_frame_t
        self.last_frame_t = now
        if dt > 1e-4:
            inst_fps = 1.0 / dt
            self.frame_times.append(inst_fps)
            if len(self.frame_times) > self.window:
                self.frame_times.pop(0)
            self.fps = float(np.mean(self.frame_times))

    def draw_telemetry(self, frame: np.ndarray, position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Draw sleek, high-contrast performance telemetry badge that is never cropped."""
        h, w = frame.shape[:2]
        text = f"FPS: {self.fps:.1f} | TRACK: {self.track_ms:.1f}ms | RENDER: {self.render_ms:.1f}ms"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.46
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        if position is None:
            # Safely anchor to top right with 20px padding from the right edge
            x = w - tw - 24
            y = 34
        else:
            x, y = position
            # Ensure text doesn't overflow right edge
            if x + tw + 12 > w:
                x = w - tw - 24

        # High-contrast solid dark container (ROI blending - avoids full 1280x720 frame clone)
        pad_x = 10
        pad_y = 7
        bx1, by1 = max(0, x - pad_x), max(0, y - th - pad_y)
        bx2, by2 = min(w, x + tw + pad_x), min(h, y + pad_y)
        roi = frame[by1:by2, bx1:bx2]
        dark_box = np.full_like(roi, (12, 8, 18))
        cv2.addWeighted(dark_box, 0.90, roi, 0.10, 0, roi)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 230, 255), 1)

        # Drop shadow + vibrant neon text
        color = (80, 255, 120) if self.fps >= 25.0 else ((80, 230, 255) if self.fps >= 15.0 else (80, 80, 255))
        cv2.putText(frame, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return frame

