import time
from typing import Tuple
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

    def draw_telemetry(self, frame: np.ndarray, position: Tuple[int, int] = (14, 26)) -> np.ndarray:
        """Draw sleek performance telemetry badge on frame."""
        text = f"FPS: {self.fps:.1f} | Track: {self.track_ms:.1f}ms | Render: {self.render_ms:.1f}ms"
        x, y = position

        # Subtle dark pill background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 6, y - th - 6), (x + tw + 6, y + 6), (10, 8, 14), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Telemetry text
        color = (80, 240, 120) if self.fps >= 28.0 else ((80, 200, 255) if self.fps >= 18.0 else (80, 80, 255))
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return frame
