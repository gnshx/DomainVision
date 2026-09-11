import math
from typing import List, Tuple, Optional
import cv2
import numpy as np


class BarrierShockwaveEffect:
    """
    Renders visible glowing expanding circular energy shockwaves & barrier crests.
    """

    def __init__(self):
        self.rings: List[dict] = []

    def trigger(
        self,
        center: Tuple[int, int],
        primary_color: Tuple[int, int, int] = (255, 60, 180),
        secondary_color: Tuple[int, int, int] = (255, 220, 80),
        duration_frames: int = 25,
        max_radius: float = 650.0,
    ):
        """Spawn expanding shockwave ring set."""
        # Ring 1: Primary fast crest
        self.rings.append({
            "center": center,
            "color": primary_color,
            "max_radius": max_radius,
            "duration": duration_frames,
            "thickness": 8,
            "frame": 0,
            "delay": 0,
        })
        # Ring 2: Secondary core crest
        self.rings.append({
            "center": center,
            "color": secondary_color,
            "max_radius": max_radius * 0.85,
            "duration": duration_frames,
            "thickness": 5,
            "frame": 0,
            "delay": 3,
        })
        # Ring 3: White high-energy razor ring
        self.rings.append({
            "center": center,
            "color": (255, 255, 255),
            "max_radius": max_radius * 0.95,
            "duration": duration_frames - 2,
            "thickness": 2,
            "frame": 0,
            "delay": 1,
        })

    def update_and_render(self, frame: np.ndarray) -> np.ndarray:
        """Render active expanding shockwave rings with bloom."""
        if not self.rings:
            return frame

        h, w = frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        alive_rings = []

        for ring in self.rings:
            if ring["delay"] > 0:
                ring["delay"] -= 1
                alive_rings.append(ring)
                continue

            prog = ring["frame"] / float(ring["duration"])
            ring["frame"] += 1

            if prog < 1.0:
                alive_rings.append(ring)
                # Expanding radius with ease-out cubic
                ease = 1.0 - (1.0 - prog) ** 3
                cur_r = int(ring["max_radius"] * ease)
                # Alpha fades as ring expands
                alpha = 1.0 - prog
                col = (
                    int(ring["color"][0] * alpha),
                    int(ring["color"][1] * alpha),
                    int(ring["color"][2] * alpha),
                )
                thick = max(1, int(ring["thickness"] * alpha))
                cv2.circle(canvas, ring["center"], cur_r, col, thick)

        self.rings = alive_rings

        if np.count_nonzero(canvas) > 0:
            blurred = cv2.GaussianBlur(canvas, (19, 19), 0)
            out = cv2.add(frame, blurred)
            out = cv2.add(out, canvas)
            return out

        return frame
