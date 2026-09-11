import math
from typing import List, Tuple, Optional
import cv2
import numpy as np


class BarrierShockwaveEffect:
    """
    Renders visible glowing expanding circular energy shockwaves & barrier crests.

    v2 Upgrades:
    - Chromatic fringe: R/G/B ring layers offset by ±2-3px at wave boundary (prismatic effect)
    - Bright core ring: saturated narrow inner ring at current expansion radius
    - Soft glow falloff: wider downscaled bloom, additively composited
    - Improved cubic ease-out progression
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
            "thickness": 10,
            "frame": 0,
            "delay": 0,
            "chromatic": True,   # Enable chromatic fringe on this ring
        })
        # Ring 2: Secondary inner crest
        self.rings.append({
            "center": center,
            "color": secondary_color,
            "max_radius": max_radius * 0.82,
            "duration": duration_frames,
            "thickness": 5,
            "frame": 0,
            "delay": 3,
            "chromatic": False,
        })
        # Ring 3: White razor high-energy ring (bright core, narrow)
        self.rings.append({
            "center": center,
            "color": (255, 255, 255),
            "max_radius": max_radius * 0.92,
            "duration": duration_frames - 3,
            "thickness": 3,
            "frame": 0,
            "delay": 1,
            "chromatic": True,
        })

    def update_and_render(self, frame: np.ndarray) -> np.ndarray:
        """Render active expanding shockwave rings with chromatic fringe and bloom."""
        if not self.rings:
            return frame

        h, w = frame.shape[:2]
        # Bloom at 320x180 for fast convolution
        bw, bh = min(320, w // 4), min(180, h // 4)
        bloom_canvas = np.zeros((bh, bw, 3), dtype=np.uint8)
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
                # Cubic ease-out for smooth deceleration
                ease = 1.0 - (1.0 - prog) ** 3
                cur_r = int(ring["max_radius"] * ease)
                alpha = max(0.0, 1.0 - prog)

                col = (
                    int(ring["color"][0] * alpha),
                    int(ring["color"][1] * alpha),
                    int(ring["color"][2] * alpha),
                )
                thick = max(1, int(ring["thickness"] * (0.4 + alpha * 0.6)))
                cx, cy = ring["center"]

                # --- A. Chromatic fringe: separate R/G/B rings at ±chroma_offset px ---
                if ring["chromatic"] and cur_r > 5:
                    chroma_off = max(1, int(3 * alpha))  # offset grows when newer
                    # Red ring: slightly smaller radius
                    r_col = (0, 0, int(ring["color"][2] * alpha))
                    g_col = (0, int(ring["color"][1] * alpha), 0)
                    b_col = (int(ring["color"][0] * alpha), 0, 0)
                    cv2.circle(frame, (cx, cy), max(1, cur_r - chroma_off), r_col, max(1, thick - 1))
                    cv2.circle(frame, (cx, cy), cur_r,                       g_col, max(1, thick - 1))
                    cv2.circle(frame, (cx, cy), min(max(1, cur_r + chroma_off), max(h, w)), b_col, max(1, thick - 1))

                # --- B. Main ring on full-res frame ---
                cv2.circle(frame, (cx, cy), cur_r, col, thick)

                # --- C. Bright core: thin saturated ring at exact wave front ---
                bright_core = (
                    min(255, int(ring["color"][0] * min(1.0, alpha * 2.2))),
                    min(255, int(ring["color"][1] * min(1.0, alpha * 2.2))),
                    min(255, int(ring["color"][2] * min(1.0, alpha * 2.2))),
                )
                cv2.circle(frame, (cx, cy), cur_r, bright_core, max(1, thick // 3))

                # --- D. Downscaled bloom pass ---
                c_small = (cx * bw // w, cy * bh // h)
                r_small = max(1, cur_r * bw // w)
                t_small = max(1, thick // 2)
                cv2.circle(bloom_canvas, c_small, r_small, col, t_small)

        self.rings = alive_rings

        if alive_rings:
            # Wider blur for softer falloff (two-pass: tight + wide)
            blurred_tight = cv2.GaussianBlur(bloom_canvas, (9, 9), 0)
            blurred_wide  = cv2.GaussianBlur(bloom_canvas, (21, 21), 0)
            bloom_combined = cv2.addWeighted(blurred_tight, 0.65, blurred_wide, 0.35, 0)
            bloom_full = cv2.resize(bloom_combined, (w, h), interpolation=cv2.INTER_LINEAR)
            frame = cv2.add(frame, bloom_full)

        return frame



