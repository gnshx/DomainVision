import math
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


class CursedEnergyEffect:
    """
    Renders perspective-aware cursed energy anchored to hand joints and finger bones.
    Glow, electric filaments, and hand aura scale and rotate naturally with the user's hands.
    """

    BONES = [
        # Palm connections
        (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0), (0, 9),
        # Thumb
        (2, 3), (3, 4),
        # Index
        (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20),
    ]

    TIPS = [4, 8, 12, 16, 20]

    def __init__(self):
        self._glow_layer = None
        self._core_layer = None

    def render(
        self,
        frame: np.ndarray,
        hands: List[Dict[str, Any]],
        primary_color: Tuple[int, int, int] = (40, 40, 240),    # BGR
        secondary_color: Tuple[int, int, int] = (30, 160, 255),  # BGR
        timer: int = 0,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Render perspective cursed energy attached to hand skeletons.
        """
        if not hands or intensity <= 0.01:
            return frame

        h, w = frame.shape[:2]
        if self._glow_layer is None or self._glow_layer.shape[:2] != (h, w):
            self._glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
            self._core_layer = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            self._glow_layer.fill(0)
            self._core_layer.fill(0)
        glow_layer = self._glow_layer
        core_layer = self._core_layer

        for hand in hands:
            landmarks = hand["landmarks"]
            scale = max(15.0, hand["palm_scale"])
            thickness = max(2, int(scale * 0.06 * intensity))

            # 1. Draw cursed energy along finger bones
            for i1, i2 in self.BONES:
                if i1 < len(landmarks) and i2 < len(landmarks):
                    p1 = landmarks[i1]
                    p2 = landmarks[i2]
                    cv2.line(glow_layer, p1, p2, primary_color, thickness * 2 + 2)
                    cv2.line(core_layer, p1, p2, secondary_color, thickness)

            # 2. Pulsing energy motes at fingertips
            for tip_idx in self.TIPS:
                if tip_idx < len(landmarks):
                    tip_pt = landmarks[tip_idx]
                    pulse = math.sin(timer * 0.3 + tip_idx) * 2.0
                    rad = max(3, int(scale * 0.12 + pulse))
                    cv2.circle(glow_layer, tip_pt, rad + 4, primary_color, -1)
                    cv2.circle(core_layer, tip_pt, max(2, rad - 2), (255, 255, 255), -1)

            # 3. Palm core cursed vortex
            center = hand["palm_center"]
            center_rad = max(8, int(scale * 0.28 + math.sin(timer * 0.2) * 3.0))
            cv2.circle(glow_layer, center, center_rad + 8, primary_color, -1)
            cv2.circle(core_layer, center, center_rad, secondary_color, -1)
            cv2.circle(core_layer, center, max(3, int(center_rad * 0.45)), (255, 255, 255), -1)

        # 4. Electric lightning between hands when 2 hands are present
        if len(hands) >= 2:
            h1, h2 = hands[0], hands[1]
            p1 = h1["palm_center"]
            p2 = h2["palm_center"]
            dist = math.dist(p1, p2)
            if dist < max(w * 0.35, h1["palm_scale"] * 3.5):
                # Draw jagged lightning between fingertips or palms
                pts = [p1]
                steps = 4
                for s in range(1, steps):
                    frac = s / float(steps)
                    mid_x = int(p1[0] * (1 - frac) + p2[0] * frac)
                    mid_y = int(p1[1] * (1 - frac) + p2[1] * frac)
                    jitter = np.random.randint(-12, 12)
                    pts.append((mid_x + jitter, mid_y + jitter))
                pts.append(p2)

                for i in range(len(pts) - 1):
                    cv2.line(glow_layer, pts[i], pts[i + 1], secondary_color, 4)
                    cv2.line(core_layer, pts[i], pts[i + 1], (255, 255, 255), 1)

        # Fast downscaled bloom (0.8ms vs 12ms full res blur)
        small_glow = cv2.resize(glow_layer, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
        small_blur = cv2.GaussianBlur(small_glow, (11, 11), 0)
        blurred_glow = cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)

        out = cv2.add(frame, cv2.convertScaleAbs(blurred_glow, alpha=intensity))
        out = cv2.add(out, cv2.convertScaleAbs(core_layer, alpha=intensity))
        return out
