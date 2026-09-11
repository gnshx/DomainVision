import math
from typing import Tuple, Optional
import cv2
import numpy as np


class OpticalDistortionEffect:
    """
    Creates real optical shockwave ripples using cv2.remap() pixel displacement
    and dynamic screen trauma/shake.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.w = width
        self.h = height

        # Base identity coordinate maps
        self.base_grid_x, self.base_grid_y = np.meshgrid(
            np.arange(self.w, dtype=np.float32),
            np.arange(self.h, dtype=np.float32)
        )

        # Active shockwaves list
        self.shockwaves = []
        # Screen shake magnitude
        self.shake_magnitude = 0.0

    def trigger_shockwave(
        self,
        center: Tuple[int, int],
        max_radius: Optional[float] = None,
        duration_frames: int = 24,
        strength: float = 24.0,
        wave_width: float = 45.0,
    ):
        """Trigger an expanding optical refraction shockwave."""
        if max_radius is None:
            max_radius = math.hypot(self.w, self.h)

        self.shockwaves.append({
            "cx": float(center[0]),
            "cy": float(center[1]),
            "max_radius": float(max_radius),
            "duration": duration_frames,
            "frame": 0,
            "strength": float(strength),
            "width": float(wave_width),
        })

    def trigger_shake(self, magnitude: float = 12.0):
        """Trigger screen shake."""
        self.shake_magnitude = max(self.shake_magnitude, magnitude)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply active optical shockwaves and screen shake."""
        h, w = frame.shape[:2]
        if w != self.w or h != self.h:
            self.w, self.h = w, h
            self.base_grid_x, self.base_grid_y = np.meshgrid(
                np.arange(self.w, dtype=np.float32),
                np.arange(self.h, dtype=np.float32)
            )

        has_shockwaves = len(self.shockwaves) > 0
        has_shake = self.shake_magnitude > 0.5

        if not has_shockwaves and not has_shake:
            return frame

        map_x = self.base_grid_x.copy()
        map_y = self.base_grid_y.copy()

        # 1. Optical Shockwave Remapping
        alive_waves = []
        for sw in self.shockwaves:
            progress = sw["frame"] / float(sw["duration"])
            sw["frame"] += 1

            if progress < 1.0:
                alive_waves.append(sw)
                current_radius = sw["max_radius"] * progress
                # Attenuation over distance
                current_strength = sw["strength"] * (1.0 - progress)
                wave_w = sw["width"]

                # Vectorized distance calculation
                dx = self.base_grid_x - sw["cx"]
                dy = self.base_grid_y - sw["cy"]
                dist = np.sqrt(dx * dx + dy * dy)
                # Avoid div by 0
                dist_safe = np.maximum(dist, 1e-4)

                # Find pixels within the wave band
                delta_r = dist - current_radius
                mask = np.abs(delta_r) < (wave_w * 0.5)

                if np.any(mask):
                    # Sine displacement profile
                    factor = np.sin((delta_r[mask] / (wave_w * 0.5)) * math.pi) * current_strength
                    # Displace outward along unit direction
                    dir_x = dx[mask] / dist_safe[mask]
                    dir_y = dy[mask] / dist_safe[mask]

                    map_x[mask] += (dir_x * factor).astype(np.float32)
                    map_y[mask] += (dir_y * factor).astype(np.float32)

        self.shockwaves = alive_waves

        # 2. Screen Shake Offset
        if has_shake:
            dx_shake = np.random.uniform(-self.shake_magnitude, self.shake_magnitude)
            dy_shake = np.random.uniform(-self.shake_magnitude, self.shake_magnitude)
            map_x += float(dx_shake)
            map_y += float(dy_shake)
            self.shake_magnitude *= 0.82  # Rapid dampening
            if self.shake_magnitude < 0.5:
                self.shake_magnitude = 0.0

        # Perform remapping
        distorted = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return distorted
