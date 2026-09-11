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

        # High-speed fast path: Screen shake only (0.3ms vs 22ms remap)
        if not has_shockwaves and has_shake:
            dx_shake = float(np.random.uniform(-self.shake_magnitude, self.shake_magnitude))
            dy_shake = float(np.random.uniform(-self.shake_magnitude, self.shake_magnitude))
            self.shake_magnitude *= 0.82
            if self.shake_magnitude < 0.5:
                self.shake_magnitude = 0.0
            M = np.float32([[1, 0, dx_shake], [0, 1, dy_shake]])
            return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Half-scale grid for shockwave calculation (3.5ms vs 35ms)
        sh, sw_w = h // 2, w // 2
        scale = 0.5
        small_frame = cv2.resize(frame, (sw_w, sh), interpolation=cv2.INTER_LINEAR)
        gx, gy = np.meshgrid(np.arange(sw_w, dtype=np.float32), np.arange(sh, dtype=np.float32))
        map_x = gx.copy()
        map_y = gy.copy()

        # 1. Optical Shockwave Remapping
        alive_waves = []
        for sw in self.shockwaves:
            progress = sw["frame"] / float(sw["duration"])
            sw["frame"] += 1

            if progress < 1.0:
                alive_waves.append(sw)
                current_radius = sw["max_radius"] * scale * progress
                current_strength = sw["strength"] * scale * (1.0 - progress)
                wave_w = sw["width"] * scale

                dx = gx - (sw["cx"] * scale)
                dy = gy - (sw["cy"] * scale)
                dist = np.sqrt(dx * dx + dy * dy)
                dist_safe = np.maximum(dist, 1e-4)

                delta_r = dist - current_radius
                mask = np.abs(delta_r) < (wave_w * 0.5)

                if np.any(mask):
                    factor = np.sin((delta_r[mask] / (wave_w * 0.5)) * math.pi) * current_strength
                    dir_x = dx[mask] / dist_safe[mask]
                    dir_y = dy[mask] / dist_safe[mask]

                    map_x[mask] += (dir_x * factor).astype(np.float32)
                    map_y[mask] += (dir_y * factor).astype(np.float32)

        self.shockwaves = alive_waves

        if has_shake:
            dx_shake = float(np.random.uniform(-self.shake_magnitude, self.shake_magnitude)) * scale
            dy_shake = float(np.random.uniform(-self.shake_magnitude, self.shake_magnitude)) * scale
            map_x += dx_shake
            map_y += dy_shake
            self.shake_magnitude *= 0.82
            if self.shake_magnitude < 0.5:
                self.shake_magnitude = 0.0

        remapped_small = cv2.remap(
            small_frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return cv2.resize(remapped_small, (w, h), interpolation=cv2.INTER_LINEAR)
