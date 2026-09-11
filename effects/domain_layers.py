"""
2.5D Parallax Domain Environment Renderer for DomainVision.

Splits the domain background image into three virtual depth layers:
  Layer 0 (FAR)   - Sky, upper backdrop: slowest parallax (0.02x camera delta)
  Layer 1 (MID)   - Main shrine/void structure: medium parallax (0.08x)
  Layer 2 (FRONT) - Ground mist, foreground elements: fastest parallax (0.25x)

Even without a camera motion estimator, layers respond to a slow synthetic
cinematic drift for depth cue. When camera (dx, dy) offsets are provided,
they are additively applied on top of the drift.

Also renders:
  - Domain-specific light rays (radial streaks from bright upper source point)
  - Ground mist turbulence (row-shifted noise bands near the base of frame)
  - Atmospheric depth haze (darkens far regions relative to camera)
"""
import math
import os
from typing import Tuple, Optional
import cv2
import numpy as np

from config import THEMES


class DomainLayerRenderer:
    """
    2.5D parallax layered domain environment renderer.
    Replaces the flat DomainEnvironmentRenderer for active domain states.
    """

    # Parallax multipliers per layer [FAR, MID, FRONT]
    PARALLAX_FACTORS = [0.02, 0.10, 0.30]

    def __init__(self, width: int = 1280, height: int = 720, theme: str = "malevolent_shrine"):
        self.w = width
        self.h = height
        self.theme_key = theme
        self.theme = THEMES.get(theme, THEMES["malevolent_shrine"])

        # Synthetic cinematic drift parameters (slow oscillation even without camera)
        self._drift_freq_x = 0.0018
        self._drift_freq_y = 0.0011
        self._drift_amp = 8.0  # pixels peak amplitude

        # Camera motion offset (updated externally by motion estimator)
        self._cam_dx: float = 0.0
        self._cam_dy: float = 0.0

        # Pre-compute mist noise map (persistent turbulence)
        np.random.seed(7)
        self._noise = np.random.uniform(-1.0, 1.0, (self.h, self.w)).astype(np.float32)
        # Pre-compute atmospheric depth haze gradient (darker at top = far)
        self._haze_mask = self._build_haze_mask(width, height)

        # Load and split background asset into three layers
        self._layers = {}  # theme_key -> [far, mid, front] image arrays
        self._load_layers()

    @staticmethod
    def _build_haze_mask(w: int, h: int) -> np.ndarray:
        """Top of frame is 'far away' and gets slightly more haze."""
        y_arr = np.linspace(0.22, 0.0, h, dtype=np.float32)  # top=0.22, bottom=0.0
        haze = np.tile(y_arr[:, np.newaxis], (1, w))[:, :, np.newaxis]
        return haze  # (h, w, 1)

    def _load_layers(self):
        asset_map = {
            "malevolent_shrine": "assets/shrine.png",
            "infinite_void": "assets/void.png",
        }
        for theme_key, path in asset_map.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    base = cv2.resize(img, (self.w, self.h))
                    # Layer split by vertical region + brightness:
                    # FAR: top 60% of frame
                    # MID: full frame (base)
                    # FRONT: bottom 40% darkened slightly
                    far = base.copy()
                    far[int(self.h * 0.6):] = (far[int(self.h * 0.6):].astype(np.float32) * 0.5).astype(np.uint8)

                    mid = base.copy()

                    front = np.zeros_like(base)
                    front[int(self.h * 0.6):] = base[int(self.h * 0.6):]
                    # Add slight warm color overlay to foreground
                    col = np.array(THEMES[theme_key]["primary_bgr"], dtype=np.uint8)
                    front_tint = np.zeros_like(front)
                    front_tint[int(self.h * 0.6):] = col
                    front = cv2.addWeighted(front, 0.85, front_tint, 0.15, 0)

                    self._layers[theme_key] = [far, mid, front]

    def set_theme(self, theme_key: str):
        if theme_key in THEMES:
            self.theme_key = theme_key
            self.theme = THEMES[theme_key]

    def set_camera_offset(self, dx: float, dy: float):
        """Update camera translation from motion estimator (pixels at full res)."""
        # Apply EMA to smooth out jitter from optical flow
        self._cam_dx = 0.7 * self._cam_dx + 0.3 * dx
        self._cam_dy = 0.7 * self._cam_dy + 0.3 * dy

    def _shift_layer(self, layer: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Translate a layer by (dx, dy) pixels using warpAffine with border replication."""
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            layer, M, (self.w, self.h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _render_light_rays(self, base: np.ndarray, timer: int) -> np.ndarray:
        """
        Renders subtle light ray streaks emanating from upper center of the frame.
        Uses a small bright point + radial blur approximation via repeated resize.
        Very cheap: ~0.5ms.
        """
        ray_layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        col = self.theme["secondary_bgr"]
        cx = self.w // 2 + int(20 * math.sin(timer * 0.007))
        cy = int(self.h * 0.08)

        num_rays = 6
        for i in range(num_rays):
            angle = math.pi * 0.25 + (i / num_rays) * math.pi * 0.5 + timer * 0.003
            length = int(self.h * (0.5 + 0.2 * math.sin(timer * 0.012 + i)))
            brightness = int(60 + 30 * math.sin(timer * 0.018 + i * 1.1))
            ray_col = (
                min(255, int(col[0] * brightness / 80)),
                min(255, int(col[1] * brightness / 80)),
                min(255, int(col[2] * brightness / 80)),
            )
            ex = int(cx + length * math.cos(angle))
            ey = int(cy + length * math.sin(angle))
            cv2.line(ray_layer, (cx, cy), (ex, ey), ray_col, 3, cv2.LINE_AA)

        # Soft blur to make rays look volumetric
        ray_blurred = cv2.GaussianBlur(ray_layer, (0, 0), sigmaX=6, sigmaY=6)
        return cv2.add(base, ray_blurred)

    def _render_ground_mist(self, base: np.ndarray, timer: int) -> np.ndarray:
        """
        Renders animated ground mist using pre-computed noise field with row-offsets.
        Turbulence effect makes the mist look swirling and volumetric.
        """
        mist_h = int(self.h * 0.40)
        mist_y1 = self.h - mist_h
        col = np.array(self.theme["primary_bgr"], dtype=np.float32)

        y_arr = np.arange(mist_h, dtype=np.float32)
        # Fade: strongest at the very bottom, zero at mist_h
        base_fade = (1.0 - y_arr / float(mist_h)) ** 1.6

        # Noise-turbulence: row-shifted horizontal noise band
        noise_row = self._noise[mist_y1:, self.w // 4:self.w // 4 + mist_h]
        if noise_row.shape[1] > 0:
            noise_col = np.mean(noise_row, axis=1)
        else:
            noise_col = np.zeros(mist_h, dtype=np.float32)

        drift = np.sin((timer * 0.025) + (y_arr * 0.06)) * 0.18
        alpha = np.clip((base_fade + noise_col * 0.08 + drift) * 0.55, 0.0, 1.0)
        alpha_full = alpha[:, np.newaxis, np.newaxis]  # (mist_h, 1, 1)

        mist_overlay = (alpha_full * col.reshape(1, 1, 3)).astype(np.uint8)
        mist_layer = np.tile(mist_overlay, (1, self.w, 1))

        result = base.copy()
        result[mist_y1:, :] = cv2.add(result[mist_y1:, :], mist_layer)
        return result

    def render(self, timer: int = 0, cam_dx: float = 0.0, cam_dy: float = 0.0) -> np.ndarray:
        """
        Render the composite parallax domain frame.

        Args:
            timer: global frame counter for animation.
            cam_dx, cam_dy: optional external camera translation offset (pixels).
        """
        self.set_camera_offset(cam_dx, cam_dy)

        layers = self._layers.get(self.theme_key)
        if layers is None:
            # Fallback: procedural void background
            fallback = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            fallback[:, :] = self.theme["void_bg"]
            return self._render_ground_mist(fallback, timer)

        far_img, mid_img, front_img = layers

        # Synthetic cinematic drift (even with no camera input)
        drift_x = self._drift_amp * math.sin(timer * self._drift_freq_x)
        drift_y = self._drift_amp * 0.4 * math.sin(timer * self._drift_freq_y)

        total_dx = drift_x + self._cam_dx
        total_dy = drift_y + self._cam_dy

        # Apply per-layer parallax offsets
        pf = self.PARALLAX_FACTORS
        far_shifted  = self._shift_layer(far_img,  total_dx * pf[0], total_dy * pf[0])
        mid_shifted  = self._shift_layer(mid_img,  total_dx * pf[1], total_dy * pf[1])
        front_shifted = self._shift_layer(front_img, total_dx * pf[2], total_dy * pf[2])

        # Atmospheric light-breathing pulse on mid layer
        pulse = 0.93 + 0.07 * math.sin(timer * 0.05)
        mid_shifted = cv2.convertScaleAbs(mid_shifted, alpha=pulse)

        # Composite: far (base) -> additive mid and front blends
        # Far is the base background, mid adds the main structure detail,
        # front adds foreground ground elements
        composite = far_shifted.copy()
        composite = cv2.addWeighted(composite, 0.4, mid_shifted, 0.6, 0)
        # Front layer: only blend bottom portion
        fy1 = int(self.h * 0.55)
        composite[fy1:] = cv2.addWeighted(
            composite[fy1:], 0.7, front_shifted[fy1:], 0.3, 0
        )

        # Add atmospheric depth haze (top = slightly darker / more opaque)
        haze_color = np.array(self.theme["void_bg"], dtype=np.float32)
        haze_overlay = (self._haze_mask * haze_color).astype(np.uint8)
        haze_layer = np.tile(haze_overlay, (1, self.w, 1)) if haze_overlay.shape[1] == 1 else haze_overlay
        if haze_layer.shape != composite.shape:
            haze_layer = np.broadcast_to(haze_overlay, composite.shape).copy()
        composite = cv2.add(composite, haze_layer.astype(np.uint8))

        # Light rays
        composite = self._render_light_rays(composite, timer)

        # Ground mist turbulence
        composite = self._render_ground_mist(composite, timer)

        return composite
