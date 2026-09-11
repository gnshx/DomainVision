import math
from typing import Tuple
import cv2
import numpy as np


class CursedAuraEffect:
    """
    Renders an animated, edge-aware 3-layer cursed energy aura around the segmented user.

    Three distinct aura layers (v2):
    1. Inner Rim: 1-2px bright contour line directly on the silhouette edge (erosion-based)
    2. Mid Glow: 8-14px morphological rim at primary domain color
    3. Outer Halo: 24-32px wide very-low-opacity atmospheric secondary color

    All bloom operations are performed at downscaled resolution then upscaled,
    keeping total aura cost under 4ms on CPU.
    """

    def __init__(self, aura_thickness: int = 18):
        self.thickness = aura_thickness
        self._kern_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._bloom_w = 320
        self._bloom_h = 180

    def composite_with_aura(
        self,
        foreground_frame: np.ndarray,
        background_frame: np.ndarray,
        person_mask: np.ndarray,
        primary_color: Tuple[int, int, int] = (40, 20, 220),
        secondary_color: Tuple[int, int, int] = (0, 80, 180),
        timer: int = 0,
        aura_intensity: float = 1.0,
        aura_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Composites foreground person over domain background with a 3-layer edge-aware aura.

        Args:
            aura_scale: quality controller override (0.25-1.0). Scales bloom resolution.
        """
        if person_mask is None or np.count_nonzero(person_mask) == 0:
            return background_frame.copy()

        h, w = foreground_frame.shape[:2]

        # 1. Full-resolution binary mask for clean compositing
        bin_mask = (person_mask > 100).astype(np.uint8) * 255
        mask_inv = cv2.bitwise_not(bin_mask)

        fg_part = cv2.bitwise_and(foreground_frame, foreground_frame, mask=bin_mask)
        bg_part = cv2.bitwise_and(background_frame, background_frame, mask=mask_inv)

        # 2. Inner rim — crisp 1-2px bright edge directly on silhouette (full resolution)
        pulse = 0.80 + 0.20 * math.sin(timer * 0.14)
        rim_inner = cv2.erode(bin_mask, self._kern_erode, iterations=1)
        rim_edge = cv2.subtract(bin_mask, rim_inner)

        rim_brightness = min(255, int(255 * aura_intensity * pulse))
        inner_color = (
            min(255, int(primary_color[0] * 0.4 + rim_brightness * 0.6)),
            min(255, int(primary_color[1] * 0.4 + rim_brightness * 0.6)),
            min(255, int(primary_color[2] * 0.4 + rim_brightness * 0.6)),
        )
        inner_rim_layer = np.zeros((h, w, 3), dtype=np.uint8)
        inner_rim_layer[rim_edge > 0] = inner_color

        # 3. Mid glow + outer halo at downscaled bloom resolution
        bw = max(80, int(self._bloom_w * aura_scale))
        bh = max(45, int(self._bloom_h * aura_scale))
        mask_small = cv2.resize(bin_mask, (bw, bh), interpolation=cv2.INTER_NEAREST)

        # Mid glow: ~9px dilation at bloom scale
        mid_thick = max(2, int(9 * max(0.25, aura_scale)))
        ksize_mid = mid_thick * 2 + 1
        kern_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_mid, ksize_mid))
        dilated_mid = cv2.dilate(mask_small, kern_mid, iterations=1)
        rim_mid = cv2.subtract(dilated_mid, mask_small)
        glow_mid = cv2.GaussianBlur(rim_mid, (7, 7), 0)

        # Outer halo: wider dilation, heavier blur
        halo_thick = max(4, int(22 * max(0.25, aura_scale)))
        ksize_halo = halo_thick * 2 + 1
        kern_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_halo, ksize_halo))
        dilated_halo = cv2.dilate(mask_small, kern_halo, iterations=1)
        rim_halo = cv2.subtract(dilated_halo, dilated_mid)
        glow_halo = cv2.GaussianBlur(rim_halo, (13, 13), 0)

        col_pri = np.array(primary_color, dtype=np.float32) / 255.0
        col_sec = np.array(secondary_color, dtype=np.float32) / 255.0

        mid_strength = min(1.4, 1.1 * aura_intensity)
        halo_strength = min(0.5, 0.35 * aura_intensity)

        glow_mid_3c = cv2.cvtColor(glow_mid, cv2.COLOR_GRAY2BGR).astype(np.float32)
        mid_colored = np.clip(glow_mid_3c * col_pri * mid_strength, 0, 255).astype(np.uint8)

        glow_halo_3c = cv2.cvtColor(glow_halo, cv2.COLOR_GRAY2BGR).astype(np.float32)
        halo_colored = np.clip(glow_halo_3c * col_sec * halo_strength, 0, 255).astype(np.uint8)

        bloom_combined = cv2.add(mid_colored, halo_colored)
        aura_full = cv2.resize(bloom_combined, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Final composite: background + bloom aura + inner rim + foreground
        bg_with_bloom = cv2.add(bg_part, aura_full)
        bg_with_rim = cv2.add(bg_with_bloom, cv2.bitwise_and(inner_rim_layer, inner_rim_layer, mask=mask_inv))
        return cv2.add(fg_part, bg_with_rim)
