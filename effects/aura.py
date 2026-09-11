from typing import Tuple
import math
import cv2
import numpy as np


class CursedAuraEffect:
    """
    Renders an animated glowing cursed energy aura around the segmented user:
    - Multi-scale dilated rim masks
    - Layered Gaussian bloom
    - Pulsing energy frequency modulation
    - Edge tinting for atmospheric integration
    """

    def __init__(self, aura_thickness: int = 18):
        self.thickness = aura_thickness
        # Pre-create morphological structuring elements
        self.kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (aura_thickness * 2 + 1, aura_thickness * 2 + 1))

    def composite_with_aura(
        self,
        foreground_frame: np.ndarray,
        background_frame: np.ndarray,
        person_mask: np.ndarray,
        primary_color: Tuple[int, int, int] = (255, 60, 180),  # BGR
        secondary_color: Tuple[int, int, int] = (255, 220, 80),  # BGR
        timer: int = 0,
        aura_intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Composites foreground person over domain background with a surging cursed aura.
        """
        if person_mask is None or np.count_nonzero(person_mask) == 0:
            # No person detected, return background
            return background_frame.copy()

        h, w = foreground_frame.shape[:2]

        # Smooth the raw segmentation mask to eliminate pixelated jagged edges
        feathered_mask = cv2.GaussianBlur(person_mask, (7, 7), 0)
        norm_mask = (feathered_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

        # 1. Create dilated aura envelopes
        pulse = 0.85 + 0.15 * math.sin(timer * 0.12)
        cur_thickness = max(5, int(self.thickness * pulse * aura_intensity))
        ksize = cur_thickness * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        dilated_mask = cv2.dilate(person_mask, kernel, iterations=1)
        # Rim mask is the outer region strictly behind the foreground
        rim_mask = cv2.subtract(dilated_mask, person_mask)

        # 2. Build multi-scale colored aura layers
        # Outer soft wide bloom (secondary color: cyan / gold)
        soft_bloom = cv2.GaussianBlur(rim_mask, (41, 41), 0)
        # Inner sharp intense flame (primary color: violet / crimson)
        sharp_glow = cv2.GaussianBlur(rim_mask, (15, 15), 0)

        aura_layer = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            aura_layer[:, :, c] = (
                (soft_bloom.astype(np.float32) / 255.0 * secondary_color[c] * 0.6) +
                (sharp_glow.astype(np.float32) / 255.0 * primary_color[c] * 1.2)
            )

        aura_layer = np.clip(aura_layer * aura_intensity, 0, 255).astype(np.uint8)

        # 3. Composite Background + Aura
        bg_with_aura = cv2.add(background_frame, aura_layer)

        # 4. Composite Person Foreground over (Background + Aura)
        # Also add a slight colored rim light tint onto the person's outer edges
        edge_rim = cv2.subtract(person_mask, cv2.erode(person_mask, self.kernel_inner, iterations=2))
        tint_layer = np.zeros_like(foreground_frame)
        for c in range(3):
            tint_layer[:, :, c] = (edge_rim.astype(np.float32) / 255.0 * primary_color[c] * 0.4).astype(np.uint8)

        tinted_fg = cv2.add(foreground_frame, tint_layer)

        # Alpha blend foreground onto background
        final_output = (tinted_fg.astype(np.float32) * norm_mask +
                        bg_with_aura.astype(np.float32) * (1.0 - norm_mask))

        return np.clip(final_output, 0, 255).astype(np.uint8)
