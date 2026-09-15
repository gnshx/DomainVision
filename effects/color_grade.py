"""
Cinematic color grading and post-processing pass for DomainVision.

Provides:
- Domain-themed color spill on person silhouette edges
- High-speed SIMD Vignette (precomputed uint8 multiplier)
- Low-overhead Bloom (320x180 working buffer with integer scaling)
- Chromatic aberration fringe

Target latency: < 2.5ms at 1280x720 on CPU.
"""
import math
from typing import Tuple, Optional
import cv2
import numpy as np


class CinematicColorGrader:
    """
    Applies domain-atmosphere color grading and post-processing to the composite frame.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height
        # Pre-compute uint8 vignette mask (radial falloff)
        self._vignette_u8 = self._build_vignette(width, height, strength=0.55, power=2.2)
        # Bloom working dimensions
        self._bloom_w = 320
        self._bloom_h = 180
        self._kern_spill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    @staticmethod
    def _build_vignette(w: int, h: int, strength: float = 0.55, power: float = 2.0) -> np.ndarray:
        """Precompute a uint8 vignette mask [0, 255] for SIMD cv2.multiply."""
        cx, cy = w / 2.0, h / 2.0
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt(((x_idx - cx) / cx) ** 2 + ((y_idx - cy) / cy) ** 2)
        vignette_f = np.clip(1.0 - (dist ** power) * strength, 0.0, 1.0)
        vignette_u8 = (vignette_f * 255.0).astype(np.uint8)
        return cv2.merge([vignette_u8, vignette_u8, vignette_u8])

    def apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """High-speed SIMD vignette multiply (< 1.0ms at 1280x720)."""
        if self._vignette_u8.shape[:2] != frame.shape[:2]:
            self._vignette_u8 = self._build_vignette(frame.shape[1], frame.shape[0])
        return cv2.multiply(frame, self._vignette_u8, scale=1.0 / 255.0)

    def apply_bloom(
        self, frame: np.ndarray, threshold: int = 200, bloom_strength: float = 0.45
    ) -> np.ndarray:
        """
        Downscaled bloom pass (< 1.2ms at 1280x720):
        Isolates bright pixels on downscaled buffer, blurs, scales, upscales, and adds.
        """
        bw, bh = self._bloom_w, self._bloom_h
        small = cv2.resize(frame, (bw, bh), interpolation=cv2.INTER_AREA)
        _, bright = cv2.threshold(small, threshold, 255, cv2.THRESH_BINARY)
        bright_masked = cv2.bitwise_and(small, bright)
        blurred = cv2.GaussianBlur(bright_masked, (15, 15), 0)
        scaled_bloom = cv2.convertScaleAbs(blurred, alpha=bloom_strength)
        bloom_full = cv2.resize(scaled_bloom, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.add(frame, bloom_full)

    def apply_color_spill(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray,
        domain_color: Tuple[int, int, int],
        spill_strength: float = 0.18,
        timer: int = 0,
    ) -> np.ndarray:
        """
        High-speed edge spill (< 0.8ms at 1280x720):
        Morphological dilation on downscaled mask, colored at low res, upscaled once.
        """
        if person_mask is None or cv2.countNonZero(person_mask) == 0:
            return frame

        h, w = frame.shape[:2]
        dw, dh = 320, 180
        mask_small = cv2.resize(person_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

        dilated = cv2.dilate(mask_small, self._kern_spill, iterations=2)
        edge_ring = cv2.subtract(dilated, mask_small)

        if cv2.countNonZero(edge_ring) == 0:
            return frame

        pulse = 0.85 + 0.15 * math.sin(timer * 0.10)
        actual_strength = spill_strength * pulse

        # Build colored spill at low res
        spill_small = np.zeros((dh, dw, 3), dtype=np.uint8)
        spill_small[:] = domain_color
        spill_colored_small = cv2.bitwise_and(spill_small, spill_small, mask=edge_ring)
        spill_scaled = cv2.convertScaleAbs(spill_colored_small, alpha=actual_strength)

        spill_full = cv2.resize(spill_scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(frame, spill_full)

    def apply_chromatic_aberration(
        self, frame: np.ndarray, magnitude: float = 2.0
    ) -> np.ndarray:
        """
        RGB channel offset producing prismatic fringe.
        magnitude: pixel offset.
        """
        if magnitude < 0.5:
            return frame
        m = int(round(magnitude))
        h, w = frame.shape[:2]
        b, g, r = cv2.split(frame)
        M_r = np.float32([[1, 0, -m], [0, 1, 0]])
        M_b = np.float32([[1, 0,  m], [0, 1, 0]])
        r_shift = cv2.warpAffine(r, M_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b_shift = cv2.warpAffine(b, M_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.merge([b_shift, g, r_shift])

    def apply(
        self,
        frame: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
        domain_color: Tuple[int, int, int] = (40, 20, 220),
        timer: int = 0,
        apply_spill: bool = True,
        apply_bloom: bool = True,
        apply_vignette: bool = True,
        chromatic_magnitude: float = 0.0,
        spill_strength: float = 0.15,
        bloom_strength: float = 0.40,
        bloom_threshold: int = 195,
    ) -> np.ndarray:
        """
        Full grading pass (< 2.5ms total).
        """
        out = frame

        # 1. Color spill
        if apply_spill and person_mask is not None:
            out = self.apply_color_spill(
                out, person_mask, domain_color, spill_strength=spill_strength, timer=timer
            )

        # 2. Bloom
        if apply_bloom:
            out = self.apply_bloom(
                out, threshold=bloom_threshold, bloom_strength=bloom_strength
            )

        # 3. Vignette
        if apply_vignette:
            out = self.apply_vignette(out)

        # 4. Chromatic aberration
        if chromatic_magnitude >= 0.5:
            out = self.apply_chromatic_aberration(out, magnitude=chromatic_magnitude)

        return out
