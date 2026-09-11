"""
Cinematic color grading and post-processing pass for DomainVision.

Provides:
- Domain-themed color spill on person silhouette edges (localized, not a whole-body tint)
- Vignette (precomputed radial falloff darkening corners)
- Bloom (bright pixel threshold -> downscaled blur -> additive blend)
- Chromatic aberration fringe (subtle R/G/B channel offset during FLASH/SHOCKWAVE)

All operations use precomputed buffers or downscaled working surfaces.
Budget target: < 3ms per frame at 1280x720 on CPU.
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
        # Pre-compute vignette mask (radial falloff, darkens corners)
        self._vignette = self._build_vignette(width, height, strength=0.55, power=2.2)
        # Bloom working size
        self._bloom_w = 320
        self._bloom_h = 180
        self._bloom_buf = np.zeros((self._bloom_h, self._bloom_w, 3), dtype=np.uint8)

    @staticmethod
    def _build_vignette(w: int, h: int, strength: float = 0.55, power: float = 2.0) -> np.ndarray:
        """Precompute a float32 vignette mask [0,1] — 0 at corners, 1 at center."""
        cx, cy = w / 2.0, h / 2.0
        y_idx, x_idx = np.ogrid[:h, :w]
        # Normalized distance from center (0=center, 1=corner)
        dist = np.sqrt(((x_idx - cx) / cx) ** 2 + ((y_idx - cy) / cy) ** 2)
        vignette = np.clip(1.0 - (dist ** power) * strength, 0.0, 1.0).astype(np.float32)
        return vignette[:, :, np.newaxis]  # (h, w, 1) for broadcast over BGR

    def apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Apply precomputed vignette. Very cheap (one multiply + clip)."""
        if self._vignette.shape[:2] != frame.shape[:2]:
            self._vignette = self._build_vignette(frame.shape[1], frame.shape[0])
        result = (frame.astype(np.float32) * self._vignette)
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_bloom(
        self, frame: np.ndarray, threshold: int = 200, bloom_strength: float = 0.45
    ) -> np.ndarray:
        """
        Bloom pass: isolate bright pixels, downscale, blur, upscale, additively blend.
        Budget: ~1-1.5ms at 1280x720.
        """
        bw, bh = self._bloom_w, self._bloom_h
        # Downscale
        small = cv2.resize(frame, (bw, bh), interpolation=cv2.INTER_AREA)
        # Isolate bright pixels only
        _, bright = cv2.threshold(small, threshold, 255, cv2.THRESH_BINARY)
        bright_masked = cv2.bitwise_and(small, bright)
        # Blur the bright regions
        blurred = cv2.GaussianBlur(bright_masked, (15, 15), 0)
        # Upscale bloom back to full frame
        bloom_full = cv2.resize(blurred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Additively blend at reduced strength
        bloom_scaled = (bloom_full.astype(np.float32) * bloom_strength).astype(np.uint8)
        return cv2.add(frame, bloom_scaled)

    def apply_color_spill(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray,
        domain_color: Tuple[int, int, int],
        spill_strength: float = 0.18,
        timer: int = 0,
    ) -> np.ndarray:
        """
        Applies domain-colored light spill to the person's silhouette edges (not the entire body).
        This simulates the environment light illuminating the person from the domain atmosphere.

        Method: dilate the person mask slightly, subtract original mask to get edge ring,
        then additively blend domain color into that edge area only.
        """
        if person_mask is None or np.count_nonzero(person_mask) == 0:
            return frame

        h, w = frame.shape[:2]
        # Downscale for speed
        dw, dh = 320, 180
        mask_small = cv2.resize(person_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask_small, kern, iterations=2)
        edge_ring = cv2.subtract(dilated, mask_small)  # ~12px edge on person boundary

        # Convert edge ring to full resolution
        edge_full = cv2.resize(edge_ring, (w, h), interpolation=cv2.INTER_LINEAR)

        # Animated pulse for spill intensity
        pulse = 0.85 + 0.15 * math.sin(timer * 0.10)
        actual_strength = spill_strength * pulse

        # Build color spill layer
        spill = np.zeros((h, w, 3), dtype=np.uint8)
        spill[:] = domain_color
        # Mask to edge ring only
        edge_norm = (edge_full.astype(np.float32) / 255.0) * actual_strength
        spill_masked = (spill.astype(np.float32) * edge_norm[:, :, np.newaxis]).astype(np.uint8)

        return cv2.add(frame, spill_masked)

    def apply_chromatic_aberration(
        self, frame: np.ndarray, magnitude: float = 2.0
    ) -> np.ndarray:
        """
        Subtle RGB channel offset producing prismatic fringe on edges.
        Used during FLASH and SHOCKWAVE states.
        magnitude: pixel offset (1.0-4.0 recommended)
        """
        if magnitude < 0.5:
            return frame
        m = int(round(magnitude))
        h, w = frame.shape[:2]
        b, g, r = cv2.split(frame)
        # Shift red channel left, blue channel right
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
        Full grading pass. Call this after all other effects are composited.
        Disable individual sub-passes via flags for quality scaling.
        """
        result = frame

        if apply_spill and person_mask is not None:
            result = self.apply_color_spill(result, person_mask, domain_color,
                                            spill_strength=spill_strength, timer=timer)

        if apply_bloom:
            result = self.apply_bloom(result, threshold=bloom_threshold, bloom_strength=bloom_strength)

        if chromatic_magnitude > 0.5:
            result = self.apply_chromatic_aberration(result, magnitude=chromatic_magnitude)

        if apply_vignette:
            result = self.apply_vignette(result)

        return result
