import cv2
import numpy as np


def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend overlay onto base using an alpha mask (float [0, 1] or uint8 [0, 255])."""
    if alpha.dtype == np.uint8:
        alpha = alpha.astype(np.float32) / 255.0

    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]

    base_f = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    blended = base_f * (1.0 - alpha) + overlay_f * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def additive_blend(base: np.ndarray, glow: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Add glow to base with saturation."""
    if strength == 1.0:
        return cv2.add(base, glow)
    glow_scaled = cv2.convertScaleAbs(glow, alpha=strength)
    return cv2.add(base, glow_scaled)


def screen_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Photoshop-style screen blending mode for intense bright energy."""
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    res = 1.0 - (1.0 - base_f) * (1.0 - overlay_f)
    return (res * 255.0).clip(0, 255).astype(np.uint8)


def apply_color_glow(mask: np.ndarray, color: tuple, ksize: int = 25) -> np.ndarray:
    """Generate a colored glow layer from a single-channel mask."""
    if ksize % 2 == 0:
        ksize += 1
    blurred_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    glow = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for c in range(3):
        glow[:, :, c] = (blurred_mask.astype(np.float32) / 255.0 * color[c]).astype(np.uint8)
    return glow
