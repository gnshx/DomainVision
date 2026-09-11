from typing import Tuple
import cv2
import numpy as np


class DomainFlashEffect:
    """
    Cinematic multi-stage flash sequence:
    Black Vignette -> Cursed Energy Surge -> Blinding White Flash -> Dissolve into Domain
    """

    def __init__(self, duration_frames: int = 18):
        self.duration = duration_frames
        self.active = False
        self.frame_idx = 0
        self.theme_color = (255, 50, 180)  # BGR default purple

    def trigger(self, theme_color: Tuple[int, int, int] = (255, 50, 180)):
        self.active = True
        self.frame_idx = 0
        self.theme_color = theme_color

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply flash to frame.
        Returns: (processed_frame, is_still_active)
        """
        if not self.active:
            return frame, False

        h, w = frame.shape[:2]
        progress = self.frame_idx / float(self.duration)
        self.frame_idx += 1

        if self.frame_idx >= self.duration:
            self.active = False

        # Stage 1: frames 0-4 (Darkening / vignette constriction)
        if progress < 0.25:
            darken = 1.0 - (progress / 0.25) * 0.7
            out = cv2.convertScaleAbs(frame, alpha=darken)

        # Stage 2: frames 4-8 (Intense theme-color surge: deep purple/crimson)
        elif progress < 0.5:
            t = (progress - 0.25) / 0.25
            surge = np.zeros_like(frame)
            surge[:, :] = self.theme_color
            surge_scaled = cv2.convertScaleAbs(surge, alpha=t * 0.85)
            out = cv2.add(cv2.convertScaleAbs(frame, alpha=0.3), surge_scaled)

        # Stage 3: frames 8-11 (Blinding Whiteout Flash)
        elif progress < 0.7:
            t = (progress - 0.5) / 0.2
            white = np.full_like(frame, 255)
            white_strength = 1.0 - (t * 0.3)
            out = cv2.addWeighted(frame, 1.0 - white_strength, white, white_strength, 0)

        # Stage 4: frames 11-duration (Fade out flash into new domain)
        else:
            t = (progress - 0.7) / 0.3
            alpha = max(0.0, 1.0 - t)
            white = np.full_like(frame, 255)
            out = cv2.addWeighted(frame, 1.0 - alpha, white, alpha, 0)

        return out, self.active
