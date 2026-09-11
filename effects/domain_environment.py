import os
import math
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from config import THEMES


class DomainEnvironmentRenderer:
    """
    Renders realistic, perspective-aware Domain Expansion environments:
    - Sukuna's Malevolent Shrine (⛩️ Demonic Pagoda, Skulls, Blood Moon, Mist)
    - Gojo's Infinite Void (Singularity Black Hole, Celestial Filaments)
    """

    def __init__(self, width: int = 640, height: int = 480, theme: str = "malevolent_shrine"):
        self.w = width
        self.h = height
        self.theme_key = theme
        self.theme = THEMES.get(theme, THEMES["malevolent_shrine"])

        # Load and pre-scale environment backdrops
        self.assets = {}
        shrine_path = "assets/shrine.png"
        void_path = "assets/void.png"

        if os.path.exists(shrine_path):
            img = cv2.imread(shrine_path)
            if img is not None:
                self.assets["malevolent_shrine"] = cv2.resize(img, (self.w, self.h))

        if os.path.exists(void_path):
            img = cv2.imread(void_path)
            if img is not None:
                self.assets["infinite_void"] = cv2.resize(img, (self.w, self.h))

    def set_theme(self, theme_key: str):
        if theme_key in THEMES:
            self.theme_key = theme_key
            self.theme = THEMES[theme_key]

    def render(self, timer: int = 0) -> np.ndarray:
        """
        Renders the active domain environment with subtle atmospheric animation.
        """
        bg = self.assets.get(self.theme_key)
        if bg is None:
            # Fallback procedural void if asset is missing
            bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            bg[:, :] = self.theme["void_bg"]

        frame = bg.copy()

        # Subtle atmospheric pulse / ominous light breathing
        pulse = 0.92 + 0.08 * math.sin(timer * 0.05)
        if pulse != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=pulse)

        # Drifting cursed energy mist at the base (vectorized)
        mist_h = int(self.h * 0.35)
        mist_y1 = self.h - mist_h
        color = self.theme["primary_bgr"]

        y_arr = np.arange(mist_h, dtype=np.float32)
        fade = (y_arr / float(mist_h)) ** 1.8
        shift = np.sin((timer * 0.03) + (y_arr * 0.08)) * 0.3
        alpha = np.clip((fade + shift) * 0.45, 0.0, 1.0)[:, None, None]
        col_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        mist_1col = (alpha * col_arr).astype(np.uint8)
        mist_layer = np.repeat(mist_1col, self.w, axis=1)

        frame[mist_y1:, :] = cv2.add(frame[mist_y1:, :], mist_layer)
        return frame
