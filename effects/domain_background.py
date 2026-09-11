import math
from typing import Tuple, Dict, Any
import cv2
import numpy as np


class DomainBackgroundRenderer:
    """
    Generates procedural animated Domain Expansion environments:
    - Infinite Void (無量空処): Deep cosmic space, violet radial gradients, rotating celestial rings.
    - Malevolent Shrine (伏魔御廚子): Blood-red abyss, obsidian horizon, dark crimson barrier sigils.
    """

    THEMES: Dict[str, Dict[str, Any]] = {
        "infinite_void": {
            "name_ja": "無量空処",
            "name_en": "INFINITE VOID",
            "primary_bgr": (255, 60, 160),      # Bright Violet / Purple
            "secondary_bgr": (255, 220, 80),    # Cyan / Electric Blue
            "bg_dark": (35, 5, 25),             # Dark void
            "bg_accent": (70, 15, 60),          # Inner aura
            "stars_count": 80,
        },
        "malevolent_shrine": {
            "name_ja": "伏魔御廚子",
            "name_en": "MALEVOLENT SHRINE",
            "primary_bgr": (40, 40, 240),       # Blood Crimson
            "secondary_bgr": (30, 140, 255),    # Hellish Orange / Gold
            "bg_dark": (15, 10, 30),            # Obsidian
            "bg_accent": (25, 20, 85),          # Deep blood red
            "stars_count": 50,
        }
    }

    def __init__(self, width: int = 640, height: int = 480, theme: str = "infinite_void"):
        self.w = width
        self.h = height
        self.theme_key = theme
        self.theme = self.THEMES.get(theme, self.THEMES["infinite_void"])
        self.frame_count = 0

        # Pre-compute static distance matrix for fast radial gradients
        y, x = np.ogrid[:self.h, :self.w]
        self.cx, self.cy = self.w // 2, self.h // 2
        self.dist_map = np.sqrt((x - self.cx)**2 + (y - self.cy)**2).astype(np.float32)
        self.max_dist = math.sqrt(self.cx**2 + self.cy**2)

        # Star / cosmic dust positions
        np.random.seed(42)
        self.stars = [
            (
                np.random.randint(0, self.w),
                np.random.randint(0, self.h),
                np.random.uniform(1.0, 2.5),
                np.random.uniform(0.5, 2.0)  # twinkle speed
            )
            for _ in range(self.theme["stars_count"])
        ]

    def set_theme(self, theme_key: str):
        if theme_key in self.THEMES:
            self.theme_key = theme_key
            self.theme = self.THEMES[theme_key]

    def render(self, timer: int = 0) -> np.ndarray:
        """Render a single frame of the animated domain background."""
        self.frame_count = timer

        # 1. Base radial gradient
        norm_dist = np.clip(self.dist_map / self.max_dist, 0.0, 1.0)
        # Pulse center brightness
        pulse = 0.85 + 0.15 * math.sin(timer * 0.06)

        bg_dark = np.array(self.theme["bg_dark"], dtype=np.float32)
        bg_accent = np.array(self.theme["bg_accent"], dtype=np.float32) * pulse

        # Inner center has bg_accent, outer edges fade to bg_dark
        grad = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for c in range(3):
            ch = bg_accent[c] * (1.0 - norm_dist**1.5) + bg_dark[c] * (norm_dist**1.5)
            grad[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # 2. Celestial / cosmic dust particles
        for sx, sy, size, speed in self.stars:
            twinkle = 0.5 + 0.5 * math.sin(timer * 0.1 * speed + sx)
            c_val = int(220 * twinkle)
            color = (
                min(255, int(self.theme["secondary_bgr"][0] * 0.4 + c_val * 0.6)),
                min(255, int(self.theme["secondary_bgr"][1] * 0.4 + c_val * 0.6)),
                min(255, int(self.theme["secondary_bgr"][2] * 0.4 + c_val * 0.6)),
            )
            cv2.circle(grad, (sx, sy), max(1, int(size * twinkle)), color, -1)

        # 3. Concentric animated barrier rings
        ring_canvas = np.zeros_like(grad)
        color_primary = self.theme["primary_bgr"]
        color_sec = self.theme["secondary_bgr"]

        # Outer barrier ring
        r_outer = int(min(self.w, self.h) * 0.45 + 10 * math.sin(timer * 0.04))
        cv2.circle(ring_canvas, (self.cx, self.cy), r_outer, color_primary, 3)

        # Mid ring with rotating tick marks
        r_mid = int(min(self.w, self.h) * 0.33)
        cv2.circle(ring_canvas, (self.cx, self.cy), r_mid, color_sec, 2)

        # Rotating tick marks (astrolabe / cursed seal)
        num_ticks = 24
        rot_angle = timer * 0.02
        for i in range(num_ticks):
            angle = rot_angle + (2 * math.pi * i / num_ticks)
            tick_len = 10 if i % 3 == 0 else 5
            x1 = int(self.cx + (r_mid - tick_len) * math.cos(angle))
            y1 = int(self.cy + (r_mid - tick_len) * math.sin(angle))
            x2 = int(self.cx + (r_mid + tick_len) * math.cos(angle))
            y2 = int(self.cy + (r_mid + tick_len) * math.sin(angle))
            cv2.line(ring_canvas, (x1, y1), (x2, y2), color_sec, 2 if i % 3 == 0 else 1)

        # Inner pulsing core ring
        r_inner = int(min(self.w, self.h) * 0.2 + 8 * math.cos(timer * 0.08))
        cv2.circle(ring_canvas, (self.cx, self.cy), r_inner, color_primary, 2)

        # Additional counter-rotating dashed ring
        r_dash = int(min(self.w, self.h) * 0.26)
        num_dashes = 16
        dash_angle = -timer * 0.015
        for i in range(num_dashes):
            a1 = dash_angle + (2 * math.pi * i / num_dashes)
            a2 = a1 + (math.pi / num_dashes) * 0.7
            cv2.ellipse(
                ring_canvas,
                (self.cx, self.cy),
                (r_dash, r_dash),
                0,
                math.degrees(a1),
                math.degrees(a2),
                color_sec,
                2
            )

        # Apply multi-pass bloom glow to barrier rings
        blurred_rings = cv2.GaussianBlur(ring_canvas, (21, 21), 0)
        grad = cv2.add(grad, blurred_rings)
        grad = cv2.add(grad, ring_canvas)

        return grad
