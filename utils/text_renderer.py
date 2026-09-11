import os
import math
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class JapaneseTextRenderer:
    """Renders authentic Japanese typography with glow, drop shadow, and alpha blending."""

    CANDIDATE_FONTS = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_best_font()
        self._font_cache = {}

    def _find_best_font(self) -> Optional[str]:
        for path in self.CANDIDATE_FONTS:
            if os.path.exists(path):
                return path
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            if self.font_path and os.path.exists(self.font_path):
                try:
                    self._font_cache[size] = ImageFont.truetype(self.font_path, size)
                except Exception:
                    self._font_cache[size] = ImageFont.load_default()
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def render_domain_banner(
        self,
        frame_shape: Tuple[int, int],
        main_text: str = "領域展開",
        sub_text: str = "無量空処",
        en_text: str = "DOMAIN EXPANSION",
        color_glow: Tuple[int, int, int] = (255, 50, 180),  # BGR
        color_fill: Tuple[int, int, int] = (255, 255, 255),  # BGR
        progress: float = 1.0,  # 0.0 to 1.0
    ) -> np.ndarray:
        """
        Renders a cinematic domain activation banner on a transparent RGBA overlay.
        Returns BGR overlay and single-channel alpha mask.
        """
        h, w = frame_shape[:2]
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        main_size = max(36, int(w * 0.075))
        sub_size = max(24, int(w * 0.045))
        en_size = max(14, int(w * 0.022))

        main_font = self._get_font(main_size)
        sub_font = self._get_font(sub_size)
        en_font = self._get_font(en_size)

        # Calculate bounding boxes
        main_bbox = draw.textbbox((0, 0), main_text, font=main_font)
        main_w = main_bbox[2] - main_bbox[0]
        main_h = main_bbox[3] - main_bbox[1]

        sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_h = sub_bbox[3] - sub_bbox[1]

        en_bbox = draw.textbbox((0, 0), en_text, font=en_font)
        en_w = en_bbox[2] - en_bbox[0]
        en_h = en_bbox[3] - en_bbox[1]

        # Positions (centered in upper third or center)
        center_x = w // 2
        center_y = int(h * 0.28)

        main_x = center_x - main_w // 2
        main_y = center_y - main_h // 2

        sub_x = center_x - sub_w // 2
        sub_y = main_y + main_h + int(h * 0.015)

        en_x = center_x - en_w // 2
        en_y = main_y - en_h - int(h * 0.02)

        # Convert BGR to RGBA for Pillow
        glow_rgba = (color_glow[2], color_glow[1], color_glow[0], int(220 * progress))
        fill_rgba = (color_fill[2], color_fill[1], color_fill[0], int(255 * progress))
        shadow_rgba = (0, 0, 0, int(200 * progress))

        # 1. Soft feathered backing glow band
        band_h = int((main_h + sub_h + en_h) * 1.8)
        band_y1 = max(0, en_y - int(h * 0.04))
        band_y2 = min(h, sub_y + sub_h + int(h * 0.04))
        # Draw gradient backing that smoothly fades at top and bottom
        for by in range(band_y1, band_y2):
            rel_y = (by - band_y1) / float(max(1, band_y2 - band_y1))
            fade = math.sin(rel_y * math.pi)  # 0 -> 1 -> 0 sine envelope
            draw.line([(0, by), (w, by)], fill=(12, 6, 18, int(160 * progress * fade)))

        # 2. Draw thick glow outline
        glow_radius = max(3, int(main_size * 0.08))
        for dx in range(-glow_radius, glow_radius + 1, 2):
            for dy in range(-glow_radius, glow_radius + 1, 2):
                if dx * dx + dy * dy <= glow_radius * glow_radius:
                    draw.text((main_x + dx, main_y + dy), main_text, font=main_font, fill=glow_rgba)
                    draw.text((sub_x + dx, sub_y + dy), sub_text, font=sub_font, fill=glow_rgba)

        # 3. Draw drop shadows
        draw.text((en_x + 2, en_y + 2), en_text, font=en_font, fill=shadow_rgba)
        draw.text((main_x + 4, main_y + 4), main_text, font=main_font, fill=shadow_rgba)
        draw.text((sub_x + 3, sub_y + 3), sub_text, font=sub_font, fill=shadow_rgba)

        # 4. Draw crisp foreground text
        draw.text((en_x, en_y), en_text, font=en_font, fill=glow_rgba)
        draw.text((main_x, main_y), main_text, font=main_font, fill=fill_rgba)
        draw.text((sub_x, sub_y), sub_text, font=sub_font, fill=fill_rgba)

        # Convert to numpy array (RGBA -> BGR + Alpha)
        arr = np.array(canvas)
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = arr[:, :, 3]

        return bgr, alpha
