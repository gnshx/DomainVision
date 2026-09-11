import math
from typing import Tuple, Optional
import cv2
import numpy as np


class SyntheticDemoCamera:
    """
    Generates a realistic animated synthetic video feed of a person performing
    the Domain Expansion gesture. Allows complete end-to-end testing and demonstration
    even in headless environments or systems without a physical webcam.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.w = width
        self.h = height
        self.fps = fps
        self.frame_idx = 0
        self.is_open = True
        self.state = "NORMAL"

    def set_state(self, state: str):
        """Syncs demo character pose with domain expansion state."""
        self.state = state

    def isOpened(self) -> bool:
        return self.is_open

    def release(self):
        self.is_open = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open:
            return False, None

        # Background: Modern studio room gradient (vectorized)
        ratios = np.linspace(0, 1, self.h, endpoint=False, dtype=np.float32)[:, None, None]
        bg_top = np.array([45, 40, 35], dtype=np.float32).reshape(1, 1, 3)
        bg_bot = np.array([25, 20, 18], dtype=np.float32).reshape(1, 1, 3)
        frame = (bg_top * (1.0 - ratios) + bg_bot * ratios).astype(np.uint8)
        frame = np.repeat(frame, self.w, axis=1)

        # Subtle room ambient details (shelves / lines)
        cv2.line(frame, (0, int(self.h * 0.7)), (self.w, int(self.h * 0.7)), (35, 30, 25), 2)
        cv2.line(frame, (int(self.w * 0.2), 0), (int(self.w * 0.2), self.h), (30, 25, 22), 1)
        cv2.line(frame, (int(self.w * 0.8), 0), (int(self.w * 0.8), self.h), (30, 25, 22), 1)

        t = self.frame_idx
        # Character center
        cx = self.w // 2
        cy = int(self.h * 0.55)

        # Breathing / subtle body sway
        sway_y = int(3 * math.sin(t * 0.08))
        sway_x = int(2 * math.sin(t * 0.04))
        cx += sway_x
        cy += sway_y

        left_shoulder = (cx - 70, cy - 60)
        right_shoulder = (cx + 70, cy - 60)

        # Determine arm pose based on Domain Expansion state:
        # NORMAL: Natural relaxed idle stance (NEVER auto-triggers expansion)
        # CHARGING / EXPANSION / DOMAIN_ACTIVE: Hands clasped at chest in mudra pose
        # COLLAPSE: Lowering hands back down
        if self.state == "NORMAL":
            left_hand = (cx - 85, cy + 90)
            right_hand = (cx + 85, cy + 90)
            left_elbow = (cx - 95, cy + 15)
            right_elbow = (cx + 95, cy + 15)
        elif self.state in ["CHARGING", "FLASH", "EXPANSION", "DOMAIN_ACTIVE"]:
            pulse = int(2 * math.sin(t * 0.3))
            left_hand = (cx - 15 + pulse, cy - 40)
            right_hand = (cx + 15 - pulse, cy - 40)
            left_elbow = (cx - 75, cy - 15)
            right_elbow = (cx + 75, cy - 15)
        elif self.state == "COLLAPSE":
            left_hand = (cx - 60, cy + 50)
            right_hand = (cx + 60, cy + 50)
            left_elbow = (cx - 85, cy + 15)
            right_elbow = (cx + 85, cy + 15)
        else:
            left_hand = (cx - 85, cy + 90)
            right_hand = (cx + 85, cy + 90)
            left_elbow = (cx - 95, cy + 15)
            right_elbow = (cx + 95, cy + 15)

        # Draw character:
        # Torso / Clothes (dark jujutsu tech uniform style: navy/black)
        body_color = (25, 20, 20)
        torso_pts = np.array([
            [cx - 70, cy - 60],
            [cx + 70, cy - 60],
            [cx + 95, self.h],
            [cx - 95, self.h]
        ], np.int32)
        cv2.fillPoly(frame, [torso_pts], body_color)

        # High collar / uniform jacket
        collar_pts = np.array([
            [cx - 40, cy - 85],
            [cx + 40, cy - 85],
            [cx + 25, cy - 50],
            [cx - 25, cy - 50]
        ], np.int32)
        cv2.fillPoly(frame, [collar_pts], (35, 28, 28))
        cv2.circle(frame, (cx, cy - 55), 4, (180, 160, 60), -1)  # Gold jujutsu tech button

        # Head & Hair (Gojo-style white/silver spiked hair or dark hair)
        head_center = (cx, cy - 125)
        # Neck
        cv2.rectangle(frame, (cx - 22, cy - 95), (cx + 22, cy - 65), (175, 195, 225), -1)
        # Face (flesh tone in BGR)
        skin_color = (185, 205, 235)
        cv2.ellipse(frame, head_center, (35, 45), 0, 0, 360, skin_color, -1)

        # Hair (stylish anime spikes)
        hair_color = (230, 235, 245)  # Silver/white hair
        hair_pts = np.array([
            [cx - 42, cy - 120],
            [cx - 55, cy - 145],
            [cx - 35, cy - 150],
            [cx - 30, cy - 175],
            [cx - 10, cy - 155],
            [cx, cy - 180],
            [cx + 15, cy - 155],
            [cx + 35, cy - 170],
            [cx + 40, cy - 145],
            [cx + 52, cy - 120],
            [cx + 38, cy - 105],
            [cx - 38, cy - 105]
        ], np.int32)
        cv2.fillPoly(frame, [hair_pts], hair_color)

        # Blindfold / Eyes (Gojo's iconic dark blindfold)
        cv2.rectangle(frame, (cx - 33, cy - 135), (cx + 33, cy - 118), (15, 15, 18), -1)
        # Subtle nose and mouth
        cv2.line(frame, (cx, cy - 112), (cx, cy - 108), (150, 170, 200), 1)
        cv2.line(frame, (cx - 6, cy - 100), (cx + 6, cy - 100), (140, 160, 190), 2)

        # Draw Arms (sleeves)
        cv2.line(frame, left_shoulder, left_elbow, body_color, 24)
        cv2.line(frame, left_elbow, left_hand, body_color, 20)
        cv2.line(frame, right_shoulder, right_elbow, body_color, 24)
        cv2.line(frame, right_elbow, right_hand, body_color, 20)

        # Draw Hands (flesh tone circles/ellipses)
        cv2.circle(frame, left_hand, 16, skin_color, -1)
        cv2.circle(frame, right_hand, 16, skin_color, -1)
        # Fingers
        cv2.circle(frame, (left_hand[0] + 5, left_hand[1] - 8), 6, skin_color, -1)
        cv2.circle(frame, (right_hand[0] - 5, right_hand[1] - 8), 6, skin_color, -1)

        self.frame_idx += 1
        return True, frame
