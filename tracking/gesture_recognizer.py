import math
from typing import List, Dict, Any, Tuple, Optional
from config import SIGN_HOLD_FRAMES_REQUIRED, SIGN_CONFIDENCE_THRESHOLD


class CanonicalGestureRecognizer:
    """
    Recognizes canonical Jujutsu Kaisen Domain Expansion hand signs:
    1. Sukuna's Malevolent Shrine (Enma-ten / Yama Mudra)
    2. Gojo's Infinite Void (Taishakuten Mudra)
    Includes temporal stability hold counter and confidence scoring.
    """

    def __init__(
        self,
        hold_frames_required: int = SIGN_HOLD_FRAMES_REQUIRED,
        confidence_threshold: float = SIGN_CONFIDENCE_THRESHOLD,
    ):
        self.hold_frames_required = hold_frames_required
        self.confidence_threshold = confidence_threshold
        self.stable_frames = 0
        self.detected_sign_name = ""
        self.cooldown = 0

    def evaluate_sukuna_mudra(self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Sukuna's Malevolent Shrine:
        - Two hands clasped/touching in front of chest
        - Hands pointing upward (dir_y < -0.3)
        - Thumbs upright/extended
        - Index fingers pointing up / touching
        - Lower fingers (middle, ring, pinky) curled or bent inward
        """
        if len(hands) < 2:
            return 0.0, None

        h1, h2 = hands[0], hands[1]
        p1, p2 = h1["palm_center"], h2["palm_center"]
        avg_scale = (h1["palm_scale"] + h2["palm_scale"]) / 2.0

        score = 0.0

        # 1. Palms proximity (should be close together)
        palm_dist = math.dist(p1, p2)
        if palm_dist < avg_scale * 2.2:
            score += 0.25
        elif palm_dist < avg_scale * 3.5:
            score += 0.12

        # 2. Upward orientation (hands pointing up)
        if h1["pointing_dir"][1] < -0.2 and h2["pointing_dir"][1] < -0.2:
            score += 0.25
        elif h1["pointing_dir"][1] < 0.1 and h2["pointing_dir"][1] < 0.1:
            score += 0.12

        # 3. Thumbs extended
        t1_ext = h1["finger_states"]["thumb"] in ["EXTENDED", "BENT"]
        t2_ext = h2["finger_states"]["thumb"] in ["EXTENDED", "BENT"]
        if t1_ext and t2_ext:
            score += 0.20
        elif t1_ext or t2_ext:
            score += 0.10

        # 4. Index fingers extended or touching
        idx_dist = math.dist(h1["index_tip"], h2["index_tip"])
        i1_ext = h1["finger_states"]["index"] == "EXTENDED"
        i2_ext = h2["finger_states"]["index"] == "EXTENDED"
        if (i1_ext and i2_ext) or (idx_dist < avg_scale * 1.5):
            score += 0.20

        # 5. Lower fingers curled (ring / pinky bent or curled)
        curled_count = 0
        for h in [h1, h2]:
            if h["finger_states"]["ring"] in ["CURLED", "BENT"]:
                curled_count += 1
            if h["finger_states"]["pinky"] in ["CURLED", "BENT"]:
                curled_count += 1
        score += min(0.10, (curled_count / 4.0) * 0.10)

        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        return min(1.0, score), center

    def evaluate_gojo_mudra(self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Gojo's Infinite Void:
        - Index & Middle fingers extended with fingertips crossed/overlapping
        - Ring & Pinky curled tightly into palm
        - Thumb folded
        """
        best_score = 0.0
        best_center = None

        for h in hands:
            score = 0.0
            fs = h["finger_states"]

            # 1. Index & Middle extended
            if fs["index"] == "EXTENDED" and fs["middle"] == "EXTENDED":
                score += 0.35
            elif fs["index"] == "EXTENDED" or fs["middle"] == "EXTENDED":
                score += 0.15

            # 2. Ring & Pinky curled tightly
            if fs["ring"] == "CURLED" and fs["pinky"] == "CURLED":
                score += 0.35
            elif fs["ring"] in ["CURLED", "BENT"] and fs["pinky"] in ["CURLED", "BENT"]:
                score += 0.20

            # 3. Crossing index and middle fingers (tips close together)
            cross_dist = math.dist(h["index_tip"], h["middle_tip"])
            if cross_dist < h["palm_scale"] * 0.55:
                score += 0.20

            # 4. Pointing generally upward
            if h["pointing_dir"][1] < -0.3:
                score += 0.10

            if score > best_score:
                best_score = score
                best_center = h["palm_center"]

        return min(1.0, best_score), best_center

    def update(
        self,
        hands: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        target_theme: str = "malevolent_shrine",
    ) -> Dict[str, Any]:
        """
        Update temporal hold counter and evaluate whether the domain triggers.
        """
        if self.cooldown > 0:
            self.cooldown -= 1

        # Evaluate target hand sign
        if target_theme == "malevolent_shrine":
            score, center = self.evaluate_sukuna_mudra(hands, frame_shape)
            sign_name = "Sukuna (Enma-ten Mudra)"
        else:
            score, center = self.evaluate_gojo_mudra(hands, frame_shape)
            sign_name = "Gojo (Taishakuten Mudra)"

        # Fallback check: if hands are together or held near chest with high score on other mudra
        if score < 0.65:
            alt_score, alt_center = (
                self.evaluate_gojo_mudra(hands, frame_shape)
                if target_theme == "malevolent_shrine"
                else self.evaluate_sukuna_mudra(hands, frame_shape)
            )
            if alt_score > score:
                score = alt_score
                center = alt_center
                sign_name = "Domain Hand Sign"

        is_matching = score >= self.confidence_threshold
        if is_matching:
            self.stable_frames += 1
            self.detected_sign_name = sign_name
        else:
            # Graceful decay: doesn't reset instantly if 1 frame drops tracking
            self.stable_frames = max(0, self.stable_frames - 2)

        hold_progress = min(1.0, self.stable_frames / float(self.hold_frames_required))
        triggered = False

        if self.stable_frames >= self.hold_frames_required and self.cooldown == 0:
            triggered = True
            self.cooldown = 45

        return {
            "sign_detected": is_matching,
            "sign_name": self.detected_sign_name if is_matching else "",
            "confidence": score,
            "hold_progress": hold_progress,
            "stable_frames": self.stable_frames,
            "required_frames": self.hold_frames_required,
            "trigger": triggered,
            "energy_center": center or (frame_shape[1] // 2, int(frame_shape[0] * 0.5)),
        }

    def reset(self):
        self.stable_frames = 0
        self.cooldown = 0
