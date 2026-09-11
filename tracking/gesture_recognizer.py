import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from config import SIGN_HOLD_FRAMES_REQUIRED, SIGN_CONFIDENCE_THRESHOLD


class CanonicalGestureRecognizer:
    """
    Recognizes canonical Jujutsu Kaisen Domain Expansion hand signs:
    1. Sukuna's Malevolent Shrine (Enma-ten / Yama Mudra - 伏魔御廚子)
    2. Gojo's Infinite Void (Taishakuten Mudra - 無量空処)

    Evaluates exact 21-landmark geometry:
    - Thumb, index, middle, ring, pinky 3D joint angles
    - Inter-finger distances & crossing proximities
    - Palm normal & upward direction
    - Wrist orientation vector
    - Temporal hold stability and match percentage (0-100%)
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
        self.last_match_pct = 0
        self.last_sukuna_pct = 0
        self.last_gojo_pct = 0

    def evaluate_sukuna_mudra(
        self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> Tuple[float, Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        Sukuna's Enma-ten Mudra (伏魔御廚子):
        - Two hands clasped / brought together in front of chest/face
        - Thumbs upright and extended (angle > 140°)
        - Index fingers extended upward and touching / close
        - Middle fingers bent inward or touching at tips
        - Ring and pinky fingers curled into palm (angle < 125°)
        - Palms oriented facing each other, wrists pointing up
        """
        if len(hands) < 2:
            # Single hand visible: partial score if one hand forms half of the mudra
            if len(hands) == 1:
                h = hands[0]
                ang = h.get("finger_angles", {})
                t_ext = ang.get("thumb", 0) > 135
                i_ext = ang.get("index", 0) > 140
                r_cur = ang.get("ring", 180) < 130
                p_cur = ang.get("pinky", 180) < 130
                single_score = 0.25 * t_ext + 0.25 * i_ext + 0.15 * r_cur + 0.15 * p_cur
                return single_score * 0.55, h.get("palm_center"), {"hands_count": 1}
            return 0.0, None, {"hands_count": 0}

        h1, h2 = hands[0], hands[1]
        p1, p2 = h1["palm_center"], h2["palm_center"]
        avg_scale = (h1.get("palm_scale", 30) + h2.get("palm_scale", 30)) / 2.0

        metrics = {}
        score = 0.0

        # 1. Palm-to-palm proximity & center calculation
        palm_dist = math.dist(p1, p2)
        metrics["palm_dist_ratio"] = palm_dist / avg_scale
        if palm_dist < avg_scale * 2.2:
            score += 0.22
        elif palm_dist < avg_scale * 3.4:
            score += 0.12

        # 2. Upward wrist and hand orientation
        dir1_y = h1.get("pointing_dir", (0, 0))[1]
        dir2_y = h2.get("pointing_dir", (0, 0))[1]
        metrics["upward_dir"] = (dir1_y + dir2_y) / 2.0
        if dir1_y < -0.20 and dir2_y < -0.20:
            score += 0.18
        elif dir1_y < 0.05 and dir2_y < 0.05:
            score += 0.10

        # 3. Palm orientation (facing each other)
        norm1 = h1.get("palm_normal", (0, 0, 1))
        norm2 = h2.get("palm_normal", (0, 0, 1))
        dot_normals = norm1[0] * norm2[0] + norm1[1] * norm2[1] + norm1[2] * norm2[2]
        metrics["dot_normals"] = dot_normals
        # When palms face each other, normal vectors point towards each other (dot < 0)
        if dot_normals < -0.15:
            score += 0.15
        elif dot_normals < 0.20:
            score += 0.08

        # 4. Thumbs upright and extended
        ang1 = h1.get("finger_angles", {})
        ang2 = h2.get("finger_angles", {})
        t1_ext = ang1.get("thumb", 0) > 135 or h1.get("finger_states", {}).get("thumb") in ["EXTENDED", "BENT"]
        t2_ext = ang2.get("thumb", 0) > 135 or h2.get("finger_states", {}).get("thumb") in ["EXTENDED", "BENT"]
        if t1_ext and t2_ext:
            score += 0.15
        elif t1_ext or t2_ext:
            score += 0.08

        # 5. Index fingers extended & touching
        idx_dist = math.dist(h1["index_tip"], h2["index_tip"])
        metrics["index_dist_ratio"] = idx_dist / avg_scale
        i1_ext = ang1.get("index", 0) > 140
        i2_ext = ang2.get("index", 0) > 140
        if (i1_ext and i2_ext) or (idx_dist < avg_scale * 1.6):
            score += 0.18
        elif i1_ext or i2_ext:
            score += 0.09

        # 6. Lower fingers curled (ring & pinky)
        curled_count = 0
        for h in [h1, h2]:
            fs = h.get("finger_states", {})
            fa = h.get("finger_angles", {})
            if fs.get("ring") in ["CURLED", "BENT"] or fa.get("ring", 180) < 130:
                curled_count += 1
            if fs.get("pinky") in ["CURLED", "BENT"] or fa.get("pinky", 180) < 130:
                curled_count += 1
        score += min(0.12, (curled_count / 4.0) * 0.12)

        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        return min(1.0, score), center, metrics

    def evaluate_gojo_mudra(
        self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> Tuple[float, Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        Gojo's Taishakuten Mudra (無量空処):
        - Index & Middle fingers extended with fingertips crossed / overlapping
        - Ring & Pinky curled tightly into palm
        - Thumb folded inward
        - Hand held vertically pointing upward
        """
        if not hands:
            return 0.0, None, {"hands_count": 0}

        best_score = 0.0
        best_center = None
        best_metrics = {}

        for h in hands:
            score = 0.0
            fs = h.get("finger_states", {})
            fa = h.get("finger_angles", {})
            scale = max(10.0, h.get("palm_scale", 30))

            # 1. Index & Middle extended (angles > 145°)
            i_ext = fa.get("index", 0) > 145 or fs.get("index") == "EXTENDED"
            m_ext = fa.get("middle", 0) > 145 or fs.get("middle") == "EXTENDED"
            if i_ext and m_ext:
                score += 0.32
            elif i_ext or m_ext:
                score += 0.15

            # 2. Ring & Pinky curled tightly (angles < 120°)
            r_cur = fa.get("ring", 180) < 120 or fs.get("ring") == "CURLED"
            p_cur = fa.get("pinky", 180) < 120 or fs.get("pinky") == "CURLED"
            if r_cur and p_cur:
                score += 0.30
            elif r_cur or p_cur:
                score += 0.16

            # 3. Crossing index and middle fingers (tips close together)
            cross_dist = math.dist(h["index_tip"], h["middle_tip"])
            cross_ratio = cross_dist / scale
            if cross_ratio < 0.50:
                score += 0.22
            elif cross_ratio < 0.85:
                score += 0.12

            # 4. Hand pointing upward
            dir_y = h.get("pointing_dir", (0, 0))[1]
            if dir_y < -0.28:
                score += 0.16
            elif dir_y < 0.0:
                score += 0.08

            if score > best_score:
                best_score = score
                best_center = h["palm_center"]
                best_metrics = {
                    "cross_ratio": cross_ratio,
                    "dir_y": dir_y,
                    "index_angle": fa.get("index", 0),
                    "middle_angle": fa.get("middle", 0),
                }

        return min(1.0, best_score), best_center, best_metrics

    def update(
        self,
        hands: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        target_theme: str = "malevolent_shrine",
    ) -> Dict[str, Any]:
        """
        Evaluates current hand gesture against canonical mudras.
        Returns match percentage, status message, hold progress, and trigger flag.
        """
        if self.cooldown > 0:
            self.cooldown -= 1

        # Evaluate both Sukuna and Gojo canonical finger signs concurrently
        score_sukuna, center_sukuna, metrics_sukuna = self.evaluate_sukuna_mudra(hands, frame_shape)
        score_gojo, center_gojo, metrics_gojo = self.evaluate_gojo_mudra(hands, frame_shape)

        # Automatically select character theme based on which hand sign is being performed
        if score_sukuna >= score_gojo:
            active_theme = "malevolent_shrine"
            score = score_sukuna
            center = center_sukuna
            sign_name = "Sukuna (Enma-ten Mudra)"
            metrics = metrics_sukuna
        else:
            active_theme = "infinite_void"
            score = score_gojo
            center = center_gojo
            sign_name = "Gojo (Taishakuten Mudra)"
            metrics = metrics_gojo

        # Reset hold stability if user dynamically switches between the two hand signs
        if hasattr(self, "_last_theme") and self._last_theme != active_theme and score >= self.confidence_threshold:
            self.stable_frames = 0
        self._last_theme = active_theme

        match_pct = int(score * 100.0)
        self.last_match_pct = match_pct
        self.last_sukuna_pct = int(score_sukuna * 100.0)
        self.last_gojo_pct = int(score_gojo * 100.0)
        is_matching = score >= self.confidence_threshold

        # Update stable frames hold counter
        if is_matching:
            self.stable_frames += 1
            self.detected_sign_name = sign_name
        else:
            # Graceful decay: drops by 2 rather than instant reset if 1 frame drops
            self.stable_frames = max(0, self.stable_frames - 2)

        hold_progress = min(1.0, self.stable_frames / float(self.hold_frames_required))
        triggered = False

        if self.stable_frames >= self.hold_frames_required and self.cooldown == 0:
            triggered = True
            self.cooldown = 50

        # State text generation
        if not hands:
            status_text = "Make hand sign (Sukuna or Gojo)"
        elif not is_matching:
            if match_pct > 35:
                status_text = f"{sign_name.split()[0]}: {match_pct}% (adjust sign)"
            else:
                status_text = "Make hand sign (Sukuna or Gojo)"
        else:
            if triggered:
                status_text = f">> DOMAIN EXPANSION: {active_theme.upper().replace('_', ' ')} <<"
            elif hold_progress >= 0.95:
                status_text = f"HOLD... {sign_name.split()[0]} {match_pct}% (ACTIVATING)"
            else:
                status_text = f"HOLD... {sign_name.split()[0]} {match_pct}% ({self.stable_frames}/{self.hold_frames_required})"

        fallback_center = (frame_shape[1] // 2, int(frame_shape[0] * 0.52))
        return {
            "sign_detected": is_matching,
            "detected_theme": active_theme,
            "sign_name": self.detected_sign_name if is_matching else sign_name,
            "confidence": score,
            "match_pct": match_pct,
            "sukuna_score": score_sukuna,
            "gojo_score": score_gojo,
            "status_text": status_text,
            "hold_progress": hold_progress,
            "stable_frames": self.stable_frames,
            "required_frames": self.hold_frames_required,
            "trigger": triggered,
            "energy_center": center or fallback_center,
            "metrics": metrics,
        }

    def reset(self):
        self.stable_frames = 0
        self.cooldown = 0
        self.detected_sign_name = ""
        self.last_match_pct = 0
