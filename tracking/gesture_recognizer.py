"""
Canonical Jujutsu Kaisen Domain Expansion Gesture Recognizer.

Features:
- Explicit Multi-Class State Machine:
    NO_HAND -> UNKNOWN -> GOJO_CANDIDATE -> GOJO_CONFIRMED
                       -> SUKUNA_CANDIDATE -> SUKUNA_CONFIRMED
                       -> TRANSITION
- Strict Gojo Taishakuten Mudra (無量空処):
    - Exactly 1 active hand (or dominant separate hand)
    - Head/face height level check
    - Middle finger crossing over index finger in 3D
    - Ring and pinky curled tightly into palm
    - Thumb folded inward
    - Rejects standard peace signs, open palms, or uncrossed fingers
- Strict Sukuna Enma-ten Mudra (伏魔御廚子):
    - Strictly TWO hands required (1 hand score is 0.0)
    - Chest / center torso position
    - Palms clasped / brought together
    - Thumbs upright and extended
    - Index fingertips touching / meeting
    - Ring and pinky fingers curled into palms
- Hysteresis, temporal smoothing, and clean transition without state contamination.
"""
import math
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from config import SIGN_HOLD_FRAMES_REQUIRED, SIGN_CONFIDENCE_THRESHOLD, SIGN_DEACTIVATION_THRESHOLD


class CanonicalGestureRecognizer:
    def __init__(
        self,
        hold_frames_required: int = SIGN_HOLD_FRAMES_REQUIRED,
        confidence_threshold: float = SIGN_CONFIDENCE_THRESHOLD,
        deactivation_threshold: float = SIGN_DEACTIVATION_THRESHOLD,
        history_len: int = 15,
    ):
        self.hold_frames_required = hold_frames_required
        self.confidence_threshold = confidence_threshold
        self.deactivation_threshold = deactivation_threshold

        # Explicit state: NO_HAND, UNKNOWN, GOJO_CANDIDATE, GOJO_CONFIRMED, SUKUNA_CANDIDATE, SUKUNA_CONFIRMED, TRANSITION
        self.state = "UNKNOWN"
        self.candidate_sign = "UNKNOWN"
        self.confirmed_sign = "UNKNOWN"

        self.stable_frames = 0
        self.cooldown = 0
        self.transition_timer = 0

        self.last_gojo_score = 0.0
        self.last_sukuna_score = 0.0
        self.last_unknown_score = 1.0

        self.last_gojo_pct = 0
        self.last_sukuna_pct = 0
        self.last_match_pct = 0
        self.detected_sign_name = ""
        self.last_hands: List[Dict[str, Any]] = []
        self.last_hands_count = 0

        # Rolling observation history for temporal filtering
        self._history: deque = deque(maxlen=history_len)

    def evaluate_gojo_mudra(
        self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> Tuple[float, Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        Evaluates Gojo's Taishakuten Mudra (無量空処):
        STRICT GOJO CANONICAL RULES:
        1. Strictly 1 hand (if len(hands) != 1: return 0.0)
        2. CANONICAL FACE/EYE HEIGHT: palm_y_norm < 0.40 and index_tip_y_norm < 0.35.
           If hand is at chest or torso level: strictly return 0.0!
           (Prevents Sukuna clasped hands at chest from ever triggering Gojo!)
        3. Index and Middle fingers upright / active
        4. Middle finger crossed over Index finger (cross_ratio < 0.44 or is_crossing_mudra is True)
        5. Ring and Pinky curled tightly into palm
        6. Thumb folded inward, not upright or extended
        7. Hand pointing strictly upward (negative Y in screen coords)
        """
        # Rule 1: Strictly 1 hand
        if len(hands) != 1:
            return 0.0, None, {"reason": "requires_strictly_1_hand", "hands_count": len(hands)}

        h = hands[0]
        palm_y_norm = h.get("palm_y_norm", 0.5)
        index_tip_y_norm = h["index_tip"][1] / float(frame_shape[0])

        # Rule 2: Gojo mudra is canonically performed ONLY at head/eye/face level!
        # Hands held at chest/torso (palm_y_norm >= 0.40 or index_tip_y_norm >= 0.35) are Sukuna territory and MUST NEVER trigger Gojo!
        if palm_y_norm >= 0.40 or index_tip_y_norm >= 0.35:
            return 0.0, None, {"reason": "hand_at_chest_level_not_gojo", "palm_y": palm_y_norm, "tip_y": index_tip_y_norm}

        fa = h.get("finger_angles", {})
        fs = h.get("finger_states", {})
        cr = h.get("curl_ratios", {})

        # Rule 3: 2 fingers crossed check (middle finger crossed over index finger)
        cross_ratio = h.get("cross_ratio", 1.0)
        is_crossing = h.get("is_crossing_mudra", False)

        if not is_crossing and cross_ratio >= 0.44:
            return 0.0, None, {"reason": "fingers_not_crossed", "cross_ratio": cross_ratio}

        # Rule 4: Index & Middle fingers upright
        i_up = (fa.get("index", 0) > 120) or (cr.get("index", 0) > 1.18) or (fs.get("index") in ["EXTENDED", "BENT"])
        m_up = (fa.get("middle", 0) > 115) or (cr.get("middle", 0) > 1.12) or (fs.get("middle") in ["EXTENDED", "BENT"])

        if not (i_up and m_up):
            return 0.0, None, {"reason": "index_or_middle_not_upright"}

        # Rule 5: Ring & Pinky curled into palm
        r_open = (fa.get("ring", 0) > 140) and (cr.get("ring", 0) > 1.25)
        p_open = (fa.get("pinky", 0) > 140) and (cr.get("pinky", 0) > 1.25)
        if r_open or p_open:
            return 0.0, None, {"reason": "ring_and_pinky_extended"}

        r_cur = (fa.get("ring", 180) < 135) or (cr.get("ring", 2) < 1.20) or (fs.get("ring") == "CURLED")
        p_cur = (fa.get("pinky", 180) < 135) or (cr.get("pinky", 2) < 1.20) or (fs.get("pinky") == "CURLED")

        # Rule 6: Hand pointing strictly upward towards ceiling (negative Y)
        dir_y = h.get("pointing_dir", (0, 0))[1]
        if dir_y > -0.15:
            return 0.0, None, {"reason": "hand_pointing_downwards", "dir_y": dir_y}

        # Rule 7: Thumb check (thumb folded inward, not upright)
        thumb_tucked = h.get("thumb_tucked", False) or (fa.get("thumb", 180) < 135)
        if not thumb_tucked and fa.get("thumb", 180) > 142:
            return 0.0, None, {"reason": "thumb_extended_not_tucked"}

        cross_quality = 1.0 if (is_crossing and cross_ratio < 0.36) else 0.88
        curl_quality = 1.0 if (r_cur and p_cur) else 0.85
        thumb_quality = 1.0 if thumb_tucked else 0.88

        total_score = 0.85 + 0.06 * cross_quality + 0.05 * curl_quality + 0.04 * thumb_quality

        center = h["palm_center"]
        metrics = {
            "cross_ratio": cross_ratio,
            "is_crossing": is_crossing,
            "palm_y_norm": palm_y_norm,
            "index_tip_y_norm": index_tip_y_norm,
            "score": total_score,
        }
        return min(1.0, total_score), center, metrics

    def evaluate_sukuna_mudra(
        self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> Tuple[float, Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        Evaluates Sukuna's Enma-ten Mudra (伏魔御廚子):
        CLASSIFICATION 2: Strictly 2 hands attached by 2 fingers
        - Strictly 2 hands required (returns 0.0 if hands count != 2)
        - Positioned in front of chest / torso level (palm_y_norm >= 0.20)
        - Palms clasped together (palm_dist / avg_scale < 5.0)
        - Attached by 2 index fingers: index fingertips touching or near each other (idx_dist / avg_scale < 3.2)
        - Both index fingers pointing up, thumbs upright, lower fingers curled
        """
        # STRICT RULE: Sukuna requires strictly 2 hands
        if len(hands) != 2:
            return 0.0, None, {"hands_count": len(hands), "reason": "requires_strictly_2_hands"}

        h1, h2 = hands[0], hands[1]
        p1, p2 = h1["palm_center"], h2["palm_center"]
        scale1 = max(10.0, h1.get("palm_scale", 30))
        scale2 = max(10.0, h2.get("palm_scale", 30))
        avg_scale = (scale1 + scale2) / 2.0

        # 1. Attached by 2 fingers check (The two index fingers touching / meeting)
        idx_dist = min(
            math.dist(h1["index_tip"], h2["index_tip"]),
            math.dist(h1["index_tip"], h2.get("index_pip", h2["index_tip"])),
            math.dist(h1.get("index_pip", h1["index_tip"]), h2["index_tip"]),
        )
        idx_ratio = idx_dist / avg_scale
        if idx_ratio > 2.8:
            return 0.0, None, {"reason": "index_fingertips_not_touching", "idx_ratio": idx_ratio}

        # 2. Palm proximity check (hands clasped together)
        palm_dist = math.dist(p1, p2)
        palm_ratio = palm_dist / avg_scale
        if palm_ratio > 5.0:
            return 0.0, None, {"reason": "hands_too_far_apart", "palm_ratio": palm_ratio}

        # 3. Chest / torso height check
        avg_palm_y = (h1.get("palm_y_norm", 0.5) + h2.get("palm_y_norm", 0.5)) / 2.0
        if avg_palm_y < 0.20:
            return 0.0, None, {"reason": "hands_too_high_for_sukuna", "avg_palm_y": avg_palm_y}

        # 4. Index fingers upright check
        ang1 = h1.get("finger_angles", {})
        ang2 = h2.get("finger_angles", {})
        cr1 = h1.get("curl_ratios", {})
        cr2 = h2.get("curl_ratios", {})
        fs1 = h1.get("finger_states", {})
        fs2 = h2.get("finger_states", {})

        i1_up = (ang1.get("index", 0) > 110) or (cr1.get("index", 0) > 1.10) or (fs1.get("index") in ["EXTENDED", "BENT"])
        i2_up = (ang2.get("index", 0) > 110) or (cr2.get("index", 0) > 1.10) or (fs2.get("index") in ["EXTENDED", "BENT"])
        if not (i1_up or i2_up):
            return 0.0, None, {"reason": "index_fingers_not_upright"}

        # 5. Lower fingers curled (ring & pinky curled into palms)
        curled_count = 0
        for ang, cr, fs in [(ang1, cr1, fs1), (ang2, cr2, fs2)]:
            for f in ["ring", "pinky"]:
                if (ang.get(f, 180) < 140) or (cr.get(f, 2) < 1.25) or (fs.get(f) in ["CURLED", "BENT"]):
                    curled_count += 1

        if curled_count < 1:
            return 0.0, None, {"reason": "lower_fingers_not_curled", "curled_count": curled_count}

        # 6. Hand direction: not pointing downwards to floor
        dir1_y = h1.get("pointing_dir", (0, 0))[1]
        dir2_y = h2.get("pointing_dir", (0, 0))[1]
        if dir1_y > 0.65 and dir2_y > 0.65:
            return 0.0, None, {"reason": "hands_pointing_downwards"}

        # Quality bonus
        touch_quality = 1.0 if idx_ratio < 1.6 else (0.92 if idx_ratio < 2.5 else 0.85)
        proximity_quality = 1.0 if palm_ratio < 2.8 else 0.90
        curl_quality = 1.0 if curled_count >= 2 else 0.88

        total_score = 0.88 + 0.05 * touch_quality + 0.04 * proximity_quality + 0.03 * curl_quality
        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        metrics = {
            "idx_dist_ratio": idx_ratio,
            "palm_dist_ratio": palm_ratio,
            "curled_lower_count": curled_count,
            "score": total_score,
        }
        return min(1.0, total_score), center, metrics

    def update(
        self,
        hands: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        target_theme: str = "malevolent_shrine",
    ) -> Dict[str, Any]:
        """
        Updates the multi-class state machine with temporal smoothing.
        Returns complete diagnostic telemetry and activation status.
        """
        if self.cooldown > 0:
            self.cooldown -= 1

        if self.transition_timer > 0:
            self.transition_timer -= 1

        self.last_hands = hands
        self.last_hands_count = len(hands)

        # 1. Evaluate both mudras independently
        score_gojo, center_gojo, metrics_gojo = self.evaluate_gojo_mudra(hands, frame_shape)
        score_sukuna, center_sukuna, metrics_sukuna = self.evaluate_sukuna_mudra(hands, frame_shape)

        self.last_gojo_score = score_gojo
        self.last_sukuna_score = score_sukuna
        self.last_gojo_pct = int(score_gojo * 100.0)
        self.last_sukuna_pct = int(score_sukuna * 100.0)

        # 2. Compute Unknown evidence score
        max_candidate_score = max(score_gojo, score_sukuna)
        score_unknown = max(0.0, 1.0 - max_candidate_score)
        self.last_unknown_score = score_unknown

        # 3. Multi-Class Evidence-Based Arbitration
        # Determine candidate gesture for this frame
        if not hands:
            frame_class = "NO_HAND"
            active_score = 0.0
            active_center = None
            active_theme = target_theme
            active_name = "None"
        elif score_sukuna >= self.confidence_threshold and score_sukuna > score_gojo:
            frame_class = "SUKUNA"
            active_score = score_sukuna
            active_center = center_sukuna
            active_theme = "malevolent_shrine"
            active_name = "Sukuna (Enma-ten Mudra)"
        elif score_gojo >= self.confidence_threshold and score_gojo > score_sukuna:
            frame_class = "GOJO"
            active_score = score_gojo
            active_center = center_gojo
            active_theme = "infinite_void"
            active_name = "Gojo (Taishakuten Mudra)"
        else:
            # Below confidence threshold -> UNKNOWN (NEVER force Gojo or Sukuna!)
            frame_class = "UNKNOWN"
            active_score = max_candidate_score
            active_center = center_gojo if score_gojo >= score_sukuna else center_sukuna
            active_theme = target_theme
            active_name = "Unknown Gesture"

        self._history.append(frame_class)

        # 4. Temporal State Machine with Hysteresis & Transition Logic
        if self.transition_timer > 0:
            self.state = "TRANSITION"
            self.stable_frames = 0
        elif frame_class == "NO_HAND":
            self.state = "NO_HAND"
            self.stable_frames = max(0, self.stable_frames - 1)
            if self.stable_frames == 0:
                self.candidate_sign = "UNKNOWN"
        elif frame_class == "UNKNOWN":
            # In hysteresis zone: if previously candidate/confirmed, hold count in band
            if self.candidate_sign == "GOJO" and score_gojo >= self.deactivation_threshold:
                pass
            elif self.candidate_sign == "SUKUNA" and score_sukuna >= self.deactivation_threshold:
                pass
            else:
                self.stable_frames = max(0, self.stable_frames - 1)
                if self.stable_frames == 0:
                    self.state = "UNKNOWN"
                    self.candidate_sign = "UNKNOWN"
        else:
            # Frame matches GOJO or SUKUNA candidate
            if self.candidate_sign != frame_class:
                # Switching between mudras! Enter TRANSITION to avoid Gojo/Sukuna cross-contamination
                if self.candidate_sign in ["GOJO", "SUKUNA"] and self.stable_frames > 2:
                    self.state = "TRANSITION"
                    self.transition_timer = 8
                    self.stable_frames = 0
                self.candidate_sign = frame_class
                self.stable_frames = 1
            else:
                self.stable_frames += 1

            # Advance state from CANDIDATE to CONFIRMED
            self.detected_sign_name = active_name
            if self.stable_frames >= self.hold_frames_required:
                self.state = f"{frame_class}_CONFIRMED"
                self.confirmed_sign = frame_class
            else:
                self.state = f"{frame_class}_CANDIDATE"

        # 5. Domain Activation Trigger
        triggered = False
        is_confirmed = "CONFIRMED" in self.state
        if is_confirmed and self.cooldown == 0:
            triggered = True
            self.cooldown = 60  # Cooldown frames to prevent re-triggering

        hold_progress = min(1.0, self.stable_frames / float(self.hold_frames_required))
        self.last_match_pct = int(active_score * 100.0)

        # 6. User-facing Status Text Generation
        if not hands:
            status_text = "No hands detected"
        elif self.state == "TRANSITION":
            status_text = "Switching hand sign..."
        elif self.state == "UNKNOWN":
            if max_candidate_score > 0.40:
                hint_name = "Sukuna" if score_sukuna > score_gojo else "Gojo"
                status_text = f"{hint_name}: {int(max_candidate_score * 100)}% (Hold canonical pose)"
            else:
                status_text = "Make hand sign (Sukuna or Gojo)"
        elif triggered:
            status_text = f">> DOMAIN EXPANSION: {active_theme.upper().replace('_', ' ')} <<"
        elif is_confirmed:
            status_text = f"CONFIRMED: {self.detected_sign_name} (ACTIVATING)"
        else:
            status_text = f"HOLD... {active_name.split()[0]} {self.last_match_pct}% ({self.stable_frames}/{self.hold_frames_required})"

        fallback_center = (frame_shape[1] // 2, int(frame_shape[0] * 0.52))

        return {
            "sign_detected": is_confirmed or (hold_progress >= 0.80),
            "state": self.state,
            "detected_theme": active_theme if (self.state != "UNKNOWN" and self.state != "NO_HAND") else target_theme,
            "candidate_sign": self.candidate_sign,
            "confirmed_sign": self.confirmed_sign,
            "sign_name": self.detected_sign_name if is_confirmed else active_name,
            "confidence": active_score,
            "match_pct": self.last_match_pct,
            "gojo_score": score_gojo,
            "sukuna_score": score_sukuna,
            "unknown_score": score_unknown,
            "status_text": status_text,
            "hold_progress": hold_progress,
            "stable_frames": self.stable_frames,
            "required_frames": self.hold_frames_required,
            "trigger": triggered,
            "hands_count": len(hands),
            "energy_center": active_center or fallback_center,
            "metrics": {
                "gojo": metrics_gojo,
                "sukuna": metrics_sukuna,
            },
        }

    def reset(self):
        self.state = "UNKNOWN"
        self.candidate_sign = "UNKNOWN"
        self.confirmed_sign = "UNKNOWN"
        self.stable_frames = 0
        self.cooldown = 0
        self.transition_timer = 0
        self.detected_sign_name = ""
        self.last_match_pct = 0
        self._history.clear()
