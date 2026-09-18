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
        1. 1 primary active hand (suppressed if 2 hands are clasped together in Sukuna territory)
        2. Vertical position: hand held near face/head level (palm_y < h * 0.68)
        3. Hand pointing upright: pointing_dir[1] < -0.30
        4. Index & Middle fingers extended: angles > 135°, curl_ratio > 1.30
        5. Middle finger crossing Index finger: cross_ratio < 0.40 or crossing swap
        6. Ring & Pinky curled: angles < 125°, curl_ratio < 1.15
        7. Thumb folded: tucked into palm / ring MCP
        """
        if not hands:
            return 0.0, None, {"status": "no_hands"}

        # If two hands are clasped close together, this is Sukuna territory — suppress Gojo
        if len(hands) >= 2:
            p1 = hands[0]["palm_center"]
            p2 = hands[1]["palm_center"]
            avg_scale = (hands[0].get("palm_scale", 30) + hands[1].get("palm_scale", 30)) / 2.0
            if math.dist(p1, p2) < avg_scale * 3.4:
                return 0.0, None, {"suppressed": "dual_hands_clasped"}

        best_score = 0.0
        best_center = None
        best_metrics = {}

        for h in hands:
            scale = max(10.0, h.get("palm_scale", 30))
            fa = h.get("finger_angles", {})
            fs = h.get("finger_states", {})
            cr = h.get("curl_ratios", {})

            # 1. Height check: Gojo sign is canonically held near face/eyes/head
            palm_y_norm = h.get("palm_y_norm", 0.5)
            # Full score if held above 0.55 frame height; penalty if held low at chest/waist
            if palm_y_norm < 0.52:
                h_score = 1.0
            elif palm_y_norm < 0.68:
                h_score = 0.75
            else:
                h_score = 0.30

            # 2. Hand pointing upright
            dir_y = h.get("pointing_dir", (0, 0))[1]
            if dir_y < -0.40:
                dir_score = 1.0
            elif dir_y < -0.15:
                dir_score = 0.65
            else:
                dir_score = 0.15

            # 3. Index & Middle extended
            i_ext = (fa.get("index", 0) > 135) or (cr.get("index", 0) > 1.30)
            m_ext = (fa.get("middle", 0) > 135) or (cr.get("middle", 0) > 1.30)
            if i_ext and m_ext:
                ext_score = 1.0
            elif i_ext or m_ext:
                ext_score = 0.25
            else:
                ext_score = 0.0

            # 4. Canonical Crossing Check (THE CRUCIAL DIFFERENTIATOR)
            # Standard uncrossed fingers have cross_ratio > 0.48; crossed fingers have < 0.38
            cross_ratio = h.get("cross_ratio", 1.0)
            is_crossing = h.get("is_crossing_mudra", False)

            if is_crossing and cross_ratio < 0.36:
                cross_score = 1.0
            elif is_crossing or cross_ratio < 0.42:
                cross_score = 0.85
            elif cross_ratio < 0.50:
                cross_score = 0.40
            else:
                # Parallel fingers (e.g. peace sign or two fingers point)
                cross_score = 0.0

            # 5. Ring & Pinky curled tightly into palm
            r_cur = (fa.get("ring", 180) < 125) or (cr.get("ring", 2) < 1.15) or (fs.get("ring") == "CURLED")
            p_cur = (fa.get("pinky", 180) < 125) or (cr.get("pinky", 2) < 1.15) or (fs.get("pinky") == "CURLED")
            if r_cur and p_cur:
                curl_score = 1.0
            elif r_cur or p_cur:
                curl_score = 0.45
            else:
                curl_score = 0.0

            # 6. Thumb folded inward
            thumb_tucked = h.get("thumb_tucked", False) or (fa.get("thumb", 180) < 130)
            thumb_score = 1.0 if thumb_tucked else 0.40

            # Compute weighted geometric evidence
            # Hard rejection gates: must have extension and curl
            if ext_score < 0.5 or curl_score < 0.4:
                cand_score = 0.0
            else:
                cand_score = (
                    0.30 * cross_score
                    + 0.22 * ext_score
                    + 0.20 * curl_score
                    + 0.12 * h_score
                    + 0.10 * dir_score
                    + 0.06 * thumb_score
                )

                lm = h.get("landmarks", [])
                thumb_mcp_y = lm[2][1] if len(lm) > 2 else h["wrist"][1]
                thumb_up = (fa.get("thumb", 0) > 120) and (h["thumb_tip"][1] < thumb_mcp_y)
                if thumb_up:
                    # Sukuna signature: upright thumb must NOT trigger Gojo
                    cand_score = max(0.0, cand_score - 0.40)

                # Cap score if crossing is absent — Gojo CANNOT activate without finger crossing!
                if cross_score < 0.4:
                    cand_score = min(0.35, cand_score)

            if cand_score > best_score:
                best_score = cand_score
                best_center = h["palm_center"]
                best_metrics = {
                    "cross_ratio": cross_ratio,
                    "cross_score": cross_score,
                    "ext_score": ext_score,
                    "curl_score": curl_score,
                    "height_score": h_score,
                    "dir_y": dir_y,
                    "palm_y_norm": palm_y_norm,
                }

        return min(1.0, best_score), best_center, best_metrics

    def evaluate_sukuna_mudra(
        self, hands: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> Tuple[float, Optional[Tuple[int, int]], Dict[str, Any]]:
        """
        Evaluates Sukuna's Enma-ten Mudra (伏魔御廚子):
        1. STRICTLY 2 HANDS REQUIRED. (1 hand score is strictly 0.0)
        2. Hands positioned at chest / center torso (0.28 < y_norm < 0.85)
        3. Palms brought together: palm_dist < 2.5 * avg_scale
        4. Upward wrist and hand orientation: pointing_dir_y < 0.15 for both hands
        5. Thumbs upright and extended: thumb_angle > 125° or curl_ratio > 1.15
        6. Index fingertips meeting: dist(tip1, tip2) < 1.15 * avg_scale
        7. Lower fingers curled: at least 3 of 4 ring/pinky fingers curled
        8. Palms facing each other
        """
        # STRICT RULE: Sukuna CANNOT be classified from a single hand
        if len(hands) != 2:
            return 0.0, None, {"hands_count": len(hands), "reason": "requires_strictly_2_hands"}

        h1, h2 = hands[0], hands[1]
        p1, p2 = h1["palm_center"], h2["palm_center"]
        scale1 = max(10.0, h1.get("palm_scale", 30))
        scale2 = max(10.0, h2.get("palm_scale", 30))
        avg_scale = (scale1 + scale2) / 2.0

        metrics = {}

        # 1. Height check: Sukuna mudra is held around chest / center region
        avg_y_norm = (h1.get("palm_y_norm", 0.5) + h2.get("palm_y_norm", 0.5)) / 2.0
        metrics["avg_y_norm"] = avg_y_norm
        if 0.28 <= avg_y_norm <= 0.85:
            h_score = 1.0
        else:
            h_score = 0.50

        # 2. Palm proximity: hands clasped together
        palm_dist = math.dist(p1, p2)
        palm_ratio = palm_dist / avg_scale
        metrics["palm_dist_ratio"] = palm_ratio

        if palm_ratio < 1.9:
            proximity_score = 1.0
        elif palm_ratio < 2.6:
            proximity_score = 0.75
        elif palm_ratio < 3.4:
            proximity_score = 0.40
        else:
            proximity_score = 0.0

        # 3. Upright hand orientation: both hands pointing up or toward center
        dir1_y = h1.get("pointing_dir", (0, 0))[1]
        dir2_y = h2.get("pointing_dir", (0, 0))[1]
        metrics["dir1_y"] = dir1_y
        metrics["dir2_y"] = dir2_y
        if dir1_y < -0.10 and dir2_y < -0.10:
            dir_score = 1.0
        elif dir1_y < 0.15 and dir2_y < 0.15:
            dir_score = 0.75
        else:
            dir_score = 0.20

        # 4. Thumbs upright & extended on both hands
        ang1 = h1.get("finger_angles", {})
        ang2 = h2.get("finger_angles", {})
        cr1 = h1.get("curl_ratios", {})
        cr2 = h2.get("curl_ratios", {})

        t1_ext = (ang1.get("thumb", 0) > 125) or (cr1.get("thumb", 0) > 1.12)
        t2_ext = (ang2.get("thumb", 0) > 125) or (cr2.get("thumb", 0) > 1.12)
        if t1_ext and t2_ext:
            thumb_score = 1.0
        elif t1_ext or t2_ext:
            thumb_score = 0.55
        else:
            thumb_score = 0.10

        # 5. Index fingertips touching / meeting at top
        idx_dist = math.dist(h1["index_tip"], h2["index_tip"])
        idx_ratio = idx_dist / avg_scale
        metrics["index_dist_ratio"] = idx_ratio

        i1_ext = (ang1.get("index", 0) > 130) or (cr1.get("index", 0) > 1.25)
        i2_ext = (ang2.get("index", 0) > 130) or (cr2.get("index", 0) > 1.25)

        if idx_ratio < 1.10 and (i1_ext or i2_ext):
            index_score = 1.0
        elif idx_ratio < 1.80 and (i1_ext or i2_ext):
            index_score = 0.75
        elif idx_ratio < 2.50:
            index_score = 0.35
        else:
            index_score = 0.0

        # 6. Lower fingers (Ring & Pinky) curled on BOTH hands
        curled_count = 0
        for h, ang, cr in [(h1, ang1, cr1), (h2, ang2, cr2)]:
            for f in ["ring", "pinky"]:
                if (ang.get(f, 180) < 135) or (cr.get(f, 2) < 1.20):
                    curled_count += 1
        metrics["curled_lower_count"] = curled_count

        if curled_count >= 3:
            lower_score = 1.0
        elif curled_count == 2:
            lower_score = 0.60
        else:
            lower_score = 0.20

        # 7. Palms facing each other
        norm1 = h1.get("palm_normal", (0, 0, 1))
        norm2 = h2.get("palm_normal", (0, 0, 1))
        dot_normals = norm1[0] * norm2[0] + norm1[1] * norm2[1] + norm1[2] * norm2[2]
        metrics["dot_normals"] = dot_normals

        # Palms facing each other have negative dot product or opposite horizontal vectors
        if dot_normals < 0.10:
            facing_score = 1.0
        elif dot_normals < 0.40:
            facing_score = 0.70
        else:
            facing_score = 0.35

        # Hard rejection: hands too far apart or all fingers flat/open
        if proximity_score < 0.30:
            total_score = 0.0
        else:
            total_score = (
                0.25 * proximity_score
                + 0.22 * index_score
                + 0.18 * thumb_score
                + 0.15 * lower_score
                + 0.10 * dir_score
                + 0.05 * facing_score
                + 0.05 * h_score
            )

        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
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
            self.stable_frames = max(0, self.stable_frames - 3)
            self.candidate_sign = "UNKNOWN"
        elif frame_class == "UNKNOWN":
            # In hysteresis zone: if previously candidate/confirmed, only decay slowly
            if self.candidate_sign == "GOJO" and score_gojo >= self.deactivation_threshold:
                # Hold count in hysteresis band
                pass
            elif self.candidate_sign == "SUKUNA" and score_sukuna >= self.deactivation_threshold:
                # Hold count in hysteresis band
                pass
            else:
                self.stable_frames = max(0, self.stable_frames - 2)
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
            if self.stable_frames >= self.hold_frames_required:
                self.state = f"{frame_class}_CONFIRMED"
                self.confirmed_sign = frame_class
                self.detected_sign_name = active_name
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
