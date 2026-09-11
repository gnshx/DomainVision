import math
from typing import Dict, Any, Optional, Tuple


class DomainGestureDetector:
    """
    Detects classic Domain Expansion activation gestures:
    1. Hands clasped / brought together in front of the chest/face.
    2. Both hands raised above shoulders.
    Uses temporal smoothing to prevent spurious activations.
    """

    def __init__(self, activation_frames: int = 4, cooldown_frames: int = 60):
        self.activation_frames = activation_frames
        self.cooldown_frames = cooldown_frames
        self.hold_count = 0
        self.cooldown_timer = 0
        self.last_trigger_pos = None

    def update(self, tracking_data: Dict[str, Any], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Evaluate current tracking data.
        Returns:
            {
                "gesture_active": bool,      # True on the exact frame the activation triggers
                "is_holding_pose": bool,     # True while the user is actively holding the pose
                "hold_progress": float,      # 0.0 to 1.0 (fill bar for charging)
                "energy_point": Tuple[x, y], # Focus point between hands for charging FX
            }
        """
        h, w = frame_shape[:2]
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        is_pose = False
        center_pt = None

        hands = tracking_data.get("hands", [])
        pose = tracking_data.get("pose")

        # 1. Check dual-hand proximity
        if len(hands) >= 2:
            p1 = hands[0]["palm_center"]
            p2 = hands[1]["palm_center"]
            dist = math.dist(p1, p2)
            # If hands are within 20% of screen width of each other
            if dist < w * 0.22:
                is_pose = True
                center_pt = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

        # 2. Check pose wrists if hands detector missed
        elif pose and pose.get("left_wrist") and pose.get("right_wrist"):
            lw = pose["left_wrist"]
            rw = pose["right_wrist"]
            dist = math.dist(lw, rw)
            # Compare to shoulder distance if available
            shoulder_dist = w * 0.3
            if pose.get("left_shoulder") and pose.get("right_shoulder"):
                shoulder_dist = math.dist(pose["left_shoulder"], pose["right_shoulder"])

            # Wrists close together in front of chest/torso
            if dist < max(w * 0.18, shoulder_dist * 0.65):
                # Also check wrists are above waist level (below chin / near chest)
                is_pose = True
                center_pt = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))

            # Or both wrists raised above shoulders (Gojo / Sukuna dramatic pose)
            elif (pose.get("left_shoulder") and pose.get("right_shoulder") and
                  lw[1] < pose["left_shoulder"][1] and rw[1] < pose["right_shoulder"][1]):
                is_pose = True
                center_pt = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))

        if is_pose:
            self.hold_count = min(self.activation_frames * 2, self.hold_count + 1)
            self.last_trigger_pos = center_pt or (w // 2, int(h * 0.5))
        else:
            self.hold_count = max(0, self.hold_count - 1)

        progress = min(1.0, self.hold_count / float(self.activation_frames))
        trigger = False

        if progress >= 1.0 and self.cooldown_timer == 0:
            trigger = True
            self.cooldown_timer = self.cooldown_frames

        return {
            "gesture_active": trigger,
            "is_holding_pose": is_pose and (self.hold_count > 1),
            "hold_progress": progress,
            "energy_point": self.last_trigger_pos or (w // 2, int(h * 0.5)),
        }

    def reset(self):
        self.hold_count = 0
        self.cooldown_timer = 0
