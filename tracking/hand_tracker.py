import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from utils.smoothing import LandmarkSmoother
from config import FINGER_EXTENDED_ANGLE, FINGER_CURLED_ANGLE, SMOOTHING_ALPHA


def compute_angle_3d(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
    """
    Computes 3D joint angle at b (between vectors ba and bc) in degrees.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=np.float32)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class AdvancedHandTracker:
    """
    Precision hand and finger tracking:
    - 21 3D joint landmark extraction
    - Temporal EMA smoothing to eliminate tracking jitter
    - Mathematical joint-angle vector calculation
    - Finger state classification (EXTENDED, BENT, CURLED)
    - Palm normal vector, scale, and spatial orientation
    """

    FINGER_JOINTS = {
        "thumb": [1, 2, 3, 4],     # CMC, MCP, IP, TIP
        "index": [5, 6, 7, 8],     # MCP, PIP, DIP, TIP
        "middle": [9, 10, 11, 12], # MCP, PIP, DIP, TIP
        "ring": [13, 14, 15, 16],  # MCP, PIP, DIP, TIP
        "pinky": [17, 18, 19, 20], # MCP, PIP, DIP, TIP
    }

    def __init__(self, smoothing_alpha: float = SMOOTHING_ALPHA):
        self.smoothers = [LandmarkSmoother(alpha=smoothing_alpha), LandmarkSmoother(alpha=smoothing_alpha)]

    def analyze_hand(self, raw_landmarks: List[Any], frame_shape: Tuple[int, int], hand_idx: int = 0) -> Dict[str, Any]:
        """
        Takes raw MediaPipe landmarks, smooths them, and calculates finger joint angles.
        """
        h, w = frame_shape[:2]

        # Convert normalized coordinates to pixel/3D space
        points_3d = [(lm.x * w, lm.y * h, lm.z * w) for lm in raw_landmarks]

        # Apply temporal smoothing
        if hand_idx < len(self.smoothers):
            smoothed_2d = self.smoothers[hand_idx].update(points_3d)
        else:
            smoothed_2d = [(int(p[0]), int(p[1])) for p in points_3d]

        # Compute finger angles and states
        finger_states = {}
        finger_angles = {}

        wrist_3d = points_3d[0]

        for fname, joints in self.FINGER_JOINTS.items():
            if fname == "thumb":
                # Thumb angle between CMC, MCP, IP
                ang1 = compute_angle_3d(points_3d[joints[0]], points_3d[joints[1]], points_3d[joints[2]])
                ang2 = compute_angle_3d(points_3d[joints[1]], points_3d[joints[2]], points_3d[joints[3]])
                mean_ang = (ang1 + ang2) / 2.0
                tip_dist = math.dist(points_3d[joints[3]][:2], wrist_3d[:2])
                mcp_dist = math.dist(points_3d[joints[1]][:2], wrist_3d[:2])
                is_extended = (mean_ang > 140.0) and (tip_dist > mcp_dist * 1.1)
                is_curled = (mean_ang < 115.0) or (tip_dist < mcp_dist)
            else:
                # Finger angle at PIP (MCP, PIP, DIP) and DIP (PIP, DIP, TIP)
                ang_pip = compute_angle_3d(points_3d[joints[0]], points_3d[joints[1]], points_3d[joints[2]])
                ang_dip = compute_angle_3d(points_3d[joints[1]], points_3d[joints[2]], points_3d[joints[3]])
                mean_ang = (ang_pip + ang_dip) / 2.0

                tip_dist = math.dist(points_3d[joints[3]][:2], wrist_3d[:2])
                pip_dist = math.dist(points_3d[joints[1]][:2], wrist_3d[:2])

                is_extended = (mean_ang >= FINGER_EXTENDED_ANGLE) and (tip_dist > pip_dist)
                is_curled = (mean_ang <= FINGER_CURLED_ANGLE) or (tip_dist < pip_dist * 0.92)

            state = "EXTENDED" if is_extended else ("CURLED" if is_curled else "BENT")
            finger_states[fname] = state
            finger_angles[fname] = mean_ang

        # Palm center & scale
        palm_x = int((smoothed_2d[0][0] + smoothed_2d[5][0] + smoothed_2d[17][0]) / 3.0)
        palm_y = int((smoothed_2d[0][1] + smoothed_2d[5][1] + smoothed_2d[17][1]) / 3.0)
        palm_center = (palm_x, palm_y)
        palm_scale = max(10.0, math.dist(smoothed_2d[0], smoothed_2d[9]))  # wrist to middle MCP distance

        # Hand direction vector / Wrist orientation (wrist to middle MCP)
        dir_x = smoothed_2d[9][0] - smoothed_2d[0][0]
        dir_y = smoothed_2d[9][1] - smoothed_2d[0][1]
        dir_len = math.hypot(dir_x, dir_y) + 1e-5
        pointing_dir = (dir_x / dir_len, dir_y / dir_len)
        wrist_orientation = pointing_dir

        # Inter-finger distances (normalized by palm scale)
        inter_finger_dists = {
            "thumb_to_index": math.dist(smoothed_2d[4], smoothed_2d[8]) / palm_scale,
            "index_to_middle": math.dist(smoothed_2d[8], smoothed_2d[12]) / palm_scale,
            "middle_to_ring": math.dist(smoothed_2d[12], smoothed_2d[16]) / palm_scale,
            "ring_to_pinky": math.dist(smoothed_2d[16], smoothed_2d[20]) / palm_scale,
            "thumb_to_pinky": math.dist(smoothed_2d[4], smoothed_2d[20]) / palm_scale,
        }

        # 3D Palm normal vector: cross product of (wrist->middle_mcp) and (pinky_mcp->index_mcp)
        v_long = np.array([points_3d[9][0] - points_3d[0][0], points_3d[9][1] - points_3d[0][1], points_3d[9][2] - points_3d[0][2]], dtype=np.float32)
        v_lat = np.array([points_3d[5][0] - points_3d[17][0], points_3d[5][1] - points_3d[17][1], points_3d[5][2] - points_3d[17][2]], dtype=np.float32)
        norm_v = np.cross(v_long, v_lat)
        norm_len = np.linalg.norm(norm_v) + 1e-6
        palm_normal = (float(norm_v[0] / norm_len), float(norm_v[1] / norm_len), float(norm_v[2] / norm_len))

        return {
            "landmarks": smoothed_2d,
            "raw_3d": points_3d,
            "palm_center": palm_center,
            "palm_scale": palm_scale,
            "pointing_dir": pointing_dir,
            "wrist_orientation": wrist_orientation,
            "palm_normal": palm_normal,
            "inter_finger_dists": inter_finger_dists,
            "finger_states": finger_states,
            "finger_angles": finger_angles,
            "wrist": smoothed_2d[0],
            "thumb_tip": smoothed_2d[4],
            "index_tip": smoothed_2d[8],
            "middle_tip": smoothed_2d[12],
            "ring_tip": smoothed_2d[16],
            "pinky_tip": smoothed_2d[20],
            "index_pip": smoothed_2d[6],
            "middle_pip": smoothed_2d[10],
        }

    def reset(self):
        for s in self.smoothers:
            s.reset()


def generate_synthetic_mudra_hands(
    frame_shape: Tuple[int, int],
    timer: int,
    theme: str = "malevolent_shrine"
) -> List[Dict[str, Any]]:
    """
    Generates synthetic hand landmarks matching the demo character's timeline
    so the full gesture recognition and skeletal cursed energy pipeline can be tested
    and demonstrated cleanly in demo mode.
    """
    h, w = frame_shape
    cx = w // 2
    cy = int(h * 0.55)
    t = timer % 360

    if t < 50 or t > 290:
        return []

    # Calculate hand positions matching SyntheticDemoCamera
    if t < 85:
        prog = (t - 50) / 35.0
        p1 = (int((cx - 85) * (1 - prog) + (cx - 18) * prog), int((cy + 90) * (1 - prog) + (cy - 35) * prog))
        p2 = (int((cx + 85) * (1 - prog) + (cx + 18) * prog), int((cy + 90) * (1 - prog) + (cy - 35) * prog))
    elif t <= 250:
        p1 = (cx - 16, cy - 38)
        p2 = (cx + 16, cy - 38)
    else:
        prog = (t - 250) / 40.0
        p1 = (int((cx - 16) * (1 - prog) + (cx - 85) * prog), int((cy - 38) * (1 - prog) + (cy + 90) * prog))
        p2 = (int((cx + 16) * (1 - prog) + (cx + 85) * prog), int((cy - 38) * (1 - prog) + (cy + 90) * prog))

    palm_scale = 36.0

    def make_hand(palm: Tuple[int, int], is_left: bool, theme_name: str) -> Dict[str, Any]:
        px, py = palm
        sgn = -1 if is_left else 1
        w_pt = (px, py + 22)
        mcp_idx = (px + sgn * 4, py - 6)
        mcp_mid = (px, py - 8)
        mcp_rng = (px - sgn * 4, py - 6)
        mcp_pnk = (px - sgn * 8, py - 4)

        if theme_name == "malevolent_shrine":
            thumb_tip = (px + sgn * 12, py - 28)
            index_tip = (cx + sgn * 2, py - 38)
            middle_tip = (px + sgn * 2, py - 2)
            ring_tip = (px - sgn * 2, py - 1)
            pinky_tip = (px - sgn * 6, py)

            fstates = {
                "thumb": "EXTENDED",
                "index": "EXTENDED",
                "middle": "CURLED",
                "ring": "CURLED",
                "pinky": "CURLED",
            }
        else:
            thumb_tip = (px - sgn * 6, py - 6)
            index_tip = (px + sgn * 4, py - 42)
            middle_tip = (px + sgn * 2, py - 42)
            ring_tip = (px - sgn * 4, py - 2)
            pinky_tip = (px - sgn * 8, py - 1)

            fstates = {
                "thumb": "BENT",
                "index": "EXTENDED",
                "middle": "EXTENDED",
                "ring": "CURLED",
                "pinky": "CURLED",
            }

        lms = [
            w_pt,
            (px + sgn * 8, py + 12), (px + sgn * 12, py + 2), (px + sgn * 13, py - 14), thumb_tip,
            mcp_idx, (px + sgn * 4, py - 18), (px + sgn * 3, py - 28), index_tip,
            mcp_mid, (px + sgn * 1, py - 16), (px + sgn * 1, py - 8), middle_tip,
            mcp_rng, (px - sgn * 3, py - 14), (px - sgn * 3, py - 7), ring_tip,
            mcp_pnk, (px - sgn * 7, py - 12), (px - sgn * 7, py - 6), pinky_tip,
        ]

        return {
            "landmarks": lms,
            "raw_3d": [[float(p[0]), float(p[1]), 0.0] for p in lms],
            "palm_center": palm,
            "palm_scale": palm_scale,
            "pointing_dir": (0.0, -1.0),
            "finger_states": fstates,
            "finger_angles": {f: (165.0 if s == "EXTENDED" else 80.0) for f, s in fstates.items()},
            "wrist": w_pt,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
            "middle_tip": middle_tip,
            "ring_tip": ring_tip,
            "pinky_tip": pinky_tip,
            "index_pip": lms[6],
            "middle_pip": lms[10],
        }

    return [make_hand(p1, is_left=True, theme_name=theme), make_hand(p2, is_left=False, theme_name=theme)]
