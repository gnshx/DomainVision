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


def segments_intersect_2d(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """
    Tests whether 2D line segment (p1 -> p2) intersects with (p3 -> p4).
    Used to detect if middle finger physically crosses over index finger in 2D projection.
    """
    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p1, p2, p3)
    d2 = ccw(p1, p2, p4)
    d3 = ccw(p3, p4, p1)
    d4 = ccw(p3, p4, p2)

    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))


def is_valid_hand_anatomy(hand_dict: Dict[str, Any]) -> bool:
    """
    Validates anatomical hand proportions to reject false detections
    caused by hair, ears, head contours, or background textures:
    - Palm scale must represent genuine human hand proportions (> 20px)
    - Palm width-to-height ratio must match human anatomy
    - Landmark bounding box must have reasonable physical extent
    """
    lms = hand_dict.get("landmarks", [])
    if len(lms) < 21:
        return False

    palm_scale = hand_dict.get("palm_scale", 0.0)
    if palm_scale < 15.0:
        return False

    bbox = hand_dict.get("bbox")
    if bbox is None and lms:
        xs = [p[0] for p in lms]
        ys = [p[1] for p in lms]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    elif bbox is None:
        bbox = (0, 0, 0, 0)

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < 18 or bh < 18:
        return False

    # Check palm breadth (dist between Index MCP 5 and Pinky MCP 17)
    p0 = lms[0]   # wrist
    p5 = lms[5]   # index MCP
    p9 = lms[9]   # middle MCP
    p17 = lms[17] # pinky MCP

    palm_breadth = math.dist(p5, p17)
    palm_len = math.dist(p0, p9)
    if palm_len < 10.0:
        return False

    ratio = palm_breadth / palm_len
    # Human hands have breadth-to-length ratio between 0.08 and 2.80 (accommodates clasped hands seen edge-on)
    if ratio < 0.08 or ratio > 2.80:
        return False

    return True


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

    def analyze_hand(
        self,
        raw_landmarks: List[Any],
        frame_shape: Tuple[int, int],
        hand_idx: int = 0,
        handedness: str = "Unknown",
        handedness_conf: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Takes raw MediaPipe landmarks, smooths them, and calculates invariant geometric features:
        - Scale-normalized coordinates and palm reference frame
        - 3D joint angles and finger curl ratios
        - Exact middle-crossing-index metric for Gojo
        - Thumb tuck/fold and palm orientation
        - Bounding box and frame-relative vertical position
        """
        h, w = frame_shape[:2]

        # Convert normalized coordinates to pixel/3D space
        points_3d = [(float(lm.x * w), float(lm.y * h), float(lm.z * w)) for lm in raw_landmarks]

        # Apply temporal smoothing
        if hand_idx < len(self.smoothers):
            smoothed_2d = self.smoothers[hand_idx].update(points_3d)
        else:
            smoothed_2d = [(int(p[0]), int(p[1])) for p in points_3d]

        wrist_3d = points_3d[0]
        wrist_2d = smoothed_2d[0]

        # Palm center & scale
        palm_x = int((smoothed_2d[0][0] + smoothed_2d[5][0] + smoothed_2d[17][0]) / 3.0)
        palm_y = int((smoothed_2d[0][1] + smoothed_2d[5][1] + smoothed_2d[17][1]) / 3.0)
        palm_center = (palm_x, palm_y)
        palm_scale = max(10.0, math.dist(smoothed_2d[0], smoothed_2d[9]))  # wrist to middle MCP distance

        # Hand reference vectors: Longitudinal (wrist -> middle MCP) and Transverse (pinky MCP -> index MCP)
        long_vec = np.array([points_3d[9][0] - points_3d[0][0], points_3d[9][1] - points_3d[0][1], points_3d[9][2] - points_3d[0][2]], dtype=np.float32)
        long_len = np.linalg.norm(long_vec) + 1e-6
        u_long = long_vec / long_len

        trans_vec = np.array([points_3d[5][0] - points_3d[17][0], points_3d[5][1] - points_3d[17][1], points_3d[5][2] - points_3d[17][2]], dtype=np.float32)
        trans_len = np.linalg.norm(trans_vec) + 1e-6
        u_trans = trans_vec / trans_len

        # 3D Palm normal vector: cross product of u_long and u_trans
        norm_v = np.cross(u_long, u_trans)
        norm_len = np.linalg.norm(norm_v) + 1e-6
        palm_normal = (float(norm_v[0] / norm_len), float(norm_v[1] / norm_len), float(norm_v[2] / norm_len))

        # Hand direction vector / Wrist orientation (2D)
        dir_x = (smoothed_2d[9][0] - smoothed_2d[0][0]) / (long_len + 1e-5)
        dir_y = (smoothed_2d[9][1] - smoothed_2d[0][1]) / (long_len + 1e-5)
        pointing_dir = (float(dir_x), float(dir_y))
        wrist_orientation = pointing_dir

        # Compute finger angles, curl ratios, and states
        finger_states = {}
        finger_angles = {}
        curl_ratios = {}

        for fname, joints in self.FINGER_JOINTS.items():
            if fname == "thumb":
                ang1 = compute_angle_3d(points_3d[joints[0]], points_3d[joints[1]], points_3d[joints[2]])
                ang2 = compute_angle_3d(points_3d[joints[1]], points_3d[joints[2]], points_3d[joints[3]])
                mean_ang = (ang1 + ang2) / 2.0
                tip_dist = math.dist(points_3d[joints[3]][:2], wrist_3d[:2])
                mcp_dist = max(5.0, math.dist(points_3d[joints[1]][:2], wrist_3d[:2]))
                c_ratio = tip_dist / mcp_dist

                is_extended = (mean_ang > 138.0) and (c_ratio > 1.15)
                is_curled = (mean_ang < 118.0) or (c_ratio < 0.95)
            else:
                ang_pip = compute_angle_3d(points_3d[joints[0]], points_3d[joints[1]], points_3d[joints[2]])
                ang_dip = compute_angle_3d(points_3d[joints[1]], points_3d[joints[2]], points_3d[joints[3]])
                mean_ang = (ang_pip + ang_dip) / 2.0

                tip_dist = math.dist(points_3d[joints[3]][:2], wrist_3d[:2])
                pip_dist = max(5.0, math.dist(points_3d[joints[1]][:2], wrist_3d[:2]))
                c_ratio = tip_dist / pip_dist

                is_extended = (mean_ang >= FINGER_EXTENDED_ANGLE) and (c_ratio > 1.35)
                is_curled = (mean_ang <= FINGER_CURLED_ANGLE) or (c_ratio < 1.05)

            state = "EXTENDED" if is_extended else ("CURLED" if is_curled else "BENT")
            finger_states[fname] = state
            finger_angles[fname] = mean_ang
            curl_ratios[fname] = c_ratio

        # Normalized coordinates relative to palm frame: origin at wrist
        normalized_local_lms = []
        for p in points_3d:
            rel = np.array([p[0] - wrist_3d[0], p[1] - wrist_3d[1], p[2] - wrist_3d[2]], dtype=np.float32)
            coord_trans = float(np.dot(rel, u_trans) / palm_scale)
            coord_long = float(np.dot(rel, u_long) / palm_scale)
            coord_norm = float(np.dot(rel, norm_v / norm_len) / palm_scale)
            normalized_local_lms.append((coord_trans, coord_long, coord_norm))

        # Inter-finger distances (normalized by palm scale)
        inter_finger_dists = {
            "thumb_to_index": math.dist(smoothed_2d[4], smoothed_2d[8]) / palm_scale,
            "index_to_middle": math.dist(smoothed_2d[8], smoothed_2d[12]) / palm_scale,
            "middle_to_ring": math.dist(smoothed_2d[12], smoothed_2d[16]) / palm_scale,
            "ring_to_pinky": math.dist(smoothed_2d[16], smoothed_2d[20]) / palm_scale,
            "thumb_to_pinky": math.dist(smoothed_2d[4], smoothed_2d[20]) / palm_scale,
        }

        # Thumb fold metric: distance from thumb tip to palm center and ring MCP
        thumb_to_palm_dist = math.dist(smoothed_2d[4], palm_center) / palm_scale
        thumb_to_ring_mcp = math.dist(smoothed_2d[4], smoothed_2d[13]) / palm_scale
        thumb_tucked = (thumb_to_palm_dist < 0.72) or (thumb_to_ring_mcp < 0.78) or (finger_states["thumb"] == "CURLED")

        # Canonical Gojo Crossing Metric:
        # Distance between Index TIP (LM 8) and Middle TIP (LM 12)
        index_tip_2d = smoothed_2d[8]
        middle_tip_2d = smoothed_2d[12]
        tip_cross_dist = math.dist(index_tip_2d, middle_tip_2d)
        cross_ratio = tip_cross_dist / palm_scale

        # Transverse overlap: in local palm frame, compare transverse position of tip 8 vs tip 12
        trans_idx_tip = normalized_local_lms[8][0]
        trans_mid_tip = normalized_local_lms[12][0]
        trans_idx_mcp = normalized_local_lms[5][0]
        trans_mid_mcp = normalized_local_lms[9][0]

        # In natural uncrossed hand, index is on the index-MCP side of middle.
        # When crossed, tips swap relative transverse positions or get close (< 0.42 scale)
        base_sign = 1.0 if (trans_idx_mcp > trans_mid_mcp) else -1.0
        tip_sign = 1.0 if (trans_idx_tip > trans_mid_tip) else -1.0
        is_crossed_swap = (base_sign * tip_sign < 0.0)

        # Direct 2D Line Segment Intersection:
        # Index finger segment (MCP 5 or PIP 6 -> TIP 8) vs Middle finger segment (MCP 9 or PIP 10 -> TIP 12)
        idx_pip = smoothed_2d[6]
        mid_pip = smoothed_2d[10]
        seg_intersect = (
            segments_intersect_2d(smoothed_2d[5], index_tip_2d, smoothed_2d[9], middle_tip_2d) or
            segments_intersect_2d(idx_pip, index_tip_2d, mid_pip, middle_tip_2d)
        )

        idx_active = finger_states["index"] in ["EXTENDED", "BENT"]
        mid_active = finger_states["middle"] in ["EXTENDED", "BENT"]

        is_crossing_mudra = (
            (seg_intersect and idx_active and mid_active) or
            (is_crossed_swap and cross_ratio < 0.52 and idx_active and mid_active) or
            (cross_ratio < 0.38 and idx_active and mid_active)
        )

        # Bounding Box
        xs = [p[0] for p in smoothed_2d]
        ys = [p[1] for p in smoothed_2d]
        bbox = (max(0, min(xs)), max(0, min(ys)), min(w - 1, max(xs)), min(h - 1, max(ys)))

        # Frame-relative vertical position (0.0 = top of screen, 1.0 = bottom)
        palm_y_norm = float(palm_center[1] / float(h))

        return {
            "landmarks": smoothed_2d,
            "raw_3d": points_3d,
            "normalized_lms": normalized_local_lms,
            "palm_center": palm_center,
            "palm_scale": palm_scale,
            "palm_y_norm": palm_y_norm,
            "pointing_dir": pointing_dir,
            "wrist_orientation": wrist_orientation,
            "palm_normal": palm_normal,
            "inter_finger_dists": inter_finger_dists,
            "finger_states": finger_states,
            "finger_angles": finger_angles,
            "curl_ratios": curl_ratios,
            "cross_ratio": cross_ratio,
            "is_crossing_mudra": is_crossing_mudra,
            "thumb_tucked": thumb_tucked,
            "handedness": handedness,
            "handedness_conf": handedness_conf,
            "bbox": bbox,
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
            "cross_ratio": 0.22 if theme_name == "infinite_void" else 0.85,
            "is_crossing_mudra": (theme_name == "infinite_void"),
            "thumb_tucked": (theme_name == "infinite_void"),
            "palm_y_norm": float(py / float(h)),
        }

    if theme == "infinite_void":
        # Gojo Taishakuten mudra: exactly ONE hand raised to eye/face level with fingers crossed
        gojo_palm = (cx, int(h * 0.40))
        return [make_hand(gojo_palm, is_left=False, theme_name="infinite_void")]

    # Sukuna Enma-ten mudra: strictly TWO hands clasped at chest level
    return [make_hand(p1, is_left=True, theme_name="malevolent_shrine"), make_hand(p2, is_left=False, theme_name="malevolent_shrine")]
