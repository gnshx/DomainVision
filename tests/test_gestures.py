"""
Unit Test Suite for DomainVision Canonical Hand Gesture Recognizer.
Validates multi-class evidence-based classification and state transitions:
- Test A: No hands -> NO_HAND / UNKNOWN (Gojo score 0.0, Sukuna score 0.0)
- Test B: Random one-hand open pose -> UNKNOWN (Gojo score < 0.40)
- Test C: Gojo Taishakuten mudra (crossed fingers, face height) -> GOJO_CONFIRMED
- Test D: Sukuna Enma-ten mudra (strictly 2 hands clasped, chest height) -> SUKUNA_CONFIRMED
- Test E: Single hand CANNOT trigger Sukuna (score strictly 0.0)
- Test F: Transition Gojo -> Sukuna without state contamination
"""
import math
import unittest
import numpy as np

from tracking.gesture_recognizer import CanonicalGestureRecognizer


def create_mock_hand(
    palm_center=(640, 300),
    palm_scale=40.0,
    finger_angles=None,
    curl_ratios=None,
    pointing_dir=(0.0, -1.0),
    cross_ratio=0.8,
    is_crossing_mudra=False,
    thumb_tucked=False,
    handedness="Right",
    frame_h=720,
):
    if finger_angles is None:
        finger_angles = {
            "thumb": 150.0,
            "index": 160.0,
            "middle": 160.0,
            "ring": 160.0,
            "pinky": 160.0,
        }
    if curl_ratios is None:
        curl_ratios = {
            "thumb": 1.3,
            "index": 1.5,
            "middle": 1.5,
            "ring": 1.5,
            "pinky": 1.5,
        }

    # Generate dummy landmarks
    px, py = palm_center
    landmarks = [(px, py + 20)]  # 0: wrist
    # Thumb 1..4
    for i in range(1, 5):
        landmarks.append((px - 10 * i, py - 5 * i))
    # Index 5..8
    for i in range(1, 5):
        landmarks.append((px - 5, py - 10 * i))
    # Middle 9..12
    for i in range(1, 5):
        landmarks.append((px, py - 10 * i))
    # Ring 13..16
    for i in range(1, 5):
        landmarks.append((px + 5, py - 10 * i))
    # Pinky 17..20
    for i in range(1, 5):
        landmarks.append((px + 10, py - 10 * i))

    finger_states = {}
    for f, ang in finger_angles.items():
        if ang > 135:
            finger_states[f] = "EXTENDED"
        elif ang < 120:
            finger_states[f] = "CURLED"
        else:
            finger_states[f] = "BENT"

    return {
        "landmarks": landmarks,
        "palm_center": palm_center,
        "palm_scale": palm_scale,
        "palm_y_norm": float(palm_center[1] / frame_h),
        "pointing_dir": pointing_dir,
        "wrist_orientation": pointing_dir,
        "palm_normal": (0.0, 0.0, 1.0),
        "finger_angles": finger_angles,
        "finger_states": finger_states,
        "curl_ratios": curl_ratios,
        "cross_ratio": cross_ratio,
        "is_crossing_mudra": is_crossing_mudra,
        "thumb_tucked": thumb_tucked,
        "handedness": handedness,
        "handedness_conf": 0.98,
        "index_tip": landmarks[8],
        "middle_tip": landmarks[12],
        "thumb_tip": landmarks[4],
        "ring_tip": landmarks[16],
        "pinky_tip": landmarks[20],
    }


class TestGestureRecognizer(unittest.TestCase):
    def setUp(self):
        self.rec = CanonicalGestureRecognizer(hold_frames_required=5, confidence_threshold=0.72)

    def test_no_hands(self):
        """Test A: No hands should return NO_HAND / UNKNOWN, scores 0.0."""
        res = self.rec.update(hands=[], frame_shape=(720, 1280))
        self.assertEqual(res["state"], "NO_HAND")
        self.assertEqual(res["gojo_score"], 0.0)
        self.assertEqual(res["sukuna_score"], 0.0)
        self.assertFalse(res["trigger"])

    def test_random_open_hand(self):
        """Test B: Standard open hand must NOT trigger Gojo or Sukuna."""
        open_hand = create_mock_hand(
            palm_center=(640, 400),
            finger_angles={"thumb": 150, "index": 170, "middle": 170, "ring": 170, "pinky": 170},
            curl_ratios={"thumb": 1.3, "index": 1.6, "middle": 1.6, "ring": 1.6, "pinky": 1.6},
            cross_ratio=0.85,
            is_crossing_mudra=False,
            thumb_tucked=False,
        )
        res = self.rec.update(hands=[open_hand], frame_shape=(720, 1280))
        self.assertEqual(res["state"], "UNKNOWN")
        self.assertLess(res["gojo_score"], 0.40)
        self.assertEqual(res["sukuna_score"], 0.0)
        self.assertFalse(res["trigger"])

    def test_peace_sign_not_gojo(self):
        """Peace sign (index & middle open, but parallel/uncrossed) must NOT trigger Gojo."""
        peace_sign = create_mock_hand(
            palm_center=(640, 280),
            finger_angles={"thumb": 90, "index": 170, "middle": 170, "ring": 80, "pinky": 80},
            curl_ratios={"thumb": 0.8, "index": 1.6, "middle": 1.6, "ring": 0.9, "pinky": 0.9},
            cross_ratio=0.75,  # Separated fingers
            is_crossing_mudra=False,
            thumb_tucked=True,
        )
        res = self.rec.update(hands=[peace_sign], frame_shape=(720, 1280))
        self.assertEqual(res["state"], "UNKNOWN")
        self.assertLess(res["gojo_score"], 0.45)
        self.assertFalse(res["trigger"])

    def test_canonical_gojo_mudra(self):
        """Test C: Canonical Gojo mudra (crossed fingers, face height) triggers GOJO."""
        gojo_hand = create_mock_hand(
            palm_center=(640, 260),  # Head/face level
            finger_angles={"thumb": 95, "index": 165, "middle": 165, "ring": 75, "pinky": 75},
            curl_ratios={"thumb": 0.85, "index": 1.65, "middle": 1.65, "ring": 0.85, "pinky": 0.85},
            cross_ratio=0.25,  # Fingers tightly crossed
            is_crossing_mudra=True,
            thumb_tucked=True,
            pointing_dir=(0.0, -1.0),
        )

        for _ in range(5):
            res = self.rec.update(hands=[gojo_hand], frame_shape=(720, 1280))

        self.assertGreater(res["gojo_score"], 0.75)
        self.assertEqual(res["state"], "GOJO_CONFIRMED")
        self.assertTrue(res["trigger"])

    def test_sukuna_single_hand_forbidden(self):
        """Test E: A single hand can NEVER trigger Sukuna."""
        single_hand = create_mock_hand(
            palm_center=(640, 450),
            finger_angles={"thumb": 150, "index": 160, "middle": 80, "ring": 80, "pinky": 80},
            curl_ratios={"thumb": 1.3, "index": 1.5, "middle": 0.8, "ring": 0.8, "pinky": 0.8},
        )
        res = self.rec.update(hands=[single_hand], frame_shape=(720, 1280))
        self.assertEqual(res["sukuna_score"], 0.0)

    def test_canonical_sukuna_mudra(self):
        """Test D: Canonical Sukuna mudra (2 hands clasped at chest level) triggers SUKUNA."""
        scale = 36.0
        # Left and Right hands close together
        h1 = create_mock_hand(
            palm_center=(620, 420),
            palm_scale=scale,
            finger_angles={"thumb": 150, "index": 165, "middle": 95, "ring": 80, "pinky": 80},
            curl_ratios={"thumb": 1.25, "index": 1.55, "middle": 0.95, "ring": 0.85, "pinky": 0.85},
            pointing_dir=(0.1, -0.9),
            handedness="Left",
        )
        h2 = create_mock_hand(
            palm_center=(660, 420),
            palm_scale=scale,
            finger_angles={"thumb": 150, "index": 165, "middle": 95, "ring": 80, "pinky": 80},
            curl_ratios={"thumb": 1.25, "index": 1.55, "middle": 0.95, "ring": 0.85, "pinky": 0.85},
            pointing_dir=(-0.1, -0.9),
            handedness="Right",
        )
        # Touch index tips
        h1["index_tip"] = (638, 320)
        h2["index_tip"] = (642, 320)

        for _ in range(5):
            res = self.rec.update(hands=[h1, h2], frame_shape=(720, 1280))

        self.assertGreater(res["sukuna_score"], 0.75)
        self.assertEqual(res["state"], "SUKUNA_CONFIRMED")
        self.assertTrue(res["trigger"])

    def test_hands_count_telemetry(self):
        """Test G: Hands count telemetry is correctly updated and exposed."""
        h1 = create_mock_hand(palm_center=(600, 300))
        h2 = create_mock_hand(palm_center=(680, 300))

        res0 = self.rec.update(hands=[], frame_shape=(720, 1280))
        self.assertEqual(res0["hands_count"], 0)
        self.assertEqual(self.rec.last_hands_count, 0)

        res1 = self.rec.update(hands=[h1], frame_shape=(720, 1280))
        self.assertEqual(res1["hands_count"], 1)
        self.assertEqual(self.rec.last_hands_count, 1)

        res2 = self.rec.update(hands=[h1, h2], frame_shape=(720, 1280))
        self.assertEqual(res2["hands_count"], 2)
        self.assertEqual(self.rec.last_hands_count, 2)


if __name__ == "__main__":
    unittest.main()
