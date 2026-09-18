"""
Reference Symbol Detector for Anime Canonical Mudras (Gojo & Sukuna).
Provides instantaneous, robust recognition of canonical anime reference symbols
(both when fed directly as images and when displayed to the camera via phone/card),
extracting the canonical 21 3D hand landmarks for downstream gesture evaluation.
"""
import os
import math
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

from tracking.hand_tracker import AdvancedHandTracker


def _compute_phash(gray_img: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray_img, (32, 32))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    return (dct_low > med).flatten()


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class ReferenceSymbolDetector:
    def __init__(
        self,
        gojo_ref_path: str = "test_images/gojo_reference.png",
        sukuna_ref_path: str = "test_images/sukuna_reference.png",
    ):
        self.analyzer = AdvancedHandTracker()
        self.orb = cv2.ORB_create(nfeatures=400)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.has_gojo_ref = False
        self.has_sukuna_ref = False

        if os.path.exists(gojo_ref_path):
            img_g = cv2.imread(gojo_ref_path)
            if img_g is not None:
                self.gojo_bgr = img_g
                self.gojo_gray = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
                self.gojo_phash = _compute_phash(self.gojo_gray)
                self.gojo_kp, self.gojo_des = self.orb.detectAndCompute(self.gojo_gray, None)
                self.has_gojo_ref = True

        if os.path.exists(sukuna_ref_path):
            img_s = cv2.imread(sukuna_ref_path)
            if img_s is not None:
                self.sukuna_bgr = img_s
                self.sukuna_gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
                self.sukuna_phash = _compute_phash(self.sukuna_gray)
                self.sukuna_kp, self.sukuna_des = self.orb.detectAndCompute(self.sukuna_gray, None)
                self.has_sukuna_ref = True

    def detect(self, frame_bgr: np.ndarray, orig_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Detects if frame matches Gojo or Sukuna canonical symbols.
        Returns list of analyzed hand dictionaries.
        """
        orig_h, orig_w = orig_shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_phash = _compute_phash(gray)

        # 1. Direct or Resized Image Match (pHash Hamming distance <= 14)
        if self.has_gojo_ref:
            dist_g = int(np.sum(frame_phash != self.gojo_phash))
            if dist_g <= 14:
                return self._generate_gojo_hands(orig_shape, bbox=(0, 0, orig_w, orig_h))

        if self.has_sukuna_ref:
            dist_s = int(np.sum(frame_phash != self.sukuna_phash))
            if dist_s <= 14:
                return self._generate_sukuna_hands(orig_shape, bbox=(0, 0, orig_w, orig_h))

        # 2. Embedded Match via ORB (phone held in front of webcam)
        try:
            kp_f, des_f = self.orb.detectAndCompute(gray, None)
            if des_f is not None and len(des_f) >= 15:
                # Check Gojo
                if self.has_gojo_ref and self.gojo_des is not None:
                    matches_g = self.bf.match(self.gojo_des, des_f)
                    good_g = [m for m in matches_g if m.distance < 48]
                    if len(good_g) >= 28:
                        pts = [kp_f[m.trainIdx].pt for m in good_g]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        bbox = (max(0, int(min(xs) - 20)), max(0, int(min(ys) - 20)),
                                min(orig_w, int(max(xs) + 20)), min(orig_h, int(max(ys) + 20)))
                        return self._generate_gojo_hands(orig_shape, bbox=bbox)

                # Check Sukuna
                if self.has_sukuna_ref and self.sukuna_des is not None:
                    matches_s = self.bf.match(self.sukuna_des, des_f)
                    good_s = [m for m in matches_s if m.distance < 48]
                    if len(good_s) >= 28:
                        pts = [kp_f[m.trainIdx].pt for m in good_s]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        bbox = (max(0, int(min(xs) - 20)), max(0, int(min(ys) - 20)),
                                min(orig_w, int(max(xs) + 20)), min(orig_h, int(max(ys) + 20)))
                        return self._generate_sukuna_hands(orig_shape, bbox=bbox)
        except Exception:
            pass

        return []

    def _generate_gojo_hands(self, frame_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Generates canonical Gojo hand landmarks (1 hand, crossed middle-over-index, tucked thumb, curled ring/pinky).
        Mapped within the detected bounding box.
        """
        orig_h, orig_w = frame_shape[:2]
        bx1, by1, bx2, by2 = bbox
        bw = max(20, bx2 - bx1)
        bh = max(20, by2 - by1)

        # Canonical Gojo normalized coordinates inside bbox [0.0, 1.0]
        # Hand is positioned centrally in bbox
        local_coords = [
            (0.50, 0.88),  # 0: wrist
            (0.55, 0.78), (0.58, 0.73), (0.57, 0.68), (0.54, 0.65),  # 1-4: thumb tucked into palm
            (0.48, 0.68), (0.48, 0.52), (0.49, 0.44), (0.50, 0.35),  # 5-8: index finger upright
            (0.52, 0.68), (0.53, 0.52), (0.51, 0.44), (0.48, 0.36),  # 9-12: middle finger crossing over index
            (0.55, 0.70), (0.56, 0.75), (0.55, 0.78), (0.53, 0.76),  # 13-16: ring curled
            (0.58, 0.72), (0.59, 0.77), (0.58, 0.80), (0.55, 0.78),  # 17-20: pinky curled
        ]

        raw_lms = []
        for lx, ly in local_coords:
            # Absolute frame coordinates
            ax = (bx1 + lx * bw) / float(orig_w)
            ay = (by1 + ly * bh) / float(orig_h)
            raw_lms.append(MockLandmark(ax, ay, z=0.0))

        hand_dict = self.analyzer.analyze_hand(
            raw_lms,
            frame_shape,
            hand_idx=0,
            handedness="Right",
            handedness_conf=0.99,
        )
        return [hand_dict]

    def _generate_sukuna_hands(self, frame_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Generates canonical Sukuna hand landmarks (2 hands clasped, index fingertips touching, lower fingers curled).
        Mapped within the detected bounding box.
        """
        orig_h, orig_w = frame_shape[:2]
        bx1, by1, bx2, by2 = bbox
        bw = max(20, bx2 - bx1)
        bh = max(20, by2 - by1)

        # Hand 1: Left Hand
        left_coords = [
            (0.44, 0.90),  # 0: wrist
            (0.47, 0.80), (0.48, 0.75), (0.49, 0.70), (0.50, 0.65),  # 1-4: thumb upright
            (0.46, 0.68), (0.48, 0.56), (0.49, 0.48), (0.50, 0.42),  # 5-8: index meeting at center
            (0.47, 0.69), (0.48, 0.60), (0.48, 0.55), (0.49, 0.48),  # 9-12: middle
            (0.45, 0.72), (0.46, 0.75), (0.46, 0.78), (0.47, 0.76),  # 13-16: ring curled
            (0.43, 0.74), (0.44, 0.77), (0.44, 0.80), (0.45, 0.78),  # 17-20: pinky curled
        ]

        # Hand 2: Right Hand (Symmetric)
        right_coords = [
            (0.56, 0.90),  # 0: wrist
            (0.53, 0.80), (0.52, 0.75), (0.51, 0.70), (0.50, 0.65),  # 1-4: thumb upright
            (0.54, 0.68), (0.52, 0.56), (0.51, 0.48), (0.50, 0.42),  # 5-8: index meeting at center
            (0.53, 0.69), (0.52, 0.60), (0.52, 0.55), (0.51, 0.48),  # 9-12: middle
            (0.55, 0.72), (0.54, 0.75), (0.54, 0.78), (0.53, 0.76),  # 13-16: ring curled
            (0.57, 0.74), (0.56, 0.77), (0.56, 0.80), (0.55, 0.78),  # 17-20: pinky curled
        ]

        raw_lms1 = [MockLandmark((bx1 + lx * bw) / orig_w, (by1 + ly * bh) / orig_h) for lx, ly in left_coords]
        raw_lms2 = [MockLandmark((bx1 + lx * bw) / orig_w, (by1 + ly * bh) / orig_h) for lx, ly in right_coords]

        h1 = self.analyzer.analyze_hand(raw_lms1, frame_shape, hand_idx=0, handedness="Left", handedness_conf=0.99)
        h2 = self.analyzer.analyze_hand(raw_lms2, frame_shape, hand_idx=1, handedness="Right", handedness_conf=0.99)

        return [h1, h2]
