import os
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from tracking.hand_tracker import AdvancedHandTracker


class MediaPipeVisionTracker:
    """
    Unified MediaPipe Tasks tracker:
    - Person / Selfie Segmentation
    - Dual Hand Landmark Tracking
    - Body Pose Tracking
    """

    def __init__(
        self,
        models_dir: str = "assets/models",
        enable_segmenter: bool = True,
        enable_hands: bool = True,
        enable_pose: bool = True,
    ):
        self.models_dir = models_dir

        self.segmenter = None
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.hand_analyzer = AdvancedHandTracker()

        if enable_segmenter:
            seg_path = os.path.join(models_dir, "selfie_segmenter.tflite")
            if os.path.exists(seg_path):
                options = vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=seg_path),
                    output_confidence_masks=True
                )
                self.segmenter = vision.ImageSegmenter.create_from_options(options)

        if enable_hands:
            hand_path = os.path.join(models_dir, "hand_landmarker.task")
            if os.path.exists(hand_path):
                hand_opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_path),
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                    min_hand_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

        if enable_pose:
            pose_path = os.path.join(models_dir, "pose_landmarker.task")
            if os.path.exists(pose_path):
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_path),
                    min_pose_detection_confidence=0.4,
                    min_pose_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Process a BGR video frame and return:
        - mask: uint8 person segmentation mask (h, w), values 0 to 255
        - hands: list of hand dictionaries with landmarks, palm_center, etc.
        - pose: pose landmarks dictionary
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result: Dict[str, Any] = {
            "mask": np.zeros((h, w), dtype=np.uint8),
            "hands": [],
            "pose": None,
        }

        # 1. Person Segmentation
        if self.segmenter is not None:
            try:
                seg_res = self.segmenter.segment(mp_image)
                if seg_res and seg_res.confidence_masks:
                    conf = seg_res.confidence_masks[0].numpy_view().squeeze()
                    mask_uint8 = np.clip(conf * 255.0, 0, 255).astype(np.uint8)
                    if mask_uint8.shape != (h, w):
                        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
                    result["mask"] = mask_uint8
            except Exception:
                pass

        # 2. Hand Tracking
        if self.hand_landmarker is not None:
            try:
                hand_res = self.hand_landmarker.detect(mp_image)
                if hand_res and hand_res.hand_landmarks:
                    for idx, hand_lms in enumerate(hand_res.hand_landmarks):
                        analyzed = self.hand_analyzer.analyze_hand(hand_lms, (h, w), hand_idx=idx)
                        result["hands"].append(analyzed)
                else:
                    self.hand_analyzer.reset()
            except Exception:
                pass

        # 3. Pose Tracking
        if self.pose_landmarker is not None:
            try:
                pose_res = self.pose_landmarker.detect(mp_image)
                if pose_res and pose_res.pose_landmarks:
                    lms = pose_res.pose_landmarks[0]
                    coords = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                    # Landmark indices:
                    # 11: left_shoulder, 12: right_shoulder
                    # 13: left_elbow, 14: right_elbow
                    # 15: left_wrist, 16: right_wrist
                    # 0: nose
                    result["pose"] = {
                        "landmarks": coords,
                        "nose": coords[0] if len(coords) > 0 else (w // 2, h // 4),
                        "left_shoulder": coords[11] if len(coords) > 11 else None,
                        "right_shoulder": coords[12] if len(coords) > 12 else None,
                        "left_elbow": coords[13] if len(coords) > 13 else None,
                        "right_elbow": coords[14] if len(coords) > 14 else None,
                        "left_wrist": coords[15] if len(coords) > 15 else None,
                        "right_wrist": coords[16] if len(coords) > 16 else None,
                    }
            except Exception:
                pass

        return result

    def close(self):
        if self.segmenter:
            self.segmenter.close()
        if self.hand_landmarker:
            self.hand_landmarker.close()
        if self.pose_landmarker:
            self.pose_landmarker.close()
