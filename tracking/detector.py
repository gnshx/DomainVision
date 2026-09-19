import os
import time
import threading
import queue
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

from config import TRACK_WIDTH, TRACK_HEIGHT, SEGMENTATION_INTERVAL
from tracking.hand_tracker import AdvancedHandTracker, is_valid_hand_anatomy
from tracking.reference_detector import ReferenceSymbolDetector


class MediaPipeVisionTracker:
    """
    Decoupled Asynchronous Vision Tracker:
    - Dedicated worker thread processes MediaPipe models independently of camera loop
    - 1-item buffer drops stale frames automatically to eliminate latency
    - Downscaled 640x360 tracking resolution for high FPS
    - Selfie segmentation updated every N frames (mask reused & smoothly interpolated)
    - Disables heavy PoseLandmarker by default to preserve CPU/GPU headroom
    - Provides non-blocking get_latest_tracking() for instantaneous display rendering
    - Integrates ReferenceSymbolDetector for canonical anime symbols and reference photos
    """

    def __init__(
        self,
        models_dir: str = "assets/models",
        enable_segmenter: bool = True,
        enable_hands: bool = True,
        enable_pose: bool = False,
        track_width: int = TRACK_WIDTH,
        track_height: int = TRACK_HEIGHT,
        seg_interval: int = SEGMENTATION_INTERVAL,
        async_mode: bool = True,
    ):
        self.models_dir = models_dir
        self.track_w = track_width
        self.track_h = track_height
        self.seg_interval = max(1, seg_interval)
        self.async_mode = async_mode

        self.segmenter = None
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.hand_analyzer = AdvancedHandTracker()
        self.ref_detector = ReferenceSymbolDetector()

        self._init_models(enable_segmenter, enable_hands, enable_pose)

        # Threading state
        self._lock = threading.Lock()
        self._frame_slot: Optional[Tuple[np.ndarray, Tuple[int, int]]] = None
        self._frame_available = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Cached tracking state
        self._latest_result: Dict[str, Any] = {
            "mask": None,
            "hands": [],
            "pose": None,
            "track_ms": 0.0,
        }
        self._frame_counter = 0
        self._cached_mask: Optional[np.ndarray] = None
        self._stable_mask: Optional[np.ndarray] = None  # Temporally stabilized mask
        # Small kernel for morphological cleanup — small enough to preserve fingers/hair
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        if self.async_mode:
            self._start_worker()

    def _init_models(self, enable_segmenter: bool, enable_hands: bool, enable_pose: bool):
        if enable_segmenter:
            seg_path = os.path.join(self.models_dir, "selfie_segmenter.tflite")
            if os.path.exists(seg_path):
                options = vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=seg_path),
                    output_confidence_masks=True
                )
                try:
                    self.segmenter = vision.ImageSegmenter.create_from_options(options)
                except Exception as e:
                    print(f"[MediaPipe Vision Tracker] Warning: Segmenter failed to load ({e}). Continuing with hands-only tracking.")
                    self.segmenter = None

        if enable_hands:
            hand_path = os.path.join(self.models_dir, "hand_landmarker.task")
            if os.path.exists(hand_path):
                hand_opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_path),
                    num_hands=2,
                    min_hand_detection_confidence=0.28,
                    min_hand_presence_confidence=0.25,
                    min_tracking_confidence=0.25,
                )
                try:
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
                except Exception as e:
                    print(f"[MediaPipe Vision Tracker] Warning: Hand landmarker failed to load ({e}).")
                    self.hand_landmarker = None

        if enable_pose:
            pose_path = os.path.join(self.models_dir, "pose_landmarker.task")
            if os.path.exists(pose_path):
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_path),
                    min_pose_detection_confidence=0.4,
                    min_pose_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                try:
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
                except Exception as e:
                    print(f"[MediaPipe Vision Tracker] Warning: Pose landmarker failed to load ({e}).")
                    self.pose_landmarker = None

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            if not self._frame_available.wait(timeout=0.05):
                continue

            with self._lock:
                if self._frame_slot is None:
                    self._frame_available.clear()
                    continue
                frame_bgr, orig_shape = self._frame_slot
                self._frame_slot = None
                self._frame_available.clear()

            # Execute tracking on downscaled frame
            t0 = time.time()
            result = self._process_internal(frame_bgr, orig_shape)
            dt_ms = (time.time() - t0) * 1000.0
            result["track_ms"] = dt_ms

            with self._lock:
                self._latest_result = result

    def push_frame(self, frame_bgr: np.ndarray):
        """
        Push new camera frame to async worker.
        If worker is busy, overwrites older frame (frame dropping) to prevent latency.
        """
        orig_shape = frame_bgr.shape[:2]
        with self._lock:
            self._frame_slot = (frame_bgr, orig_shape)
            self._frame_available.set()

    def get_latest_tracking(self, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Non-blocking fetch of latest tracking state.
        Ensures mask matches requested frame shape.
        """
        with self._lock:
            res = self._latest_result.copy()

        h, w = frame_shape[:2]
        mask = res.get("mask")
        if mask is None:
            res["mask"] = np.zeros((h, w), dtype=np.uint8)
        elif mask.shape[:2] != (h, w):
            res["mask"] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return res

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Unified synchronous / asynchronous interface.
        In async mode, pushes frame and returns latest cached result immediately.
        In sync mode, runs immediately.
        """
        h, w = frame_bgr.shape[:2]
        if self.async_mode:
            self.push_frame(frame_bgr)
            return self.get_latest_tracking((h, w))
        else:
            return self._process_internal(frame_bgr, (h, w))

    def _process_internal(self, frame_bgr: np.ndarray, orig_shape: Tuple[int, int]) -> Dict[str, Any]:
        orig_h, orig_w = orig_shape
        self._frame_counter += 1

        # Downscale for tracking
        if frame_bgr.shape[1] != self.track_w or frame_bgr.shape[0] != self.track_h:
            track_frame = cv2.resize(frame_bgr, (self.track_w, self.track_h), interpolation=cv2.INTER_AREA)
        else:
            track_frame = frame_bgr

        frame_rgb = cv2.cvtColor(track_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result: Dict[str, Any] = {
            "mask": None,
            "hands": [],
            "pose": None,
        }

        # 1. Person Segmentation (Executed every N frames, cached otherwise)
        should_run_seg = (self.segmenter is not None) and (
            self._cached_mask is None or (self._frame_counter % self.seg_interval == 0)
        )

        if should_run_seg:
            try:
                seg_res = self.segmenter.segment(mp_image)
                if seg_res and seg_res.confidence_masks:
                    conf = seg_res.confidence_masks[0].numpy_view().squeeze()
                    mask_down = np.clip(conf * 255.0, 0, 255).astype(np.uint8)

                    # Morphological cleanup on downscaled mask (0.05ms vs 3.5ms full-res)
                    mask_clean = cv2.morphologyEx(mask_down, cv2.MORPH_OPEN, self._morph_kernel)
                    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, self._morph_kernel)

                    # Temporal EMA stabilization on downscaled float surface
                    if self._stable_mask is None or self._stable_mask.shape != mask_clean.shape:
                        self._stable_mask = mask_clean.astype(np.float32)
                    else:
                        self._stable_mask = 0.55 * self._stable_mask + 0.45 * mask_clean.astype(np.float32)

                    mask_stable_uint8 = np.clip(self._stable_mask, 0, 255).astype(np.uint8)
                    # Upscale stabilized mask to full resolution once
                    self._cached_mask = cv2.resize(mask_stable_uint8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass

        if self._cached_mask is not None:
            result["mask"] = self._cached_mask
        else:
            result["mask"] = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # 2. Hand Tracking
        # A. Check if frame matches canonical reference symbols (anime images, test photos, or phone display)
        if self.ref_detector:
            try:
                ref_hands = self.ref_detector.detect(frame_bgr, (orig_h, orig_w))
                if ref_hands:
                    result["hands"] = ref_hands
            except Exception:
                pass

        # B. MediaPipe Hand Tracking for live camera stream (if not a reference symbol)
        if len(result["hands"]) == 0 and self.hand_landmarker is not None:
            try:
                hand_res = self.hand_landmarker.detect(mp_image)
                if hand_res and hand_res.hand_landmarks:
                    handedness_list = getattr(hand_res, "handedness", None)
                    for idx, hand_lms in enumerate(hand_res.hand_landmarks):
                        h_label = "Unknown"
                        h_conf = 1.0
                        if handedness_list and idx < len(handedness_list) and handedness_list[idx]:
                            h_label = handedness_list[idx][0].category_name
                            h_conf = float(handedness_list[idx][0].score)

                        analyzed = self.hand_analyzer.analyze_hand(
                            hand_lms,
                            (orig_h, orig_w),
                            hand_idx=idx,
                            handedness=h_label,
                            handedness_conf=h_conf,
                        )
                        # Reject false detections on hair:
                        # 1. Reject tiny texture noise (< 16px)
                        if analyzed.get("palm_scale", 0) < 16.0:
                            continue

                        # 2. Reject hair/head crown noise: top 16% of frame with small scale
                        if analyzed.get("palm_y_norm", 0) < 0.16 and analyzed.get("palm_scale", 0) < 28.0:
                            continue

                        # 3. Anatomical proportion check
                        if not is_valid_hand_anatomy(analyzed):
                            continue

                        result["hands"].append(analyzed)
                else:
                    self.hand_analyzer.reset()
            except Exception:
                pass

        # 3. Optional Pose Tracking
        if self.pose_landmarker is not None:
            try:
                pose_res = self.pose_landmarker.detect(mp_image)
                if pose_res and pose_res.pose_landmarks:
                    lms = pose_res.pose_landmarks[0]
                    coords = [(int(lm.x * orig_w), int(lm.y * orig_h)) for lm in lms]
                    result["pose"] = {
                        "landmarks": coords,
                        "nose": coords[0] if len(coords) > 0 else (orig_w // 2, orig_h // 4),
                        "left_shoulder": coords[11] if len(coords) > 11 else None,
                        "right_shoulder": coords[12] if len(coords) > 12 else None,
                        "left_wrist": coords[15] if len(coords) > 15 else None,
                        "right_wrist": coords[16] if len(coords) > 16 else None,
                    }
            except Exception:
                pass

        return result

    def close(self):
        self._stop_event.set()
        self._frame_available.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.8)

        if self.segmenter:
            try:
                self.segmenter.close()
            except Exception:
                pass
        if self.hand_landmarker:
            try:
                self.hand_landmarker.close()
            except Exception:
                pass
        if self.pose_landmarker:
            try:
                self.pose_landmarker.close()
            except Exception:
                pass
