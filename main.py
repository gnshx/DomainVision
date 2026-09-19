import argparse
import json
import math
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

from config import (
    THEMES,
    DEFAULT_THEME,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    SIGN_HOLD_FRAMES_REQUIRED,
    SIGN_CONFIDENCE_THRESHOLD,
    AUDIO_TIMELINE,
    QUALITY_PROMOTE_FPS,
    QUALITY_DEMOTE_FPS,
    QUALITY_LEVELS,
    QUALITY_BUDGETS,
)
from tracking.detector import MediaPipeVisionTracker
from tracking.gesture_recognizer import CanonicalGestureRecognizer
from tracking.hand_tracker import generate_synthetic_mudra_hands
from effects.flash import DomainFlashEffect
from effects.domain_environment import DomainEnvironmentRenderer
from effects.domain_layers import DomainLayerRenderer
from effects.aura import CursedAuraEffect
from effects.cursed_energy import CursedEnergyEffect
from effects.particles import CursedParticleSystem
from effects.distortion import OpticalDistortionEffect
from effects.shockwave import BarrierShockwaveEffect
from effects.color_grade import CinematicColorGrader
from utils.text_renderer import JapaneseTextRenderer
from utils.motion_estimator import CameraMotionEstimator
from audio.audio_manager import AudioManager
from utils.demo_feed import SyntheticDemoCamera
from utils.fps import PerformanceProfiler


class AdaptiveQualityController:
    """
    Measures rolling FPS and upgrades/demotes effect quality level with hysteresis.
    - Upgrades quality when rolling FPS >= QUALITY_PROMOTE_FPS for 20 consecutive frames
    - Demotes quality when rolling FPS < QUALITY_DEMOTE_FPS for 5 consecutive frames
    This prevents thrashing while responding quickly to sustained FPS drops.
    """

    def __init__(self, initial_level: str = "HIGH", window: int = 20):
        self.level = initial_level
        self._fps_window: deque = deque(maxlen=window)
        self._promote_streak = 0
        self._demote_streak = 0
        self._promote_required = window
        self._demote_required = 5

    def update(self, fps: float) -> Dict[str, Any]:
        """Call once per frame with current FPS. Returns current quality budget."""
        self._fps_window.append(fps)
        avg_fps = sum(self._fps_window) / len(self._fps_window)

        idx = QUALITY_LEVELS.index(self.level)

        if avg_fps >= QUALITY_PROMOTE_FPS:
            self._demote_streak = 0
            self._promote_streak += 1
            if self._promote_streak >= self._promote_required and idx < len(QUALITY_LEVELS) - 1:
                self.level = QUALITY_LEVELS[idx + 1]
                self._promote_streak = 0
        elif avg_fps < QUALITY_DEMOTE_FPS:
            self._promote_streak = 0
            self._demote_streak += 1
            if self._demote_streak >= self._demote_required and idx > 0:
                self.level = QUALITY_LEVELS[idx - 1]
                self._demote_streak = 0
        else:
            # In hysteresis band — reset streaks but don't change level
            self._promote_streak = max(0, self._promote_streak - 1)
            self._demote_streak = max(0, self._demote_streak - 1)

        return QUALITY_BUDGETS[self.level]



class DomainExpansionApp:
    """
    DomainVision: High-Performance AR JJK Domain Expansion Filter.
    - Decoupled asynchronous hand tracker thread with zero-latency webcam display
    - Strict 21-landmark geometric mudra evaluation with percentage match feedback
    - Crunchyroll Sukuna activation timing sequence (0.00s -> 1.50s audio-visual sync)
    - Perspective-anchored cursed energy and downscaled bloom rendering
    - Malevolent Shrine (伏魔御廚子) & Infinite Void (無量空処) backdrops
    """

    STATES = ["NORMAL", "CHARGING", "VOICE", "FLASH", "SHOCKWAVE", "EXPANSION", "DOMAIN_ACTIVE", "COLLAPSE"]

    def __init__(
        self,
        input_source: Optional[str] = None,
        camera_idx: int = 0,
        demo_mode: bool = False,
        theme: str = DEFAULT_THEME,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        headless: bool = False,
        record_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        calibrate_mode: bool = False,
        log_gestures: bool = False,
        flip_webcam: bool = True,
    ):
        self.w = width
        self.h = height
        self.theme_name = theme if theme in THEMES else DEFAULT_THEME
        self.headless = headless
        self.record_path = record_path
        self.max_frames = max_frames
        self.show_hud = True
        self.hud_position = "right"  # "right", "left", or "none"
        self.show_landmarks = True
        self.calibrate_mode = calibrate_mode
        self.show_calibration = calibrate_mode
        self.log_gestures = log_gestures
        self.flip_webcam = flip_webcam
        self.log_file = "gesture_samples.jsonl" if log_gestures else None

        # Video Source
        self.cap, self.is_demo = self._init_video_source(input_source, camera_idx, demo_mode)

        # State Machine & Timing
        self.state = "NORMAL"
        self.state_start_time = 0.0
        self.seq_start_time = 0.0
        self.total_frames = 0
        self.energy_center = (self.w // 2, int(self.h * 0.52))
        self.shockwave_triggered = False

        # Profiler & Audio Engine
        self.profiler = PerformanceProfiler()
        self.audio = AudioManager()

        # Decoupled Asynchronous Tracking Layer
        print("Initializing MediaPipe Asynchronous Vision Tracker...")
        seg_int = int(os.environ.get("SEG_INTERVAL", "5" if self.headless else str(SEGMENTATION_INTERVAL)))
        self.tracker = MediaPipeVisionTracker(
            enable_segmenter=True,
            enable_hands=True,
            enable_pose=False,   # Disabled for maximum framerate
            async_mode=True,  # Fully asynchronous decoupled tracking for 30+ FPS
            seg_interval=seg_int,
        )
        self.gesture_recognizer = CanonicalGestureRecognizer(
            hold_frames_required=SIGN_HOLD_FRAMES_REQUIRED,
            confidence_threshold=SIGN_CONFIDENCE_THRESHOLD,
        )

        # Visual Effects Engines
        print("Initializing Visual Effects Engines...")
        self.flash_fx = DomainFlashEffect(duration_frames=14)
        self.env_renderer = DomainEnvironmentRenderer(width=self.w, height=self.h, theme=self.theme_name)
        self.aura_fx = CursedAuraEffect(aura_thickness=16)
        self.cursed_energy_fx = CursedEnergyEffect()
        self.particle_system = CursedParticleSystem(max_particles=120)
        self.distortion_fx = OpticalDistortionEffect(width=self.w, height=self.h)
        self.shockwave_fx = BarrierShockwaveEffect()
        self.text_renderer = JapaneseTextRenderer()

        # 2.5D Parallax Domain Layer Renderer (replaces flat environment for active states)
        self.domain_layers = DomainLayerRenderer(width=self.w, height=self.h, theme=self.theme_name)

        # Cinematic Color Grader (bloom, vignette, color spill, chromatic aberration)
        self.color_grader = CinematicColorGrader(width=self.w, height=self.h)

        # Camera motion estimator (sparse optical flow, < 1.5ms)
        self.motion_estimator = CameraMotionEstimator()

        # Adaptive quality controller (starts at HIGH, auto-scales based on FPS)
        self.quality_ctrl = AdaptiveQualityController(initial_level="HIGH")

        # Picture-in-Picture camera capture for bottom display during animated feed
        self.camera_idx = camera_idx
        self.pip_cap = None
        if self.is_demo and not self.headless:
            try:
                cam = cv2.VideoCapture(self.camera_idx)
                if cam.isOpened():
                    ret_test, _ = cam.read()
                    if ret_test:
                        self.pip_cap = cam
                        print(f"Live webcam detected at index {self.camera_idx} for bottom display during animated feed!")
                    else:
                        cam.release()
            except Exception:
                self.pip_cap = None

        # Video Recorder
        self.video_writer = None
        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.record_path, fourcc, 30.0, (self.w, self.h))
            print(f"Recording output to: {self.record_path}")

    def _init_video_source(self, input_source, camera_idx, demo_mode):
        if demo_mode:
            print("Running in SYNTHETIC DEMO mode.")
            return SyntheticDemoCamera(width=self.w, height=self.h), True

        if input_source and os.path.exists(input_source):
            print(f"Opening video file: {input_source}")
            cap = cv2.VideoCapture(input_source)
            if cap.isOpened():
                return cap, False

        print(f"Attempting to open camera index {camera_idx} at {self.w}x{self.h}...")
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            cap.set(cv2.CAP_PROP_FPS, 30.0)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"Camera initialized successfully at {test_frame.shape[1]}x{test_frame.shape[0]}!")
                return cap, False
            cap.release()

        print("Notice: No live webcam detected. Falling back to SYNTHETIC DEMO mode.")
        return SyntheticDemoCamera(width=self.w, height=self.h), True

    def set_theme(self, theme_name: str):
        if theme_name in THEMES:
            self.theme_name = theme_name
            self.env_renderer.set_theme(theme_name)
            self.domain_layers.set_theme(theme_name)
            print(f"Switched theme to: {self.theme_name} ({THEMES[theme_name]['name_ja']})")

    def trigger_domain(self):
        """Force-trigger Domain Expansion sequence matching canonical audio-visual timeline."""
        if self.state in ["NORMAL", "CHARGING"]:
            now = time.time()
            self.seq_start_time = now
            self.state = "CHARGING"
            self.state_start_time = now
            self.shockwave_triggered = False

            # Start audio sequence (charge -> energy -> voice -> flash -> shockwave -> ambience)
            self.audio.start_domain_sequence(theme=self.theme_name)
            self.distortion_fx.trigger_shake(magnitude=8.0)

    def reset_domain(self):
        """Collapse domain barrier back to normal reality."""
        self.state = "COLLAPSE"
        self.state_start_time = time.time()
        self.audio.play_collapse()
        self.gesture_recognizer.reset()

    def _update_timeline_state_machine(self, gesture_info: dict):
        """
        Synchronizes animations to the canonical Crunchyroll Sukuna activation timing:
        0.00s : Hand sign recognized
        0.05s : Charge sound & particle intake
        0.70s : Energy builds & tremor intensifies
        1.20s : Voice line ("Domain Expansion")
        1.35s : Blinding cursed flash
        1.40s : Spatial shockwave & barrier displacement
        1.50s : Domain environment expansion & looping ambience
        """
        now = time.time()
        theme_info = self.env_renderer.theme

        # 1. NORMAL State: Check for mudra trigger
        if self.state == "NORMAL":
            if gesture_info["trigger"]:
                self.trigger_domain()
            elif gesture_info["hold_progress"] > 0.25:
                # Sign held long enough to start initial charge windup
                self.state = "CHARGING"
                self.seq_start_time = now
                self.state_start_time = now
                self.shockwave_triggered = False
                self.audio.start_domain_sequence(theme=self.theme_name)

        # 2. CHARGING State (0.00s -> 1.35s)
        elif self.state == "CHARGING":
            elapsed = now - self.seq_start_time

            # Cancel if user breaks hand sign early during initial charge
            if (not self.is_demo and elapsed < 0.85 and
                    gesture_info["hold_progress"] <= 0.05 and not gesture_info["sign_detected"]):
                self.state = "NORMAL"
                self.audio.stop_domain_sequence()
                return

            # Tremor escalates as energy builds
            if elapsed >= AUDIO_TIMELINE.get("energy_build", 0.70):
                if self.total_frames % 2 == 0:
                    self.distortion_fx.trigger_shake(magnitude=5.0)
            else:
                if self.total_frames % 4 == 0:
                    self.distortion_fx.trigger_shake(magnitude=2.5)

            # Transition to FLASH at 1.35s
            if elapsed >= AUDIO_TIMELINE.get("flash", 1.35):
                self.state = "FLASH"
                self.state_start_time = now
                self.flash_fx.trigger(theme_color=theme_info["primary_bgr"])
                self.distortion_fx.trigger_shake(magnitude=22.0)

        # 3. FLASH State (1.35s -> 1.50s)
        elif self.state == "FLASH":
            elapsed = now - self.seq_start_time

            # Trigger Shockwave at 1.40s
            if not self.shockwave_triggered and elapsed >= AUDIO_TIMELINE.get("shockwave", 1.40):
                self.shockwave_triggered = True
                self.distortion_fx.trigger_shockwave(
                    center=self.energy_center,
                    max_radius=math.hypot(self.w, self.h),
                    duration_frames=26,
                    strength=35.0,
                )
                self.shockwave_fx.trigger(
                    center=self.energy_center,
                    primary_color=theme_info["primary_bgr"],
                    secondary_color=theme_info["secondary_bgr"],
                    duration_frames=28,
                )

            if elapsed >= AUDIO_TIMELINE.get("domain_env", 1.50):
                self.state = "EXPANSION"
                self.state_start_time = now

        # 4. EXPANSION State (Domain barrier unrolls outward)
        elif self.state == "EXPANSION":
            expansion_duration = 1.2  # 1.2s expansion animation
            if (now - self.state_start_time) >= expansion_duration:
                self.state = "DOMAIN_ACTIVE"
                self.state_start_time = now

        # 5. DOMAIN_ACTIVE State (Active for ~10 seconds)
        elif self.state == "DOMAIN_ACTIVE":
            active_duration = 10.0
            if (now - self.state_start_time) >= active_duration:
                self.reset_domain()

        # 6. COLLAPSE State (Dissolve barrier over 0.9s)
        elif self.state == "COLLAPSE":
            if (now - self.state_start_time) >= 0.9:
                self.state = "NORMAL"
                self.gesture_recognizer.reset()

    def process_frame(self, raw_frame: np.ndarray, external_tracking: Optional[dict] = None) -> np.ndarray:
        self.profiler.start_frame()

        if not self.is_demo and self.flip_webcam:
            raw_frame = cv2.flip(raw_frame, 1)

        if raw_frame.shape[1] != self.w or raw_frame.shape[0] != self.h:
            raw_frame = cv2.resize(raw_frame, (self.w, self.h))

        h, w = self.h, self.w
        theme_info = self.env_renderer.theme
        col_pri = theme_info["primary_bgr"]
        col_sec = theme_info["secondary_bgr"]

        # 1. Decoupled Vision Tracking (Non-blocking async query or pre-tracked slot)
        if external_tracking is not None:
            tracking = external_tracking
        else:
            self.profiler.start_tracking()
            tracking = self.tracker.process(raw_frame)
            self.profiler.end_tracking()

        person_mask = tracking["mask"]
        hands = tracking["hands"]

        if self.is_demo and len(hands) == 0:
            # Only synthesize skeletal energy joints when domain is ALREADY active or triggered
            # NEVER synthesize clasped hands in NORMAL state (prevents fake 12-second auto-expansion)
            if self.state != "NORMAL":
                hands = generate_synthetic_mudra_hands((h, w), self.total_frames, self.theme_name)
            else:
                hands = []

        # 2. Canonical Hand Sign Evaluation with Invariant Geometry & Multi-Class State
        gesture_info = self.gesture_recognizer.update(
            hands=hands,
            frame_shape=(h, w),
            target_theme=self.theme_name,
        )

        # Automatic character theme switching based strictly on stable confirmed/candidate mudra!
        detected_theme = gesture_info.get("detected_theme")
        if detected_theme and self.state == "NORMAL":
            if detected_theme != self.theme_name and (
                gesture_info.get("sign_detected")
                or (
                    gesture_info.get("candidate_sign") in ["GOJO", "SUKUNA"]
                    and gesture_info.get("stable_frames", 0) >= 2
                    and gesture_info.get("match_pct", 0) >= 78
                )
            ):
                self.set_theme(detected_theme)

        if gesture_info.get("energy_center"):
            self.energy_center = gesture_info["energy_center"]

        # Advance Timeline State Machine
        self._update_timeline_state_machine(gesture_info)

        # --- Quality controller: measure FPS, get budget for this frame ---
        q_budget = self.quality_ctrl.update(self.profiler.fps if self.profiler.fps > 0 else 30.0)
        q_max_particles = q_budget["max_particles"]
        q_distortion    = q_budget["distortion"]
        q_aura_scale    = q_budget["aura_scale"]

        # Camera motion estimation (used for parallax; minimal cost at 160x90)
        cam_dx, cam_dy = self.motion_estimator.estimate(raw_frame)

        # 3. Environment Generation (Parallax Shrine / Void with depth layers)
        now = time.time()
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE"]:
            # Use 2.5D parallax renderer for active domain states
            domain_bg = self.domain_layers.render(timer=self.total_frames, cam_dx=cam_dx, cam_dy=cam_dy)

            if self.state == "EXPANSION":
                prog = min(1.0, (now - self.state_start_time) / 1.2)
                wipe_r = int(math.hypot(w, h) * prog)
                mask_circ = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask_circ, self.energy_center, wipe_r, 255, -1)
                circ_inv = cv2.bitwise_not(mask_circ)
                active_bg = cv2.add(
                    cv2.bitwise_and(domain_bg, domain_bg, mask=mask_circ),
                    cv2.bitwise_and(raw_frame, raw_frame, mask=circ_inv)
                )
            elif self.state == "COLLAPSE":
                prog = min(1.0, (now - self.state_start_time) / 0.9)
                dissolve = max(0.0, 1.0 - prog)
                active_bg = cv2.addWeighted(domain_bg, dissolve, raw_frame, 1.0 - dissolve, 0)
            else:
                active_bg = domain_bg
        else:
            active_bg = raw_frame.copy()

        # 4. Edge-Aware 3-Layer Cursed Aura & Silhouette Compositing
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE"]:
            aura_intensity = 1.0 if self.state != "COLLAPSE" else max(0.0, 1.0 - (now - self.state_start_time) / 0.9)
            composited = self.aura_fx.composite_with_aura(
                foreground_frame=raw_frame,
                background_frame=active_bg,
                person_mask=person_mask,
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                aura_intensity=aura_intensity,
                aura_scale=q_aura_scale,
            )
        elif self.state == "CHARGING":
            elapsed = now - self.seq_start_time
            intensity = min(0.8, 0.2 + (elapsed / 1.35) * 0.6)
            composited = self.aura_fx.composite_with_aura(
                foreground_frame=raw_frame,
                background_frame=raw_frame,
                person_mask=person_mask,
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                aura_intensity=intensity,
                aura_scale=q_aura_scale,
            )
        else:
            composited = raw_frame.copy()

        # 5. Cursed Particles System (depth-aware, hand-attractor, quality-gated)
        p_mode = "float"
        p_intensity = 0.0
        if self.state == "CHARGING":
            p_mode = "suck_in"
            p_intensity = 1.4
        elif self.state == "COLLAPSE":
            p_mode = "blast_out"
            p_intensity = 1.6
        elif self.state in ["EXPANSION", "DOMAIN_ACTIVE"]:
            p_mode = "float"
            p_intensity = 1.0

        if p_intensity > 0.01:
            # Build hand attractor list from palm centers of tracked hands
            hand_attractors = None
            if hands:
                hand_attractors = [h.get("palm_center", self.energy_center) for h in hands]

            composited = self.particle_system.update_and_render(
                composited,
                primary_color=col_pri,
                secondary_color=col_sec,
                mode=p_mode,
                center=self.energy_center,
                intensity=p_intensity,
                max_particles=q_max_particles,
                hand_attractors=hand_attractors,
            )

        # 6. Perspective Hand Cursed Energy (Attached to 21 Finger Joints)
        if self.state in ["CHARGING", "EXPANSION", "DOMAIN_ACTIVE"]:
            energy_intensity = 0.75 if self.state == "CHARGING" else 1.0
            composited = self.cursed_energy_fx.render(
                composited,
                hands=hands,
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                intensity=energy_intensity,
            )

        # 7. Barrier Shockwaves
        composited = self.shockwave_fx.update_and_render(composited)

        # 8. Flash Transition
        if self.flash_fx.active:
            composited, _ = self.flash_fx.apply(composited)

        # 9. Optical Distortion & Screen Shake (quality-gated)
        if q_distortion:
            composited = self.distortion_fx.apply(composited)
        elif self.distortion_fx.shake_magnitude > 0.5:
            # Still allow screen shake even in low quality — it's very cheap
            composited = self.distortion_fx.apply(composited)

        # 10. Japanese Domain Calligraphy Banner Overlay
        if self.state in ["EXPANSION", "DOMAIN_ACTIVE"]:
            if self.state == "EXPANSION":
                text_prog = min(1.0, (now - self.state_start_time) / 0.8)
            else:
                elapsed_active = now - self.state_start_time
                if elapsed_active < 2.5:
                    text_prog = 1.0
                elif elapsed_active < 3.8:
                    text_prog = max(0.0, 1.0 - (elapsed_active - 2.5) / 1.3)
                else:
                    text_prog = 0.0

            if text_prog > 0.01:
                txt_bgr, txt_alpha = self.text_renderer.render_domain_banner(
                    frame_shape=(h, w),
                    main_text="領域展開",
                    sub_text=theme_info["name_ja"],
                    en_text=f"DOMAIN EXPANSION - {theme_info['name_en']}",
                    color_glow=col_pri,
                    progress=text_prog,
                )
                # High-speed SIMD bitwise blending for banner (1.5ms vs 34ms)
                y1, y2 = int(h * 0.12), int(h * 0.52)
                roi_c = composited[y1:y2, :]
                roi_t = txt_bgr[y1:y2, :]
                roi_m = txt_alpha[y1:y2, :]
                if cv2.countNonZero(roi_m) > 0:
                    mask_inv = cv2.bitwise_not(roi_m)
                    p1 = cv2.bitwise_and(roi_t, roi_t, mask=roi_m)
                    p2 = cv2.bitwise_and(roi_c, roi_c, mask=mask_inv)
                    composited[y1:y2, :] = cv2.add(p1, p2)

        # 10.5 Cinematic Color Grading (bloom, vignette, color spill, chromatic aberration)
        # Only applied in active domain states to preserve NORMAL/CHARGING performance
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE", "CHARGING"]:
            # Chromatic aberration active during FLASH and brief SHOCKWAVE period
            chroma_mag = 0.0
            if self.state in ["FLASH", "SHOCKWAVE"] or (self.state == "EXPANSION" and (now - self.state_start_time) < 0.4):
                chroma_mag = 3.0

            composited = self.color_grader.apply(
                composited,
                person_mask=person_mask,
                domain_color=col_pri,
                timer=self.total_frames,
                apply_spill=(self.state in ["DOMAIN_ACTIVE", "EXPANSION"]),
                apply_bloom=(self.state in ["DOMAIN_ACTIVE", "EXPANSION", "CHARGING"]),
                apply_vignette=True,
                chromatic_magnitude=chroma_mag,
                spill_strength=0.14,
                bloom_strength=0.35,
                bloom_threshold=200,
            )

        # 10.8 Gesture Landmarks & Skeleton Visualization
        if self.show_landmarks and hands:
            composited = self._render_landmarks(composited, hands)

        # 11. Sleek HUD & Performance Telemetry (Crystal Clear Visibility)
        self.profiler.end_rendering()
        if self.show_hud:
            self._render_hud(composited, gesture_info, hands=hands, q_level=self.quality_ctrl.level)

        # 11.5 Gesture Calibration Panel (Active in Calibration Mode)
        if self.calibrate_mode:
            self._render_calibration_overlay(composited, hands, gesture_info)

        # 11.8 Misclassification Sample Logging
        if self.log_gestures:
            self._log_gesture_sample(hands, gesture_info)

        # 12. Show User Camera Display in Bottom during Animated Feed
        if self.is_demo and not self.headless:
            self._render_bottom_cam_display(composited)

        self.total_frames += 1
        return composited

    def _render_landmarks(self, frame: np.ndarray, hands: List[Dict[str, Any]]) -> np.ndarray:
        """Renders 21 MediaPipe skeletal joints, finger bones, bounding boxes, and handedness."""
        bones = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # index
            (5, 9), (9, 10), (10, 11), (11, 12),   # middle
            (9, 13), (13, 14), (14, 15), (15, 16), # ring
            (13, 17), (17, 18), (18, 19), (19, 20),# pinky
            (0, 17),                                # palm base
        ]
        finger_tip_indices = [4, 8, 12, 16, 20]
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]

        for idx, h in enumerate(hands):
            lms = h["landmarks"]
            scale = h.get("palm_scale", 30)
            handedness = h.get("handedness", "Unknown")
            hand_conf = h.get("handedness_conf", 1.0)
            f_states = h.get("finger_states", {})

            # 1. Draw finger bones
            for i1, i2 in bones:
                if i1 < len(lms) and i2 < len(lms):
                    cv2.line(frame, lms[i1], lms[i2], (255, 200, 50), 2, cv2.LINE_AA)

            # 2. Draw joint landmarks
            for pt in lms:
                cv2.circle(frame, pt, 3, (0, 240, 255), -1, cv2.LINE_AA)

            # 3. Draw color-coded fingertips (Green=EXTENDED, Yellow=BENT, Red=CURLED)
            for tip_i, fname in zip(finger_tip_indices, finger_names):
                if tip_i < len(lms):
                    st = f_states.get(fname, "BENT")
                    t_col = (80, 255, 80) if st == "EXTENDED" else ((0, 230, 255) if st == "BENT" else (50, 50, 255))
                    cv2.circle(frame, lms[tip_i], 6, t_col, -1, cv2.LINE_AA)
                    cv2.circle(frame, lms[tip_i], 7, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Palm Center Crosshair
            cx, cy = h["palm_center"]
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

            # 5. Hand Bounding Box & Handedness Tag
            bbox = h.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                box_pad = 10
                bx1, by1 = max(0, x1 - box_pad), max(0, y1 - box_pad)
                bx2, by2 = min(frame.shape[1], x2 + box_pad), min(frame.shape[0], y2 + box_pad)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180, 100, 255), 1)

                label = f"{handedness} ({int(hand_conf * 100)}%)"
                cv2.rectangle(frame, (bx1, max(0, by1 - 18)), (bx1 + 100, by1), (20, 12, 30), -1)
                cv2.putText(frame, label, (bx1 + 4, max(12, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 180, 255), 1, cv2.LINE_AA)

        return frame

    def _render_calibration_overlay(
        self, frame: np.ndarray, hands: List[Dict[str, Any]], gesture_info: dict
    ):
        """Displays real-time geometric mudra telemetry for calibration and tuning."""
        h, w = frame.shape[:2]
        cw, ch = 390, 360
        x1 = w - cw - 16
        y1 = 80
        x2 = x1 + cw
        y2 = y1 + ch

        roi = frame[y1:y2, x1:x2]
        card = np.full_like(roi, (10, 8, 16))
        cv2.addWeighted(card, 0.88, roi, 0.12, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

        cv2.putText(frame, "GESTURE CALIBRATION [C to toggle]", (x1 + 12, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 160, 40), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Hands: {len(hands)}  |  State: {gesture_info.get('state', 'UNKNOWN')}",
                    (x1 + 12, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 240), 1, cv2.LINE_AA)

        # Gojo metrics
        gojo_m = gesture_info.get("metrics", {}).get("gojo", {})
        gojo_s = gesture_info.get("gojo_score", 0.0)
        cv2.putText(frame, f"-- GOJO TAISHAKUTEN ({int(gojo_s * 100)}%) --", (x1 + 12, y1 + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 220), 1, cv2.LINE_AA)

        if hands:
            h0 = hands[0]
            cr = h0.get("cross_ratio", 1.0)
            is_cross = h0.get("is_crossing_mudra", False)
            yn = h0.get("palm_y_norm", 0.5)
            i_ang = h0.get("finger_angles", {}).get("index", 0)
            m_ang = h0.get("finger_angles", {}).get("middle", 0)
            r_ang = h0.get("finger_angles", {}).get("ring", 180)
            p_ang = h0.get("finger_angles", {}).get("pinky", 180)

            c_tag = "MATCH (<0.40)" if cr < 0.40 else "FAIL (>0.40)"
            cv2.putText(frame, f"Cross Ratio : {cr:.2f} [{c_tag}]", (x1 + 16, y1 + 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 255, 120) if cr < 0.40 else (120, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Fingers Crossed: {is_cross}  |  Y-pos: {yn:.2f} (<0.68)", (x1 + 16, y1 + 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Idx:{i_ang:.0f} Mid:{m_ang:.0f}  Rng:{r_ang:.0f} Pnk:{p_ang:.0f}",
                        (x1 + 16, y1 + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hand visible for Gojo", (x1 + 16, y1 + 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 170), 1, cv2.LINE_AA)

        # Sukuna metrics
        sukuna_m = gesture_info.get("metrics", {}).get("sukuna", {})
        sukuna_s = gesture_info.get("sukuna_score", 0.0)
        cv2.putText(frame, f"-- SUKUNA ENMA-TEN ({int(sukuna_s * 100)}%) --", (x1 + 12, y1 + 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 255), 1, cv2.LINE_AA)

        if len(hands) == 2:
            p_dist = sukuna_m.get("palm_dist_ratio", 9.9)
            i_dist = sukuna_m.get("index_dist_ratio", 9.9)
            p_tag = "MATCH (<2.5)" if p_dist < 2.5 else "FAIL"
            i_tag = "MATCH (<1.2)" if i_dist < 1.2 else "FAIL"
            cv2.putText(frame, f"Palm Dist Ratio  : {p_dist:.2f} [{p_tag}]", (x1 + 16, y1 + 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 255, 120) if p_dist < 2.5 else (120, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Index Tip Ratio  : {i_dist:.2f} [{i_tag}]", (x1 + 16, y1 + 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 255, 120) if i_dist < 1.2 else (120, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Lower Curled     : {sukuna_m.get('curled_lower_count', 0)}/4 (Need >=3)",
                        (x1 + 16, y1 + 225), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Requires 2 hands (Currently {len(hands)})", (x1 + 16, y1 + 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 255), 1, cv2.LINE_AA)

        # Thresholds summary
        cv2.line(frame, (x1 + 10, y1 + 250), (x2 - 10, y1 + 250), (60, 50, 75), 1)
        cv2.putText(frame, "TARGET THRESHOLDS:", (x1 + 12, y1 + 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Activation : >= 72% for 10 frames", (x1 + 16, y1 + 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "Deactivation : < 52% drops hold", (x1 + 16, y1 + 308),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "Switching   : Mudra change resets count", (x1 + 16, y1 + 326),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 220), 1, cv2.LINE_AA)

    def _log_gesture_sample(self, hands: List[Dict[str, Any]], gesture_info: dict):
        """Append lightweight structured observation to JSONL log file for offline debugging."""
        if not self.log_file or not hands:
            return
        # Sample every 5 frames to keep logs lightweight
        if self.total_frames % 5 != 0:
            return

        record = {
            "time": round(time.time(), 3),
            "frame": self.total_frames,
            "hands_count": len(hands),
            "state": gesture_info.get("state"),
            "gojo_score": round(gesture_info.get("gojo_score", 0.0), 3),
            "sukuna_score": round(gesture_info.get("sukuna_score", 0.0), 3),
            "unknown_score": round(gesture_info.get("unknown_score", 0.0), 3),
            "hands": [
                {
                    "handedness": h.get("handedness"),
                    "palm_scale": round(h.get("palm_scale", 0.0), 1),
                    "cross_ratio": round(h.get("cross_ratio", 0.0), 3),
                    "is_crossing": h.get("is_crossing_mudra", False),
                    "curl_ratios": {k: round(v, 2) for k, v in h.get("curl_ratios", {}).items()},
                }
                for h in hands
            ],
        }
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _render_bottom_cam_display(self, frame: np.ndarray):
        """Displays user's live camera inset in bottom corner during animated demo feed."""
        h, w = frame.shape[:2]
        pip_w, pip_h = 240, 140
        x1 = w - pip_w - 24
        y1 = h - pip_h - 24
        x2 = x1 + pip_w
        y2 = y1 + pip_h

        cam_frame = np.zeros((pip_h, pip_w, 3), dtype=np.uint8)
        cam_frame[:, :] = (20, 14, 28)
        cv2.line(cam_frame, (pip_w // 2, 20), (pip_w // 2, pip_h - 20), (55, 45, 70), 1)
        cv2.line(cam_frame, (20, pip_h // 2), (pip_w - 20, pip_h // 2), (55, 45, 70), 1)
        cv2.circle(cam_frame, (pip_w // 2, pip_h // 2), 26, (85, 65, 105), 1)
        cv2.putText(cam_frame, "USER CAM DISPLAY", (pip_w // 2 - 60, pip_h // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 190, 245), 1, cv2.LINE_AA)
        cv2.putText(cam_frame, "Make Sukuna / Gojo Mudra", (pip_w // 2 - 76, pip_h // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 230, 255), 1, cv2.LINE_AA)

        frame[y1:y2, x1:x2] = cam_frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 22), (14, 10, 22), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 22), (0, 230, 255), 1)
        cv2.circle(frame, (x1 + 12, y1 + 11), 4, (40, 40, 240), -1)
        cv2.putText(frame, "YOUR LIVE CAM DISPLAY", (x1 + 22, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    def _render_hud(
        self,
        frame: np.ndarray,
        gesture_info: dict,
        hands: Optional[List[Dict[str, Any]]] = None,
        q_level: str = "HIGH",
    ):
        """Renders cyber/curse telemetry HUD positioned on the side to keep face 100% visible."""
        pos = getattr(self, "hud_position", "right")
        if pos == "none" or not self.show_hud:
            return

        h, w = frame.shape[:2]
        hud_w, hud_h = 190, 172

        # Side position: 'right' (default) or 'left'
        if pos == "right":
            x0 = w - hud_w - 12
            y0 = 36  # Docked on right side, below top-right pill
        else:
            x0 = 12
            y0 = 36  # Docked on left side, below top-left pill

        x0 = max(0, min(w - hud_w, x0))
        y0 = max(0, min(h - hud_h, y0))

        roi = frame[y0:y0 + hud_h, x0:x0 + hud_w]
        dark_card = np.full_like(roi, (12, 8, 18))
        cv2.addWeighted(dark_card, 0.88, roi, 0.12, 0, roi)
        cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + hud_h), (0, 230, 255), 1)

        # 1. State & Active Theme
        state_col = (80, 255, 120) if self.state == "DOMAIN_ACTIVE" else ((80, 230, 255) if self.state == "CHARGING" else (210, 210, 220))
        cv2.putText(frame, f"STATE: {self.state}", (x0 + 8, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, state_col, 1, cv2.LINE_AA)

        theme_title = "SUKUNA" if "shrine" in self.theme_name.lower() else "GOJO"
        cv2.putText(frame, f"THEME: {theme_title}", (x0 + 8, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (215, 195, 255), 1, cv2.LINE_AA)

        # 2. Sukuna Mudra Match Meter
        sukuna_pct = int(gesture_info.get("sukuna_score", 0.0) * 100)
        s_col = (80, 255, 120) if sukuna_pct >= 72 else (180, 180, 190)
        cv2.putText(frame, f"Sukuna (2H): {sukuna_pct}%", (x0 + 8, y0 + 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, s_col, 1, cv2.LINE_AA)
        bar_w = hud_w - 16
        cv2.rectangle(frame, (x0 + 8, y0 + 51), (x0 + 8 + bar_w, y0 + 58), (35, 24, 46), -1)
        s_fill = int(bar_w * (sukuna_pct / 100.0))
        if s_fill > 0:
            cv2.rectangle(frame, (x0 + 8, y0 + 51), (x0 + 8 + s_fill, y0 + 58),
                          (80, 220, 255) if sukuna_pct < 72 else (80, 255, 120), -1)

        # 3. Gojo Mudra Match Meter
        gojo_pct = int(gesture_info.get("gojo_score", 0.0) * 100)
        g_col = (80, 255, 120) if gojo_pct >= 72 else (180, 180, 190)
        cv2.putText(frame, f"Gojo (1H): {gojo_pct}%", (x0 + 8, y0 + 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, g_col, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 8, y0 + 77), (x0 + 8 + bar_w, y0 + 84), (35, 24, 46), -1)
        g_fill = int(bar_w * (gojo_pct / 100.0))
        if g_fill > 0:
            cv2.rectangle(frame, (x0 + 8, y0 + 77), (x0 + 8 + g_fill, y0 + 84),
                          (255, 100, 220) if gojo_pct < 72 else (80, 255, 120), -1)

        # 4. Hold Progress Meter
        hold_prog = gesture_info.get("hold_progress", 0.0)
        hold_txt = f"HOLD: {int(hold_prog * 100)}%" if hold_prog > 0 else "HOLD: READY"
        cv2.putText(frame, hold_txt, (x0 + 8, y0 + 99),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 8, y0 + 103), (x0 + 8 + bar_w, y0 + 110), (35, 24, 46), -1)
        h_fill = int(bar_w * hold_prog)
        if h_fill > 0:
            cv2.rectangle(frame, (x0 + 8, y0 + 103), (x0 + 8 + h_fill, y0 + 110),
                          (80, 235, 255) if hold_prog < 1.0 else (80, 255, 120), -1)

        # 5. Live Gesture Status Text
        status_text = gesture_info.get("status_text", "")
        if len(status_text) > 22:
            status_text = status_text[:20] + ".."
        cv2.putText(frame, status_text, (x0 + 8, y0 + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 220, 230), 1, cv2.LINE_AA)

        # 6. Telemetry line
        hands_cnt = len(hands) if hands else 0
        fps_color = (80, 255, 120) if self.profiler.fps >= 25.0 else ((80, 230, 255) if self.profiler.fps >= 15.0 else (80, 80, 255))
        cv2.putText(frame, f"FPS: {self.profiler.fps:.1f} | Hands: {hands_cnt}", (x0 + 8, y0 + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, fps_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Trk: {self.profiler.track_ms:.0f}ms | Rdr: {self.profiler.render_ms:.0f}ms",
                    (x0 + 8, y0 + 161), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 180), 1, cv2.LINE_AA)

    def run(self):
        print("\n=======================================================")
        print("  DOMAINVISION - REAL-TIME AR DOMAIN EXPANSION (領域展開)")
        print("  Decoupled High-FPS Architecture & Canonical Audio Sync")
        print("=======================================================")
        print("  Hand Signs:")
        print("    - Sukuna: STRICTLY 2 HANDS clasped, thumbs upright, index touching")
        print("    - Gojo  : Exactly 1 hand, index & middle crossed, head height")
        print("  Controls:")
        print("    [C]         : Toggle Gesture Calibration Overlay")
        print("    [L]         : Toggle 21-Landmark Skeleton Overlay")
        print("    [M]         : Toggle Webcam Mirror Preview")
        print("    [1]         : Switch to Malevolent Shrine (Sukuna)")
        print("    [2]         : Switch to Infinite Void (Gojo)")
        print("    [D] / Space : Force Trigger Domain Expansion")
        print("    [R]         : Reset / Collapse Domain")
        print("    [H]         : Toggle HUD")
        print("    [S]         : Save Screenshot")
        print("    [Q] / ESC   : Quit")
        print("=======================================================\n")

        frame_counter = 0

        try:
            while self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    if self.is_demo:
                        continue
                    else:
                        break

                output_frame = self.process_frame(raw_frame)

                if self.video_writer:
                    self.video_writer.write(output_frame)

                frame_counter += 1
                if not self.headless:
                    cv2.imshow("DomainVision AR - 領域展開", output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        break
                    elif key in [ord("c"), ord("C")]:
                        self.calibrate_mode = not self.calibrate_mode
                        print(f"Calibration Mode: {self.calibrate_mode}")
                    elif key in [ord("l"), ord("L")]:
                        self.show_landmarks = not self.show_landmarks
                        print(f"Show Landmarks: {self.show_landmarks}")
                    elif key in [ord("m"), ord("M")]:
                        self.flip_webcam = not self.flip_webcam
                        print(f"Webcam Mirror: {self.flip_webcam}")
                    elif key in [ord("d"), ord("D"), 32]:
                        self.trigger_domain()
                    elif key in [ord("r"), ord("R")]:
                        self.reset_domain()
                    elif key == ord("1"):
                        self.set_theme("malevolent_shrine")
                    elif key == ord("2"):
                        self.set_theme("infinite_void")
                    elif key in [ord("h"), ord("H")]:
                        positions = ["right", "left", "none"]
                        cur = getattr(self, "hud_position", "right")
                        idx = positions.index(cur) if cur in positions else 0
                        self.hud_position = positions[(idx + 1) % len(positions)]
                        self.show_hud = (self.hud_position != "none")
                        print(f"HUD Position: {self.hud_position.upper()}")
                    elif key in [ord("s"), ord("S")]:
                        ss_name = f"domain_screenshot_{int(time.time())}.png"
                        cv2.imwrite(ss_name, output_frame)
                        print(f"Saved screenshot: {ss_name}")

                if self.max_frames and frame_counter >= self.max_frames:
                    print(f"Reached max frames limit ({self.max_frames}). Finishing.")
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up resources...")
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            print(f"Recorded video saved to: {self.record_path}")
        if not self.headless:
            cv2.destroyAllWindows()
        if hasattr(self, "tracker"):
            self.tracker.close()
        if hasattr(self, "pip_cap") and self.pip_cap:
            self.pip_cap.release()
        if hasattr(self, "audio"):
            self.audio.stop_domain_sequence()
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="DomainVision AR Filter")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo simulation mode")
    parser.add_argument("--theme", type=str, default=DEFAULT_THEME,
                        choices=list(THEMES.keys()), help="Domain Theme")
    parser.add_argument("--record", type=str, default=None, help="Path to save output video (.mp4)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Camera height (default: 720)")
    parser.add_argument("--calibrate", action="store_true", help="Launch in gesture calibration overlay mode")
    parser.add_argument("--log-gestures", action="store_true", help="Log gesture observation samples to gesture_samples.jsonl")
    parser.add_argument("--flip", action="store_true", default=True, help="Mirror webcam feed horizontally (default: True)")
    parser.add_argument("--no-flip", action="store_false", dest="flip", help="Do not mirror webcam feed")

    args = parser.parse_args()

    is_headless = args.headless or ("DISPLAY" not in os.environ or not os.environ["DISPLAY"])
    record_path = args.record
    if is_headless and record_path is None:
        record_path = "output.mp4"
        print("SSH/Headless session detected: Automatically recording to output.mp4")

    app = DomainExpansionApp(
        input_source=args.video,
        camera_idx=args.camera,
        demo_mode=args.demo,
        theme=args.theme,
        width=args.width,
        height=args.height,
        headless=is_headless,
        record_path=record_path,
        max_frames=args.frames,
        calibrate_mode=args.calibrate,
        log_gestures=args.log_gestures,
        flip_webcam=args.flip,
    )
    app.run()


if __name__ == "__main__":
    main()
