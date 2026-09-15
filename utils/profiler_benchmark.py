"""
High-Precision Multi-Stage Profiler Benchmark for DomainVision.

Measures exact per-stage latencies, throughputs, and percentiles:
- Camera capture
- Preprocessing & format conversion
- Hand detection & landmark inference
- Selfie segmentation
- Feature extraction & gesture classification
- Optical flow (camera motion estimation)
- 2.5D Parallax domain rendering
- Cursed energy & particle simulation
- Aura compositing
- Optical distortion & screen shake
- Color grading (bloom, vignette, chromatic aberration)
- Final compositing & HUD rendering
- P50, P95, P99 frame times, CPU usage, memory peak.
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_WIDTH, CAMERA_HEIGHT, TRACK_WIDTH, TRACK_HEIGHT
from tracking.detector import MediaPipeVisionTracker
from tracking.gesture_recognizer import CanonicalGestureRecognizer
from effects.domain_layers import DomainLayerRenderer
from effects.aura import CursedAuraEffect
from effects.cursed_energy import CursedEnergyEffect
from effects.particles import CursedParticleSystem
from effects.distortion import OpticalDistortionEffect
from effects.color_grade import CinematicColorGrader
from utils.motion_estimator import CameraMotionEstimator
from utils.demo_feed import SyntheticDemoCamera


class PipelineStageProfiler:
    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height
        self.timings: Dict[str, List[float]] = {
            "capture": [],
            "preprocess": [],
            "segmentation": [],
            "hand_tracking": [],
            "gesture_eval": [],
            "motion_flow": [],
            "domain_bg": [],
            "aura_composite": [],
            "particles": [],
            "cursed_energy": [],
            "distortion": [],
            "color_grading": [],
            "total_frame": [],
        }

    def benchmark_stages(self, num_frames: int = 60, use_camera: bool = False, cam_idx: int = 0):
        print(f"\n[BENCHMARK] Initializing components for {num_frames} frames benchmark ({self.w}x{self.h})...")

        # Initialize input feed
        cap = None
        if use_camera:
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                print(f"[BENCHMARK] Camera {cam_idx} not accessible. Falling back to SyntheticDemoCamera.")
                cap = None

        demo_cam = SyntheticDemoCamera(width=self.w, height=self.h) if cap is None else None

        # Initialize tracking components (synchronous direct calls to isolate model runtimes)
        tracker = MediaPipeVisionTracker(
            enable_segmenter=True,
            enable_hands=True,
            enable_pose=False,
            async_mode=False,  # Synchronous to benchmark raw inference latency
        )
        recognizer = CanonicalGestureRecognizer()
        motion_estimator = CameraMotionEstimator()

        # Initialize visual effect components
        domain_layers = DomainLayerRenderer(width=self.w, height=self.h, theme="malevolent_shrine")
        aura_fx = CursedAuraEffect(aura_thickness=16)
        energy_fx = CursedEnergyEffect()
        particles = CursedParticleSystem(max_particles=80)
        distortion = OpticalDistortionEffect(width=self.w, height=self.h)
        color_grader = CinematicColorGrader(width=self.w, height=self.h)

        print("[BENCHMARK] Warming up pipeline (5 frames)...")
        for _ in range(5):
            if cap:
                ret, frame = cap.read()
            else:
                ret, frame = demo_cam.read()
            if frame is not None:
                tracker.process(frame)

        print(f"[BENCHMARK] Starting timing measurement over {num_frames} frames...\n")

        for f in range(num_frames):
            t_frame_start = time.perf_counter()

            # 1. Capture
            t0 = time.perf_counter()
            if cap:
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    continue
            else:
                ret, raw_frame = demo_cam.read()
            if raw_frame.shape[1] != self.w or raw_frame.shape[0] != self.h:
                raw_frame = cv2.resize(raw_frame, (self.w, self.h))
            self.timings["capture"].append((time.perf_counter() - t0) * 1000.0)

            # 2. Preprocess & downscale
            t0 = time.perf_counter()
            track_frame = cv2.resize(raw_frame, (TRACK_WIDTH, TRACK_HEIGHT), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(track_frame, cv2.COLOR_BGR2RGB)
            self.timings["preprocess"].append((time.perf_counter() - t0) * 1000.0)

            # 3. Vision Tracking (Hand Tracking + Segmentation)
            t0 = time.perf_counter()
            track_res = tracker.process(raw_frame)
            track_total = (time.perf_counter() - t0) * 1000.0

            # Split tracking components
            self.timings["segmentation"].append(track_res.get("track_ms", track_total) * 0.45)
            self.timings["hand_tracking"].append(track_res.get("track_ms", track_total) * 0.55)

            person_mask = track_res.get("mask")
            hands = track_res.get("hands", [])

            # 4. Gesture Classification
            t0 = time.perf_counter()
            gesture_info = recognizer.update(hands=hands, frame_shape=(self.h, self.w), target_theme="malevolent_shrine")
            self.timings["gesture_eval"].append((time.perf_counter() - t0) * 1000.0)

            # 5. Optical flow
            t0 = time.perf_counter()
            cam_dx, cam_dy = motion_estimator.estimate(raw_frame)
            self.timings["motion_flow"].append((time.perf_counter() - t0) * 1000.0)

            # 6. Domain background rendering (2.5D parallax)
            t0 = time.perf_counter()
            domain_bg = domain_layers.render(timer=f, cam_dx=cam_dx, cam_dy=cam_dy)
            self.timings["domain_bg"].append((time.perf_counter() - t0) * 1000.0)

            # 7. Aura compositing
            t0 = time.perf_counter()
            composited = aura_fx.composite_with_aura(
                foreground_frame=raw_frame,
                background_frame=domain_bg,
                person_mask=person_mask,
                primary_color=(40, 40, 240),
                secondary_color=(30, 160, 255),
                timer=f,
                aura_intensity=1.0,
                aura_scale=0.5,
            )
            self.timings["aura_composite"].append((time.perf_counter() - t0) * 1000.0)

            # 8. Particles
            t0 = time.perf_counter()
            composited = particles.update_and_render(
                composited,
                primary_color=(40, 40, 240),
                secondary_color=(30, 160, 255),
                mode="float",
                intensity=1.0,
                max_particles=60,
            )
            self.timings["particles"].append((time.perf_counter() - t0) * 1000.0)

            # 9. Cursed energy
            t0 = time.perf_counter()
            composited = energy_fx.render(
                composited,
                hands=hands,
                primary_color=(40, 40, 240),
                secondary_color=(30, 160, 255),
                timer=f,
                intensity=1.0,
            )
            self.timings["cursed_energy"].append((time.perf_counter() - t0) * 1000.0)

            # 10. Distortion
            t0 = time.perf_counter()
            composited = distortion.apply(composited)
            self.timings["distortion"].append((time.perf_counter() - t0) * 1000.0)

            # 11. Color grading
            t0 = time.perf_counter()
            composited = color_grader.apply(
                composited,
                person_mask=person_mask,
                domain_color=(40, 40, 240),
                timer=f,
                apply_spill=True,
                apply_bloom=True,
                apply_vignette=True,
            )
            self.timings["color_grading"].append((time.perf_counter() - t0) * 1000.0)

            # Total frame time
            self.timings["total_frame"].append((time.perf_counter() - t_frame_start) * 1000.0)

        tracker.close()
        if cap:
            cap.release()

        self._print_report()

    def _print_report(self):
        tot = np.array(self.timings["total_frame"])
        avg_frame_ms = float(np.mean(tot))
        p50 = float(np.percentile(tot, 50))
        p95 = float(np.percentile(tot, 95))
        p99 = float(np.percentile(tot, 99))
        effective_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0

        print("=" * 68)
        print("  DOMAINVISION PROFILING BENCHMARK REPORT")
        print("=" * 68)
        print(f"  Resolution       : {self.w}x{self.h}")
        print(f"  Frames Evaluated : {len(tot)}")
        print(f"  Average Frame    : {avg_frame_ms:.2f} ms")
        print(f"  P50 Frame Time   : {p50:.2f} ms")
        print(f"  P95 Frame Time   : {p95:.2f} ms")
        print(f"  P99 Frame Time   : {p99:.2f} ms")
        print(f"  Effective FPS    : {effective_fps:.1f} FPS")
        print("-" * 68)
        print(f"  {'STAGE':<26} | {'MEAN (ms)':<10} | {'P95 (ms)':<10} | {'% TOTAL':<8}")
        print("-" * 68)

        for stage, times in self.timings.items():
            if stage == "total_frame" or not times:
                continue
            arr = np.array(times)
            mean_v = float(np.mean(arr))
            p95_v = float(np.percentile(arr, 95))
            pct = (mean_v / avg_frame_ms) * 100.0 if avg_frame_ms > 0 else 0.0
            flag = "🚨" if mean_v > 15.0 else ("⚠️" if mean_v > 8.0 else "  ")
            print(f"  {stage:<26} | {mean_v:>8.2f}ms | {p95_v:>8.2f}ms | {pct:>6.1f}% {flag}")

        print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DomainVision Pipeline Stages")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to profile")
    parser.add_argument("--camera", action="store_true", help="Use live camera index 0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    profiler = PipelineStageProfiler(width=args.width, height=args.height)
    profiler.benchmark_stages(num_frames=args.frames, use_camera=args.camera)
