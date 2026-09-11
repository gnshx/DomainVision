import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from config import THEMES, DEFAULT_THEME
from tracking.detector import MediaPipeVisionTracker
from tracking.gesture_recognizer import CanonicalGestureRecognizer
from tracking.hand_tracker import generate_synthetic_mudra_hands
from effects.flash import DomainFlashEffect
from effects.domain_environment import DomainEnvironmentRenderer
from effects.aura import CursedAuraEffect
from effects.cursed_energy import CursedEnergyEffect
from effects.particles import CursedParticleSystem
from effects.distortion import OpticalDistortionEffect
from effects.shockwave import BarrierShockwaveEffect
from utils.text_renderer import JapaneseTextRenderer
from audio.audio_manager import AudioManager
from utils.demo_feed import SyntheticDemoCamera
from utils.fps import PerformanceProfiler


class DomainExpansionApp:
    """
    DomainVision 2.0: AR-grade JJK Domain Expansion filter.
    - Precision 3D finger tracking & canonical JJK hand signs (Sukuna & Gojo)
    - Temporal landmark smoothing without tracking jitter
    - Perspective-anchored skeletal cursed energy
    - Realistic Malevolent Shrine (⛩️ Pagoda) backdrop
    - Frame-accurate audio state synchronization
    """

    STATES = ["NORMAL", "CHARGING", "FLASH", "EXPANSION", "DOMAIN_ACTIVE", "COLLAPSE"]

    def __init__(
        self,
        input_source: Optional[str] = None,
        camera_idx: int = 0,
        demo_mode: bool = False,
        theme: str = DEFAULT_THEME,
        width: int = 640,
        height: int = 480,
        headless: bool = False,
        record_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        self.w = width
        self.h = height
        self.theme_name = theme if theme in THEMES else DEFAULT_THEME
        self.headless = headless
        self.record_path = record_path
        self.max_frames = max_frames
        self.show_hud = True

        # Video source
        self.cap, self.is_demo = self._init_video_source(input_source, camera_idx, demo_mode)

        # State Machine
        self.state = "NORMAL"
        self.state_timer = 0
        self.total_frames = 0
        self.energy_center = (self.w // 2, int(self.h * 0.5))

        # Profiler & Audio
        self.profiler = PerformanceProfiler()
        self.audio = AudioManager()

        # Tracking Layer
        print("Initializing MediaPipe Precision Vision Tracker...")
        self.tracker = MediaPipeVisionTracker()
        self.gesture_recognizer = CanonicalGestureRecognizer()

        # Visual Effects Engines
        print("Initializing Perspective-Aware Visual Effects...")
        self.flash_fx = DomainFlashEffect(duration_frames=16)
        self.env_renderer = DomainEnvironmentRenderer(width=self.w, height=self.h, theme=self.theme_name)
        self.aura_fx = CursedAuraEffect(aura_thickness=18)
        self.cursed_energy_fx = CursedEnergyEffect()
        self.particle_system = CursedParticleSystem(max_particles=120)
        self.distortion_fx = OpticalDistortionEffect(width=self.w, height=self.h)
        self.shockwave_fx = BarrierShockwaveEffect()
        self.text_renderer = JapaneseTextRenderer()

        # Video Writer
        self.video_writer = None
        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.record_path, fourcc, 30.0, (self.w, self.h))
            print(f"Recording output to {self.record_path}")

    def _init_video_source(self, input_source, camera_idx, demo_mode):
        if demo_mode:
            print("Running in SYNTHETIC DEMO mode.")
            return SyntheticDemoCamera(width=self.w, height=self.h), True

        if input_source and os.path.exists(input_source):
            print(f"Opening video file: {input_source}")
            cap = cv2.VideoCapture(input_source)
            if cap.isOpened():
                return cap, False

        print(f"Attempting to open camera index {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("Camera initialized successfully!")
                return cap, False
            cap.release()

        print("Notice: No live webcam detected. Falling back to SYNTHETIC DEMO mode.")
        return SyntheticDemoCamera(width=self.w, height=self.h), True

    def set_theme(self, theme_name: str):
        if theme_name in THEMES:
            self.theme_name = theme_name
            self.env_renderer.set_theme(theme_name)
            print(f"Switched theme to: {self.theme_name} ({THEMES[theme_name]['name_ja']})")

    def trigger_domain(self):
        """Force-trigger Domain Expansion sequence."""
        if self.state in ["NORMAL", "CHARGING"]:
            self.state = "FLASH"
            self.state_timer = 0
            theme_info = self.env_renderer.theme
            self.flash_fx.trigger(theme_color=theme_info["primary_bgr"])
            self.audio.play("activation")
            self.distortion_fx.trigger_shake(magnitude=18.0)

    def reset_domain(self):
        """Collapse domain back to normal."""
        self.state = "COLLAPSE"
        self.state_timer = 0
        self.audio.play("collapse")

    def _update_state_machine(self, gesture_info: dict):
        self.state_timer += 1
        theme_info = self.env_renderer.theme

        # 1. NORMAL State
        if self.state == "NORMAL":
            if gesture_info["trigger"]:
                self.trigger_domain()
            elif gesture_info["hold_progress"] > 0.25:
                self.state = "CHARGING"
                self.state_timer = 0
                self.audio.play("charge")

        # 2. CHARGING State
        elif self.state == "CHARGING":
            if self.state_timer % 3 == 0:
                self.distortion_fx.trigger_shake(magnitude=3.0)

            if gesture_info["trigger"] or self.state_timer >= 22:
                self.state = "FLASH"
                self.state_timer = 0
                self.flash_fx.trigger(theme_color=theme_info["primary_bgr"])
                self.audio.play("activation")
                self.distortion_fx.trigger_shake(magnitude=20.0)
            elif gesture_info["hold_progress"] <= 0.05 and not self.is_demo and self.state_timer > 8:
                # User stopped holding the sign early
                self.state = "NORMAL"
                self.state_timer = 0

        # 3. FLASH State
        elif self.state == "FLASH":
            if self.state_timer == 10:
                # Shockwaves erupt
                self.distortion_fx.trigger_shockwave(
                    center=self.energy_center,
                    max_radius=math.hypot(self.w, self.h),
                    duration_frames=26,
                    strength=32.0
                )
                self.shockwave_fx.trigger(
                    center=self.energy_center,
                    primary_color=theme_info["primary_bgr"],
                    secondary_color=theme_info["secondary_bgr"],
                    duration_frames=28
                )
                self.audio.play("impact")

            if self.state_timer >= 16:
                self.state = "EXPANSION"
                self.state_timer = 0

        # 4. EXPANSION State
        elif self.state == "EXPANSION":
            if self.state_timer >= 32:
                self.state = "DOMAIN_ACTIVE"
                self.state_timer = 0

        # 5. DOMAIN_ACTIVE State
        elif self.state == "DOMAIN_ACTIVE":
            if self.state_timer >= 280:
                self.reset_domain()

        # 6. COLLAPSE State
        elif self.state == "COLLAPSE":
            if self.state_timer >= 25:
                self.state = "NORMAL"
                self.state_timer = 0
                self.gesture_recognizer.reset()

    def process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        self.profiler.start_frame()

        if raw_frame.shape[1] != self.w or raw_frame.shape[0] != self.h:
            raw_frame = cv2.resize(raw_frame, (self.w, self.h))

        h, w = self.h, self.w
        theme_info = self.env_renderer.theme
        col_pri = theme_info["primary_bgr"]
        col_sec = theme_info["secondary_bgr"]

        # 1. MediaPipe Vision Tracking
        self.profiler.start_tracking()
        tracking = self.tracker.process(raw_frame)
        self.profiler.end_tracking()

        person_mask = tracking["mask"]
        hands = tracking["hands"]

        if self.is_demo and len(hands) == 0:
            hands = generate_synthetic_mudra_hands((h, w), self.total_frames, self.theme_name)

        # 2. Canonical Hand Sign Evaluation
        gesture_info = self.gesture_recognizer.update(
            hands=hands,
            frame_shape=(h, w),
            target_theme=self.theme_name
        )

        if gesture_info.get("energy_center"):
            self.energy_center = gesture_info["energy_center"]

        # Advance State Machine
        self._update_state_machine(gesture_info)

        # 3. Environment Generation (Shrine / Void)
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE"]:
            domain_bg = self.env_renderer.render(timer=self.total_frames)
            if self.state == "EXPANSION":
                wipe_r = int(math.hypot(w, h) * (self.state_timer / 32.0))
                mask_circ = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask_circ, self.energy_center, wipe_r, 255, -1)
                mask_blurred = cv2.GaussianBlur(mask_circ, (31, 31), 0)
                norm_wipe = (mask_blurred.astype(np.float32) / 255.0)[:, :, np.newaxis]
                active_bg = (domain_bg.astype(np.float32) * norm_wipe +
                             raw_frame.astype(np.float32) * (1.0 - norm_wipe)).astype(np.uint8)
            elif self.state == "COLLAPSE":
                dissolve = max(0.0, 1.0 - (self.state_timer / 25.0))
                active_bg = cv2.addWeighted(domain_bg, dissolve, raw_frame, 1.0 - dissolve, 0)
            else:
                active_bg = domain_bg
        else:
            active_bg = raw_frame.copy()

        # 4. Cursed Aura & Depth Compositing (Environment Behind -> Aura -> Person Foreground)
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE"]:
            composited = self.aura_fx.composite_with_aura(
                foreground_frame=raw_frame,
                background_frame=active_bg,
                person_mask=person_mask,
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                aura_intensity=1.0 if self.state != "COLLAPSE" else (1.0 - self.state_timer / 25.0)
            )
        elif self.state == "CHARGING":
            composited = self.aura_fx.composite_with_aura(
                foreground_frame=raw_frame,
                background_frame=raw_frame,
                person_mask=person_mask,
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                aura_intensity=0.5
            )
        else:
            composited = raw_frame.copy()

        # 5. Cursed Particles System
        p_mode = "float"
        p_intensity = 0.0
        if self.state == "CHARGING":
            p_mode = "suck_in"
            p_intensity = 1.2
        elif self.state == "COLLAPSE":
            p_mode = "blast_out"
            p_intensity = 1.5
        elif self.state in ["EXPANSION", "DOMAIN_ACTIVE"]:
            p_mode = "float"
            p_intensity = 1.0

        if p_intensity > 0.01:
            composited = self.particle_system.update_and_render(
                composited,
                primary_color=col_pri,
                secondary_color=col_sec,
                mode=p_mode,
                center=self.energy_center,
                intensity=p_intensity
            )

        # 6. Perspective Hand Cursed Energy (Attached to Finger Joints)
        if self.state in ["CHARGING", "EXPANSION", "DOMAIN_ACTIVE"]:
            energy_intensity = 0.7 if self.state == "CHARGING" else 1.0
            composited = self.cursed_energy_fx.render(
                composited,
                hands=hands,
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                intensity=energy_intensity
            )

        # 7. Barrier Shockwaves
        composited = self.shockwave_fx.update_and_render(composited)

        # 8. Flash Transition
        if self.flash_fx.active:
            composited, _ = self.flash_fx.apply(composited)

        # 9. Optical Distortion & Shake
        composited = self.distortion_fx.apply(composited)

        # 10. Japanese Calligraphy Overlay
        if self.state in ["EXPANSION", "DOMAIN_ACTIVE"]:
            if self.state == "EXPANSION":
                text_prog = min(1.0, self.state_timer / 20.0)
            elif self.state == "DOMAIN_ACTIVE":
                if self.state_timer < 60:
                    text_prog = 1.0
                elif self.state_timer < 100:
                    text_prog = max(0.0, 1.0 - (self.state_timer - 60) / 40.0)
                else:
                    text_prog = 0.0
            else:
                text_prog = 0.0

            if text_prog > 0.01:
                txt_bgr, txt_alpha = self.text_renderer.render_domain_banner(
                    frame_shape=(h, w),
                    main_text="領域展開",
                    sub_text=theme_info["name_ja"],
                    en_text=f"DOMAIN EXPANSION - {theme_info['name_en']}",
                    color_glow=col_pri,
                    progress=text_prog
                )
                alpha_norm = (txt_alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]
                composited = (txt_bgr.astype(np.float32) * alpha_norm +
                              composited.astype(np.float32) * (1.0 - alpha_norm)).astype(np.uint8)

        # 11. HUD & Performance Telemetry
        self.profiler.end_rendering()
        if self.show_hud:
            self._render_hud(composited, gesture_info)

        self.total_frames += 1
        return composited

    def _render_hud(self, frame: np.ndarray, gesture_info: dict):
        h, w = frame.shape[:2]
        hud_w, hud_h = 320, 105
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (16, 12, 22), -1)
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (70, 45, 90), 1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # State & Theme
        cv2.putText(frame, f"STATUS : {self.state}", (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"DOMAIN : {self.env_renderer.theme['name_en']} ({self.env_renderer.theme['character']})",
                    (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 180, 240), 1, cv2.LINE_AA)

        # Sign Tracking Progress
        if gesture_info["sign_name"]:
            sign_text = f"SIGN   : {gesture_info['sign_name']}"
            bar_color = (80, 240, 120)
        else:
            sign_text = "SIGN   : Make Sukuna / Gojo Hand Mudra"
            bar_color = (180, 180, 180)

        cv2.putText(frame, sign_text, (22, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_color, 1, cv2.LINE_AA)

        # Hold Progress Bar
        bar_bg_w = hud_w - 20
        cv2.rectangle(frame, (22, 84), (22 + bar_bg_w, 94), (40, 30, 50), -1)
        fill_w = int(bar_bg_w * gesture_info["hold_progress"])
        if fill_w > 0:
            fill_col = (80, 230, 255) if gesture_info["hold_progress"] < 1.0 else (80, 255, 120)
            cv2.rectangle(frame, (22, 84), (22 + fill_w, 94), fill_col, -1)

        # Performance Telemetry Badge (Top-Right)
        self.profiler.draw_telemetry(frame, position=(w - 285, 30))

    def run(self):
        print("\n=======================================================")
        print("  DOMAINVISION 2.0 - CANONICAL AR FILTER (領域展開)")
        print("=======================================================")
        print("  Hand Signs:")
        print("    - Sukuna (Enma-ten Mudra): Palms pressed, thumbs upright, index touching")
        print("    - Gojo   (Taishakuten Mudra): Index & Middle fingers crossed")
        print("  Controls:")
        print("    [1]         : Switch to Malevolent Shrine (伏魔御廚子 - Sukuna)")
        print("    [2]         : Switch to Infinite Void (無量空処 - Gojo)")
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
                    cv2.imshow("DomainVision AR", output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        break
                    elif key in [ord("d"), ord("D"), 32]:
                        self.trigger_domain()
                    elif key in [ord("r"), ord("R")]:
                        self.reset_domain()
                    elif key == ord("1"):
                        self.set_theme("malevolent_shrine")
                    elif key == ord("2"):
                        self.set_theme("infinite_void")
                    elif key in [ord("h"), ord("H")]:
                        self.show_hud = not self.show_hud
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
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="DomainVision 2.0 AR Filter")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo simulation mode")
    parser.add_argument("--theme", type=str, default=DEFAULT_THEME,
                        choices=list(THEMES.keys()), help="Domain Theme")
    parser.add_argument("--record", type=str, default=None, help="Path to save output video (.mp4)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--width", type=int, default=640, help="Processing width")
    parser.add_argument("--height", type=int, default=480, help="Processing height")

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
    )
    app.run()


if __name__ == "__main__":
    main()
