import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from tracking.detector import MediaPipeVisionTracker
from tracking.gesture import DomainGestureDetector
from effects.flash import DomainFlashEffect
from effects.domain_background import DomainBackgroundRenderer
from effects.aura import CursedAuraEffect
from effects.hand_energy import HandEnergyEffect
from effects.particles import CursedParticleSystem
from effects.distortion import OpticalDistortionEffect
from effects.shockwave import BarrierShockwaveEffect
from utils.text_renderer import JapaneseTextRenderer
from utils.audio import CursedSoundSynthesizer
from utils.demo_feed import SyntheticDemoCamera


class DomainExpansionApp:
    """
    Main controller for the JJK Domain Expansion AR Filter.
    Orchestrates tracking, state machine, effects compositing, and rendering.
    """

    STATES = ["NORMAL", "CHARGING", "FLASH", "EXPANSION", "DOMAIN_ACTIVE", "COLLAPSE"]

    def __init__(
        self,
        input_source: Optional[str] = None,
        camera_idx: int = 0,
        demo_mode: bool = False,
        theme: str = "infinite_void",
        width: int = 640,
        height: int = 480,
        headless: bool = False,
        record_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        self.w = width
        self.h = height
        self.theme_name = theme
        self.headless = headless
        self.record_path = record_path
        self.max_frames = max_frames
        self.show_hud = True

        # Initialize Video Capture
        self.cap, self.is_demo = self._init_video_source(input_source, camera_idx, demo_mode)

        # State Machine variables
        self.state = "NORMAL"
        self.state_timer = 0
        self.total_frames = 0
        self.energy_center = (self.w // 2, int(self.h * 0.5))

        # Audio synthesizer
        self.audio = CursedSoundSynthesizer()

        # MediaPipe Tracking
        print("Initializing MediaPipe Vision Tracker...")
        self.tracker = MediaPipeVisionTracker()
        self.gesture_detector = DomainGestureDetector(activation_frames=4)

        # Visual Effects
        print("Initializing Visual Effects Engines...")
        self.flash_fx = DomainFlashEffect(duration_frames=16)
        self.bg_renderer = DomainBackgroundRenderer(width=self.w, height=self.h, theme=self.theme_name)
        self.aura_fx = CursedAuraEffect(aura_thickness=18)
        self.hand_fx = HandEnergyEffect()
        self.particle_system = CursedParticleSystem(max_particles=120)
        self.distortion_fx = OpticalDistortionEffect(width=self.w, height=self.h)
        self.shockwave_fx = BarrierShockwaveEffect()
        self.text_renderer = JapaneseTextRenderer()

        # Video Recorder
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

        # Attempt to open webcam
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

        # Fallback to demo mode if no camera found
        print("Notice: No live webcam detected. Falling back to SYNTHETIC DEMO mode.")
        return SyntheticDemoCamera(width=self.w, height=self.h), True

    def set_theme(self, theme_name: str):
        if theme_name in self.bg_renderer.THEMES:
            self.theme_name = theme_name
            self.bg_renderer.set_theme(theme_name)
            print(f"Switched theme to: {self.theme_name}")

    def trigger_domain(self):
        """Force-trigger Domain Expansion sequence."""
        if self.state in ["NORMAL", "CHARGING"]:
            self.state = "FLASH"
            self.state_timer = 0
            theme_info = self.bg_renderer.theme
            self.flash_fx.trigger(theme_color=theme_info["primary_bgr"])
            self.audio.play("flash")
            self.distortion_fx.trigger_shake(magnitude=16.0)

    def reset_domain(self):
        """Reset domain back to normal."""
        self.state = "COLLAPSE"
        self.state_timer = 0
        self.audio.play("collapse")

    def _update_state_machine(self, gesture_info: dict):
        """Advance the cinematic state machine."""
        self.state_timer += 1
        theme_info = self.bg_renderer.theme

        # 1. NORMAL State
        if self.state == "NORMAL":
            if gesture_info["gesture_active"] or gesture_info["is_holding_pose"]:
                self.state = "CHARGING"
                self.state_timer = 0
                self.audio.play("charge")

        # 2. CHARGING State
        elif self.state == "CHARGING":
            # Screen vibration during charge
            if self.state_timer % 2 == 0:
                self.distortion_fx.trigger_shake(magnitude=2.5)

            # Auto-advance after charge duration (approx 20-25 frames)
            if self.state_timer >= 22 or gesture_info["gesture_active"]:
                self.state = "FLASH"
                self.state_timer = 0
                self.flash_fx.trigger(theme_color=theme_info["primary_bgr"])
                self.audio.play("flash")
                self.distortion_fx.trigger_shake(magnitude=18.0)
            elif not gesture_info["is_holding_pose"] and self.state_timer > 10 and not self.is_demo:
                # Cancelled charging if user lowers hands early
                self.state = "NORMAL"
                self.state_timer = 0

        # 3. FLASH State
        elif self.state == "FLASH":
            # Midway through flash, ignite shockwaves
            if self.state_timer == 10:
                self.distortion_fx.trigger_shockwave(
                    center=self.energy_center,
                    max_radius=math.hypot(self.w, self.h),
                    duration_frames=26,
                    strength=30.0
                )
                self.shockwave_fx.trigger(
                    center=self.energy_center,
                    primary_color=theme_info["primary_bgr"],
                    secondary_color=theme_info["secondary_bgr"],
                    duration_frames=28
                )
                self.audio.play("expansion")

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
            # Domain stays open for 300 frames (~10 sec) then collapses
            if self.state_timer >= 280:
                self.state = "COLLAPSE"
                self.state_timer = 0
                self.audio.play("collapse")

        # 6. COLLAPSE State
        elif self.state == "COLLAPSE":
            if self.state_timer >= 25:
                self.state = "NORMAL"
                self.state_timer = 0
                self.gesture_detector.reset()

    def process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Execute full visual pipeline on a single frame."""
        # Ensure correct dimensions
        if raw_frame.shape[1] != self.w or raw_frame.shape[0] != self.h:
            raw_frame = cv2.resize(raw_frame, (self.w, self.h))

        h, w = self.h, self.w
        theme_info = self.bg_renderer.theme
        col_pri = theme_info["primary_bgr"]
        col_sec = theme_info["secondary_bgr"]

        # 1. MediaPipe Vision Tracking
        tracking = self.tracker.process(raw_frame)
        person_mask = tracking["mask"]
        hands = tracking["hands"]

        # 2. Gesture Evaluation
        gesture_info = self.gesture_detector.update(tracking, (h, w))
        if gesture_info.get("energy_point"):
            self.energy_center = gesture_info["energy_point"]

        # Advance state
        self._update_state_machine(gesture_info)

        # 3. Background Generation
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE"]:
            domain_bg = self.bg_renderer.render(timer=self.total_frames)
            if self.state == "EXPANSION":
                # Radial wipe expanding outward
                wipe_radius = int(math.hypot(w, h) * (self.state_timer / 32.0))
                mask_circ = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask_circ, self.energy_center, wipe_radius, 255, -1)
                mask_circ_blurred = cv2.GaussianBlur(mask_circ, (31, 31), 0)
                norm_wipe = (mask_circ_blurred.astype(np.float32) / 255.0)[:, :, np.newaxis]
                active_bg = (domain_bg.astype(np.float32) * norm_wipe +
                             raw_frame.astype(np.float32) * (1.0 - norm_wipe)).astype(np.uint8)
            elif self.state == "COLLAPSE":
                dissolve = max(0.0, 1.0 - (self.state_timer / 25.0))
                active_bg = cv2.addWeighted(domain_bg, dissolve, raw_frame, 1.0 - dissolve, 0)
            else:
                active_bg = domain_bg
        else:
            active_bg = raw_frame.copy()

        # 4. Person Foreground & Cursed Aura Compositing
        if self.state in ["DOMAIN_ACTIVE", "EXPANSION", "COLLAPSE"]:
            # Full aura compositing
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
            # Subtle charging aura on person
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
        particle_mode = "float"
        p_intensity = 1.0
        p_center = self.energy_center
        if self.state == "CHARGING":
            particle_mode = "suck_in"
            p_intensity = 1.2
        elif self.state == "COLLAPSE":
            particle_mode = "blast_out"
            p_intensity = 1.5
        elif self.state == "NORMAL":
            p_intensity = 0.0  # Clean normal video until domain activates

        if p_intensity > 0.01:
            composited = self.particle_system.update_and_render(
                composited,
                primary_color=col_pri,
                secondary_color=col_sec,
                mode=particle_mode,
                center=p_center,
                intensity=p_intensity
            )

        # 6. Hand Energy Orbs & Electric Lightning
        if self.state in ["CHARGING", "EXPANSION", "DOMAIN_ACTIVE"]:
            composited = self.hand_fx.render_hand_effects(
                composited,
                hands=hands,
                pose=tracking.get("pose"),
                primary_color=col_pri,
                secondary_color=col_sec,
                timer=self.total_frames,
                is_charging=(self.state == "CHARGING"),
                energy_center=self.energy_center
            )

        # 7. Barrier Shockwaves
        composited = self.shockwave_fx.update_and_render(composited)

        # 8. Flash Transition
        if self.flash_fx.active:
            composited, _ = self.flash_fx.apply(composited)

        # 9. Optical Distortion & Screen Shake
        composited = self.distortion_fx.apply(composited)

        # 10. Japanese Typography Overlay
        if self.state in ["EXPANSION", "DOMAIN_ACTIVE"]:
            # Text fades in during expansion, stays during domain, then fades
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

        # 11. HUD Overlay
        if self.show_hud and not self.headless:
            self._render_hud(composited, gesture_info)

        self.total_frames += 1
        return composited

    def _render_hud(self, frame: np.ndarray, gesture_info: dict):
        """Render a sleek modern AR HUD."""
        # Top-left status pill
        h, w = frame.shape[:2]
        hud_w, hud_h = 290, 85
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (18, 14, 22), -1)
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (70, 50, 90), 1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # State badge color
        state_colors = {
            "NORMAL": (180, 180, 180),
            "CHARGING": (80, 210, 255),
            "FLASH": (255, 255, 255),
            "EXPANSION": (255, 100, 220),
            "DOMAIN_ACTIVE": (255, 60, 180),
            "COLLAPSE": (100, 100, 200),
        }
        color = state_colors.get(self.state, (200, 200, 200))

        cv2.putText(frame, f"STATUS: {self.state}", (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"THEME : {self.bg_renderer.theme['name_en']}", (22, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "[D/Space] Expand  [1/2] Theme  [R] Reset", (22, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        # Charging bar when charging
        if self.state == "CHARGING":
            charge_ratio = min(1.0, self.state_timer / 22.0)
            bar_w = int(hud_w * charge_ratio)
            cv2.rectangle(frame, (12, 12 + hud_h - 4), (12 + bar_w, 12 + hud_h), (80, 220, 255), -1)

    def run(self):
        """Main application loop."""
        print("\n=======================================================")
        print("  JUJUTSU KAISEN - DOMAIN EXPANSION AR FILTER (領域展開)")
        print("=======================================================")
        print("  Controls:")
        print("    [D] / [Space] : Trigger Domain Expansion")
        print("    [1]           : Switch to Infinite Void (無量空処)")
        print("    [2]           : Switch to Malevolent Shrine (伏魔御廚子)")
        print("    [R]           : Reset / Collapse Domain")
        print("    [H]           : Toggle HUD Overlay")
        print("    [S]           : Save Screenshot")
        print("    [Q] / [ESC]   : Quit")
        print("=======================================================\n")

        fps_timer = time.time()
        frame_counter = 0

        try:
            while self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    if self.is_demo:
                        # Demo loops indefinitely unless max_frames specified
                        pass
                    else:
                        break

                output_frame = self.process_frame(raw_frame)

                if self.video_writer:
                    self.video_writer.write(output_frame)

                frame_counter += 1
                if not self.headless:
                    cv2.imshow("Domain Expansion AR", output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        break
                    elif key in [ord("d"), ord("D"), 32]:  # Space or D
                        self.trigger_domain()
                    elif key in [ord("r"), ord("R")]:
                        self.reset_domain()
                    elif key == ord("1"):
                        self.set_theme("infinite_void")
                    elif key == ord("2"):
                        self.set_theme("malevolent_shrine")
                    elif key in [ord("h"), ord("H")]:
                        self.show_hud = not self.show_hud
                    elif key in [ord("s"), ord("S")]:
                        ss_name = f"domain_screenshot_{int(time.time())}.png"
                        cv2.imwrite(ss_name, output_frame)
                        print(f"Saved screenshot: {ss_name}")

                if self.max_frames and frame_counter >= self.max_frames:
                    print(f"Reached max frames limit ({self.max_frames}). Finishing.")
                    break

                # FPS tracking
                if frame_counter % 30 == 0:
                    elapsed = time.time() - fps_timer
                    fps = 30.0 / elapsed if elapsed > 0 else 0
                    if self.headless:
                        print(f"Frame {frame_counter} | State: {self.state} | FPS: {fps:.1f}")
                    fps_timer = time.time()

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
    parser = argparse.ArgumentParser(description="JJK Domain Expansion AR Filter")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo simulation mode")
    parser.add_argument("--theme", type=str, default="infinite_void",
                        choices=["infinite_void", "malevolent_shrine"], help="Domain Theme")
    parser.add_argument("--record", type=str, default=None, help="Path to save output video (.mp4)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--width", type=int, default=640, help="Processing width")
    parser.add_argument("--height", type=int, default=480, help="Processing height")

    args = parser.parse_args()

    # Auto-detect headless if DISPLAY is not set
    is_headless = args.headless or ("DISPLAY" not in os.environ or not os.environ["DISPLAY"])
    record_path = args.record
    if is_headless and record_path is None:
        record_path = "output.mp4"
        print("SSH/Headless session detected: Automatically recording to output.mp4")
        print("You can click output.mp4 in your editor explorer to view it directly!")

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
