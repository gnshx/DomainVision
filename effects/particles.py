import random
import math
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np


class CursedParticleSystem:
    """
    Simulates floating cursed energy motes, embers, and dynamic inward/outward vortexes.
    """

    def __init__(self, max_particles: int = 120):
        self.max_particles = max_particles
        self.particles: List[Dict] = []
        self._glow_layer = None

    def spawn_particle(
        self,
        w: int,
        h: int,
        primary_color: Tuple[int, int, int],
        secondary_color: Tuple[int, int, int],
        mode: str = "float",
        center: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Spawn a new particle depending on current mode."""
        if mode == "suck_in" and center:
            cx, cy = center
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(80, 240)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            speed = random.uniform(3.0, 7.0)
            vx = -speed * math.cos(angle)
            vy = -speed * math.sin(angle)
        elif mode == "blast_out" and center:
            cx, cy = center
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(5, 30)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            speed = random.uniform(6.0, 15.0)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
        else:
            # Normal upward float
            x = random.uniform(0, w)
            y = random.uniform(h * 0.4, h)
            vx = random.uniform(-0.8, 0.8)
            vy = -random.uniform(1.2, 3.8)

        color_choice = primary_color if random.random() < 0.65 else secondary_color
        # Add random brightness variation
        brightness = random.uniform(0.7, 1.3)
        p_color = (
            min(255, int(color_choice[0] * brightness)),
            min(255, int(color_choice[1] * brightness)),
            min(255, int(color_choice[2] * brightness)),
        )

        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "size": random.uniform(1.5, 4.0),
            "color": p_color,
            "life": 1.0,
            "decay": random.uniform(0.012, 0.035),
            "sway_offset": random.uniform(0, 2 * math.pi),
        }

    def update_and_render(
        self,
        frame: np.ndarray,
        primary_color: Tuple[int, int, int] = (255, 60, 180),
        secondary_color: Tuple[int, int, int] = (255, 220, 80),
        mode: str = "float",
        center: Optional[Tuple[int, int]] = None,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Update particle physics and draw with additive glow."""
        h, w = frame.shape[:2]

        target_count = int(self.max_particles * intensity)
        while len(self.particles) < target_count:
            self.particles.append(
                self.spawn_particle(w, h, primary_color, secondary_color, mode=mode, center=center)
            )

        # Pre-allocate or draw directly using fast additive blending
        if self._glow_layer is None or self._glow_layer.shape[:2] != (h, w):
            self._glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            self._glow_layer.fill(0)
        glow_layer = self._glow_layer
        alive_particles = []

        for p in self.particles:
            # Update position
            if mode == "float":
                p["x"] += p["vx"] + 0.6 * math.sin(p["sway_offset"] + p["y"] * 0.05)
                p["y"] += p["vy"]
            else:
                p["x"] += p["vx"]
                p["y"] += p["vy"]

            p["life"] -= p["decay"]

            # Boundary checks
            if (
                p["life"] > 0
                and 0 <= p["x"] < w
                and 0 <= p["y"] < h
            ):
                alive_particles.append(p)
                alpha = max(0.0, p["life"])
                r_core = max(1, int(p["size"]))
                pt = (int(p["x"]), int(p["y"]))

                # Multi-halo glowing particle: inner white-hot core, middle vibrant, outer faint halo
                c_outer = (
                    int(p["color"][0] * alpha * 0.35),
                    int(p["color"][1] * alpha * 0.35),
                    int(p["color"][2] * alpha * 0.35),
                )
                c_mid = (
                    int(p["color"][0] * alpha * 0.85),
                    int(p["color"][1] * alpha * 0.85),
                    int(p["color"][2] * alpha * 0.85),
                )
                c_core = (
                    min(255, int(c_mid[0] + 80)),
                    min(255, int(c_mid[1] + 80)),
                    min(255, int(c_mid[2] + 80)),
                )

                # Outer soft halo
                cv2.circle(glow_layer, pt, r_core + 4, c_outer, -1)
                # Vibrant flame body
                cv2.circle(glow_layer, pt, r_core + 1, c_mid, -1)
                # Hot center
                cv2.circle(glow_layer, pt, max(1, r_core - 1), c_core, -1)

        self.particles = alive_particles

        # Direct additive blending without full-canvas blur overhead
        return cv2.add(frame, glow_layer)

