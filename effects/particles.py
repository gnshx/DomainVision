import random
import math
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np


class CursedParticleSystem:
    """
    Simulates floating cursed energy motes, embers, and dynamic inward/outward vortexes.

    Enhancements over v1:
    - Depth field (z: 0.0=far, 1.0=close): size, speed, and brightness scale with depth
    - Velocity-field hand attractors: particles drift toward hand positions during CHARGING
    - Adaptive particle cap: accepts max_particles override per-frame from quality controller
    - All allocations avoided inside the inner loop; glow_layer is pre-allocated and reused
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
        """Spawn a new particle with depth, velocity, and attractor fields."""
        depth = random.uniform(0.15, 1.0)  # z: far=0, near=1
        speed_scale = 0.5 + depth * 1.0    # near particles move faster

        if mode == "suck_in" and center:
            cx, cy = center
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(60, 280)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            speed = random.uniform(2.5, 6.0) * speed_scale
            vx = -speed * math.cos(angle)
            vy = -speed * math.sin(angle)
        elif mode == "blast_out" and center:
            cx, cy = center
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(5, 25)
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            speed = random.uniform(5.0, 14.0) * speed_scale
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
        else:
            # Normal upward float — near particles start lower and move faster
            x = random.uniform(0, w)
            y = random.uniform(h * (0.3 + (1.0 - depth) * 0.3), h)
            vx = random.uniform(-0.5, 0.5) * speed_scale
            vy = -random.uniform(1.0, 3.5) * speed_scale

        color_choice = primary_color if random.random() < 0.65 else secondary_color
        # Near particles are brighter
        brightness = random.uniform(0.7, 1.0) + depth * 0.4
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
            "depth": depth,
            "size": random.uniform(1.0, 2.5) + depth * 2.5,  # near = bigger
            "color": p_color,
            "life": 1.0,
            "decay": random.uniform(0.010, 0.030) * (0.7 + depth * 0.6),  # near particles die faster
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
        max_particles: Optional[int] = None,
        hand_attractors: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Update particle physics with optional hand attractors and draw with additive glow.

        Args:
            max_particles: Override default max from quality controller. None = use class default.
            hand_attractors: List of (x, y) positions of detected hand joints — particles
                             drift toward nearest attractor during CHARGING mode.
        """
        h, w = frame.shape[:2]
        cap = int(min(self.max_particles, max_particles or self.max_particles) * intensity)

        # Spawn new particles to reach target count
        while len(self.particles) < cap:
            self.particles.append(
                self.spawn_particle(w, h, primary_color, secondary_color, mode=mode, center=center)
            )

        # Pre-allocate or reuse glow layer — no allocation inside the loop
        if self._glow_layer is None or self._glow_layer.shape[:2] != (h, w):
            self._glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            self._glow_layer.fill(0)
        glow_layer = self._glow_layer
        alive_particles = []

        for p in self.particles:
            depth = p["depth"]

            if mode == "float":
                # Lateral sway + upward drift; near particles sway more
                p["x"] += p["vx"] + (0.4 + depth * 0.4) * math.sin(p["sway_offset"] + p["y"] * 0.04)
                p["y"] += p["vy"]

                # Hand attractor: gently pull float particles toward nearest hand joint
                if hand_attractors:
                    ax, ay = hand_attractors[0]
                    dx = ax - p["x"]
                    dy = ay - p["y"]
                    dist = math.hypot(dx, dy)
                    if dist > 5:
                        attract_strength = 0.08 * depth  # near particles respond more
                        p["x"] += attract_strength * dx / dist
                        p["y"] += attract_strength * dy / dist
            else:
                p["x"] += p["vx"]
                p["y"] += p["vy"]

            p["life"] -= p["decay"]

            if (
                p["life"] > 0
                and 0 <= p["x"] < w
                and 0 <= p["y"] < h
            ):
                alive_particles.append(p)
                alpha = max(0.0, p["life"])
                r_core = max(1, int(p["size"]))
                pt = (int(p["x"]), int(p["y"]))

                # Multi-halo: inner white-hot core, mid vibrant color, outer soft halo
                # Near (high depth) particles have larger halos for parallax feel
                halo_r = r_core + int(3 + depth * 3)
                c_outer = (
                    int(p["color"][0] * alpha * 0.30),
                    int(p["color"][1] * alpha * 0.30),
                    int(p["color"][2] * alpha * 0.30),
                )
                c_mid = (
                    int(p["color"][0] * alpha * 0.85),
                    int(p["color"][1] * alpha * 0.85),
                    int(p["color"][2] * alpha * 0.85),
                )
                c_core = (
                    min(255, int(c_mid[0] + 70)),
                    min(255, int(c_mid[1] + 70)),
                    min(255, int(c_mid[2] + 70)),
                )

                cv2.circle(glow_layer, pt, halo_r, c_outer, -1)
                cv2.circle(glow_layer, pt, r_core + 1, c_mid, -1)
                cv2.circle(glow_layer, pt, max(1, r_core - 1), c_core, -1)

        self.particles = alive_particles

        # Additive blending — no full-canvas blur overhead
        return cv2.add(frame, glow_layer)

