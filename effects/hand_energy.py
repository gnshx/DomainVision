import math
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class HandEnergyEffect:
    """
    Renders cursed energy orbs on palms, chaotic electric lightning arcs between hands,
    and charging convergence spirals.
    """

    def __init__(self):
        pass

    def _draw_lightning_bolt(
        self,
        canvas: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        displace: float = 18.0,
        subdivisions: int = 4,
    ):
        """Draws a jagged procedural lightning branch between two points."""
        points = [p1, p2]
        for _ in range(subdivisions):
            new_points = []
            for i in range(len(points) - 1):
                pt_a = np.array(points[i], dtype=np.float32)
                pt_b = np.array(points[i + 1], dtype=np.float32)
                mid = (pt_a + pt_b) / 2.0

                # Normal vector
                diff = pt_b - pt_a
                norm = np.array([-diff[1], diff[0]], dtype=np.float32)
                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-4:
                    norm = norm / norm_len

                offset = np.random.uniform(-displace, displace)
                jitter_pt = mid + norm * offset
                new_points.append(tuple(points[i]))
                new_points.append((int(jitter_pt[0]), int(jitter_pt[1])))
            new_points.append(points[-1])
            points = new_points
            displace *= 0.55

        # Draw outer glowing bolt
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], color, 4)
        # Draw crisp bright core bolt
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (255, 255, 255), 1)

    def render_hand_effects(
        self,
        frame: np.ndarray,
        hands: List[Dict[str, Any]],
        pose: Optional[Dict[str, Any]] = None,
        primary_color: Tuple[int, int, int] = (255, 60, 180),
        secondary_color: Tuple[int, int, int] = (255, 220, 80),
        timer: int = 0,
        is_charging: bool = False,
        energy_center: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Render glowing hand orbs, lightning arcs, and charging spirals."""
        active_hands = list(hands) if hands else []
        if not active_hands and pose:
            if pose.get("left_wrist"):
                active_hands.append({"palm_center": pose["left_wrist"]})
            if pose.get("right_wrist"):
                active_hands.append({"palm_center": pose["right_wrist"]})

        if not active_hands and not energy_center:
            return frame

        h, w = frame.shape[:2]
        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Energy orbs on each hand palm
        for hand in active_hands:
            center = hand["palm_center"]
            pulse = math.sin(timer * 0.2 + center[0]) * 6.0
            base_radius = 28 + int(pulse)

            # Outer colored glow
            cv2.circle(glow_layer, center, base_radius + 12, primary_color, -1)
            cv2.circle(glow_layer, center, base_radius, secondary_color, -1)
            # Inner white core
            cv2.circle(glow_layer, center, max(6, int(base_radius * 0.45)), (255, 255, 255), -1)

            # Orbiting energy sparks
            for i in range(4):
                angle = (timer * 0.15) + (i * math.pi / 2.0)
                orbit_r = base_radius + 18 + int(5 * math.sin(timer * 0.3 + i))
                ox = int(center[0] + orbit_r * math.cos(angle))
                oy = int(center[1] + orbit_r * math.sin(angle))
                if 0 <= ox < w and 0 <= oy < h:
                    cv2.circle(glow_layer, (ox, oy), 3, (255, 255, 255), -1)
                    cv2.circle(glow_layer, (ox, oy), 6, secondary_color, 1)

        # 2. Electric lightning arcs connecting hands
        if len(hands) >= 2:
            p1 = hands[0]["palm_center"]
            p2 = hands[1]["palm_center"]
            # Draw main connecting lightning bolt
            self._draw_lightning_bolt(glow_layer, p1, p2, primary_color, displace=22.0)

            # Secondary crackling arc
            if timer % 3 == 0:
                t1 = hands[0].get("index_tip", p1)
                t2 = hands[1].get("index_tip", p2)
                self._draw_lightning_bolt(glow_layer, t1, t2, secondary_color, displace=14.0)

        # 3. Charging energy convergence spirals
        if is_charging and energy_center:
            cx, cy = energy_center
            # Draw converging energy spiral arcs
            num_arms = 3
            for arm in range(num_arms):
                spiral_pts = []
                arm_offset = arm * (2 * math.pi / num_arms)
                for step in range(15):
                    frac = step / 14.0
                    rad = (1.0 - frac) * 90.0
                    ang = arm_offset + frac * 4.0 - timer * 0.25
                    px = int(cx + rad * math.cos(ang))
                    py = int(cy + rad * math.sin(ang))
                    spiral_pts.append((px, py))

                for i in range(len(spiral_pts) - 1):
                    alpha_factor = i / len(spiral_pts)
                    col = (
                        int(primary_color[0] * alpha_factor),
                        int(primary_color[1] * alpha_factor),
                        int(primary_color[2] * alpha_factor)
                    )
                    cv2.line(glow_layer, spiral_pts[i], spiral_pts[i + 1], col, 2)

            # Charging bright focal point
            charge_rad = 12 + int(6 * math.sin(timer * 0.4))
            cv2.circle(glow_layer, (cx, cy), charge_rad + 15, primary_color, -1)
            cv2.circle(glow_layer, (cx, cy), charge_rad, (255, 255, 255), -1)

        # Blur glow layer and blend additively
        blurred = cv2.GaussianBlur(glow_layer, (21, 21), 0)
        output = cv2.add(frame, blurred)
        output = cv2.add(output, glow_layer)
        return output
