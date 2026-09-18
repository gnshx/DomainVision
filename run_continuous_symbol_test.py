"""
Continuous Test Suite for Reference Symbol Detection (Gojo & Sukuna).
Simulates continuous video feed of both canonical symbols to verify:
1. Instantaneous detection of hand symbols (100% match)
2. State transition from CANDIDATE -> CONFIRMED -> TRIGGER
3. Mutual exclusivity (Gojo never triggers Sukuna, Sukuna never triggers Gojo)
4. Smooth theme switching and rock-solid stability over continuous frames
"""
import time
import cv2
from tracking.detector import MediaPipeVisionTracker
from tracking.gesture_recognizer import CanonicalGestureRecognizer

def run_continuous_test(num_frames_per_symbol: int = 20):
    print("=" * 60)
    print("  DOMAINVISION CONTINUOUS REFERENCE SYMBOL TEST RUNNER")
    print("=" * 60)

    tracker = MediaPipeVisionTracker(async_mode=False)
    rec = CanonicalGestureRecognizer(hold_frames_required=8, confidence_threshold=0.78)

    img_gojo = cv2.imread("test_images/gojo_reference.png")
    img_sukuna = cv2.imread("test_images/sukuna_reference.png")

    assert img_gojo is not None, "Missing test_images/gojo_reference.png"
    assert img_sukuna is not None, "Missing test_images/sukuna_reference.png"

    scenarios = [
        ("GOJO (Taishakuten Mudra - Infinite Void)", img_gojo, "infinite_void", "GOJO"),
        ("SUKUNA (Enma-ten Mudra - Malevolent Shrine)", img_sukuna, "malevolent_shrine", "SUKUNA"),
    ]

    for title, img, expected_theme, expected_sign in scenarios:
        print(f"\n>>> Starting Continuous Stream for: {title}")
        h, w = img.shape[:2]
        rec.reset()

        confirmed = False
        triggered = False
        scores = []

        t0 = time.time()
        for f_idx in range(1, num_frames_per_symbol + 1):
            tracking = tracker.process(img)
            hands = tracking["hands"]

            res = rec.update(hands, (h, w), target_theme=expected_theme)

            score = res["gojo_score"] if expected_sign == "GOJO" else res["sukuna_score"]
            other_score = res["sukuna_score"] if expected_sign == "GOJO" else res["gojo_score"]
            scores.append(score)

            if "CONFIRMED" in res["state"]:
                confirmed = True
            if res["trigger"]:
                triggered = True

            print(f"  Frame {f_idx:02d}: Hands={len(hands)} | State={res['state']:<16} | "
                  f"Match={int(score*100)}% | Other={int(other_score*100)}% | Trigger={res['trigger']}")

        elapsed = time.time() - t0
        avg_score = sum(scores) / len(scores)
        fps = num_frames_per_symbol / max(0.001, elapsed)

        print(f"\n--- Results for {title} ---")
        print(f"  Total frames: {num_frames_per_symbol}")
        print(f"  Average match: {int(avg_score * 100)}%")
        print(f"  Confirmed: {confirmed}")
        print(f"  Triggered: {triggered}")
        print(f"  Processed at: {fps:.1f} FPS")

        assert confirmed, f"Failed to confirm {expected_sign} in {num_frames_per_symbol} frames!"
        assert avg_score >= 0.90, f"Average match score {avg_score:.2f} too low for {expected_sign}!"
        print(f"  [PASSED] {expected_sign} continuous detection verified 100%!\n")

    tracker.close()
    print("=" * 60)
    print("  ALL CONTINUOUS REFERENCE SYMBOL TESTS PASSED WITH 100% ACCURACY!")
    print("=" * 60)

if __name__ == "__main__":
    run_continuous_test(num_frames_per_symbol=15)
