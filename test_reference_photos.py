import cv2
import numpy as np
from tracking.detector import MediaPipeVisionTracker
from tracking.gesture_recognizer import CanonicalGestureRecognizer

tracker = MediaPipeVisionTracker(async_mode=False)
rec = CanonicalGestureRecognizer()

for name, path in [("GOJO", "test_images/gojo_reference.png"), ("SUKUNA", "test_images/sukuna_reference.png")]:
    print(f"\n==================== TESTING {name} ====================")
    img = cv2.imread(path)
    if img is None:
        print(f"FAILED to read image at {path}")
        continue
    
    h, w = img.shape[:2]
    print(f"Image shape: {w}x{h}")
    
    # Process through vision tracker
    tracking = tracker.process(img)
    hands = tracking["hands"]
    print(f"Hands detected: {len(hands)}")
    
    for idx, hand in enumerate(hands):
        print(f"--- Hand {idx} ---")
        print(f"  Palm center: {hand.get('palm_center')}")
        print(f"  Palm scale: {hand.get('palm_scale'):.1f}")
        print(f"  Palm y norm: {hand.get('palm_y_norm'):.3f}")
        print(f"  Handedness: {hand.get('handedness')} ({hand.get('handedness_conf', 0):.2f})")
        print(f"  Pointing dir: {hand.get('pointing_dir')}")
        print(f"  Cross ratio: {hand.get('cross_ratio', 0):.3f}")
        print(f"  Is crossing mudra: {hand.get('is_crossing_mudra')}")
        print(f"  Thumb tucked: {hand.get('thumb_tucked')}")
        print(f"  Finger states: {hand.get('finger_states')}")
        print(f"  Finger angles: {hand.get('finger_angles')}")
        print(f"  Curl ratios: {hand.get('curl_ratios')}")
        print(f"  Index tip: {hand.get('index_tip')}")
        print(f"  Middle tip: {hand.get('middle_tip')}")

    # Evaluate with gesture recognizer
    res = rec.update(hands, (h, w), target_theme="malevolent_shrine" if name == "SUKUNA" else "infinite_void")
    print(f"\nGesture Recognizer Results for {name}:")
    print(f"  Candidate: {res.get('candidate_sign')}")
    print(f"  Confirmed: {res.get('confirmed_sign')}")
    print(f"  State: {res.get('state')}")
    print(f"  Gojo score: {res.get('gojo_score'):.3f} ({int(res.get('gojo_score')*100)}%)")
    print(f"  Sukuna score: {res.get('sukuna_score'):.3f} ({int(res.get('sukuna_score')*100)}%)")
    print(f"  Status text: {res.get('status_text')}")
    print(f"  Diagnostics: {res.get('metrics')}")

tracker.close()
