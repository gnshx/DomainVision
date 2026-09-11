# DomainVision - Jujutsu Kaisen Domain Expansion AR Filter (領域展開)

A real-time, cinematic **Jujutsu Kaisen (JJK)-inspired "Domain Expansion" (領域展開) Augmented Reality filter** built in Python with **OpenCV**, **MediaPipe Tasks**, **NumPy**, and **Pillow**.

DomainVision 2.0 transitions from a basic 2D overlay to an **authentic AR filter** centered around **accurate 3D finger tracking**, **canonical mudra gesture recognition**, **neural person segmentation**, **perspective-anchored skeletal energy**, and **synchronized audio**.

---

## ⛩️ Canonical Domain Themes & Hand Signs

Domain activation requires the user to form and hold the canonical hand sign:

### 1. 🩸 Malevolent Shrine (伏魔御廚子 - Ryomen Sukuna) [DEFAULT]
- **Hand Sign**: **Enma-ten Mudra (閻魔天印 / Yama Mudra)**.
- **Gesture**: Bring both palms together at chest level with thumbs pointing straight up, index fingertips pressed together, and lower fingers (middle, ring, pinky) curled inward.
- **Environment**: Demonic pagoda shrine flanked by horns, skulls, and a blood-crimson stormy sky positioned **behind** the segmented user.
- **Effects**: Burning ember particles, fiery crimson aura, skeletal cursed energy along finger bones, lightning arcs across fingertips, and Japanese calligraphy banner (`領域展開 伏魔御廚子`).

### 2. 🌌 Infinite Void (無量空処 - Satoru Gojo)
- **Hand Sign**: **Taishakuten Mudra (帝釈天印 / Indra Mudra)**.
- **Gesture**: Single hand raised to eye level. Cross your middle finger over your index finger, holding your ring and pinky fingers down with your thumb.
- **Environment**: Infinite black hole singularity and celestial cosmic void.
- **Effects**: Radiant purple & electric blue cursed aura, astrolabe barrier rings, inward gravity motes, and Japanese calligraphy banner (`領域展開 無量空処`).

---

## ⚡ Core Technical Features

```
Camera / Browser Webcam (640×480)
       ↓
MediaPipe 3D Landmark & Segmentation Tracking
       ↓
Vector Cosine Joint-Angle Analysis (21 joints per hand)
       ↓
Canonical Hand Sign Recognizer (Sukuna / Gojo)
       ↓
Temporal Hold Meter (12 frames stable hold)
       ↓
State Machine (NORMAL → CHARGING → FLASH → EXPANSION → DOMAIN_ACTIVE → COLLAPSE)
       ↓
Depth Layering: [BACKGROUND SHRINE] → [AURA] → [YOU] → [SKELETAL ENERGY & PARTICLES]
       ↓
Synchronized Audio SFX + Real-Time Telemetry HUD (FPS, Track ms, Render ms)
```

1. **Precision 3D Finger-Angle Analysis**:
   Uses 3D vectors and cosine angles rather than 2D pixel coordinates:
   $$\cos(\theta) = \frac{\vec{ba} \cdot \vec{bc}}{\|\vec{ba}\| \|\vec{bc}\|}$$
   Categorizes each finger dynamically into `EXTENDED`, `BENT`, or `CURLED`.

2. **Velocity-Adaptive Landmark Smoothing**:
   Exponential moving average ($$\alpha = 0.40$$) with velocity-adaptive scaling completely removes camera jitter while preserving snappy hand movements.

3. **Temporal Pose Requirement**:
   Prevents accidental triggering by requiring the user to hold the canonical mudra for 12 consecutive frames, visualized via a charging HUD meter.

4. **Depth-Layered Compositing**:
   The user is cleanly segmented using MediaPipe's neural selfie segmenter, allowing the Malevolent Shrine environment to exist realistically **behind** the user while cursed aura and skeletal hand energy wrap around them in the foreground.

5. **Perspective-Anchored Skeletal Energy**:
   Glowing cursed energy filaments connect directly to the 21 finger joints, wrist, and palm center, scaling and rotating as the user moves closer or turns their hands.

6. **SSH Browser Streaming**:
   Run `python web_app.py` on a remote headless server to stream the live AR filter in your laptop browser using your local webcam over WebRTC/HTTP without needing `/dev/video0` on the server!

---

## 🚀 Quickstart

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run with Live Webcam
```bash
python main.py
```
*(Specify camera index if needed: `python main.py --camera 1`)*

### 3. Remote Server / SSH (Browser Webcam Stream)
When running over SSH without a physical webcam connected to the Linux server:
```bash
python web_app.py --port 8080
```
Open `http://localhost:8080/` in your browser. Select **📷 My Local Webcam** to use your laptop camera in real time, or **🤖 Animated Demo Feed** to watch the automated demonstration!

### 4. Automated Demo & Video Recording
```bash
python main.py --demo --theme malevolent_shrine --frames 130 --record demo_sukuna_v2.mp4
```

---

## 🎮 Interactive Controls

| Key | Action |
| :---: | :--- |
| **`D` / `Space`** | **Force Trigger Domain Expansion** |
| **`1`** | Switch to **Malevolent Shrine (伏魔御廚子 - Sukuna)** |
| **`2`** | Switch to **Infinite Void (無量空処 - Gojo)** |
| **`R`** | **Reset / Collapse** Domain back to Normal |
| **`H`** | Toggle Telemetry HUD Overlay |
| **`S`** | Save high-res screenshot (`domain_screenshot_<timestamp>.png`) |
| **`Q` / `ESC`** | Quit application |

---

## 📂 Project Architecture

```
domaincv/
├── main.py                     # Application state machine, pipeline orchestration, CLI, HUD
├── config.py                   # Resolutions, smoothing factors, angle thresholds, domain themes
├── web_app.py                  # Threading HTTP server with browser webcam & demo stream
├── requirements.txt            # Python dependencies
│
├── tracking/
│   ├── detector.py             # Unified MediaPipe tracker (Segmentation, Hands, Pose)
│   ├── hand_tracker.py         # 21-joint 3D angle analyzer & synthetic mudra generator
│   └── gesture_recognizer.py   # Canonical Sukuna & Gojo mudra recognizer with hold meter
│
├── effects/
│   ├── domain_environment.py   # Layered Malevolent Shrine & Infinite Void environment renderer
│   ├── aura.py                 # Neural segmentation boundary aura with multi-scale Gaussian glow
│   ├── cursed_energy.py        # Perspective skeletal hand filaments and fingertip lightning
│   ├── particles.py            # Physics particle system (floating embers, vortex pull, shock blast)
│   ├── flash.py                # Multi-stage blinding flash transition
│   ├── shockwave.py            # Concentric expanding barrier rings
│   └── distortion.py           # Optical refraction shockwave via cv2.remap() & camera shake
│
├── utils/
│   ├── smoothing.py            # Velocity-adaptive Exponential Moving Average (EMA)
│   ├── fps.py                  # Real-time PerformanceProfiler (FPS, Track ms, Render ms)
│   ├── text_renderer.py        # Pillow Japanese typography with glow and drop shadow
│   ├── demo_feed.py            # Synthetic animated character video stream
│   └── blending.py             # Additive, screen, and alpha compositing utilities
│
├── audio/
│   └── audio_manager.py        # Synchronized SFX player (charge, activation, impact, slash, collapse)
│
└── assets/
    ├── shrine.png              # Sukuna's Malevolent Shrine environment backdrop
    ├── void.png                # Gojo's Infinite Void environment backdrop
    ├── models/                 # MediaPipe task models (selfie_segmenter, hand, pose)
    └── sounds/                 # 16-bit PCM WAV sound effects
```
