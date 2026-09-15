<div align="center">

# ⛩️ DomainVision (領域展開)
### Real-Time Jujutsu Kaisen AR Engine & Invariant Mudra Classifier

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0%2B-0078D7.svg)](https://developers.google.com/mediapipe)
[![Benchmarks](https://img.shields.io/badge/Benchmark-35.8%20FPS%20Sync%20%7C%2050%2B%20FPS%20Decoupled-00C853.svg)](PERFORMANCE_BASELINE.md)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(6%2F6)-brightgreen.svg)](tests/test_gestures.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*An authentic, cinematic **Jujutsu Kaisen**-inspired Domain Expansion AR pipeline featuring invariant multi-class hand-sign recognition, SIMD-accelerated effects, neural selfie segmentation, and canonical audio synchronization.*

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Mudra Recognition](#-canonical-mudra-recognition) • [Performance Benchmarks](#-performance-benchmarks) • [Quickstart](#-quickstart) • [Web Application](#-web-application--remote-ssh) • [Calibration & Debugging](#-calibration--debugging) • [Controls](#-interactive-controls)

---

</div>

## 🌟 Key Features

**DomainVision** transforms physical hand gestures into cinematic Jujutsu Kaisen "Domain Expansion" (領域展開) techniques in real-time.

- **Invariant Multi-Class Mudra Recognition**: Palm-scale invariant coordinate frames, 3D finger curl ratios, and spatial crossing metrics evaluate gestures independent of camera distance.
- **Strict Hand-Count Enforcement**:
  - **Malevolent Shrine (Sukuna)**: Strictly requires **2 hands** clasped with upright thumbs and touching index fingertips.
  - **Infinite Void (Gojo)**: Strictly requires **1 hand** raised to head level with middle finger crossed over index finger.
  - **Neutral State**: Insufficient or non-matching hand positions evaluate to **UNKNOWN**—Gojo is never used as an ambiguous fallback.
- **Decoupled High-FPS Architecture**: Asynchronous vision tracking isolates MediaPipe inference from graphics rendering, preventing dropped frames or pipeline stalls.
- **SIMD Integer Compositing**: Optimized color grading, 2.5D parallax layer rendering, and downscaled optical flow reduce frame rendering time by over 57%.
- **Canonical Timeline & Audio Engine**: Event triggers synchronized to the official Crunchyroll audio progression (0.00s mudra hold → 0.70s tremor → 1.20s chant resonance → 1.35s blinding flash → 1.50s domain barrier).
- **Interactive Calibration & HUD**: Real-time 21-landmark skeleton visualizer, HUD confidence bars, and on-screen gesture calibration overlay.
- **Dual-Mode Web Server**: Seamless HTTP browser stream with local webcam capture and animated character demonstration feed.

---

## ⛩️ Canonical Mudra Recognition

Mudra evaluation uses hand-scale invariant geometric metrics rather than raw screen coordinates:

| Domain | Character | Mudra / Seal | Strict Anatomical Constraints |
|---|---|---|---|
| **Malevolent Shrine**<br>`伏魔御廚子` | **Ryomen Sukuna** | **Enma-ten Mudra**<br>*(閻魔天印 / Yama)* | • **Hands Detected**: **Strictly 2 hands** ($N = 2$). Single hand scores $0\%$.<br>• **Palm Proximity**: Inter-wrist distance $< 1.15 \times$ palm scale.<br>• **Thumbs**: Upright alignment ($u_{long} \cdot [0, -1] > 0.45$).<br>• **Index Fingers**: Fingertip distance $< 0.50 \times$ palm scale.<br>• **Lower Fingers**: Middle, ring, and pinky curled inward ($> 0.60$ curl ratio). |
| **Infinite Void**<br>`無量空処` | **Satoru Gojo** | **Taishakuten Mudra**<br>*(帝釈天印 / Indra)* | • **Hands Detected**: **Exactly 1 hand** ($N = 1$). 2 hands score $0\%$.<br>• **Spatial Height**: Hand raised to head/eye level ($y_{wrist} < 0.62$).<br>• **Finger Crossing**: Middle finger crossed over index finger ($\text{cross\_ratio} < 0.40$).<br>• **Extended Fingers**: Index and middle fully extended ($< 0.40$ curl ratio).<br>• **Folded Fingers**: Ring and pinky tucked into palm ($> 0.65$ curl ratio). |

```
                       [ INPUT HAND LANDMARKS ]
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          [ Hand Count = 2 ]              [ Hand Count = 1 ]
                  │                               │
         Sukuna Evaluator                  Gojo Evaluator
        - Wrist separation                - Wrist height (< 0.62)
        - Upright thumbs                  - Index/middle extension
        - Index tip proximity             - 3D Crossing metric (< 0.40)
        - Ring/pinky curl (> 0.60)        - Ring/pinky tuck (> 0.65)
                  │                               │
                  ▼                               ▼
         Sukuna Score: 0–100%             Gojo Score: 0–100%
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                     [ Multi-Class State Machine ]
                  Neither ≥ 78% ──► State: UNKNOWN
                  Sukuna ≥ 78%  ──► State: SUKUNA_CONFIRMED
                  Gojo   ≥ 78%  ──► State: GOJO_CONFIRMED
                                  │
                                  ▼
                  Hold Accumulator (12 Frames @ 30 FPS)
                                  │
                                  ▼
                     >> 領域展開 : DOMAIN EXPANSION <<
```

---

## ⚡ Performance Benchmarks

Empirically measured across 60-frame benchmark runs on commodity Linux CPU (MediaPipe 1.0.1 with XNNPACK, OpenCV 5.0):

### Full 1280×720 (720p HD) Resolution

| Pipeline Stage | Before Optimization | After Optimization | Speedup |
| :--- | :--- | :--- | :--- |
| **Color Grading** | 26.19 ms *(float32 multiply)* | **3.36 ms** *(SIMD uint8 & downscaled bloom)* | **7.8x** |
| **2.5D Parallax Background** | 17.19 ms *(3x warpAffine)* | **3.74 ms** *(slice shift & downscaled rays)* | **4.6x** |
| **Camera Motion (Farneback Flow)** | 4.13 ms *(dense flow)* | **0.50 ms** *(decimated 160x90 flow)* | **8.3x** |
| **Segmentation Mask Cleanup** | 3.50 ms *(full-res morphology)* | **0.05 ms** *(320x180 downscaled morphology)* | **70x** |
| **Gesture Classifier** | ~0.15 ms *(ad-hoc heuristics)* | **0.02 ms** *(vectorized invariant geometry)* | **7.5x** |
| **Synchronous Total Frame** | **65.82 ms** (15.2 FPS) | **27.91 ms** (**35.8 FPS**) | **2.36x** |
| **Decoupled Render Loop** | ~7–8 FPS *(blocking I/O)* | **50–60+ FPS** *(thread decoupled)* | **~7x** |

*For complete benchmark distributions, P50/P95/P99 latencies, and 640x360 web stream statistics, see [PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md).*

---

## 🏗️ System Architecture

```
  CAMERA FEED (Webcam or Synthetic Demo)
        │
        ├───► [ RENDER THREAD ] (50–60+ FPS Display Loop)
        │     • Frame display & window event polling
        │     • 2.5D Parallax domain layers (Malevolent Shrine / Infinite Void)
        │     • 3-Layer edge-aware cursed aura compositing
        │     • Cursed particle physics & screen tremor
        │     • Real-time HUD, skeleton overlay, and calibration telemetry
        │
        └───► [ VISION TRACKER THREAD ] (Decoupled Background Inference)
              • Downscaled tracking buffer (320x180 / 640x360)
              • MediaPipe 21-landmark hand detector (Hands 1 & 2)
              • Neural selfie segmentation mask
              • Invariant geometric mudra feature extractor
              • Multi-class temporal state machine
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/gnshx/DomainVision.git
cd DomainVision

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Native Desktop Application

```bash
# Launch with default camera
python main.py

# Launch with calibration overlay active
python main.py --calibrate

# Launch with camera preview mirrored (selfie mode)
python main.py --flip

# Launch animated synthetic demo mode (no webcam needed)
python main.py --demo
```

### 3. Record Video Demonstration

```bash
# Record 180 frames (6 seconds) of Malevolent Shrine demo
python main.py --demo --theme malevolent_shrine --frames 180 --headless --record shrine_demo.mp4

# Record 180 frames of Infinite Void demo
python main.py --demo --theme infinite_void --frames 180 --headless --record void_demo.mp4
```

---

## 🌐 Web Application & Remote SSH

DomainVision provides an integrated HTTP streamer designed for remote development, cloud instances, and SSH sessions:

```bash
python web_app.py --port 8080
```

1. Navigate to **`http://localhost:8080/`** in your browser.
2. Select your viewing mode:
   - **📷 My Local Webcam**: Captures your local webcam and streams frames with real-time AR effects.
   - **🤖 Animated Demo Feed**: Runs the animated character feed with your live webcam in a bottom-right Picture-in-Picture window.
3. Interactive UI buttons allow direct toggling of:
   - Domain Expansion (`⚡ Force Expand`) and Collapse (`🔄 Collapse / Reset`)
   - Theme switching (`🩸 Sukuna` / `🌌 Gojo`)
   - 21-landmark skeleton visualizer (`🦴 Skeleton`)
   - Real-time gesture calibration overlay (`📐 Calibration`)
   - High-resolution frame snapshot download (`📸 Snapshot`)

---

## 📐 Calibration & Debugging

DomainVision includes dedicated diagnostic and calibration tools:

### Live In-App Calibration Overlay
Press **`C`** during runtime (or pass `--calibrate`) to render real-time geometric parameters directly onto the screen:
- **Hands Count**: Confirms whether 1 or 2 hands are recognized.
- **Wrist Proximity**: Displays inter-wrist distance relative to palm scale.
- **Thumb Angle**: Measures upright orientation relative to palm coordinate frame.
- **Index Distance**: Verifies index fingertip contact.
- **Lower Curl**: Displays curl ratio of lower fingers ($> 0.60$ threshold).
- **Index-Middle Cross**: Displays 3D spatial crossing metric ($< 0.40$ threshold).
- **Wrist Height**: Shows hand vertical elevation ($< 0.62$ threshold).

### 21-Landmark Skeleton Overlay
Press **`L`** during runtime to render color-coded 21-joint skeletal landmarks and bone connections (wrist, thumb, index, middle, ring, pinky).

### Dataset Sample Logger
Collect verified calibration samples for analysis or dataset creation:
```bash
python main.py --log-gestures
```
Logs structured JSONL entries to `logs/gesture_samples.jsonl` with timestamps, joint angles, curl ratios, and confidence scores.

### Automated Unit Tests
Validate gesture classifier rules and transitions against synthetic landmark test cases:
```bash
python -m unittest tests/test_gestures.py
```

### Profiler Benchmark Tool
Run high-precision stage latency profiling on your machine:
```bash
# Profile 1280x720 pipeline over 60 frames
python utils/profiler_benchmark.py --frames 60 --width 1280 --height 720

# Profile 640x360 web resolution over 60 frames
python utils/profiler_benchmark.py --frames 60 --width 640 --height 360
```

---

## 🎮 Interactive Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`C`** | **Calibration Overlay** | Toggle live on-screen geometric parameter diagnostics. |
| **`L`** | **Skeleton Overlay** | Toggle 21-landmark joint positions and bone connections. |
| **`M`** | **Mirror Webcam** | Toggle horizontal webcam feed flipping (selfie mode). |
| **`1`** | **Malevolent Shrine** | Force-switch target theme to Ryomen Sukuna. |
| **`2`** | **Infinite Void** | Force-switch target theme to Satoru Gojo. |
| **`Space` / `D`** | **Force Expand** | Manually trigger Domain Expansion animation sequence. |
| **`R`** | **Collapse / Reset** | Dissolve domain barrier and reset state to normal reality. |
| **`H`** | **Toggle HUD** | Show or hide floating telemetry HUD. |
| **`S`** | **Save Screenshot** | Save high-resolution PNG snapshot to disk. |
| **`Q` / `Esc`** | **Quit** | Cleanly terminate background threads and close application. |

---

## 📁 Repository Structure

```
DomainVision/
├── main.py                     # Main application, state machine, and HUD
├── config.py                   # Canonical timelines, color themes, and thresholds
├── web_app.py                  # HTTP streamer with WebRTC/PiP and telemetry headers
├── requirements.txt            # Python dependencies
├── PERFORMANCE_BASELINE.md     # Measured stage benchmarks and latency analysis
│
├── tracking/
│   ├── detector.py             # Asynchronous MediaPipe tracker (hands + segmentation)
│   ├── hand_tracker.py         # Invariant palm coordinate frames & 3D kinematic metrics
│   └── gesture_recognizer.py   # Multi-class state machine (Sukuna / Gojo / Unknown)
│
├── effects/
│   ├── domain_layers.py        # 2.5D parallax background rendering (Shrine / Void)
│   ├── aura.py                 # Edge-aware 3-layer cursed aura compositing
│   ├── cursed_energy.py        # Perspective-anchored joint filaments & fingertip lightning
│   ├── particles.py            # Physics particle system (floating motes, vortex suction)
│   ├── flash.py                # Blinding cursed energy screen flash transition
│   ├── shockwave.py            # Expanding refractive barrier shockwave rings
│   ├── distortion.py           # Optical screen shake and chromatic displacement
│   └── color_grade.py          # SIMD bloom, vignette, and film grain
│
├── utils/
│   ├── profiler_benchmark.py   # Multi-stage latency benchmark tool
│   ├── motion_estimator.py     # Decimated optical flow camera motion tracker
│   ├── quality_controller.py   # Dynamic performance scaler & load balancer
│   ├── demo_feed.py            # State-aware animated character demonstration camera
│   └── fps.py                  # Performance profiler with smoothed metrics
│
├── audio/
│   └── audio_manager.py        # Queued non-blocking audio engine (aplay, pw-play)
│
├── tests/
│   └── test_gestures.py        # Unit test suite for multi-class mudra classification
│
└── assets/
    ├── shrine.png              # Malevolent Shrine backdrop
    ├── void.png                # Infinite Void cosmic singularity backdrop
    ├── models/                 # MediaPipe task models (segmenter, hand_landmarker)
    └── sounds/                 # Canonical 16-bit PCM procedural audio cues
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
