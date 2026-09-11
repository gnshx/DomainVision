# DomainVision - Jujutsu Kaisen Domain Expansion AR Filter (領域展開)

A real-time, cinematic **Jujutsu Kaisen (JJK)-inspired "Domain Expansion" (領域展開) Augmented Reality filter** built purely in Python with **OpenCV**, **MediaPipe Tasks**, **NumPy**, and **Pillow**.

No heavy game engine required. Transforms standard video or live webcam footage into an immersive cursed energy domain when triggered by physical gesture or hotkey.

---

## Visual Showcase

| Stage | Visual Effect | Description |
| :--- | :--- | :--- |
| **1. Normal** | Camera Feed | Crystal-clear live webcam / video stream. |
| **2. Charging** | Energy Vortex | Swirling particles pull inward toward hands; camera trembles. |
| **3. Flash** | Whiteout & Surge | Multi-stage vignette dip → cursed energy surge → blinding whiteout flash with screen shake. |
| **4. Expansion** | Optical Shockwave | `cv2.remap()` radial refraction wave + concentric glowing barrier rings expanding outward. |
| **5. Domain Active** | Neural Segmentation & Aura | Artificial cosmic/abyss void replaces background while keeping the user in the foreground with a glowing cursed aura, hand energy orbs, and Japanese calligraphy banner. |
| **6. Collapse** | Barrier Dissolve | Energy shatters and dissolves back to reality. |

---

## Supported Domain Themes

Switch themes on the fly by pressing **`1`** or **`2`**:
1. **Infinite Void (無量空処 - Satoru Gojo)**:
   - Deep cosmic indigo/violet void with celestial stars.
   - Concentric glowing barrier astrolabe rings with cyan runic tick marks.
   - Radiant purple & electric blue cursed aura.
   - Japanese typography: **領域展開 無量空処** (`DOMAIN EXPANSION - INFINITE VOID`).
2. **Malevolent Shrine (伏魔御廚子 - Ryomen Sukuna)**:
   - Blood-red abyss with hellish fiery embers.
   - Crimson barrier circles with gold slicing sigils.
   - Intense blood-crimson cursed energy aura.
   - Japanese typography: **領域展開 伏魔御廚子** (`DOMAIN EXPANSION - MALEVOLENT SHRINE`).

---

## Quickstart

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

### 3. Running over SSH (Remote Server)

When working over SSH without a physical webcam or local X11 display:

#### Option A: View MP4 Videos & GIFs Directly in VS Code
We have installed the VS Code video preview extension (`batchnepal.vscode-video-preview`).
- Simply click on [demo_expansion.mp4](file:///home/dlcv/domaincv/demo_expansion.mp4) or [demo_sukuna.mp4](file:///home/dlcv/domaincv/demo_sukuna.mp4) in the file explorer tree to watch the video directly in a tab!
- You can also click on [demo_expansion.gif](file:///home/dlcv/domaincv/demo_expansion.gif) or [demo_sukuna.gif](file:///home/dlcv/domaincv/demo_sukuna.gif) for native image tab playback.

#### Option B: Live Web Stream in Your Browser
Run the built-in streaming server:
```bash
python web_app.py
```
VS Code will automatically forward port 8080. Open `http://localhost:8080/` in your browser to view the **live real-time AR filter stream** with interactive buttons to trigger Domain Expansion, switch themes, and reset!

#### Option C: Record to MP4 Video
```bash
python main.py --demo --theme infinite_void --frames 180 --record output.mp4
```

### 5. Run on a Pre-recorded Video File
```bash
python main.py --video path/to/your_video.mp4 --record output.mp4
```

---

## Interactive Controls

| Key | Action |
| :---: | :--- |
| **`D` / `Space`** | **Trigger Domain Expansion** immediately |
| **`1`** | Switch to **Infinite Void (無量空処)** |
| **`2`** | Switch to **Malevolent Shrine (伏魔御廚子)** |
| **`R`** | **Reset / Collapse** Domain back to Normal |
| **`H`** | Toggle HUD overlay |
| **`S`** | Save high-res screenshot (`domain_screenshot_<timestamp>.png`) |
| **`Q` / `ESC`** | Quit application |

---

## Project Structure

```
domaincv/
├── main.py                     # Main application loop, pipeline orchestration, CLI, HUD
├── requirements.txt            # Python dependencies
├── tracking/
│   ├── detector.py             # MediaPipe Tasks (Selfie Segmentation, Hand & Pose tracking)
│   └── gesture.py              # Temporal gesture recognition (clasped hands / raised arms)
├── effects/
│   ├── domain_background.py    # Procedural cosmic void, rotating barrier astrolabe rings
│   ├── aura.py                 # Segmented body outline, multi-scale dilated Gaussian bloom
│   ├── hand_energy.py          # Pulsing hand orbs & chaotic electric lightning arcs
│   ├── particles.py            # Cursed energy motes (rising embers, inward vortex, explosion)
│   ├── flash.py                # Multi-stage cinematic flash transition
│   ├── shockwave.py            # Visible expanding concentric barrier crests
│   └── distortion.py           # Optical refraction shockwave via cv2.remap() & screen shake
├── utils/
│   ├── text_renderer.py        # Pillow Japanese typography with glow and drop shadow
│   ├── audio.py                # Procedural cursed energy SFX generator (hum, sub-bass, shatter)
│   ├── demo_feed.py            # Synthetic animated character video stream
│   └── blending.py             # Additive, screen, and alpha compositing utilities
└── assets/
    ├── models/                 # MediaPipe task models (selfie_segmenter, hand, pose)
    └── sounds/                 # Pre-synthesized audio effects
```
