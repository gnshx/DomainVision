# DomainVision Configuration

# Video & Processing Resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

# MediaPipe Async Tracking Resolution (downscaled for high FPS)
TRACK_WIDTH = 640
TRACK_HEIGHT = 360

# Segmentation Optimization
# Run selfie segmentation every N frames and reuse/smooth mask
SEGMENTATION_INTERVAL = 3

# Landmark Smoothing (Exponential Moving Average)
# 0.0 = freeze, 1.0 = raw (no smoothing). 0.35-0.45 provides smooth tracking without lag.
SMOOTHING_ALPHA = 0.40

# Finger Angle Thresholds (degrees)
# > 150 deg: straight/extended finger
# < 105 deg: bent/curled into palm
FINGER_EXTENDED_ANGLE = 150.0
FINGER_CURLED_ANGLE = 105.0

# Gesture Recognition
# Number of consecutive frames the user must hold the canonical sign to activate
SIGN_HOLD_FRAMES_REQUIRED = 12
SIGN_CONFIDENCE_THRESHOLD = 0.78          # Activation threshold
SIGN_DEACTIVATION_THRESHOLD = 0.62        # Hysteresis: lower threshold before resetting hold

# Adaptive Quality Scaling
# Controller upgrades quality when FPS >= PROMOTE_FPS, demotes when FPS < DEMOTE_FPS
QUALITY_PROMOTE_FPS = 45.0
QUALITY_DEMOTE_FPS  = 28.0
QUALITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "ULTRA"]
# Per-quality effect budgets (applied to the rendering pipeline)
# max_particles: cap on active particles; distortion: enable optical remap; aura_scale: downscale factor
QUALITY_BUDGETS = {
    "ULTRA":  {"max_particles": 120, "distortion": True,  "aura_scale": 1.0},
    "HIGH":   {"max_particles": 80,  "distortion": True,  "aura_scale": 0.5},
    "MEDIUM": {"max_particles": 40,  "distortion": False, "aura_scale": 0.35},
    "LOW":    {"max_particles": 20,  "distortion": False, "aura_scale": 0.25},
}

# Frame-Accurate Canonical JJK Audio Timeline (seconds)
AUDIO_TIMELINE = {
    "sign_recognized": 0.00,
    "charge": 0.05,
    "energy_build": 0.70,
    "voice": 1.20,
    "flash": 1.35,
    "shockwave": 1.40,
    "domain_env": 1.50,
    "ambience": 1.50,
}

# Domain Themes Configuration
THEMES = {
    "malevolent_shrine": {
        "id": "malevolent_shrine",
        "name_ja": "伏魔御廚子",
        "name_en": "MALEVOLENT SHRINE",
        "character": "Ryomen Sukuna",
        "primary_bgr": (40, 40, 240),       # Blood Crimson
        "secondary_bgr": (30, 160, 255),    # Cursed Flame Gold
        "aura_inner": (30, 20, 220),
        "aura_outer": (15, 80, 255),
        "void_bg": (12, 8, 24),
        "accent_bg": (20, 15, 80),
    },
    "infinite_void": {
        "id": "infinite_void",
        "name_ja": "無量空処",
        "name_en": "INFINITE VOID",
        "character": "Satoru Gojo",
        "primary_bgr": (255, 60, 180),      # Bright Violet
        "secondary_bgr": (255, 230, 80),    # Celestial Cyan
        "aura_inner": (240, 50, 160),
        "aura_outer": (255, 220, 70),
        "void_bg": (30, 6, 22),
        "accent_bg": (65, 12, 55),
    }
}
DEFAULT_THEME = "malevolent_shrine"
