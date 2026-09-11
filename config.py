# DomainVision Configuration

# Video & Processing Resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

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
SIGN_CONFIDENCE_THRESHOLD = 0.82

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
