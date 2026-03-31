"""
Centralized configuration for the Volleyball Analytics System.
All constants, thresholds, color definitions, and class mappings.
"""

# ────────────────────────── MODEL & VIDEO ──────────────────────────
DEFAULT_MODEL_PATH = r"C:\Volleyball01\runs\detect\YOLOv8\runs\volleyball_train\weights\best.pt"
DEFAULT_VIDEO_PATH = r"C:\Volleyball01\Input vido.mp4"
TRACKER_CONFIG = r"C:\Volleyball01\volleyball_bytetrack.yaml"
OUTPUT_VIDEO_PATH = "processed_volleyball.mp4"

# ────────────────────────── CLASS MAPPING ──────────────────────────
# From data.yaml: ['1', '2', 'ball', 'player', 'player 1', 'referee']
BALL_CLASS_ID = 2
TEAM_1_CLASS_IDS = [0, 3]       # '1', 'player'
TEAM_2_CLASS_IDS = [1, 4]       # '2', 'player 1'
PLAYER_CLASS_IDS = TEAM_1_CLASS_IDS + TEAM_2_CLASS_IDS
REFEREE_CLASS_ID = 5

CLASS_NAMES = {
    0: "Team 1", 1: "Team 2", 2: "Ball",
    3: "Player", 4: "Player 1", 5: "Referee"
}

# ────────────────────────── COLORS (BGR) ──────────────────────────
TEAM_1_COLOR = (255, 140, 40)       # Warm blue-orange
TEAM_2_COLOR = (50, 50, 255)        # Red
TEAM_1_COLOR_LIGHT = (255, 180, 100)
TEAM_2_COLOR_LIGHT = (120, 120, 255)
BALL_COLOR = (0, 255, 0)            # Green
BALL_TRAIL_COLOR = (0, 255, 255)    # Yellow
REF_COLOR = (180, 180, 180)         # Gray
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BG = (20, 20, 30)

# Action badge colors (BGR)
ACTION_COLORS = {
    "JUMP": (0, 220, 255),      # Yellow
    "SPIKE": (255, 255, 0),     # Cyan
    "BLOCK": (0, 165, 255),     # Orange
    "BALL": (0, 255, 0),        # Green
    "SERVE": (255, 100, 255),   # Pink
    "RECV": (255, 180, 100),    # Light Blue
    "SET": (100, 255, 100),     # Lime
}

# ────────────────────────── DETECTION THRESHOLDS ──────────────────────────
DEF_CONF_THRESH = 0.25
DEF_JUMP_THRESH = 25       # Increased to 25 for better precision
DEF_SPIKE_SPEED = 5        # Ball px/frame speed for spike
DEF_POSSESSION = 140       # Decreased to 140 to be stricter about touches
DEF_SPIKE_WINDOW = 12      # Frames after jump to still count a spike
DEF_BLOCK_COOLDOWN = 20    # Frames between block counts per player
DEF_EVENT_COOLDOWN = 15    # Minimum frames between same-type events

BALL_MISSING_END = 45      # Frames without ball to end rally
BALL_COAST_FRAMES = 15     # Max frames to predict ball position when missing
ID_REMAP_DIST = 120        # Px to remap lost player ID
MAX_TRAIL = 40             # Ball trail buffer size
MAX_EVENT_LOG = 200        # Maximum event log entries

# ────────────────────────── PERFORMANCE ──────────────────────────
SKIP_FRAMES_OUTSIDE_RALLY = 2  # Process every Nth frame when no rally
SKIP_FRAMES_DURING_RALLY = 1   # Process every frame during rally

# ────────────────────────── VOLLEYBALL ZONES ──────────────────────────
# Standard volleyball court zones 1-6
# Zone layout (from player's perspective facing net):
#   4 | 3 | 2   (front row - near net)
#   5 | 6 | 1   (back row - near baseline)
FRONT_ROW_ZONES = [2, 3, 4]
BACK_ROW_ZONES = [1, 5, 6]

# ────────────────────────── GAMEPLAY STATES ──────────────────────────
class RallyState:
    IDLE = "IDLE"
    SERVE = "SERVE"
    RECV = "RECV"
    SET = "SET"
    SPIKE = "SPIKE"
    BLOCK = "BLOCK"
    RALLY_END = "RALLY_END"

    FLOW = [IDLE, SERVE, RECV, SET, SPIKE, RALLY_END]
