import streamlit as st
import cv2
import tempfile
import os
import math
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ────────────────────────── CONFIGURATION ──────────────────────────
DEFAULT_MODEL_PATH = r"C:\Volleyball01\runs\detect\YOLOv8\runs\volleyball_train\weights\best.pt"
DEFAULT_VIDEO_PATH = r"C:\Volleyball01\Input vido.mp4"

# Class mapping from data.yaml: ['1', '2', 'ball', 'player', 'player 1', 'referee']
BALL_CLASS_ID = 2
TEAM_1_CLASS_IDS = [0, 3]       # '1', 'player'
TEAM_2_CLASS_IDS = [1, 4]       # '2', 'player 1'
PLAYER_CLASS_IDS = TEAM_1_CLASS_IDS + TEAM_2_CLASS_IDS
REFEREE_CLASS_ID = 5

# Colors BGR
TEAM_1_COLOR = (255, 100, 0)    # Blue
TEAM_2_COLOR = (0, 0, 255)      # Red
BALL_COLOR   = (0, 255, 0)      # Green

# Default thresholds
DEF_JUMP_THRESH   = 12     # Y-pixel rise to detect jump start
DEF_SPIKE_SPEED   = 5      # ball px/frame speed for spike (lowered for better detection)
DEF_POSSESSION    = 180    # max px distance for ball possession
DEF_SPIKE_WINDOW  = 12     # frames after jump to still count a spike
BALL_MISSING_END  = 30     # frames without ball to end rally
ID_REMAP_DIST     = 120    # px to remap lost player ID

# ────────────────────────── UTILITIES ──────────────────────────

def pdist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def box_center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def box_bottom(b):
    return ((b[0]+b[2])/2, b[3])

def box_height(b):
    return b[3] - b[1]

# ────────────────────────── ID STABILIZER ──────────────────────────

class IDStabilizer:
    """Maps raw ByteTrack IDs to stable sequential IDs."""
    def __init__(self):
        self.r2s = {}           # raw -> stable
        self.s2r = {}           # stable -> raw
        self.next_id = 1
        self.last_pos = {}      # stable -> (x,y)
        self.last_cls = {}      # stable -> cls_id
        # We will keep IDs alive infinitely instead of dropping them.

    def get(self, raw, center, cls_id):
        if raw in self.r2s:
            s = self.r2s[raw]
            self.last_pos[s] = center
            self.last_cls[s] = cls_id
            return s
            
        # Try match lost stable ID of the exact SAME TEAM (cls_id) globally
        # We ignore time buffers and distance limits. If a player was lost,
        # they get their ID back.
        best_s, best_d = None, float('inf')
        for sid, pos in self.last_pos.items():
            if sid in self.s2r and self.s2r[sid] in self.r2s:
                continue  # currently actively tracked
            if self.last_cls.get(sid) != cls_id:
                continue  # wrong team/class
                
            d = pdist(center, pos)
            if d < best_d:
                best_d, best_s = d, sid
                
        if best_s is not None:
            self.r2s[raw] = best_s
            self.s2r[best_s] = raw
            self.last_pos[best_s] = center
            self.last_cls[best_s] = cls_id
            return best_s
            
        # Only assign new ID if team is full/no lost members
        s = self.next_id; self.next_id += 1
        self.r2s[raw] = s; self.s2r[s] = raw
        self.last_pos[s] = center
        self.last_cls[s] = cls_id
        return s

    def cleanup(self, active_raws):
        for r in [r for r in self.r2s if r not in active_raws]:
            del self.r2s[r]

# ────────────────────────── HEATMAP ──────────────────────────

def make_heatmap_fig(positions, w, h, cmap, title):
    from scipy.ndimage import gaussian_filter
    fig, ax = plt.subplots(figsize=(10, 6))
    if not positions:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16, color='gray', transform=ax.transAxes)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
    else:
        xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
        hm, _, _ = np.histogram2d(xs, ys, bins=[50, 30], range=[[0, w], [0, h]])
        hm = gaussian_filter(hm.T, sigma=2.5)
        im = ax.imshow(hm, extent=[0, w, h, 0], cmap=cmap, alpha=0.9, interpolation='bilinear', aspect='auto')
        fig.colorbar(im, ax=ax, label='Density', shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('X (px)', color='white'); ax.set_ylabel('Y (px)', color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#1a1a2e')
    fig.tight_layout()
    return fig

def make_live_heatmap_overlay(positions, w, h, color_bgr):
    """Create a semi-transparent heatmap overlay as an OpenCV image."""
    from scipy.ndimage import gaussian_filter
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if not positions or len(positions) < 10:
        return overlay
    xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
    hm, _, _ = np.histogram2d(xs, ys, bins=[w//20, h//20], range=[[0, w], [0, h]])
    hm = gaussian_filter(hm.T, sigma=3)
    if hm.max() > 0:
        hm = (hm / hm.max() * 255).astype(np.uint8)
    hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    for c in range(3):
        overlay[:, :, c] = (hm_resized * (color_bgr[c] / 255.0)).astype(np.uint8)
    return overlay

def make_combined_heatmap(pos_t1, color_t1, pos_t2, color_t2, w, h):
    """Create a fast cv2 standalone heatmap with both teams on one canvas."""
    from scipy.ndimage import gaussian_filter
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, h), (26, 26, 46), -1) # Dark #1a1a2e background
    
    def process_layer(positions, color):
        layer = np.zeros((h, w, 3), dtype=np.uint8)
        if not positions or len(positions) < 5:
            return layer
        xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
        hm, _, _ = np.histogram2d(xs, ys, bins=[w//15, h//15], range=[[0, w], [0, h]])
        hm = gaussian_filter(hm.T, sigma=2)
        if hm.max() > 0:    
            hm = (hm / hm.max() * 255).astype(np.uint8)
        hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            layer[:, :, c] = (hm_resized * (color[c] / 255.0)).astype(np.uint8)
        return layer

    layer1 = process_layer(pos_t1, color_t1)
    layer2 = process_layer(pos_t2, color_t2)
    
    # Blend layers (preventing overflow by clipping bright overlapping spots)
    blended = np.clip(layer1.astype(np.int16) + layer2.astype(np.int16), 0, 255).astype(np.uint8)
    img = cv2.add(img, blended)
    return img

def draw_stats_panel(frame, players, distances, jumps, spikes):
    """Draws a live stats panel onto the right edge of the video frame."""
    H, W = frame.shape[:2]
    panel_w = 280
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (W - panel_w, 0), (W, H), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Title
    cv2.putText(frame, "LIVE PLAYER STATS", (W - panel_w + 15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(frame, (W - panel_w + 15, 40), (W - 15, 40), (200, 200, 200), 1)
    
    # Player Rows
    y_offset = 70
    # sort players by ID
    players_sorted = sorted(players, key=lambda x: x[0])
    for (sid, pb, cid) in players_sorted:
        team_color = TEAM_1_COLOR if cid in TEAM_1_CLASS_IDS else TEAM_2_COLOR
        
        # Player ID box
        cv2.rectangle(frame, (W - panel_w + 15, y_offset - 15), (W - panel_w + 45, y_offset + 5), team_color, -1)
        cv2.putText(frame, f"P{sid}", (W - panel_w + 20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
        # Stats text
        d = int(distances.get(sid, 0))
        j = jumps.get(sid, 0)
        s = spikes.get(sid, 0)
        stats_txt = f"{d}px | J:{j} | S:{s}"
        cv2.putText(frame, stats_txt, (W - panel_w + 55, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        y_offset += 30
    return frame

def get_zone(x, y, W, H):
    col = int((x / W) * 3)
    row = int((y / H) * 2)
    return row * 3 + col + 1

def draw_court_overlay(frame, players, ball_c):
    H, W = frame.shape[:2]

    court_w, court_h = 500, 250
    x0 = (W - court_w) // 2
    y0 = H - court_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+court_w,y0+court_h), (30,120,30), -1)
    frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

    # Net
    cv2.line(frame, (x0+court_w//2,y0), (x0+court_w//2,y0+court_h), (255,255,255), 2)

    # Attack lines
    cv2.line(frame, (x0+court_w//2-60,y0), (x0+court_w//2-60,y0+court_h), (200,200,200), 1)
    cv2.line(frame, (x0+court_w//2+60,y0), (x0+court_w//2+60,y0+court_h), (200,200,200), 1)

    def map_xy(x,y):
        mx = x0 + int((x/W)*court_w)
        my = y0 + int((y/H)*court_h)
        return mx,my

    # Players
    for sid, pb, cid in players:
        c = box_center(pb)
        mx,my = map_xy(c[0],c[1])
        color = TEAM_1_COLOR if cid in TEAM_1_CLASS_IDS else TEAM_2_COLOR
        cv2.circle(frame, (mx,my), 7, color, -1)

    # Ball
    if ball_c:
        mx,my = map_xy(ball_c[0], ball_c[1])
        cv2.circle(frame, (mx,my), 5, (0,255,0), -1)

    return frame

def draw_scoreboard(frame, t1_jumps, t1_spikes, t2_jumps, t2_spikes, rally_n):
    H, W = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (W,60), (20,20,20), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    cv2.putText(frame, f"RALLY {rally_n}", (W//2 - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"T1 J:{t1_jumps} S:{t1_spikes}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 2)

    cv2.putText(frame, f"T2 J:{t2_jumps} S:{t2_spikes}", (W-250,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return frame

def draw_event(frame, text):
    H, W = frame.shape[:2]
    # Small black background for event popup
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(frame, (W//2 - 100 - 10, H//2 - th - 10), (W//2 - 100 + tw + 10, H//2 + 10), (0,0,0), -1)
    cv2.putText(frame, text, (W//2 - 100, H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
    return frame

# ────────────────────────── STREAMLIT UI ──────────────────────────

st.set_page_config(page_title="🏐 Volleyball Analytics", layout="wide")
st.title("🏐 Volleyball Analytics Dashboard")
st.markdown("### AI-Powered Match Analysis — YOLOv8 + ByteTrack")

# Sidebar
st.sidebar.header("🔧 Settings")
model_path = st.sidebar.text_input("Model Path", DEFAULT_MODEL_PATH)
conf_thresh = st.sidebar.slider("Confidence", 0.01, 1.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Thresholds")
jump_thresh = st.sidebar.slider("Jump Threshold (px)", 5, 40, DEF_JUMP_THRESH,
    help="Min Y-pixel rise to count as a jump")
spike_speed = st.sidebar.slider("Spike Ball Speed (px/frame)", 2, 30, DEF_SPIKE_SPEED,
    help="Min ball speed in px/frame for spike")
spike_window = st.sidebar.slider("Spike Window (frames)", 2, 20, DEF_SPIKE_WINDOW,
    help="Frames after jump start to count spike")
poss_dist = st.sidebar.slider("Possession Distance (px)", 50, 400, DEF_POSSESSION)
show_heatmap_live = st.sidebar.checkbox("Show Live Heatmap Overlay", value=True)
show_in_video_overlays = st.sidebar.checkbox("Show Minimap & Stats Overlays", value=True)

st.sidebar.markdown("---")
st.sidebar.info("**Tracked:** Movement, Jumps, Spikes, Possession, Ball Speed, Rallies, Heatmaps")

uploaded = st.file_uploader("Upload volleyball video", type=["mp4", "avi", "mov"])

# ────────────────────────── MAIN ──────────────────────────

if uploaded or st.button("▶️ Use Sample Video"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    if uploaded:
        tfile.write(uploaded.read()); video_path = tfile.name
    else:
        video_path = DEFAULT_VIDEO_PATH

    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}"); st.stop()
    st.success("✅ Model + ByteTrack loaded")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video"); st.stop()

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    net_x = W // 2

    st.info(f"🎞️ {total} frames @ {fps}fps | {W}x{H}")

    # Set up VideoWriter to save the stream
    output_video_path = "processed_volleyball.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    # Layout
    st.markdown("## 🎥 Match Analysis")
    frame_ph = st.empty()
    
    st.markdown("## 📊 Key Insights")
    insight_box = st.empty()
    
    st.markdown("### 🧾 Event Timeline")
    event_box = st.empty()
    
    stop_btn = st.sidebar.button("⏹️ Stop")
    show_t1 = st.sidebar.checkbox("Show Team 1 Boxes", True)
    show_t2 = st.sidebar.checkbox("Show Team 2 Boxes", True)

    # ── State ──
    pos_history   = defaultdict(list)    # stable_id -> [(x,y), ...]
    bot_history   = defaultdict(list)    # stable_id -> [y_bottom, ...]
    height_history = defaultdict(list)  # stable_id -> [box_height, ...]
    distances     = defaultdict(float)
    jumps         = defaultdict(int)
    spikes        = defaultdict(int)
    teams         = {}                   # stable_id -> "Team 1"/"Team 2"
    is_jumping    = {}                   # stable_id -> bool
    jump_frame    = {}                   # stable_id -> frame when jump started
    poss_frames   = defaultdict(int)
    spike_cooldown = defaultdict(int)    # stable_id -> last spike frame (debounce)
    blocks        = defaultdict(int)     # blocks counter
    
    ball_trail = []
    MAX_TRAIL = 20
    event_log = []
    active_event = ""
    event_timer = 0

    t1_positions = []; t2_positions = []

    prev_ball = None; ball_speed = 0.0; ball_speeds = []
    rally_n = 0; rally_exch = 0; ball_side = None
    no_ball = 0; rally_on = False; rally_hist = []

    t1_jumps = 0; t1_spikes = 0; t2_jumps = 0; t2_spikes = 0
    t1_total_dist = 0.0; t2_total_dist = 0.0

    stabilizer = IDStabilizer()
    fcount = 0

    # ── LOOP ──
    while cap.isOpened():
        if stop_btn: st.warning("Stopped."); break
        ok, frame = cap.read()
        if not ok: break
        fcount += 1

        results = model.track(source=frame, conf=conf_thresh, persist=True,
                              tracker=r"C:\Volleyball01\volleyball_bytetrack.yaml", verbose=False)

        ball_box = None; ball_c = None
        players = []; refs = []
        t1_now = 0; t2_now = 0

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cid = int(box.cls[0].item())
                xy = box.xyxy[0].cpu().numpy()
                rid = int(box.id[0].item()) if box.id is not None else None

                if cid == BALL_CLASS_ID:
                    ball_box = xy; ball_c = box_center(xy)
                elif cid in PLAYER_CLASS_IDS and rid is not None:
                    c = box_center(xy)
                    sid = stabilizer.get(rid, c, cid)
                    players.append((sid, xy, cid))
                elif cid == REFEREE_CLASS_ID:
                    refs.append(xy)

        # Cleanup
        actives = set()
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is not None:
                    actives.add(int(box.id[0].item()))
        stabilizer.cleanup(actives)

        # ── Ball analytics ──
        ball_frame_speed = 0.0
        if ball_c is not None:
            ball_trail.append(ball_c)
            if len(ball_trail) > MAX_TRAIL:
                ball_trail.pop(0)
                
            no_ball = 0
            if prev_ball is not None:
                ball_frame_speed = pdist(prev_ball, ball_c)
                ball_speed = ball_frame_speed * fps
                ball_speeds.append(ball_speed)
            prev_ball = ball_c

            # Rally
            side = 'L' if ball_c[0] < net_x else 'R'
            if not rally_on:
                rally_on = True; rally_n += 1; rally_exch = 0; ball_side = side
                event_log.append(f"Frame {fcount}: SERVE detected")
                active_event = "SERVE"
                event_timer = 20
            if ball_side and side != ball_side:
                rally_exch += 1; ball_side = side
        else:
            no_ball += 1
            if rally_on and no_ball > BALL_MISSING_END:
                rally_on = False
                rally_hist.append(rally_exch)
                rally_exch = 0; ball_side = None

        # ── Player analytics ──
        poss_id = None; min_bd = float('inf')

        for (sid, pb, cid) in players:
            c = box_center(pb)
            bot = box_bottom(pb)
            bh = box_height(pb)

            t1 = cid in TEAM_1_CLASS_IDS
            t2 = cid in TEAM_2_CLASS_IDS
            teams[sid] = "Team 1" if t1 else "Team 2"

            pos_history[sid].append(c)
            bot_history[sid].append(bot[1])
            height_history[sid].append(bh)

            # Heatmap
            if t1: t1_positions.append(c)
            else:  t2_positions.append(c)

            # Distance
            ph = pos_history[sid]
            if len(ph) >= 2:
                d = pdist(ph[-2], ph[-1])
                distances[sid] += d
                if t1: t1_total_dist += d
                else:  t2_total_dist += d

            # Jump detection — using box height change + bottom Y rise
            bots = bot_history[sid]
            heights = height_history[sid]
            was_jumping = is_jumping.get(sid, False)
            now_jumping = False

            if len(bots) >= 3:
                # Compare with average of last 3 frames vs current
                recent_bot = np.mean(bots[-4:-1]) if len(bots) >= 4 else bots[-2]
                y_rise = recent_bot - bots[-1]  # positive = going up

                # Also check if box got taller (player stretching up)
                if len(heights) >= 3:
                    recent_h = np.mean(heights[-4:-1]) if len(heights) >= 4 else heights[-2]
                    h_growth = heights[-1] - recent_h  # positive = taller
                else:
                    h_growth = 0

                # Jump if feet lifted significantly OR player got significantly taller
                if y_rise > jump_thresh or (y_rise > jump_thresh * 0.6 and h_growth > 5):
                    now_jumping = True

            if now_jumping and not was_jumping:
                # New jump started
                jumps[sid] += 1
                jump_frame[sid] = fcount
                if t1: t1_jumps += 1
                else:  t2_jumps += 1

            is_jumping[sid] = now_jumping

            # Smart Possession
            if ball_c is not None:
                d = pdist(c, ball_c)
                if d < poss_dist and ball_frame_speed < 15 and d < min_bd:
                    min_bd = d; poss_id = sid

            # Spike detection — player jumped recently + ball moving fast + near player
            if ball_c is not None and ball_frame_speed > spike_speed:
                d_ball = pdist(c, ball_c)
                jumped_recently = (fcount - jump_frame.get(sid, -999)) <= spike_window
                not_cooled = (fcount - spike_cooldown.get(sid, -999)) > spike_window * 2
                if jumped_recently and d_ball < poss_dist and not_cooled:
                    spikes[sid] += 1
                    spike_cooldown[sid] = fcount
                    
                    # Log Spike Event
                    event_log.append(f"Frame {fcount}: Player {sid} SPIKE")
                    active_event = "SPIKE!"
                    event_timer = 20

                    # Draw Arrow (will be visible next frame)
                    px, py = int(c[0]), int(c[1])
                    bx, by = int(ball_c[0]), int(ball_c[1])
                    cv2.arrowedLine(frame, (px, py), (bx, by), (0, 255, 255), 4)

                    if t1: t1_spikes += 1
                    else:  t2_spikes += 1
                    
            # Block Detection
            if is_jumping.get(sid, False) and abs(c[0] - net_x) < 80:
                nearby_jumpers = 0
                for (sid2, pb2, cid2) in players:
                    if sid2 != sid:
                        c2 = box_center(pb2)
                        if abs(c2[0] - net_x) < 80 and is_jumping.get(sid2, False):
                            nearby_jumpers += 1
                if nearby_jumpers >= 1:
                    blocks[sid] += 1

        if poss_id is not None:
            poss_frames[poss_id] += 1

        # ── Draw ──
        # Live heatmap overlay
        if show_heatmap_live and fcount > 30:
            hm1 = make_live_heatmap_overlay(t1_positions[-2000:], W, H, (255, 100, 0))
            hm2 = make_live_heatmap_overlay(t2_positions[-2000:], W, H, (50, 50, 255))
            frame = cv2.addWeighted(frame, 1.0, hm1, 0.3, 0)
            frame = cv2.addWeighted(frame, 1.0, hm2, 0.3, 0)

        # Players
        for (sid, pb, cid) in players:
            t1 = cid in TEAM_1_CLASS_IDS
            t2 = cid in TEAM_2_CLASS_IDS
            if t1 and not show_t1: continue
            if t2 and not show_t2: continue

            tn = "T1" if t1 else "T2"
            color = TEAM_1_COLOR if t1 else TEAM_2_COLOR
            if t1: t1_now += 1
            else:  t2_now += 1

            # Action label
            jmp = is_jumping.get(sid, False)
            recently_spiked = (fcount - spike_cooldown.get(sid, -999)) <= 10
            if recently_spiked:
                action = "SPIKE!"
            elif jmp:
                action = "JUMP"
            elif poss_id == sid:
                action = "BALL"
            else:
                action = ""

            x1, y1, x2, y2 = map(int, pb)
            thick = 3 if (poss_id == sid or jmp or recently_spiked) else 2
            if recently_spiked:
                color = (0, 255, 255)  # Yellow highlight for spike
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

            # Label
            lbl = f"{tn} P{sid}"
            if action: lbl += f" {action}"
            lbl += f" J:{jumps.get(sid,0)} S:{spikes.get(sid,0)}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, lbl, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # Distance below
            cv2.putText(frame, f"{distances.get(sid,0):.0f}px", (x1, y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Referee
        for rb in refs:
            x1,y1,x2,y2 = map(int, rb)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (180,180,180), 1)
            cv2.putText(frame, "Ref", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

        # Ball Trail
        for i in range(1, len(ball_trail)):
            if ball_trail[i-1] and ball_trail[i]:
                cv2.line(frame,
                         (int(ball_trail[i-1][0]), int(ball_trail[i-1][1])),
                         (int(ball_trail[i][0]), int(ball_trail[i][1])),
                         (0, 255, 255), 4)

        # Ball
        if ball_box is not None:
            bx1,by1,bx2,by2 = map(int, ball_box)
            bcx, bcy = (bx1+bx2)//2, (by1+by2)//2
            cv2.circle(frame, (bcx,bcy), 10, BALL_COLOR, -1)
            cv2.circle(frame, (bcx,bcy), 12, (255,255,255), 2)
            cv2.putText(frame, f"BALL {ball_speed:.0f}px/s", (bx1, by1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)

        # Top Scoreboard Overlay
        frame = draw_scoreboard(frame, t1_jumps, t1_spikes, t2_jumps, t2_spikes, rally_n)
                    
        # In-Video Minimap & Stats Panel Overlays
        if show_in_video_overlays:
            frame = draw_court_overlay(frame, players, ball_c)
            frame = draw_stats_panel(frame, players, distances, jumps, spikes)
            
        # Draw Active Event Popup (Timer)
        if event_timer > 0 and active_event:
            frame = draw_event(frame, active_event)
            event_timer -= 1

        # ── Dashboard Markdown UI ──
        poss_status = f"{teams.get(poss_id, '?')} P{poss_id}" if poss_id else "—"
        insight_box.markdown(f"""
### 🔍 Live Insights
* **Current Rally:** {rally_n} (Exchanges: {rally_exch})
* **Ball Speed:** {ball_speed:.0f} px/s
* **Possession:** {poss_status}
* **Team 1 Total Distance:** {t1_total_dist:.0f}px
* **Team 2 Total Distance:** {t2_total_dist:.0f}px
        """)
        
        # Format the event log for the timeline
        event_box.markdown("```text\n" + "\n".join(event_log[-20:]) + "\n```")

        # Draw video frame
        out.write(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_ph.image(frame_rgb, width='stretch')

    # ── END ──
    cap.release()
    out.release()
    if rally_on: rally_hist.append(rally_exch)

    st.balloons()
    st.success("✅ Processing complete!")
    
    with open(output_video_path, "rb") as video_file:
        video_bytes = video_file.read()
    st.download_button(
        label="📥 Download Processed Video",
        data=video_bytes,
        file_name="processed_match.mp4",
        mime="video/mp4"
    )

    # ════════════════════ FINAL MATCH REPORT ════════════════════

    st.markdown("---")
    st.markdown("## 📈 Final Match Report")

    # ── Team Summary ──
    st.markdown("### 🏆 Team Comparison")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### 🔵 Team 1")
        st.metric("Total Jumps", t1_jumps)
        st.metric("Total Spikes", t1_spikes)
        st.metric("Total Distance", f"{t1_total_dist:.0f} px")
        t1_poss = sum(poss_frames[s] for s in teams if teams[s] == "Team 1")
        st.metric("Possession Time", f"{t1_poss/fps:.1f}s")
    with tc2:
        st.markdown("#### 🔴 Team 2")
        st.metric("Total Jumps", t2_jumps)
        st.metric("Total Spikes", t2_spikes)
        st.metric("Total Distance", f"{t2_total_dist:.0f} px")
        t2_poss = sum(poss_frames[s] for s in teams if teams[s] == "Team 2")
        st.metric("Possession Time", f"{t2_poss/fps:.1f}s")

    # ── Individual Player Stats ──
    st.markdown("### 🏃 Individual Player Statistics")

    all_ids = sorted(teams.keys())
    t1_rows = []; t2_rows = []
    for sid in all_ids:
        team = teams[sid]
        row = {
            "Player": f"P{sid}",
            "Distance (px)": f"{distances.get(sid,0):.0f}",
            "Jumps": jumps.get(sid, 0),
            "Spikes": spikes.get(sid, 0),
            "Possession (s)": f"{poss_frames.get(sid,0)/fps:.1f}",
            "Frames Seen": len(pos_history.get(sid, [])),
        }
        if team == "Team 1": t1_rows.append(row)
        else: t2_rows.append(row)

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("#### 🔵 Team 1 Players")
        if t1_rows: st.dataframe(t1_rows, use_container_width=True)
        else: st.write("No Team 1 data")
    with pc2:
        st.markdown("#### 🔴 Team 2 Players")
        if t2_rows: st.dataframe(t2_rows, use_container_width=True)
        else: st.write("No Team 2 data")

    # ── Top performers ──
    if all_ids:
        st.markdown("### 🌟 Top Performers")
        tp1, tp2, tp3, tp4 = st.columns(4)
        if distances:
            top_dist = max(distances, key=distances.get)
            tp1.metric("Most Distance", f"P{top_dist} ({teams.get(top_dist,'')})",
                       f"{distances[top_dist]:.0f} px")
        if jumps:
            top_jump = max(jumps, key=jumps.get)
            tp2.metric("Most Jumps", f"P{top_jump} ({teams.get(top_jump,'')})",
                       f"{jumps[top_jump]}")
        if spikes:
            top_spike = max(spikes, key=spikes.get)
            tp3.metric("Most Spikes", f"P{top_spike} ({teams.get(top_spike,'')})",
                       f"{spikes[top_spike]}")
        if poss_frames:
            top_poss = max(poss_frames, key=poss_frames.get)
            tp4.metric("Most Possession", f"P{top_poss} ({teams.get(top_poss,'')})",
                       f"{poss_frames[top_poss]/fps:.1f}s")

    # ── Rally Summary ──
    st.markdown("### 🏁 Rally Summary")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Total Rallies", rally_n)
    rc2.metric("Avg Exchanges", f"{np.mean(rally_hist):.1f}" if rally_hist else "0")
    rc3.metric("Longest Rally", f"{max(rally_hist)} exch" if rally_hist else "0")

    if rally_hist:
        fig_r, ax_r = plt.subplots(figsize=(10, 3))
        ax_r.bar(range(1, len(rally_hist)+1), rally_hist, color='#4ecdc4', edgecolor='#2c3e50')
        ax_r.set_xlabel("Rally #"); ax_r.set_ylabel("Exchanges")
        ax_r.set_title("Net Crossings per Rally", fontweight='bold')
        fig_r.patch.set_facecolor('#f0f2f6'); fig_r.tight_layout()
        st.pyplot(fig_r); plt.close(fig_r)

    # ── Ball Speed ──
    if ball_speeds:
        st.markdown("### 🏐 Ball Speed Over Time")
        fig_s, ax_s = plt.subplots(figsize=(10, 3))
        step = max(1, len(ball_speeds)//500)
        sampled = ball_speeds[::step]
        ax_s.plot(sampled, color='#27ae60', lw=1, alpha=0.8)
        ax_s.fill_between(range(len(sampled)), sampled, alpha=0.2, color='#2ecc71')
        ax_s.set_xlabel("Frame"); ax_s.set_ylabel("Speed (px/s)")
        ax_s.set_title("Ball Speed", fontweight='bold')
        fig_s.patch.set_facecolor('#f0f2f6'); fig_s.tight_layout()
        st.pyplot(fig_s); plt.close(fig_s)

    # ── Heatmaps ──
    st.markdown("### 🔥 Player Position Heatmaps (Full Match)")
    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown("#### 🔵 Team 1")
        fig1 = make_heatmap_fig(t1_positions, W, H, 'Blues', 'Team 1 Positions')
        st.pyplot(fig1); plt.close(fig1)
    with hc2:
        st.markdown("#### 🔴 Team 2")
        fig2 = make_heatmap_fig(t2_positions, W, H, 'Reds', 'Team 2 Positions')
        st.pyplot(fig2); plt.close(fig2)

    # Combined heatmap
    st.markdown("#### 🔥 Combined Heatmap (Both Teams)")
    fig_comb = make_heatmap_fig(t1_positions + t2_positions, W, H, 'hot', 'All Players')
    st.pyplot(fig_comb); plt.close(fig_comb)

else:
    st.info("💡 Upload a video or click **'▶️ Use Sample Video'** to start.")
