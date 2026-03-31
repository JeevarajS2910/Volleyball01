"""
Drawing Module — All OpenCV video overlay functions.
Clean bounding boxes, ball circles, scoreboard, court overlay, stats panel, events.
"""
import cv2
import numpy as np
from core.config import (
    TEAM_1_COLOR, TEAM_2_COLOR, TEAM_1_COLOR_LIGHT, TEAM_2_COLOR_LIGHT,
    BALL_COLOR, BALL_TRAIL_COLOR, REF_COLOR, WHITE, BLACK, DARK_BG,
    ACTION_COLORS, TEAM_1_CLASS_IDS, TEAM_2_CLASS_IDS
)
from core.utils import box_center, get_tactical_zone


def draw_player_bbox(frame, sid, bbox, cid, action, role, distance, is_poss, gameplay_label=""):
    """
    Draw a clean bounding box for a player with label bar and prominent action badge.
    gameplay_label: SET/PASS/BLOCK/SPIKE/SERVE from gameplay engine (shown as large badge).
    """
    is_t1 = cid in TEAM_1_CLASS_IDS
    tn = "T1" if is_t1 else "T2"
    base_color = TEAM_1_COLOR if is_t1 else TEAM_2_COLOR
    light_color = TEAM_1_COLOR_LIGHT if is_t1 else TEAM_2_COLOR_LIGHT

    x1, y1, x2, y2 = map(int, bbox)
    box_w = x2 - x1
    box_h = y2 - y1

    # Choose the best label to display (gameplay label takes priority)
    display_action = gameplay_label if gameplay_label else action

    # Bounding box thickness and color based on state
    thickness = 2
    box_color = base_color
    if display_action in ("SPIKE", "BLOCK", "SET", "RECV", "SERVE"):
        thickness = 3
        box_color = ACTION_COLORS.get(display_action, base_color)
    elif display_action == "JUMP":
        thickness = 3
        box_color = ACTION_COLORS.get("JUMP", base_color)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    # Draw corner accents (broadcast style)
    corner_len = min(15, box_w // 4, box_h // 4)
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + corner_len * dx, cy), WHITE, 2)
        cv2.line(frame, (cx, cy), (cx, cy + corner_len * dy), WHITE, 2)

    # Label bar (team + player ID + role)
    lbl = f"{tn} P{sid}"
    if role:
        lbl += f" [{role}]"

    (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_h = th + baseline + 8
    label_w = max(tw + 10, box_w)

    cv2.rectangle(frame, (x1, y1 - label_h), (x1 + label_w, y1), base_color, -1)
    cv2.putText(frame, lbl, (x1 + 4, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # ── LARGE ACTION BADGE (SPIKE / BLOCK / SET / PASS / SERVE) ──
    # Shown as a prominent colored badge ABOVE the label bar
    if display_action and display_action in ("SPIKE", "BLOCK", "SET", "PASS", "SERVE"):
        badge_color = ACTION_COLORS.get(display_action, (0, 255, 255))
        badge_font_scale = 0.65
        badge_thickness = 2
        (bw, bh), _ = cv2.getTextSize(display_action, cv2.FONT_HERSHEY_SIMPLEX, badge_font_scale, badge_thickness)

        # Center the badge above the bounding box
        badge_x = x1 + (box_w - bw) // 2 - 6
        badge_y = y1 - label_h - bh - 16

        # Badge background pill
        cv2.rectangle(frame, (badge_x - 6, badge_y - 4), (badge_x + bw + 6, badge_y + bh + 8),
                      badge_color, -1)
        cv2.rectangle(frame, (badge_x - 6, badge_y - 4), (badge_x + bw + 6, badge_y + bh + 8),
                      WHITE, 1)
        # Badge text
        cv2.putText(frame, display_action, (badge_x, badge_y + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, badge_font_scale, BLACK, badge_thickness, cv2.LINE_AA)

    elif display_action == "JUMP":
        # Smaller badge for JUMP
        badge_color = ACTION_COLORS.get("JUMP", (0, 220, 255))
        (bw, bh), _ = cv2.getTextSize("JUMP", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_x = x1 + (box_w - bw) // 2 - 4
        badge_y = y1 - label_h - bh - 12
        cv2.rectangle(frame, (badge_x - 4, badge_y - 2), (badge_x + bw + 4, badge_y + bh + 6),
                      badge_color, -1)
        cv2.putText(frame, "JUMP", (badge_x, badge_y + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    return frame




def draw_referee(frame, bbox):
    """Draw referee with simple labeled box."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), REF_COLOR, 1)
    cv2.putText(frame, "REF", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, REF_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_ball(frame, ball_box, speed):
    """Draw ball as circle with speed label."""
    if ball_box is None:
        return frame
    bx1, by1, bx2, by2 = map(int, ball_box)
    bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
    radius = max(8, (bx2 - bx1) // 2)

    # Outer glow
    cv2.circle(frame, (bcx, bcy), radius + 4, (0, 200, 0), 2, cv2.LINE_AA)
    # Main circle
    cv2.circle(frame, (bcx, bcy), radius, BALL_COLOR, -1, cv2.LINE_AA)
    # White ring
    cv2.circle(frame, (bcx, bcy), radius + 1, WHITE, 2, cv2.LINE_AA)

    # Speed label
    speed_txt = f"{speed:.0f} px/s"
    cv2.putText(frame, speed_txt, (bx1, by1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BALL_COLOR, 2, cv2.LINE_AA)
    return frame


def draw_ball_trail(frame, trail_points):
    """Draw ball trajectory trail with fading thickness."""
    pts = trail_points
    n = len(pts)
    for i in range(1, n):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # Fade: thicker and brighter toward recent
        alpha = i / n
        thickness = max(1, int(alpha * 4))
        color_intensity = int(alpha * 255)
        color = (0, color_intensity, 255)  # yellow fading
        cv2.line(frame,
                 (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])),
                 color, thickness, cv2.LINE_AA)
    return frame


def draw_pass_arrow(frame, arrow, timer):
    """Draw pass arrow between players."""
    if timer <= 0 or arrow is None:
        return frame
    p1, p2 = arrow
    # White with team color tip
    cv2.arrowedLine(frame,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    WHITE, 3, tipLength=0.08, line_type=cv2.LINE_AA)
    return frame


def draw_scoreboard(frame, t1_score, t2_score, t1_spikes, t2_spikes,
                    rally_n, match_time_str, possession_pct, state):
    """
    Professional broadcast-style scoreboard overlay.
    
    ┌─────────────────────────────────────────────────────────────┐
    │ ⏱ 02:34 │ TEAM 1 [3] ━━ [2] TEAM 2 │ Rally 5 │ STATE │
    └─────────────────────────────────────────────────────────────┘
    """
    H, W = frame.shape[:2]
    bar_h = 55

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (15, 15, 25), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Bottom accent line
    cv2.line(frame, (0, bar_h), (W, bar_h), (60, 60, 80), 2)

    y_text = 35

    # Time
    cv2.putText(frame, match_time_str, (15, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 1, cv2.LINE_AA)

    # Team 1 section
    t1_label = "TEAM 1"
    cv2.putText(frame, t1_label, (120, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAM_1_COLOR_LIGHT, 2, cv2.LINE_AA)

    # Score
    score_x = W // 2 - 40
    cv2.putText(frame, "VS", (score_x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    # Team 2 section
    t2_label = "TEAM 2"
    cv2.putText(frame, t2_label, (score_x + 60, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAM_2_COLOR_LIGHT, 2, cv2.LINE_AA)

    # Rally + State on right
    rally_txt = f"Rally {rally_n}"
    cv2.putText(frame, rally_txt, (W - 250, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 200), 1, cv2.LINE_AA)

    # State badge
    if state and state != "IDLE":
        state_color = ACTION_COLORS.get(state, (120, 200, 120))
        (sw, sh), _ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        sx = W - 100
        cv2.rectangle(frame, (sx - 5, 12), (sx + sw + 5, 12 + sh + 8), state_color, -1)
        cv2.putText(frame, state, (sx, 12 + sh + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)

    # Spike counts as sub-info
    info_y = 50
    cv2.putText(frame, f"S:{t1_spikes}", (120, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEAM_1_COLOR_LIGHT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"S:{t2_spikes}", (score_x + 60, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEAM_2_COLOR_LIGHT, 1, cv2.LINE_AA)

    return frame


def draw_win_prob_bar(frame, p1, p2):
    """Draw win probability bar below scoreboard."""
    H, W = frame.shape[:2]
    bar_w, bar_h = min(400, W - 100), 14
    x0 = (W - bar_w) // 2
    y0 = 62

    if p1 + p2 == 0:
        p1, p2 = 50, 50
    w_p1 = int((p1 / (p1 + p2)) * bar_w)

    # Background
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + bar_w + 1, y0 + bar_h + 1), (60, 60, 80), -1)
    # Team bars
    cv2.rectangle(frame, (x0, y0), (x0 + w_p1, y0 + bar_h), TEAM_1_COLOR, -1)
    cv2.rectangle(frame, (x0 + w_p1, y0), (x0 + bar_w, y0 + bar_h), TEAM_2_COLOR, -1)

    # Labels
    cv2.putText(frame, f"{int(p1)}%", (x0 - 35, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEAM_1_COLOR_LIGHT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(p2)}%", (x0 + bar_w + 5, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEAM_2_COLOR_LIGHT, 1, cv2.LINE_AA)

    return frame


def draw_event_popup(frame, text):
    """Draw event popup in center of frame."""
    if not text:
        return frame
    H, W = frame.shape[:2]

    event_color = ACTION_COLORS.get(text, (0, 255, 255))

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)
    cx = W // 2 - tw // 2
    cy = H // 2

    # Dark background pill
    pad = 20
    cv2.rectangle(frame, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad),
                  (10, 10, 10), -1)
    cv2.rectangle(frame, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad),
                  event_color, 2)

    # Text
    cv2.putText(frame, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, event_color, 3, cv2.LINE_AA)

    return frame


def draw_stats_panel(frame, players, distances, jumps, spikes, blocks):
    """Draws a live stats panel on the right edge of the video frame."""
    H, W = frame.shape[:2]
    panel_w = 260

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (W - panel_w, 0), (W, H), (10, 10, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Title
    cv2.putText(frame, "PLAYER STATS", (W - panel_w + 15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
    cv2.line(frame, (W - panel_w + 15, 98), (W - 15, 98), (60, 60, 80), 1)

    # Column headers
    y = 118
    cv2.putText(frame, "ID", (W - panel_w + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 170), 1)
    cv2.putText(frame, "Dist", (W - panel_w + 55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 170), 1)
    cv2.putText(frame, "J", (W - panel_w + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 170), 1)
    cv2.putText(frame, "S", (W - panel_w + 155, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 170), 1)
    cv2.putText(frame, "B", (W - panel_w + 190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 170), 1)

    y_offset = 140
    players_sorted = sorted(players, key=lambda x: x[0])
    for (sid, pb, cid) in players_sorted:
        if y_offset > H - 20:
            break
        color = TEAM_1_COLOR if cid in TEAM_1_CLASS_IDS else TEAM_2_COLOR

        # Colored dot
        cv2.circle(frame, (W - panel_w + 22, y_offset - 4), 5, color, -1)

        # Stats
        cv2.putText(frame, f"P{sid}", (W - panel_w + 32, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        cv2.putText(frame, f"{int(distances.get(sid, 0))}", (W - panel_w + 55, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"{jumps.get(sid, 0)}", (W - panel_w + 120, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"{spikes.get(sid, 0)}", (W - panel_w + 155, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"{blocks.get(sid, 0)}", (W - panel_w + 190, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        y_offset += 24

    return frame


def draw_court_overlay(frame, players, ball_c, setter_spiker_arrow=None, setter_spiker_timer=0, player_labels=None):
    """Draw tactical mini-court with setter→spiker arrows."""
    H, W = frame.shape[:2]

    court_w, court_h = 460, 220
    x0 = (W - court_w) // 2
    y0 = H - court_h - 15

    overlay = frame.copy()

    # Court background
    cv2.rectangle(overlay, (x0, y0), (x0 + court_w, y0 + court_h), (30, 30, 50), -1)
    half_w = (court_w - 40) // 2
    cv2.rectangle(overlay, (x0 + 20, y0 + 20), (x0 + 20 + half_w, y0 + court_h - 20), (40, 120, 180), -1)
    cv2.rectangle(overlay, (x0 + 20 + half_w, y0 + 20), (x0 + court_w - 20, y0 + court_h - 20), (35, 100, 160), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + court_w, y0 + court_h), WHITE, 2)

    lane_h = (court_h - 40) // 3
    for i in range(1, 3):
        cv2.line(overlay, (x0 + 20, y0 + 20 + i * lane_h),
                 (x0 + court_w - 20, y0 + 20 + i * lane_h), (180, 180, 200), 1)

    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # Net and attack lines
    net_x = x0 + court_w // 2
    cv2.line(frame, (net_x, y0 + 20), (net_x, y0 + court_h - 20), WHITE, 3)
    cv2.line(frame, (net_x - 60, y0 + 20), (net_x - 60, y0 + court_h - 20), (200, 200, 220), 1)
    cv2.line(frame, (net_x + 60, y0 + 20), (net_x + 60, y0 + court_h - 20), (200, 200, 220), 1)

    # Zone labels
    for z, (zx, zy) in {
        4: (x0 + 50, y0 + 55), 3: (x0 + 50, y0 + court_h // 2),
        2: (x0 + 50, y0 + court_h - 55),
        5: (x0 + 150, y0 + 55), 6: (x0 + 150, y0 + court_h // 2),
        1: (x0 + 150, y0 + court_h - 55),
    }.items():
        cv2.putText(frame, str(z), (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 130), 1)

    def map_xy(x, y):
        mx = x0 + int((x / W) * court_w)
        my = y0 + int((y / H) * court_h)
        return mx, my

    # Players with action labels on court
    if player_labels is None:
        player_labels = {}
    for sid, pb, cid in players:
        c = box_center(pb)
        mx, my = map_xy(c[0], c[1])
        color = TEAM_1_COLOR if cid in TEAM_1_CLASS_IDS else TEAM_2_COLOR
        cv2.circle(frame, (mx, my), 7, WHITE, -1)
        cv2.circle(frame, (mx, my), 5, color, -1)
        # Player ID
        cv2.putText(frame, str(sid), (mx - 4, my + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1)
        # Action label on mini-court
        plbl = player_labels.get(sid, "")
        if plbl:
            lbl_color = ACTION_COLORS.get(plbl, (0, 255, 255))
            cv2.putText(frame, plbl, (mx - 12, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, lbl_color, 1, cv2.LINE_AA)

    # Ball
    if ball_c:
        mx, my = map_xy(ball_c[0], ball_c[1])
        cv2.circle(frame, (mx, my), 5, BLACK, -1)
        cv2.circle(frame, (mx, my), 4, (0, 255, 255), -1)

    # ── SETTER → SPIKER ARROW on mini-court ──
    if setter_spiker_arrow and setter_spiker_timer > 0:
        sp, ep = setter_spiker_arrow
        s_mx, s_my = map_xy(sp[0], sp[1])
        e_mx, e_my = map_xy(ep[0], ep[1])
        # Bright cyan arrow from setter to spiker
        cv2.arrowedLine(frame, (s_mx, s_my), (e_mx, e_my),
                        (0, 255, 255), 3, tipLength=0.15, line_type=cv2.LINE_AA)
        # Label "SET→SPIKE"
        mid_x = (s_mx + e_mx) // 2
        mid_y = (s_my + e_my) // 2 - 8
        cv2.putText(frame, "SET>SPIKE", (mid_x - 25, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "TACTICAL MAP", (x0 + 5, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1, cv2.LINE_AA)

    return frame
