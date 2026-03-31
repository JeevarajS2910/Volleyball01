"""
Player Analytics — per-player stats tracking, jump/spike/block detection.
"""
import numpy as np
from collections import defaultdict
from core.utils import pdist, box_center, box_bottom, box_height, get_tactical_zone
from core.config import (
    TEAM_1_CLASS_IDS, TEAM_2_CLASS_IDS,
    DEF_JUMP_THRESH, DEF_SPIKE_SPEED, DEF_SPIKE_WINDOW, DEF_POSSESSION,
    DEF_BLOCK_COOLDOWN, FRONT_ROW_ZONES, BACK_ROW_ZONES
)


class PlayerTracker:
    """Tracks all per-player analytics across the match."""

    def __init__(self, jump_thresh=DEF_JUMP_THRESH, spike_speed=DEF_SPIKE_SPEED,
                 spike_window=DEF_SPIKE_WINDOW, poss_dist=DEF_POSSESSION):
        self.jump_thresh = jump_thresh
        self.spike_speed = spike_speed
        self.spike_window = spike_window
        self.poss_dist = poss_dist

        # Per-player state
        self.pos_history = defaultdict(list)     # sid -> [(x,y), ...]
        self.bot_history = defaultdict(list)     # sid -> [y_bottom, ...]
        self.height_history = defaultdict(list)  # sid -> [box_height, ...]
        self.distances = defaultdict(float)       # sid -> total px
        self.jumps = defaultdict(int)
        self.spikes = defaultdict(int)
        self.blocks = defaultdict(int)
        self.teams = {}                           # sid -> "Team 1"/"Team 2"
        self.is_jumping = {}                      # sid -> bool
        self.jump_frame = {}                      # sid -> frame# when jump started
        self.spike_cooldown = defaultdict(int)    # sid -> last spike frame
        self.block_cooldown = defaultdict(int)    # sid -> last block frame
        self.poss_frames = defaultdict(int)       # sid -> frames with possession

        # Team aggregates
        self.t1_jumps = 0
        self.t1_spikes = 0
        self.t2_jumps = 0
        self.t2_spikes = 0
        self.t1_total_dist = 0.0
        self.t2_total_dist = 0.0

        # Zone tracking
        self.zone_presence = defaultdict(lambda: {z: 0 for z in range(1, 7)})
        self.t1_attack_zones = {z: 0 for z in range(1, 7)}
        self.t2_attack_zones = {z: 0 for z in range(1, 7)}

        # Heatmap positions
        self.t1_off_positions = []
        self.t1_def_positions = []
        self.t2_off_positions = []
        self.t2_def_positions = []

        # Current frame state
        self.current_poss_id = None
        self.prev_poss_id = None

    def update_player(self, sid, bbox, cid, fcount, ball_tracker, W, H):
        """
        Update all analytics for a single player in the current frame.
        Returns a dict with current action info.
        """
        center = box_center(bbox)
        bot = box_bottom(bbox)
        bh = box_height(bbox)

        is_t1 = cid in TEAM_1_CLASS_IDS
        team = "Team 1" if is_t1 else "Team 2"
        self.teams[sid] = team

        # History
        self.pos_history[sid].append(center)
        self.bot_history[sid].append(bot[1])
        self.height_history[sid].append(bh)

        # Limit history length to prevent memory growth
        for hist in [self.pos_history[sid], self.bot_history[sid], self.height_history[sid]]:
            if len(hist) > 500:
                del hist[:200]

        # Zone presence
        zone = get_tactical_zone(center[0], center[1], is_t1, W, H)
        self.zone_presence[sid][zone] += 1

        # Offensive / Defensive heatmaps (with spatial filtering)
        ball_side = ball_tracker.side
        net_x = W // 2
        is_on_correct_side = (center[0] < net_x) if is_t1 else (center[0] > net_x)
        
        if is_on_correct_side:
            if ball_side == 'L':
                if is_t1:
                    self.t1_off_positions.append(center)
                else:
                    self.t2_def_positions.append(center)
            elif ball_side == 'R':
                if not is_t1:
                    self.t2_off_positions.append(center)
                else:
                    self.t1_def_positions.append(center)

        # Distance
        ph = self.pos_history[sid]
        if len(ph) >= 2:
            d = pdist(ph[-2], ph[-1])
            self.distances[sid] += d
            if is_t1:
                self.t1_total_dist += d
            else:
                self.t2_total_dist += d

        # Jump detection
        action = self._detect_jump(sid, fcount, is_t1)

        # Possession detection
        self._update_possession(sid, center, ball_tracker)

        # Spike detection
        spike_info = self._detect_spike(sid, center, fcount, is_t1, ball_tracker, zone)
        if spike_info:
            action = "SPIKE"

        # Current action string
        if action is None:
            recently_spiked = (fcount - self.spike_cooldown.get(sid, -999)) <= 10
            if recently_spiked:
                action = "SPIKE"
            elif self.is_jumping.get(sid, False):
                action = "JUMP"
            elif self.current_poss_id == sid:
                action = "BALL"
            else:
                action = ""

        return {
            "action": action,
            "zone": zone,
            "role": "F" if zone in FRONT_ROW_ZONES else "B",
            "is_jumping": self.is_jumping.get(sid, False),
            "team": team,
            "is_t1": is_t1,
        }

    def _detect_jump(self, sid, fcount, is_t1):
        """Detect if player just started jumping. Returns action string or None."""
        bots = self.bot_history[sid]
        heights = self.height_history[sid]
        was_jumping = self.is_jumping.get(sid, False)
        now_jumping = False

        if len(bots) >= 3:
            recent_bot = np.mean(bots[-4:-1]) if len(bots) >= 4 else bots[-2]
            y_rise = recent_bot - bots[-1]

            h_growth = 0
            if len(heights) >= 3:
                recent_h = np.mean(heights[-4:-1]) if len(heights) >= 4 else heights[-2]
                h_growth = heights[-1] - recent_h

            if y_rise > self.jump_thresh or (y_rise > self.jump_thresh * 0.7 and h_growth > 5):
                # Ensure the rise is consistent (vertical velocity check)
                now_jumping = True

        if now_jumping and not was_jumping:
            self.jumps[sid] += 1
            self.jump_frame[sid] = fcount
            if is_t1:
                self.t1_jumps += 1
            else:
                self.t2_jumps += 1

        self.is_jumping[sid] = now_jumping
        return "JUMP" if (now_jumping and not was_jumping) else None

    def _update_possession(self, sid, center, ball_tracker):
        """Track ball possession — closest player within threshold."""
        if not ball_tracker.is_detected:
            return
        d = pdist(center, ball_tracker.current_pos)
        if d < self.poss_dist and ball_tracker.speed_px_frame < 15:
            if self.current_poss_id is None or d < self._min_poss_dist:
                self._min_poss_dist = d
                self.current_poss_id = sid

    def begin_frame_possession(self):
        """Call at start of each frame to reset per-frame possession tracking."""
        self.prev_poss_id = self.current_poss_id
        self.current_poss_id = None
        self._min_poss_dist = float('inf')

    def finalize_frame_possession(self):
        """Call at end of player loop to finalize possession."""
        if self.current_poss_id is not None:
            self.poss_frames[self.current_poss_id] += 1

    def _detect_spike(self, sid, center, fcount, is_t1, ball_tracker, zone):
        """Detect spike event with cooldown."""
        if not ball_tracker.is_detected:
            return None
        if ball_tracker.speed_px_frame <= self.spike_speed:
            return None

        is_possession = (self.current_poss_id == sid)
        jumped_recently = (fcount - self.jump_frame.get(sid, -999)) <= self.spike_window
        not_cooled = (fcount - self.spike_cooldown.get(sid, -999)) > self.spike_window * 2
        d_ball = pdist(center, ball_tracker.current_pos)
        if is_possession and jumped_recently and d_ball < self.poss_dist and not_cooled:
            self.spikes[sid] += 1
            self.spike_cooldown[sid] = fcount

            if is_t1:
                self.t1_spikes += 1
                self.t1_attack_zones[zone] += 1
            else:
                self.t2_spikes += 1
                self.t2_attack_zones[zone] += 1

            return {"sid": sid, "zone": zone, "is_t1": is_t1}
        return None

    def detect_blocks(self, players, fcount, net_x):
        """Detect block events — multiple players jumping near net."""
        events = []
        for (sid, pb, cid) in players:
            c = box_center(pb)
            if not self.is_jumping.get(sid, False):
                continue
            if abs(c[0] - net_x) > 80:
                continue
            # Cooldown check
            if (fcount - self.block_cooldown.get(sid, -999)) < DEF_BLOCK_COOLDOWN:
                continue

            nearby_jumpers = 0
            for (sid2, pb2, cid2) in players:
                if sid2 == sid:
                    continue
                c2 = box_center(pb2)
                if abs(c2[0] - net_x) < 80 and self.is_jumping.get(sid2, False):
                    nearby_jumpers += 1

            if nearby_jumpers >= 1:
                self.blocks[sid] += 1
                self.block_cooldown[sid] = fcount
                events.append(sid)

        return events

    def get_possession_pct(self, fps):
        """Get possession percentage for each team."""
        t1_poss = sum(self.poss_frames[s] for s in self.teams if self.teams[s] == "Team 1")
        t2_poss = sum(self.poss_frames[s] for s in self.teams if self.teams[s] == "Team 2")
        total = t1_poss + t2_poss
        if total == 0:
            return 50.0, 50.0
        return (t1_poss / total) * 100, (t2_poss / total) * 100

    def get_player_stats(self, fps):
        """Get structured stats for all players."""
        all_ids = sorted(self.teams.keys())
        t1_rows, t2_rows = [], []

        for sid in all_ids:
            team = self.teams[sid]
            zone_pres = self.zone_presence[sid]
            fav_zone = max(zone_pres, key=zone_pres.get) if any(zone_pres.values()) else "-"
            row = {
                "Player": f"P{sid}",
                "Distance (px)": f"{self.distances.get(sid, 0):.0f}",
                "Jumps": self.jumps.get(sid, 0),
                "Spikes": self.spikes.get(sid, 0),
                "Blocks": self.blocks.get(sid, 0),
                "Pref Zone": fav_zone,
                "Possession (s)": f"{self.poss_frames.get(sid, 0) / fps:.1f}",
                "Role": "Front" if fav_zone in FRONT_ROW_ZONES else "Back",
                "Frames Seen": len(self.pos_history.get(sid, [])),
            }
            if team == "Team 1":
                t1_rows.append(row)
            else:
                t2_rows.append(row)

        return t1_rows, t2_rows

    def get_top_performers(self, fps):
        """Get top performer in each category."""
        performers = {}
        if self.distances:
            top = max(self.distances, key=self.distances.get)
            performers["distance"] = (top, self.teams.get(top, ""), self.distances[top])
        if self.jumps:
            top = max(self.jumps, key=self.jumps.get)
            performers["jumps"] = (top, self.teams.get(top, ""), self.jumps[top])
        if self.spikes:
            top = max(self.spikes, key=self.spikes.get)
            performers["spikes"] = (top, self.teams.get(top, ""), self.spikes[top])
        if self.poss_frames:
            top = max(self.poss_frames, key=self.poss_frames.get)
            performers["possession"] = (top, self.teams.get(top, ""), self.poss_frames[top] / fps)
        return performers
