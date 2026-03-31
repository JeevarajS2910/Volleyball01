"""
Gameplay Engine — Volleyball rally state machine.
Manages: IDLE → SERVE → PASS → SET → SPIKE → RALLY_END
"""
from core.config import RallyState, BALL_MISSING_END, DEF_EVENT_COOLDOWN, MAX_EVENT_LOG


class GameplayEngine:
    """
    Volleyball gameplay state machine.
    
    Transitions:
        IDLE → SERVE:     Ball appears after absence
        SERVE → RECV:     Ball possession changes (first touch)
        RECV → SET:       Possession changes to teammate near net
        SET → SPIKE:      Player jumps + ball moves fast
        SPIKE → RECV:     Ball crosses net (new rally exchange)
        Any → RALLY_END:  Ball missing for too long
        RALLY_END → IDLE: Reset for next rally
    """

    def __init__(self):
        self.state = RallyState.IDLE
        self.rally_on = False
        self.rally_number = 0
        self.rally_exchanges = 0
        self.match_over = False
        self.match_winner = None
        self.rally_history = []      # list of exchange counts per completed rally
        self.ball_side = None
        self.last_ball_pos = None
        self.last_player_tracker = None

        # Event management
        self.event_log = []
        self.active_event = ""
        self.event_timer = 0
        self.event_cooldowns = {}    # event_type -> last_frame

        # Pass tracking — structured data for pass network
        self.pass_arrow = None
        self.pass_timer = 0
        self.pass_chain = []         # [(from_sid, to_sid, type), ...] for pass network
        self.setter_spiker_pairs = []  # [(setter_sid, spiker_sid, frame), ...] for tactical map
        self.current_setter = None     # sid of last player who did a SET
        self.last_setter_pos = None    # position of setter during SET
        self.last_spiker_pos = None    # position of spiker during SPIKE
        self.setter_spiker_arrow = None  # (setter_pos, spiker_pos) for tactical map
        self.setter_spiker_timer = 0

        # Player action labels (broadcast to drawing layer)
        self.player_labels = {}  # sid -> label string (SET/PASS/BLOCK/SPIKE/SERVE)
        self.label_timers = {}   # sid -> frames remaining for label

        # Score (manual + auto-suggest)
        self.t1_score = 0
        self.t2_score = 0
        self.t1_sets = 0
        self.t2_sets = 0
        self.suggested_point = None  # ("Team 1" or "Team 2", reason)

        # FIVB specific Tracking
        self.team_touches = 0
        self.last_touch_team = None
        self.last_touch_side = None
        self.rally_on = False

    def update(self, fcount, ball_tracker, player_tracker, net_x):
        """Main state machine update loop."""
        if self.match_over:
            return

        """
        Main update — call once per frame.
        Returns list of new events generated this frame.
        """
        new_events = []

        # ── Ball present ──
        if ball_tracker.is_detected:
            self.no_ball_frames = 0
            self.last_ball_pos = ball_tracker.current_pos
            self.last_player_tracker = player_tracker
            side = ball_tracker.side

            # Start new rally if ball appears after being missing
            if not self.rally_on:
                self.rally_on = True
                self.rally_number += 1
                self.ball_side = side
                self.team_touches = 1
                # Auto-assign initial possession to the team whose court it spawns in
                self.last_touch_team = "Team 1" if side == 'R' else "Team 2"
                self.state = RallyState.SERVE
                self._log_event(fcount, "RALLY_START", f"Rally {self.rally_number} begins")
                
            # Ball crossed the net
            if side != self.ball_side:
                self.rally_exchanges += 1
                
                # Mathematical fallback: if the ball crossed the net from R->L, Team 1 MUST have been the last to hit it.
                # This guarantees we don't miss attacks due to tracker missing fast balls
                self.last_touch_team = "Team 1" if self.ball_side == 'R' else "Team 2"
                
                self.ball_side = side
                self.team_touches = 0
                self.current_setter = None
                self._log_event(fcount, "EXCHANGE", f"Ball crossed to {side}")
                # After a spike or block, if ball crosses, go to RECV for the other team
                if self.rally_on:
                    self.state = RallyState.RECV

            # Possession-based state transitions
            poss_id = player_tracker.current_poss_id
            prev_poss = player_tracker.prev_poss_id

            if poss_id is not None and poss_id != prev_poss and prev_poss is not None:
                # Possession changed
                curr_team = player_tracker.teams.get(poss_id, "")
                prev_team = player_tracker.teams.get(prev_poss, "")

                self.last_touch_team = curr_team
                self.last_touch_side = side

                if curr_team == prev_team and len(curr_team) > 0:
                    self.team_touches += 1
                else:
                    self.team_touches = 1

                # FIVB 4-hit fault check
                if self.team_touches > 3:
                    fault_team = curr_team
                    opp_team = "Team 2" if fault_team == "Team 1" else "Team 1"
                    self._log_event(fcount, "FAULT", f"4-Hit Fault by {fault_team}")
                    self.suggested_point = (opp_team, f"4-Hit Fault ({fault_team})")
                    self.accept_suggested_point(opp_team)

                # Store structured pass for network
                self.pass_chain.append((prev_poss, poss_id, self.state))

                if self.state == RallyState.SERVE:
                    self.state = RallyState.RECV
                    evt = self._log_event(fcount, "RECV", f"P{poss_id} RECV (from P{prev_poss})")
                    if evt:
                        new_events.append(evt)
                    self._set_player_label(poss_id, "RECV", 25)
                    self._set_player_label(prev_poss, "SERVE", 25)

                elif self.state == RallyState.RECV or self.state == RallyState.SET:
                    if curr_team == prev_team:
                        # Same team touch mapping
                        evt = None
                        if self.team_touches == 2:
                            # 2nd touch: typically a SET
                            self.state = RallyState.SET
                            self.current_setter = poss_id
                            evt = self._log_event(fcount, "SET", f"P{poss_id} SET (from P{prev_poss})")
                            self._set_player_label(poss_id, "SET", 25)
                        elif self.team_touches >= 3:
                            # 3rd touch, if NOT a real spike, it's just a freeball push
                            evt = self._log_event(fcount, "FREEBALL", f"P{poss_id} FREEBALL (3rd touch)")
                            self._set_player_label(poss_id, "PASS", 25)
                        
                        if evt:
                            new_events.append(evt)
                            
                        # Store setter position for tactical arrow
                        if self.state == RallyState.SET and poss_id in player_tracker.pos_history:
                            if len(player_tracker.pos_history[poss_id]) > 0:
                                self.last_setter_pos = player_tracker.pos_history[poss_id][-1]
                    else:
                        # Cross-net hit from opponent
                        self.state = RallyState.RECV
                        self.team_touches = 1
                        evt = self._log_event(fcount, "RECV", f"P{poss_id} RECV (from opponent)")
                        if evt:
                            new_events.append(evt)
                        self._set_player_label(poss_id, "RECV", 25)

                elif self.state == RallyState.SET:
                    # After set, next touch should be attack
                    self.state = RallyState.SPIKE
                    # Don't log spike here — wait for actual spike detection

                # Generate pass arrow
                if prev_poss in player_tracker.pos_history and len(player_tracker.pos_history[prev_poss]) > 0:
                    start_p = player_tracker.pos_history[prev_poss][-1]
                    end_p = player_tracker.pos_history[poss_id][-1]
                    self.pass_arrow = (start_p, end_p)
                    self.pass_timer = 20

        else:
            # Ball not detected
            if self.rally_on and ball_tracker.is_missing_long:
                self._end_rally(fcount)

        # Decay event timer
        if self.event_timer > 0:
            self.event_timer -= 1
        if self.pass_timer > 0:
            self.pass_timer -= 1
        if self.setter_spiker_timer > 0:
            self.setter_spiker_timer -= 1

        # Decay player labels
        expired = [sid for sid, t in self.label_timers.items() if t <= 0]
        for sid in expired:
            del self.label_timers[sid]
            if sid in self.player_labels:
                del self.player_labels[sid]
        for sid in self.label_timers:
            self.label_timers[sid] -= 1

        return new_events

    def on_spike(self, fcount, sid, zone, is_t1, player_tracker=None):
        """Called by player_tracker when a spike is detected."""
        # Prevent multiple spike logs in the same attack sequence
        if self.state == RallyState.SPIKE:
            return None
            
        team = "Team 1" if is_t1 else "Team 2"
        self.state = RallyState.SPIKE
        
        # Build description showing setter → spiker connection
        desc = f"P{sid} SPIKE Zone {zone}"
        if self.current_setter is not None:
            desc += f" (set by P{self.current_setter})"
            self.setter_spiker_pairs.append((self.current_setter, sid, fcount))
            # Build tactical arrow: setter → spiker
            if player_tracker and sid in player_tracker.pos_history and len(player_tracker.pos_history[sid]) > 0:
                self.last_spiker_pos = player_tracker.pos_history[sid][-1]
                if self.last_setter_pos:
                    self.setter_spiker_arrow = (self.last_setter_pos, self.last_spiker_pos)
                    self.setter_spiker_timer = 40  # Show for longer
        
        evt = self._log_event(fcount, "SPIKE", desc)
        self._set_player_label(sid, "SPIKE", 30)
        
        # Auto-suggest point
        self.suggested_point = (team, f"Spike from Zone {zone}")
        self.current_setter = None  # Reset setter for next play
        return evt

    def on_block(self, fcount, sid):
        """Called when a block is detected."""
        self.state = RallyState.BLOCK
        evt = self._log_event(fcount, "BLOCK", f"P{sid} BLOCK at net")
        self._set_player_label(sid, "BLOCK", 25)
        return evt

    def _set_player_label(self, sid, label, duration):
        """Set a visible label for a player (shown on bounding box)."""
        self.player_labels[sid] = label
        self.label_timers[sid] = duration

    def get_player_label(self, sid):
        """Get current label for a player, or empty string."""
        return self.player_labels.get(sid, "")

    def set_score(self, t1, t2, t1_sets=None, t2_sets=None):
        """Manual score update."""
        self.t1_score = t1
        self.t2_score = t2
        if t1_sets is not None:
            self.t1_sets = t1_sets
        if t2_sets is not None:
            self.t2_sets = t2_sets

    def accept_suggested_point(self, team):
        """Accept the auto-suggested point winner."""
        if team == "Team 1":
            self.t1_score += 1
        elif team == "Team 2":
            self.t2_score += 1
        self.suggested_point = None
        self._check_set_win()

    def _check_set_win(self):
        """Check if a team has reached 25 (or 15 for 5th set) with a 2-point lead."""
        total_sets = self.t1_sets + self.t2_sets
        points_to_win = 15 if total_sets == 4 else 25

        if self.t1_score >= points_to_win and (self.t1_score - self.t2_score) >= 2:
            self.t1_sets += 1
            self.t1_score = 0
            self.t2_score = 0
            if self.t1_sets == 3:
                self.match_over = True
                self.match_winner = "Team 1"
                self._log_event(0, "MATCH_WIN", "Team 1 Wins the Match!")
        elif self.t2_score >= points_to_win and (self.t2_score - self.t1_score) >= 2:
            self.t2_sets += 1
            self.t1_score = 0
            self.t2_score = 0
            if self.t2_sets == 3:
                self.match_over = True
                self.match_winner = "Team 2"
                self._log_event(0, "MATCH_WIN", "Team 2 Wins the Match!")

    def _end_rally(self, fcount):
        """End the current rally."""
        
        # ── Auto Assign Point Heuristic ──
        if self.rally_on and self.last_touch_team and self.ball_side:
            # If the ball dropped on Team 2's side, Team 1 typically gets the point (attack won).
            # If Team 1 touches it last and it drops on Team 1's side (error), Team 2 gets the point.
            # Assuming ball_side 'R' or 'L' maps to court halves. 
            # We don't distinctly know which team owns which half permanently, but we can assume
            # whoever didn't drop it on their side wins it, or if you touch it and drop it on your side, opponent wins.
            # For simplicity: if last_touch was X, and it didn't cross the net (team_touches > 0 but no cross?), 
            # we just reward the opponent if team drops it.
            
            # Simple heuristic: If last team to touch the ball is the ONLY one to touch it
            # and it drops... error.
            # A more stable heuristic based on the user's plan:
            # if T1 final touch & ball disappears on T2 side -> T1 point
            # if T1 final touch & ball disappears on T1 side -> T2 point
            # We roughly estimate "T1 side" vs "T2 side" by where they are currently standing, but lacking that,
            # we assume the opponent gets the point unless the ball made it across the net.
            
            opp_team = "Team 2" if self.last_touch_team == "Team 1" else "Team 1"
            
            # We assume a successful attack if the ball side changed after the last hit. 
            # If team_touches > 0 and didn't reset, they dropped it on their own side.
            if self.team_touches > 0:
                # Dropped on own side before crossing
                win_team = opp_team
                reason = f"Error by {self.last_touch_team}"
            else:
                # Ball crossed net and then dropped
                # Calculate distance to nearest defender to check if they 'left' the ball (Out of bounds)
                min_def_dist = float('inf')
                if self.last_ball_pos and self.last_player_tracker:
                    for sid, t_name in self.last_player_tracker.teams.items():
                        if t_name == opp_team:
                            ph = self.last_player_tracker.pos_history.get(sid, [])
                            if ph:
                                d = ((self.last_ball_pos[0] - ph[-1][0])**2 + (self.last_ball_pos[1] - ph[-1][1])**2)**0.5
                                if d < min_def_dist:
                                    min_def_dist = d
                                    
                if min_def_dist > 250:
                    win_team = opp_team
                    reason = f"Out of Bounds by {self.last_touch_team}"
                else:
                    win_team = self.last_touch_team
                    reason = f"Attack by {self.last_touch_team}"
                
            self.suggested_point = (win_team, reason)
            self.accept_suggested_point(win_team)

        self.rally_on = False
        self.rally_history.append(self.rally_exchanges)
        self.state = RallyState.IDLE
        self.rally_exchanges = 0
        self.ball_side = None
        self.current_setter = None
        self.team_touches = 0
        self.last_touch_team = None
        
        # Clear all action badges immediately so they don't stick
        self.player_labels.clear()
        self.label_timers.clear()

    def _log_event(self, fcount, event_type, description):
        """Log an event with cooldown to prevent flickering."""
        # Cooldown check
        last = self.event_cooldowns.get(event_type, -999)
        if fcount - last < DEF_EVENT_COOLDOWN:
            return None

        self.event_cooldowns[event_type] = fcount
        entry = f"Frame {fcount}: {description}"
        self.event_log.append(entry)

        # Limit log size
        if len(self.event_log) > MAX_EVENT_LOG:
            self.event_log = self.event_log[-MAX_EVENT_LOG:]

        self.active_event = event_type
        self.event_timer = 20

        return entry

    def get_rally_stats(self):
        """Get rally summary statistics."""
        import numpy as np
        hist = self.rally_history
        return {
            "total": self.rally_number,
            "avg_exchanges": float(np.mean(hist)) if hist else 0.0,
            "max_exchanges": max(hist) if hist else 0,
            "history": hist,
        }
