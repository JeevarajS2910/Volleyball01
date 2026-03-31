"""
ID Stabilizer — maps raw ByteTrack IDs to consistent team-based IDs.
Team 1: IDs 1-6, Team 2: IDs 7-12.
"""
from core.utils import pdist


class IDStabilizer:
    """Maps raw ByteTrack IDs to strict 1-6 (T1) and 7-12 (T2) IDs."""

    def __init__(self, max_per_team=6):
        self.r2s = {}           # raw_id -> stable_id
        self.s2r = {}           # stable_id -> raw_id
        self.last_pos = {}      # stable_id -> (x, y)
        self.last_team = {}     # stable_id -> "Team 1" / "Team 2"
        self.team_counts = {"Team 1": 0, "Team 2": 0}
        self.max_per_team = max_per_team

    def get(self, raw, center, team_name):
        """Get or create a stable ID for a raw tracking ID."""
        # Already mapped
        if raw in self.r2s:
            s = self.r2s[raw]
            self.last_pos[s] = center
            return s

        # Try to reclaim a lost stable ID (same team, closest position)
        best_s, best_d = None, float('inf')
        for sid, pos in self.last_pos.items():
            if sid in self.s2r and self.s2r[sid] in self.r2s:
                continue
            if self.last_team.get(sid) != team_name:
                continue
            d = pdist(center, pos)
            if d < best_d:
                best_d, best_s = d, sid

        if best_s is not None and best_d < 200:
            self.r2s[raw] = best_s
            self.s2r[best_s] = raw
            self.last_pos[best_s] = center
            return best_s

        # Assign new stable ID
        if self.team_counts[team_name] < self.max_per_team:
            self.team_counts[team_name] += 1
            offset = 0 if team_name == "Team 1" else self.max_per_team
            s = offset + self.team_counts[team_name]

            self.r2s[raw] = s
            self.s2r[s] = raw
            self.last_pos[s] = center
            self.last_team[s] = team_name
            return s

        return None

    def cleanup(self, active_raws):
        """Remove mappings for raw IDs no longer in the current frame."""
        for r in [r for r in self.r2s if r not in active_raws]:
            del self.r2s[r]

    def reset(self):
        """Full reset of all mappings."""
        self.r2s.clear()
        self.s2r.clear()
        self.last_pos.clear()
        self.last_team.clear()
        self.team_counts = {"Team 1": 0, "Team 2": 0}
