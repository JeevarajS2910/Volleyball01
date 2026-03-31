"""
Attack Zone Analysis — Zone mapping, charts, and court diagrams.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_attack_zones(zones_dict, title, color):
    """
    Bar chart of spike counts per zone (1-6).
    
    Args:
        zones_dict: {zone_num: count}
        title: chart title
        color: bar color
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    z_labels = [f"Z{i}" for i in range(1, 7)]
    z_values = [zones_dict.get(i, 0) for i in range(1, 7)]

    bars = ax.bar(z_labels, z_values, color=color, edgecolor='white', alpha=0.85, width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, z_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontsize=10, color='white')

    ax.set_title(title, fontweight='bold', color='white', fontsize=13)
    ax.set_ylabel("Spikes", color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_zone_heatmap(positions, is_team1, cmap, title, W, H, net_x):
    """
    Court zone distribution heatmap (3x2 grid).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    grid = np.zeros((3, 2))

    for c in positions:
        x, y = c
        
        # Spatial filtering: ignore wrong-side points
        is_on_wrong_side = (x > net_x) if is_team1 else (x < net_x)
        if is_on_wrong_side:
            continue

        row = int((y / H) * 3)
        row = max(0, min(2, row))
        is_left_side = (x < net_x)
        if is_left_side:
            col = 0 if x > (net_x - W * 0.15) else 1
        else:
            col = 0 if x < (net_x + W * 0.15) else 1
        
        grid[row, col] += 1

    im = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')

    # Value annotations
    max_val = np.max(grid) if np.max(grid) > 0 else 1
    for (i, j), val in np.ndenumerate(grid):
        text_color = 'white' if val > max_val / 2 else 'lightgray'
        ax.text(j, i, int(val), ha='center', va='center', color=text_color,
                fontsize=14, fontweight='bold')

    ax.set_title(title, fontweight='bold', color='white', fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Front", "Back"], color='white')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Top", "Mid", "Bot"], color='white')

    fig.tight_layout()
    return fig


def plot_court_diagram_with_zones(t1_zones, t2_zones, title="Attack Zone Distribution"):
    """
    Visual volleyball court diagram with zone heat coloring.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Court dimensions
    cw, ch = 18, 9  # meters (standard volleyball court)

    # Draw court outline
    court = plt.Rectangle((0, 0), cw, ch, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(court)

    # Center line (net)
    ax.plot([cw / 2, cw / 2], [0, ch], 'w-', linewidth=3)

    # Attack lines (3m from net)
    ax.plot([cw / 2 - 3, cw / 2 - 3], [0, ch], 'w--', linewidth=1, alpha=0.5)
    ax.plot([cw / 2 + 3, cw / 2 + 3], [0, ch], 'w--', linewidth=1, alpha=0.5)

    # Zone definitions for each half
    # Team 1 (left half): zones are relative to baseline
    t1_zone_rects = {
        4: (6, 6, 3, 3),    # Front left
        3: (6, 3, 3, 3),    # Front center
        2: (6, 0, 3, 3),    # Front right
        5: (0, 6, 6, 3),    # Back left
        6: (0, 3, 6, 3),    # Back center
        1: (0, 0, 6, 3),    # Back right
    }

    t2_zone_rects = {
        2: (9, 6, 3, 3),    # Front left
        3: (9, 3, 3, 3),    # Front center
        4: (9, 0, 3, 3),    # Front right
        1: (12, 6, 6, 3),   # Back left
        6: (12, 3, 6, 3),   # Back center
        5: (12, 0, 6, 3),   # Back right
    }

    # Color zones by spike count
    all_vals = list(t1_zones.values()) + list(t2_zones.values())
    max_val = max(all_vals) if all_vals and max(all_vals) > 0 else 1

    for z, (rx, ry, rw, rh) in t1_zone_rects.items():
        val = t1_zones.get(z, 0)
        intensity = val / max_val
        color = (0.2 + intensity * 0.5, 0.4 + intensity * 0.4, 1.0, 0.3 + intensity * 0.5)
        rect = plt.Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(rx + rw / 2, ry + rh / 2, f"Z{z}\n{val}", ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')

    for z, (rx, ry, rw, rh) in t2_zone_rects.items():
        val = t2_zones.get(z, 0)
        intensity = val / max_val
        color = (1.0, 0.3 + intensity * 0.3, 0.2 + intensity * 0.3, 0.3 + intensity * 0.5)
        rect = plt.Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(rx + rw / 2, ry + rh / 2, f"Z{z}\n{val}", ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')

    # Labels
    ax.text(cw / 4, ch + 0.5, "TEAM 1", ha='center', fontsize=14,
            fontweight='bold', color='#4ecdc4')
    ax.text(3 * cw / 4, ch + 0.5, "TEAM 2", ha='center', fontsize=14,
            fontweight='bold', color='#e74c3c')
    ax.text(cw / 2, ch + 0.5, "NET", ha='center', fontsize=10, color='white')

    ax.set_xlim(-0.5, cw + 0.5)
    ax.set_ylim(-0.5, ch + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=15, fontweight='bold', color='white')

    fig.tight_layout()
    return fig


def plot_cumulative_timeline(t1_data, t2_data, title, y_label="Value"):
    """
    Line chart tracking cumulative stats over time (e.g., Spikes Timeline).
    Matches the 'xG Timeline' style requested.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    x_vals = range(len(t1_data))
    
    # Team A & Team B Lines
    ax.plot(x_vals, t1_data, color='#4ecdc4', linewidth=2.5, label='Team 1 (T1)')
    ax.plot(x_vals, t2_data, color='#2874A6', linewidth=2.5, label='Team 2 (T2)')

    ax.set_title(title, fontweight='bold', color='white', fontsize=13)
    ax.set_xlabel("Time (seconds)", color='gray')
    ax.set_ylabel(y_label, color='gray')
    
    ax.tick_params(colors='gray')
    ax.grid(color='#333', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    leg = ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    for text in leg.get_texts():
        text.set_color("white")

    fig.tight_layout()
    return fig
