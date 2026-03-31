"""
Match Report Generator — CSV, PDF, and visual reports.
Uses matplotlib PdfPages for visual PDF reports.
"""
import csv
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def generate_csv(player_tracker, gameplay_engine, ball_tracker, fps):
    """
    Generate CSV report as a string buffer.
    
    Returns:
        StringIO buffer with CSV content
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Match Summary
    writer.writerow(["=== MATCH SUMMARY ==="])
    writer.writerow(["Metric", "Value"])
    rally_stats = gameplay_engine.get_rally_stats()
    writer.writerow(["Total Rallies", rally_stats["total"]])
    writer.writerow(["Avg Exchanges/Rally", f"{rally_stats['avg_exchanges']:.1f}"])
    writer.writerow(["Max Exchanges", rally_stats["max_exchanges"]])
    writer.writerow(["Score T1", gameplay_engine.t1_score])
    writer.writerow(["Score T2", gameplay_engine.t2_score])
    writer.writerow([])

    # Team Comparison
    writer.writerow(["=== TEAM COMPARISON ==="])
    writer.writerow(["Metric", "Team 1", "Team 2"])
    writer.writerow(["Jumps", player_tracker.t1_jumps, player_tracker.t2_jumps])
    writer.writerow(["Spikes", player_tracker.t1_spikes, player_tracker.t2_spikes])
    writer.writerow(["Distance (px)", f"{player_tracker.t1_total_dist:.0f}",
                     f"{player_tracker.t2_total_dist:.0f}"])
    t1_poss, t2_poss = player_tracker.get_possession_pct(fps)
    writer.writerow(["Possession %", f"{t1_poss:.1f}%", f"{t2_poss:.1f}%"])

    t1_blocks = sum(player_tracker.blocks[s] for s in player_tracker.teams
                    if player_tracker.teams[s] == "Team 1")
    t2_blocks = sum(player_tracker.blocks[s] for s in player_tracker.teams
                    if player_tracker.teams[s] == "Team 2")
    writer.writerow(["Blocks", t1_blocks, t2_blocks])
    writer.writerow([])

    # Player Stats
    writer.writerow(["=== PLAYER STATS ==="])
    writer.writerow(["Player", "Team", "Distance", "Jumps", "Spikes", "Blocks",
                     "Pref Zone", "Possession (s)", "Role"])

    t1_rows, t2_rows = player_tracker.get_player_stats(fps)
    for row in t1_rows + t2_rows:
        team = "Team 1" if row in t1_rows else "Team 2"
        writer.writerow([
            row["Player"], team, row["Distance (px)"], row["Jumps"],
            row["Spikes"], row["Blocks"], row["Pref Zone"],
            row["Possession (s)"], row["Role"]
        ])

    writer.writerow([])

    # Event Log
    writer.writerow(["=== EVENT LOG ==="])
    for event in gameplay_engine.event_log:
        writer.writerow([event])

    output.seek(0)
    return output


def generate_pdf_report(player_tracker, gameplay_engine, ball_tracker, fps,
                        filepath="match_report.pdf"):
    """
    Generate a multi-page visual PDF report using matplotlib PdfPages.
    
    Pages:
    1. Match Summary + Team Comparison
    2. Ball Speed Over Time
    3. Attack Zone Distribution
    4. Rally Length Distribution
    5. Player Stats Table
    """
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # ── Page 1: Match Summary ──
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.axis('off')

        rally_stats = gameplay_engine.get_rally_stats()
        t1_poss, t2_poss = player_tracker.get_possession_pct(fps)

        title = "VOLLEYBALL MATCH ANALYSIS REPORT"
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=20,
                fontweight='bold', color='white', transform=ax.transAxes)

        # Summary box
        summary_lines = [
            f"Total Rallies: {rally_stats['total']}",
            f"Avg Exchanges per Rally: {rally_stats['avg_exchanges']:.1f}",
            f"Longest Rally: {rally_stats['max_exchanges']} exchanges",
            f"Score: Team 1 [{gameplay_engine.t1_score}] — [{gameplay_engine.t2_score}] Team 2",
            f"",
            f"TEAM 1 STATS:",
            f"  Jumps: {player_tracker.t1_jumps}  |  Spikes: {player_tracker.t1_spikes}",
            f"  Distance: {player_tracker.t1_total_dist:.0f} px  |  Possession: {t1_poss:.1f}%",
            f"",
            f"TEAM 2 STATS:",
            f"  Jumps: {player_tracker.t2_jumps}  |  Spikes: {player_tracker.t2_spikes}",
            f"  Distance: {player_tracker.t2_total_dist:.0f} px  |  Possession: {t2_poss:.1f}%",
        ]

        y = 0.82
        for line in summary_lines:
            color = '#4ecdc4' if "TEAM 1" in line else '#e74c3c' if "TEAM 2" in line else 'white'
            fontsize = 13 if "TEAM" in line else 11
            ax.text(0.1, y, line, fontsize=fontsize, color=color,
                    family='monospace', transform=ax.transAxes)
            y -= 0.055

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Page 2: Ball Speed ──
        if ball_tracker.speeds_history:
            fig, ax = plt.subplots(figsize=(11, 6))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')

            step = max(1, len(ball_tracker.speeds_history) // 500)
            sampled = ball_tracker.speeds_history[::step]
            ax.plot(sampled, color='#2ecc71', lw=1, alpha=0.8)
            ax.fill_between(range(len(sampled)), sampled, alpha=0.15, color='#2ecc71')
            ax.set_xlabel("Frame (sampled)", color='white')
            ax.set_ylabel("Speed (px/s)", color='white')
            ax.set_title("Ball Speed Over Time", fontsize=16, fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Stats text
            ax.text(0.02, 0.95, f"Max: {ball_tracker.get_max_speed():.0f} px/s",
                    transform=ax.transAxes, fontsize=10, color='#e74c3c',
                    verticalalignment='top')
            ax.text(0.02, 0.88, f"Avg: {ball_tracker.get_avg_speed():.0f} px/s",
                    transform=ax.transAxes, fontsize=10, color='#3498db',
                    verticalalignment='top')

            fig.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

        # ── Page 3: Attack Zones ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor('#1a1a2e')

        for ax_i, (zones, title_i, color) in zip(
            [ax1, ax2],
            [(player_tracker.t1_attack_zones, "Team 1 Attack Zones", '#3498db'),
             (player_tracker.t2_attack_zones, "Team 2 Attack Zones", '#e74c3c')]
        ):
            ax_i.set_facecolor('#1a1a2e')
            z_labels = [f"Z{i}" for i in range(1, 7)]
            z_values = [zones.get(i, 0) for i in range(1, 7)]
            ax_i.bar(z_labels, z_values, color=color, edgecolor='white', alpha=0.85)
            ax_i.set_title(title_i, fontweight='bold', color='white')
            ax_i.set_ylabel("Spikes", color='white')
            ax_i.tick_params(colors='white')
            ax_i.spines['bottom'].set_color('#444')
            ax_i.spines['left'].set_color('#444')
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['right'].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Page 4: Rally Distribution ──
        if rally_stats["history"]:
            fig, ax = plt.subplots(figsize=(11, 5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')

            hist = rally_stats["history"]
            ax.bar(range(1, len(hist) + 1), hist, color='#4ecdc4',
                   edgecolor='white', alpha=0.85)
            ax.set_xlabel("Rally #", color='white')
            ax.set_ylabel("Net Crossings", color='white')
            ax.set_title("Rally Length Distribution", fontsize=16,
                         fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

        # ── Page 5: Player Stats Table ──
        t1_rows, t2_rows = player_tracker.get_player_stats(fps)
        all_rows = t1_rows + t2_rows

        if all_rows:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.patch.set_facecolor('#1a1a2e')
            ax.axis('off')

            cols = ["Player", "Distance (px)", "Jumps", "Spikes", "Blocks",
                    "Pref Zone", "Possession (s)", "Role"]
            cell_text = []
            cell_colors = []

            for row in all_rows:
                cell_text.append([str(row.get(c, "")) for c in cols])
                is_t1 = row in t1_rows
                bg = '#1a3a5c' if is_t1 else '#5c1a1a'
                cell_colors.append([bg] * len(cols))

            table = ax.table(cellText=cell_text, colLabels=cols,
                             cellColours=cell_colors, loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Style header
            for j in range(len(cols)):
                table[(0, j)].set_facecolor('#2a2a4e')
                table[(0, j)].set_text_props(color='white', fontweight='bold')
            # Style cells
            for i in range(1, len(cell_text) + 1):
                for j in range(len(cols)):
                    table[(i, j)].set_text_props(color='white')

            ax.set_title("Player Performance Summary", fontsize=16,
                         fontweight='bold', color='white', pad=20)

            fig.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer


def plot_ball_speed_graph(speeds, dark_theme=True):
    """Generate ball speed over time chart for Streamlit display."""
    fig, ax = plt.subplots(figsize=(10, 3))

    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        text_color = 'white'
    else:
        fig.patch.set_facecolor('#f0f2f6')
        ax.set_facecolor('#f0f2f6')
        text_color = 'black'

    if speeds:
        step = max(1, len(speeds) // 500)
        sampled = speeds[::step]
        ax.plot(sampled, color='#2ecc71', lw=1, alpha=0.8)
        ax.fill_between(range(len(sampled)), sampled, alpha=0.15, color='#2ecc71')

    ax.set_xlabel("Frame", color=text_color)
    ax.set_ylabel("Speed (px/s)", color=text_color)
    ax.set_title("Ball Speed Over Time", fontweight='bold', color=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_rally_distribution(rally_hist, dark_theme=True):
    """Generate rally length distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 3))

    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        text_color = 'white'
    else:
        fig.patch.set_facecolor('#f0f2f6')
        ax.set_facecolor('#f0f2f6')
        text_color = 'black'

    if rally_hist:
        ax.bar(range(1, len(rally_hist) + 1), rally_hist,
               color='#4ecdc4', edgecolor='white', alpha=0.85)

    ax.set_xlabel("Rally #", color=text_color)
    ax.set_ylabel("Exchanges", color=text_color)
    ax.set_title("Net Crossings per Rally", fontweight='bold', color=text_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig
