"""
Pass Network — Build and visualize player pass networks using networkx.
Nodes = players, Edges = passes, Weights = pass frequency.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build_pass_network(gameplay, teams):
    """
    Build pass adjacency data from structured pass events.
    
    Returns:
        t1_passes: dict of (from_sid, to_sid) -> count for Team 1
        t2_passes: dict of (from_sid, to_sid) -> count for Team 2
        all_passes: dict of (from_sid, to_sid) -> count for all
    """
    t1_passes = {}
    t2_passes = {}
    all_passes = {}

    # Extract all directed connections from pass chain
    edges = []
    for prev_poss, curr_poss, event_type in gameplay.pass_chain:
        edges.append((prev_poss, curr_poss))
    
    # Also add setter->spiker direct connections
    for setter_sid, spiker_sid, frame in gameplay.setter_spiker_pairs:
        edges.append((setter_sid, spiker_sid))

    for prev_poss, curr_poss in edges:
        key = (prev_poss, curr_poss)
        all_passes[key] = all_passes.get(key, 0) + 1

        prev_team = teams.get(prev_poss, "")
        curr_team = teams.get(curr_poss, "")
        if prev_team == curr_team == "Team 1":
            t1_passes[key] = t1_passes.get(key, 0) + 1
        elif prev_team == curr_team == "Team 2":
            t2_passes[key] = t2_passes.get(key, 0) + 1

    return t1_passes, t2_passes, all_passes


def plot_pass_network(passes, team_players, title, color, node_positions=None):
    """
    Create a matplotlib pass network visualization over an X/Y scatter area.
    
    Args:
        passes: dict of (from_id, to_id) -> count
        team_players: list of player IDs in this team
        title: plot title
        color: base color for nodes
        node_positions: dict of sid -> (x_pct, y_pct)
    
    Returns:
        matplotlib Figure
    """
    try:
        import networkx as nx
        has_nx = True
    except ImportError:
        has_nx = False

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    if not passes or not team_players:
        ax.text(0.5, 0.5, 'No pass data available', ha='center', va='center',
                fontsize=14, color='gray', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.tight_layout()
        return fig

    if has_nx:
        G = nx.DiGraph()

        for pid in team_players:
            G.add_node(pid)

        for (src, dst), weight in passes.items():
            if src in team_players and dst in team_players:
                G.add_edge(src, dst, weight=weight)

        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No connections', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
        else:
            # Use given physical node positions, or fallback to spring_layout
            if node_positions:
                # filter node_positions to only players in team_players
                pos = {}
                for n in G.nodes():
                    if n in node_positions:
                        pos[n] = node_positions[n]
                    else:
                        pos[n] = (50, 50)
            else:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Draw background grid for the "Map Network" look
            ax.grid(color='#333344', linestyle='-', linewidth=0.5, zorder=0)

            # Draw edges with varying width
            edges = G.edges(data=True)
            if edges:
                weights = [d.get('weight', 1) for _, _, d in edges]
                max_w = max(weights) if weights else 1
                edge_widths = [1 + (w / max_w) * 3 for w in weights]
                edge_alphas = [0.3 + (w / max_w) * 0.5 for w in weights]

                for (u, v, d), width, alpha in zip(edges, edge_widths, edge_alphas):
                    ax.annotate("", xy=pos[v], xytext=pos[u],
                                arrowprops=dict(arrowstyle="-|>", color=color,
                                                lw=width, alpha=alpha,
                                                connectionstyle="arc3,rad=0.1",
                                                zorder=1))
                    # Weight label
                    mid_x = (pos[u][0] + pos[v][0]) / 2
                    mid_y = (pos[u][1] + pos[v][1]) / 2
                    ax.text(mid_x, mid_y, str(d.get('weight', '')),
                            fontsize=8, color='white', ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='#2a2a4e', alpha=0.8),
                            zorder=2)

            # Draw nodes
            node_sizes = []
            for n in G.nodes():
                total = sum(d.get('weight', 0) for _, _, d in G.edges(n, data=True))
                total += sum(d.get('weight', 0) for _, _, d in G.in_edges(n, data=True))
                # Base size + scaled by involvement
                node_sizes.append(400 + total * 80)

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                   node_color=color, alpha=0.9, edgecolors='white', linewidths=2)
            
            labels = {n: f"P{n}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9,
                                     font_color='white', font_weight='bold')
            
            if node_positions:
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_xlabel("X Position (%)", color='gray')
                ax.set_ylabel("Y Position (%)", color='gray')
                ax.tick_params(colors='gray')
                ax.axis('on')
                # Hide top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#444')
                ax.spines['bottom'].set_color('#444')
            else:
                ax.axis('off')

    else:
        # Fallback without networkx — simple text summary
        ax.text(0.5, 0.8, title, ha='center', fontsize=14, color='white',
                transform=ax.transAxes)
        y_pos = 0.65
        for (src, dst), cnt in sorted(passes.items(), key=lambda x: -x[1])[:10]:
            ax.text(0.5, y_pos, f"P{src} → P{dst}: {cnt} passes",
                    ha='center', fontsize=11, color=color, transform=ax.transAxes)
            y_pos -= 0.08

    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    if not has_nx or not node_positions:
        ax.axis('off')
    fig.tight_layout()
    return fig
