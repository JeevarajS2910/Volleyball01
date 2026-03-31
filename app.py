"""
🏐 Professional Volleyball Analytics System
============================================
Industry-level volleyball analysis platform with:
- YOLOv8 detection + ByteTrack tracking
- Gameplay state machine (SERVE → PASS → SET → SPIKE → BLOCK)
- Broadcast-quality UI with 5 analysis tabs
- Advanced analytics (pass networks, attack zones, heatmaps)
- PDF/CSV report generation
"""
import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    # ── Core modules ──
    from core.config import (
        DEFAULT_MODEL_PATH, DEFAULT_VIDEO_PATH, TRACKER_CONFIG, OUTPUT_VIDEO_PATH,
        BALL_CLASS_ID, TEAM_1_CLASS_IDS, TEAM_2_CLASS_IDS, PLAYER_CLASS_IDS, REFEREE_CLASS_ID,
        TEAM_1_COLOR, TEAM_2_COLOR, BALL_COLOR,
        DEF_CONF_THRESH, DEF_JUMP_THRESH, DEF_SPIKE_SPEED, DEF_SPIKE_WINDOW, DEF_POSSESSION,
        SKIP_FRAMES_DURING_RALLY, SKIP_FRAMES_OUTSIDE_RALLY, RallyState
    )
    from core.utils import box_center, format_time, pdist
    from core.id_stabilizer import IDStabilizer
    from core.ball_analytics import BallTracker
    from core.player_analytics import PlayerTracker
    from core.gameplay_engine import GameplayEngine
    from core.drawing import (
        draw_player_bbox, draw_referee, draw_ball,
        draw_ball_trail, draw_pass_arrow, draw_scoreboard,
        draw_event_popup, draw_stats_panel, draw_court_overlay
    )

    # ── Analytics modules ──
    from analytics.pass_network import build_pass_network, plot_pass_network
    from analytics.attack_zones import plot_attack_zones, plot_zone_heatmap, plot_court_diagram_with_zones, plot_cumulative_timeline

    # ── Report modules ──
    from reports.match_report import (
        generate_csv, generate_pdf_report,
        plot_ball_speed_graph, plot_rally_distribution
    )
except Exception as e:
    import streamlit as st
    st.error(f"Module error: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
#                        STREAMLIT PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=" Volleyball Analytics",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state init ──
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ═══════════════════════════════════════════════════════════════════
#                        CUSTOM CSS — BROADCAST THEME
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Light professional background */
    .stApp {
        background: #f4f6f9;
        color: #1e293b;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricDelta"] {
        color: #0ea5e9 !important;
    }
    
    /* Glass-morphism cards -> Clean Light Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-radius: 12px;
        padding: 12px 16px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #f1f5f9;
        color: #0ea5e9 !important;
        border-bottom: 2px solid #0ea5e9;
    }
    
    /* Tables */
    .stDataFrame {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 700;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: #334155 !important;
    }
    .stButton > button:hover {
        border-color: #94a3b8;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0ea5e9, #60a5fa);
    }
    
    /* Headings */
    h1, h2, h3, h4, h5 {
        color: #0f172a !important;
    }
    
    /* Remove default boxiness */
    hr { border-color: #e2e8f0; }
    
    /* Top bar styling */
    .top-bar {
        background: #ffffff;
        border-bottom: 2px solid #0ea5e9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        padding: 8px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .score-display {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
    }
    .team-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .team1-color { color: #ea580c; } /* darker orange */
    .team2-color { color: #2563eb; } /* darker blue */
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
    }
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#                            SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  Volleyball Analytics")
    st.markdown("---")

    analysis_mode = st.selectbox("📺 Analysis Mode", ["Video Analysis", "Live Camera"], index=0)

    start_btn = st.button("🔴 Start Analysis", type="primary", width="stretch")
    if start_btn:
        st.session_state.run_analysis = True

    stop_btn = st.button("⏹️ Stop", width="stretch")
    if stop_btn:
        st.session_state.run_analysis = False

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    flt_team = st.selectbox("Teams", ["All", "Team 1", "Team 2"])
    flt_event = st.selectbox("Events", ["All Events", "Serve", "Pass", "Set", "Spike", "Block"])

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")
    model_path = st.text_input("Model Path", DEFAULT_MODEL_PATH)
    conf_thresh = st.slider("Confidence Threshold", 0.01, 1.0, DEF_CONF_THRESH)

    with st.expander("🔬 Detection Thresholds"):
        jump_thresh = st.slider("Jump Threshold (px)", 5, 40, DEF_JUMP_THRESH)
        spike_speed = st.slider("Spike Speed (px/f)", 2, 30, DEF_SPIKE_SPEED)
        spike_window = st.slider("Spike Window (frames)", 2, 20, DEF_SPIKE_WINDOW)
        poss_dist = st.slider("Possession Distance (px)", 50, 400, DEF_POSSESSION)

    st.markdown("---")
    st.markdown("### 📂 Video Source")
    uploaded = st.file_uploader("Upload Match Video", type=["mp4", "avi", "mov"])


# ═══════════════════════════════════════════════════════════════════
#                          MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════
if st.session_state.run_analysis or uploaded:

    # ── Video Setup ──
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    if uploaded:
        tfile.write(uploaded.read())
        video_path = tfile.name
    else:
        video_path = DEFAULT_VIDEO_PATH

    # ── Model Loading ──
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info("Check that the model path is correct and the file exists.")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        st.stop()

    # Reduce resolution for performance
    W = 1280
    H = 720
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    net_x = W // 2

    # ── Video Writer ──
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W, H))

    # ── Initialize Engines ──
    stabilizer = IDStabilizer()
    ball_tracker = BallTracker(fps=fps)
    player_tracker = PlayerTracker(
        jump_thresh=jump_thresh, spike_speed=spike_speed,
        spike_window=spike_window, poss_dist=poss_dist
    )
    gameplay = GameplayEngine()

    # ══════════════════════════════════════════════════════════════
    #                    TOP BAR + TIMELINE
    # ══════════════════════════════════════════════════════════════
    top_bar = st.container()
    with top_bar:
        tb1, tb2, tb3, tb4, tb5 = st.columns([1, 1.5, 2, 1.5, 1])
        time_ph = tb1.empty()
        t1_info_ph = tb2.empty()
        score_ph = tb3.empty()
        t2_info_ph = tb4.empty()
        ball_info_ph = tb5.empty()

    timeline_ph = st.progress(0)

    # ══════════════════════════════════════════════════════════════
    #                         TABS
    # ══════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "🔴 Live Analysis",
        "📹 Replay",
        "📋 Tactics",
        "👥 Players",
        "📊 Reports"
    ])

    # ── Tab 0: Live Analysis ──
    with tabs[0]:
        col_vid, col_live = st.columns([2.5, 1])

        with col_vid:
            frame_ph = st.empty()
            st.markdown("#### Feature Toggles")
            tog1, tog2, tog3 = st.columns(3)
            show_t1 = tog1.checkbox("Team 1", True)
            show_t2 = tog1.checkbox("Team 2", True)
            opt_ball_trail = tog2.checkbox("Ball Trail", True)
            opt_court = tog2.checkbox("Tactical Court", True)
            opt_roles = tog3.checkbox("Player Roles", True)
            opt_stats_panel = tog1.checkbox("Stats Panel", True)

            st.markdown("---")
            mc1, mc2, mc3, mc4 = st.columns(4)
            stat_players = mc1.empty()
            stat_ball = mc2.empty()
            stat_events = mc3.empty()
            stat_state = mc4.empty()

        with col_live:
            st.markdown("### 📈 Live Stats")
            live_score_ph = st.empty()
            live_poss_ph = st.empty()
            st.markdown("### 🧾 Event Timeline")
            live_events_ph = st.empty()

    # ── Tab 1: VAR / Replay (placeholders) ──
    with tabs[1]:
        var_ph = st.empty()

    # ── Tab 2: Tactics (placeholders) ──
    with tabs[2]:
        tactics_ph = st.empty()

    # ── Tab 3: Players (placeholders) ──
    with tabs[3]:
        players_ph = st.empty()

    # ── Tab 4: Reports (placeholders) ──
    with tabs[4]:
        reports_ph = st.empty()

    # ══════════════════════════════════════════════════════════════
    #                    PROCESSING LOOP
    # ══════════════════════════════════════════════════════════════
    fcount = 0
    t1_timeline_spikes = []
    t2_timeline_spikes = []
    last_results = None

    while cap.isOpened():
        if not st.session_state.run_analysis:
            break

        ok, frame = cap.read()
        if not ok:
            break
        
        # Resize frame directly to 1280x720 
        frame = cv2.resize(frame, (1280, 720))
        fcount += 1

        # ── Adaptive Frame Skipping ──
        skip_rate = SKIP_FRAMES_DURING_RALLY if gameplay.rally_on else SKIP_FRAMES_OUTSIDE_RALLY
        
        # ── YOLO Detection + Tracking ──
        if fcount % skip_rate != 0:
            results = last_results
        else:
            try:
                # tracking args changed for stability, directly passing frame
                results = model.track(
                    frame, conf=conf_thresh, persist=True,
                    tracker=TRACKER_CONFIG, verbose=False
                )
                last_results = results
            except Exception as e:
                # Skip frame on tracking error
                out.write(frame)
                continue

        # ── Parse Detections ──
        ball_box = None
        ball_c = None
        players = []
        refs = []
        is_skipped = (fcount % skip_rate != 0)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cid = int(box.cls[0].item())
                
                # For ball: SKIP extraction on skipped frames to allow prediction to take over
                if cid == BALL_CLASS_ID and is_skipped:
                    continue
                    
                xy = box.xyxy[0].cpu().numpy()
                
                # Check safely if ID exists according to fix
                if box.id is not None and len(box.id) > 0:
                    rid = int(box.id[0].item())
                else:
                    if cid not in [BALL_CLASS_ID, REFEREE_CLASS_ID]:
                        continue
                    rid = None

                if cid == BALL_CLASS_ID:
                    ball_box = xy
                    ball_c = box_center(xy)
                elif cid in PLAYER_CLASS_IDS and rid is not None:
                    c = box_center(xy)
                    t_name = "Team 1" if cid in TEAM_1_CLASS_IDS else "Team 2"
                    sid = stabilizer.get(rid, c, t_name)
                    if sid is not None:
                        players.append((sid, xy, cid))
                elif cid == REFEREE_CLASS_ID:
                    refs.append(xy)

        # ── Stabilizer Cleanup ──
        actives = set()
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is not None and len(box.id) > 0:
                    actives.add(int(box.id[0].item()))
        stabilizer.cleanup(actives)

        # ── Ball Analytics ──
        ball_tracker.update(ball_box, ball_c, net_x)
        
        # ── Update local ball coords with tracker (handling prediction) ──
        ball_c = ball_tracker.current_pos
        ball_box = ball_tracker.current_box

        # ── Player Analytics ──
        player_tracker.begin_frame_possession()

        player_actions = {}
        for (sid, pb, cid) in players:
            info = player_tracker.update_player(sid, pb, cid, fcount, ball_tracker, W, H)
            player_actions[sid] = info

        player_tracker.finalize_frame_possession()

        # ── Block Detection ──
        block_events = player_tracker.detect_blocks(players, fcount, net_x)
        for bsid in block_events:
            gameplay.on_block(fcount, bsid)

        # ── Spike Events → Gameplay ──
        for (sid, pb, cid) in players:
            info = player_actions.get(sid, {})
            if info.get("action") == "SPIKE":
                zone = info.get("zone", 0)
                is_t1 = info.get("is_t1", True)
                gameplay.on_spike(fcount, sid, zone, is_t1, player_tracker=player_tracker)

                # Draw spike arrow
                c = box_center(pb)
                if ball_c is not None:
                    cv2.arrowedLine(frame,
                                   (int(c[0]), int(c[1])),
                                   (int(ball_c[0]), int(ball_c[1])),
                                   (0, 255, 255), 3, tipLength=0.15)

        # ── Gameplay State Machine ──
        gameplay.update(fcount, ball_tracker, player_tracker, net_x)

        # ═════════════════════════════════════════════════════════
        #                   FRAME DRAWING
        # ═════════════════════════════════════════════════════════

        # ── Players (Bounding Boxes) ──
        for (sid, pb, cid) in players:
            is_t1 = cid in TEAM_1_CLASS_IDS
            is_t2 = cid in TEAM_2_CLASS_IDS
            if is_t1 and not show_t1:
                continue
            if is_t2 and not show_t2:
                continue

            info = player_actions.get(sid, {})
            action = info.get("action", "")
            role = info.get("role", "") if opt_roles else ""
            dist = player_tracker.distances.get(sid, 0)
            is_poss = (player_tracker.current_poss_id == sid)

            # Get gameplay label (SET/PASS/BLOCK/SPIKE/SERVE) from engine
            gameplay_label = gameplay.get_player_label(sid)

            frame = draw_player_bbox(frame, sid, pb, cid, action, role, dist, is_poss,
                                     gameplay_label=gameplay_label)

        # ── Referee ──
        for rb in refs:
            frame = draw_referee(frame, rb)

        # ── Ball Trail ──
        if opt_ball_trail:
            frame = draw_ball_trail(frame, ball_tracker.get_trail_points())

        # ── Ball ──
        frame = draw_ball(frame, ball_tracker.current_box, ball_tracker.speed_px_sec)

        # ── Pass Arrow ──
        frame = draw_pass_arrow(frame, gameplay.pass_arrow, gameplay.pass_timer)

        # ── Scoreboard ──
        t1_poss_pct, t2_poss_pct = player_tracker.get_possession_pct(fps)
        frame = draw_scoreboard(
            frame,
            t1_score=gameplay.t1_score,
            t2_score=gameplay.t2_score,
            t1_spikes=player_tracker.t1_spikes,
            t2_spikes=player_tracker.t2_spikes,
            rally_n=gameplay.rally_number,
            match_time_str=format_time(fcount, fps),
            possession_pct=(t1_poss_pct, t2_poss_pct),
            state=gameplay.state
        )

        # ── Tactical Court Overlay ──
        if opt_court:
            frame = draw_court_overlay(
                frame, players, ball_c,
                setter_spiker_arrow=gameplay.setter_spiker_arrow,
                setter_spiker_timer=gameplay.setter_spiker_timer,
                player_labels=gameplay.player_labels
            )

        # ── Stats Panel ──
        if opt_stats_panel:
            frame = draw_stats_panel(
                frame, players,
                player_tracker.distances,
                player_tracker.jumps,
                player_tracker.spikes,
                player_tracker.blocks
            )

        # ═════════════════════════════════════════════════════════
        #                  STREAMLIT UI UPDATES
        # ═════════════════════════════════════════════════════════

        if fcount % fps == 0:
            t1_timeline_spikes.append(player_tracker.t1_spikes)
            t2_timeline_spikes.append(player_tracker.t2_spikes)

        # Top bar
        match_time = format_time(fcount, fps)
        time_ph.markdown(f"**⏱ {match_time}**")
        t1_info_ph.markdown(f"<span class='team-label team1-color'>TEAM 1</span><br>"
                            f"<span class='stat-label'>Spikes</span> <span class='stat-value'>{player_tracker.t1_spikes}</span> · "
                            f"<span class='stat-label'>Jumps</span> <span class='stat-value'>{player_tracker.t1_jumps}</span>",
                            unsafe_allow_html=True)
        score_ph.markdown(f"<div class='score-display' style='margin-top: 15px; font-size: 1.4rem;'>"
                          f"<span style='color:#64748b;'>RALLY {gameplay.rally_number}</span>"
                          f"</div>", unsafe_allow_html=True)
        t2_info_ph.markdown(f"<span class='team-label team2-color'>TEAM 2</span><br>"
                            f"<span class='stat-label'>Spikes</span> <span class='stat-value'>{player_tracker.t2_spikes}</span> · "
                            f"<span class='stat-label'>Jumps</span> <span class='stat-value'>{player_tracker.t2_jumps}</span>",
                            unsafe_allow_html=True)
        ball_info_ph.markdown(f"**🏐 {ball_tracker.speed_px_sec:.0f}** px/s")

        # Timeline
        pct = max(0, min(100, int((fcount / max(total, 1)) * 100)))
        timeline_ph.progress(pct, text=f"⏱ {match_time} — Frame {fcount}/{total}")

        # Live stats
        stat_players.metric("Players", len(players))
        stat_ball.metric("Ball Speed", f"{ball_tracker.speed_px_sec:.0f} px/s")
        stat_events.metric("Events", len(gameplay.event_log))
        stat_state.metric("State", gameplay.state)

        # Live sidebar
        poss_str = f"P{player_tracker.current_poss_id}" if player_tracker.current_poss_id else "—"
        poss_team = player_tracker.teams.get(player_tracker.current_poss_id, "")
        live_score_ph.markdown(
            f"Rally **{gameplay.rally_number}** | Exch: **{gameplay.rally_exchanges}**"
        )
        live_poss_ph.markdown(
            f"**Possession:** {poss_team} {poss_str}\n\n"
            f"T1: {t1_poss_pct:.1f}% | T2: {t2_poss_pct:.1f}%"
        )
        live_events_ph.markdown("```text\n" + "\n".join(gameplay.event_log[-15:]) + "\n```")

        # Write frame
        out.write(frame)
        if fcount % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ph.image(frame_rgb, width="stretch")

    # ══════════════════════════════════════════════════════════════
    #               POST-PROCESSING — FINALIZE
    # ══════════════════════════════════════════════════════════════
    cap.release()
    out.release()
    if gameplay.rally_on:
        gameplay.rally_history.append(gameplay.rally_exchanges)

    st.balloons()
    timeline_ph.progress(100, text="✅ Analysis Complete")

    # ══════════════════════════════════════════════════════════════
    #               TAB 1: VAR / REPLAY
    # ══════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## 📹Rally Review")

        if os.path.exists(OUTPUT_VIDEO_PATH):
            st.markdown("### 🎬 Processed Match Video")
            st.download_button(
                "📥 Download Processed Video (MP4)",
                open(OUTPUT_VIDEO_PATH, "rb").read(),
                "volleyball_analysis.mp4",
                "video/mp4",
                width="stretch"
            )

        st.markdown("### 📝 Full Event Log")
        if gameplay.event_log:
            st.code("\n".join(gameplay.event_log), language="text")
        else:
            st.info("No events recorded")

    # ══════════════════════════════════════════════════════════════
    #               TAB 2: TACTICS
    # ══════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## 📋 Tactical Analysis Suite")

        # ── Attack Zones ──
        st.markdown("### ⚔️ Attack Zone Distribution")
        fig_court = plot_court_diagram_with_zones(
            player_tracker.t1_attack_zones,
            player_tracker.t2_attack_zones
        )
        st.pyplot(fig_court)
        plt.close(fig_court)

        zc1, zc2 = st.columns(2)
        with zc1:
            st.markdown("#### Team 1 Attacks")
            fig_z1 = plot_attack_zones(player_tracker.t1_attack_zones, "T1 Spike Zones", '#3498db')
            st.pyplot(fig_z1)
            plt.close(fig_z1)
        with zc2:
            st.markdown("#### Team 2 Attacks")
            fig_z2 = plot_attack_zones(player_tracker.t2_attack_zones, "T2 Spike Zones", '#e74c3c')
            st.pyplot(fig_z2)
            plt.close(fig_z2)

        # ── Zone Heatmaps ──
        st.markdown("### 🔥 Zone Distribution (Offense vs Defense)")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown("#### T1 Offense")
            fig_h1 = plot_zone_heatmap(player_tracker.t1_off_positions, True, 'Blues', 'T1 Offense', W, H, net_x)
            st.pyplot(fig_h1)
            plt.close(fig_h1)
        with h2:
            st.markdown("#### T1 Defense")
            fig_h2 = plot_zone_heatmap(player_tracker.t1_def_positions, True, 'Blues', 'T1 Defense', W, H, net_x)
            st.pyplot(fig_h2)
            plt.close(fig_h2)
        with h3:
            st.markdown("#### T2 Offense")
            fig_h3 = plot_zone_heatmap(player_tracker.t2_off_positions, False, 'Reds', 'T2 Offense', W, H, net_x)
            st.pyplot(fig_h3)
            plt.close(fig_h3)
        with h4:
            st.markdown("#### T2 Defense")
            fig_h4 = plot_zone_heatmap(player_tracker.t2_def_positions, False, 'Reds', 'T2 Defense', W, H, net_x)
            st.pyplot(fig_h4)
            plt.close(fig_h4)

        
    # ══════════════════════════════════════════════════════════════
    #               TAB 3: PLAYERS
    # ══════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## 👥 Player Profiles & Statistics")

        t1_rows, t2_rows = player_tracker.get_player_stats(fps)

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("#### 🔵 Team 1 Players")
            if t1_rows:
                st.dataframe(t1_rows, width="stretch")
            else:
                st.info("No Team 1 data")
        with pc2:
            st.markdown("#### 🔴 Team 2 Players")
            if t2_rows:
                st.dataframe(t2_rows, width="stretch")
            else:
                st.info("No Team 2 data")

        # ── Top Performers ──
        st.markdown("### 🌟 Top Performers")
        performers = player_tracker.get_top_performers(fps)
        tp1, tp2, tp3, tp4 = st.columns(4)

        if "distance" in performers:
            sid, team, val = performers["distance"]
            tp1.metric("Most Distance", f"P{sid} ({team})", f"{val:.0f} px")
        if "jumps" in performers:
            sid, team, val = performers["jumps"]
            tp2.metric("Most Jumps", f"P{sid} ({team})", f"{val}")
        if "spikes" in performers:
            sid, team, val = performers["spikes"]
            tp3.metric("Most Spikes", f"P{sid} ({team})", f"{val}")
        if "possession" in performers:
            sid, team, val = performers["possession"]
            tp4.metric("Most Possession", f"P{sid} ({team})", f"{val:.1f}s")

        # ── Team Totals ──
        st.markdown("### 📊 Team Comparison")
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("#### 🔵 Team 1")
            st.metric("Total Jumps", player_tracker.t1_jumps)
            st.metric("Total Spikes", player_tracker.t1_spikes)
            st.metric("Total Distance", f"{player_tracker.t1_total_dist:.0f} px")
            t1_poss_time = sum(player_tracker.poss_frames[s] for s in player_tracker.teams
                               if player_tracker.teams[s] == "Team 1")
            st.metric("Possession", f"{t1_poss_time / fps:.1f}s")
            t1_blocks = sum(player_tracker.blocks[s] for s in player_tracker.teams
                            if player_tracker.teams[s] == "Team 1")
            st.metric("Blocks", t1_blocks)

        with tc2:
            st.markdown("#### 🔴 Team 2")
            st.metric("Total Jumps", player_tracker.t2_jumps)
            st.metric("Total Spikes", player_tracker.t2_spikes)
            st.metric("Total Distance", f"{player_tracker.t2_total_dist:.0f} px")
            t2_poss_time = sum(player_tracker.poss_frames[s] for s in player_tracker.teams
                               if player_tracker.teams[s] == "Team 2")
            st.metric("Possession", f"{t2_poss_time / fps:.1f}s")
            t2_blocks = sum(player_tracker.blocks[s] for s in player_tracker.teams
                            if player_tracker.teams[s] == "Team 2")
            st.metric("Blocks", t2_blocks)

    # ══════════════════════════════════════════════════════════════
    #               TAB 4: REPORTS
    # ══════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## 📊 Match Reports & Downloads")

        # ── Rally Summary ──
        st.markdown("### 🏁 Rally Summary")
        rally_stats = gameplay.get_rally_stats()
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Total Rallies", rally_stats["total"])
        rc2.metric("Avg Exchanges", f"{rally_stats['avg_exchanges']:.1f}")
        rc3.metric("Longest Rally", f"{rally_stats['max_exchanges']} exch")

        if rally_stats["history"]:
            fig_r = plot_rally_distribution(rally_stats["history"])
            st.pyplot(fig_r)
            plt.close(fig_r)

        # ── Ball Speed ──
        if ball_tracker.speeds_history:
            st.markdown("### 🏐 Ball Speed Analysis")
            bs1, bs2, bs3 = st.columns(3)
            bs1.metric("Avg Speed", f"{ball_tracker.get_avg_speed():.0f} px/s")
            bs2.metric("Max Speed", f"{ball_tracker.get_max_speed():.0f} px/s")
            bs3.metric("Measurements", len(ball_tracker.speeds_history))

            fig_s = plot_ball_speed_graph(ball_tracker.speeds_history)
            st.pyplot(fig_s)
            plt.close(fig_s)

        # ── Downloads ──
        st.markdown("### 📥 Download Reports")
        dl1, dl2, dl3 = st.columns(3)

        # CSV
        with dl1:
            csv_data = generate_csv(player_tracker, gameplay, ball_tracker, fps)
            st.download_button(
                "📄 Download CSV Report",
                csv_data.getvalue(),
                "volleyball_report.csv",
                "text/csv",
                width="stretch"
            )

        # PDF
        with dl2:
            try:
                pdf_data = generate_pdf_report(player_tracker, gameplay, ball_tracker, fps)
                st.download_button(
                    "📕 Download PDF Report",
                    pdf_data.getvalue(),
                    "volleyball_report.pdf",
                    "application/pdf",
                    width="stretch"
                )
            except Exception as e:
                st.warning(f"PDF generation error: {e}")

        # Video
        with dl3:
            if os.path.exists(OUTPUT_VIDEO_PATH):
                st.download_button(
                    "🎬 Download Video",
                    open(OUTPUT_VIDEO_PATH, "rb").read(),
                    "volleyball_analysis.mp4",
                    "video/mp4",
                    width="stretch"
                )

else:
    # ══════════════════════════════════════════════════════════════
    #                    LANDING PAGE
    # ══════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🏐 Volleyball Analytics Pro</h1>
        <p style="font-size: 1.2rem; color: #8080a0; max-width: 600px; margin: 0 auto;">
            Professional volleyball analysis platform with AI-powered detection,
            real-time tracking, and broadcast-quality analytics.
        </p>
        <div style="margin-top: 40px; display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <div style="background: rgba(26,26,62,0.6); border: 1px solid rgba(78,205,196,0.3);
                        border-radius: 16px; padding: 30px; width: 200px; text-align: center;">
                <div style="font-size: 2rem;">🎯</div>
                <div style="font-weight: 700; margin: 10px 0; color: #4ecdc4;">Detection</div>
                <div style="font-size: 0.85rem; color: #8080a0;">YOLOv8 + ByteTrack player & ball detection</div>
            </div>
            <div style="background: rgba(26,26,62,0.6); border: 1px solid rgba(78,205,196,0.3);
                        border-radius: 16px; padding: 30px; width: 200px; text-align: center;">
                <div style="font-size: 2rem;">📊</div>
                <div style="font-weight: 700; margin: 10px 0; color: #4ecdc4;">Analytics</div>
                <div style="font-size: 0.85rem; color: #8080a0;">Pass networks, attack zones, heatmaps</div>
            </div>
            <div style="background: rgba(26,26,62,0.6); border: 1px solid rgba(78,205,196,0.3);
                        border-radius: 16px; padding: 30px; width: 200px; text-align: center;">
                <div style="font-size: 2rem;">📋</div>
                <div style="font-weight: 700; margin: 10px 0; color: #4ecdc4;">Reports</div>
                <div style="font-size: 0.85rem; color: #8080a0;">PDF, CSV, and video export</div>
            </div>
            <div style="background: rgba(26,26,62,0.6); border: 1px solid rgba(78,205,196,0.3);
                        border-radius: 16px; padding: 30px; width: 200px; text-align: center;">
                <div style="font-size: 2rem;">🏆</div>
                <div style="font-weight: 700; margin: 10px 0; color: #4ecdc4;">Gameplay</div>
                <div style="font-size: 0.85rem; color: #8080a0;">Serve → Pass → Set → Spike state machine</div>
            </div>
        </div>
        <p style="margin-top: 40px; color: #6060a0;">
            Upload a video or click <strong>Start Analysis</strong> in the sidebar to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)
