import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Predictor 25/26", page_icon="⚽", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 8px 8px 0 0; padding: 10px 20px;
        font-size: 18px; font-weight: 700; color: #4a4a4a;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #e0e2e6; color: #1f77b4; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff; border-top: 3px solid #00cc00;
        color: #00cc00; box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    .stButton button { width: 100%; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 0
def reset_page(): st.session_state.page = 0

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# --- DATA LOADING ---
@st.cache_data(ttl=1800)
def load_data():
    try:
        bootstrap = requests.get(f"{API_BASE}/bootstrap-static/").json()
    except:
        st.error("API Error: Could not fetch static data.")
        return None, None
    try:
        fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    except:
        st.error("API Error: Could not fetch fixtures.")
        return bootstrap, None
    return bootstrap, fixtures

# --- FIXTURE ENGINE (CONTEXT AWARE) ---
def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    
    # Map Team Strengths
    t_stats = {}
    for t in teams_data:
        t_stats[t['id']] = {
            'att_h': t['strength_attack_home'],
            'att_a': t['strength_attack_away'],
            'def_h': t['strength_defence_home'],
            'def_a': t['strength_defence_away']
        }
    
    # League Averages (Approximate)
    avg_att_strength = 1080.0
    avg_def_strength = 1080.0
    
    team_sched = {t['id']: {'future_opp_att': [], 'future_opp_def': [], 'display': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time'] or f['finished']: continue

        h, a = f['team_h'], f['team_a']
        
        # Store Opponent Strengths
        # For Home Team: Opponent is Away
        team_sched[h]['future_opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['future_opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        # For Away Team: Opponent is Home
        team_sched[a]['future_opp_att'].append(t_stats[h]['att_h'])
        team_sched[a]['future_opp_def'].append(t_stats[h]['def_h'])
        team_sched[a]['display'].append(f"{team_map[h]}(A)")

    return team_sched, avg_att_strength, avg_def_strength

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### Contextual Model: Risk vs Reward")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule, league_avg_att, league_avg_def = process_fixtures(fixtures, teams)
    
    # Team Defense Strength (0-10 Scale)
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    min_str = min(team_conceded.values()) if team_conceded else 1
    # Invert: High Strength ID = Good Defense (Rank 10)
    team_def_strength = {k: ((v - min_str) / (max_str - min_str)) * 10 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Settings")
    horizon_option = st.sidebar.selectbox(
        "Analyze next:", [1, 5, 10], 
        format_func=lambda x: f"{x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )
    
    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 250, key="min_mins", on_change=reset_page)

    # --- ANALYSIS ENGINE ---
    def run_analysis(player_type_ids, pos_category):
        candidates = []

        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue
            tid = p['team']
            
            # DATA EXTRACTION
            opp_att_strengths = team_schedule[tid]['future_opp_att'][:horizon_option]
            opp_def_strengths = team_schedule[tid]['future_opp_def'][:horizon_option]
            fixtures_disp = ", ".join(team_schedule[tid]['display'][:horizon_option])
            
            if not opp_att_strengths: continue

            try:
                # 1. BASE STATS
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                price = 4.0 if price <= 0 else price
                
                # 2. CONTEXT CALCULATIONS
                
                if pos_category in ["GK", "DEF"]:
                    # --- DEFENSIVE LOGIC ---
                    # Goal: Punish players heavily if they face Elite Attacks (City/Arsenal)
                    
                    cs_per_90 = float(p['clean_sheets_per_90'])
                    xgc_per_90 = float(p.get('expected_goals_conceded_per_90', 0))
                    
                    # Capability Score (0-10): How good is this player/team at defending normally?
                    t_def = team_def_strength[tid]
                    # xGC is inverted (0.5 is great, 2.0 is bad). Scale: 2.5 - xGC
                    player_skill = (cs_per_90 * 25) + (max(0, 2.5 - xgc_per_90) * 3)
                    def_capability = (player_skill * 0.6) + (t_def * 0.4)
                    
                    # FIXTURE IMPACT (The "Wipeout" Logic)
                    avg_opp_att = sum(opp_att_strengths) / len(opp_att_strengths)
                    
                    # POWER LAW RATIO
                    # If League Avg (1080) / Opp Att (1350) -> 0.8
                    # 0.8 ^ 4 = 0.40 (Severe penalty for playing City)
                    # 0.8 ^ 1 = 0.80 (Too gentle)
                    # Using Power of 4 to simulate "probability of CS loss"
                    fixture_multiplier = (league_avg_att / avg_opp_att) ** 4
                    
                    # Apply Context
                    context_score = def_capability * fixture_multiplier
                    
                    stat_disp = cs_per_90
                    
                else:
                    # --- ATTACKING LOGIC ---
                    xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    
                    # Capability Score
                    att_capability = xgi * 10 # Scale xGI (0.8 -> 8.0)
                    
                    # Fixture Impact
                    # We want Weak Opponent Defense.
                    avg_opp_def = sum(opp_def_strengths) / len(opp_def_strengths)
                    
                    # Ratio: League Avg (1080) / Opp Def (1350 - Strong) = 0.8
                    # Ratio: League Avg (1080) / Opp Def (1000 - Weak) = 1.08
                    # Attackers are less sensitive to fixtures than Defenders are to Clean Sheets.
                    # Using Power of 2.
                    fixture_multiplier = (league_avg_def / avg_opp_def) ** 2
                    
                    context_score = att_capability * fixture_multiplier
                    
                    stat_disp = xgi

                # 3. FINAL SCORE (Balance)
                # User request: Equal weight to PPM and Context
                # We normalize Context Score roughly to PPM scale (0-10)
                final_score = (context_score * 0.5) + (ppm * 0.5)
                
                # 4. ROI INDEX
                roi_index = final_score / price

                # Formatting
                status_icon = "✅" if p['status'] == 'a' else ("⚠️" if p['status'] == 'd' else "❌")

                candidates.append({
                    "Name": f"{status_icon} {p['web_name']}",
                    "Team": team_names[tid],
                    "Price": price,
                    "Key Stat": stat_disp,
                    "Upcoming Fixtures": fixtures_disp,
                    "PPM": ppm,
                    "Context Score": round(context_score, 1),
                    "ROI Index": roi_index
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df
        
        # Normalize ROI for visuals (0-10 scale)
        df['ROI Index'] = min_max_scale(df['ROI Index'])
        
        return df.sort_values(by="ROI Index", ascending=False)

    # --- RENDER TABS ---
    def render_tab(p_ids, pos_cat):
        df = run_analysis(p_ids, pos_cat)
        if df.empty: st.warning("No players found."); return

        items_per_page = 50
        total_pages = max(1, (len(df) + items_per_page - 1) // items_per_page)
        if st.session_state.page >= total_pages: st.session_state.page = total_pages - 1
        start, end = st.session_state.page * items_per_page, (st.session_state.page + 1) * items_per_page
        
        stat_label = "CS/90" if pos_cat in ["GK", "DEF"] else "xGI/90"
        
        st.dataframe(
            df.iloc[start:end], hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Key Stat": st.column_config.NumberColumn(stat_label, format="%.2f"),
                "Upcoming Fixtures": st.column_config.TextColumn("Opponents", width="medium"),
                "PPM": st.column_config.NumberColumn("Pts/G", format="%.1f"),
                "Context Score": st.column_config.NumberColumn(
                    "Ctx Rating", 
                    format="%.1f", 
                    help="Performance adjusted for Opponent Strength. (Defenders severely punished for facing Elite Attacks)"
                ),
            }
        )
        c1, _, c3 = st.columns([1, 2, 1])
        if c1.button("⬅️ Previous", key=f"p_{pos_cat}"): st.session_state.page -= 1; st.rerun()
        if c3.button("Next ➡️", key=f"n_{pos_cat}"): st.session_state.page += 1; st.rerun()

    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs(["🧤 GOALKEEPERS", "🛡️ DEFENDERS", "⚔️ MIDFIELDERS", "⚽ FORWARDS"])
    with tab_gk: render_tab([1], "GK")
    with tab_def: render_tab([2], "DEF")
    with tab_mid: render_tab([3], "MID")
    with tab_fwd: render_tab([4], "FWD")

if __name__ == "__main__":
    main()
