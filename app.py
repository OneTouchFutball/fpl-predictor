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
        # Home Team faces Away Stats
        team_sched[h]['future_opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['future_opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        # Away Team faces Home Stats
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
    st.markdown("### Contextual Model with User Controls")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule, league_avg_att, league_avg_def = process_fixtures(fixtures, teams)
    
    # Team Defense Strength (0-10 Scale)
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    min_str = min(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: ((v - min_str) / (max_str - min_str)) * 10 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Analyze next:", [1, 5, 10], 
        format_func=lambda x: f"{x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Model Weights")
    
    # GLOBAL PRICE WEIGHT
    w_budget = st.sidebar.slider("Price Importance", 0.0, 1.0, 0.5, key="price_weight", on_change=reset_page)
    
    st.sidebar.divider()
    st.sidebar.subheader("Position Adjustments")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Clean Sheet Ability", 0.1, 1.0, 0.6, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Impact", 0.1, 1.0, 0.8, help="High = Penalize hard games heavily. Low = Trust the player.", key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Clean Sheet Ability", 0.1, 1.0, 0.6, key="def_cs", on_change=reset_page)
        w_xgi_def = st.slider("Attacking Threat", 0.1, 1.0, 0.4, key="def_xgi", on_change=reset_page)
        w_ppm_def = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="def_ppm", on_change=reset_page)
        w_fix_def = st.slider("Fixture Impact", 0.1, 1.0, 0.8, help="High = Wipe out CS potential vs Elite attacks.", key="def_fix", on_change=reset_page)
        def_weights = {'cs': w_cs_def, 'xgi': w_xgi_def, 'ppm': w_ppm_def, 'fix': w_fix_def}

    # 3. MIDFIELDERS
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_xgi_mid = st.slider("Attacking Threat", 0.1, 1.0, 0.7, key="mid_xgi", on_change=reset_page)
        w_ppm_mid = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="mid_ppm", on_change=reset_page)
        w_fix_mid = st.slider("Fixture Impact", 0.1, 1.0, 0.6, key="mid_fix", on_change=reset_page)
        mid_weights = {'xgi': w_xgi_mid, 'ppm': w_ppm_mid, 'fix': w_fix_mid}

    # 4. FORWARDS
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_xgi_fwd = st.slider("Attacking Threat", 0.1, 1.0, 0.8, key="fwd_xgi", on_change=reset_page)
        w_ppm_fwd = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="fwd_ppm", on_change=reset_page)
        w_fix_fwd = st.slider("Fixture Impact", 0.1, 1.0, 0.6, key="fwd_fix", on_change=reset_page)
        fwd_weights = {'xgi': w_xgi_fwd, 'ppm': w_ppm_fwd, 'fix': w_fix_fwd}

    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 250, key="min_mins", on_change=reset_page)

    # --- ANALYSIS ENGINE ---
    def run_analysis(player_type_ids, pos_category, weights):
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
                    cs_per_90 = float(p['clean_sheets_per_90'])
                    xgi_val = float(p.get('expected_goal_involvements_per_90', 0))
                    xgc_per_90 = float(p.get('expected_goals_conceded_per_90', 0))
                    
                    # Capability (Skill) Score
                    # Invert xGC (Scale approx: 2.5 - xGC)
                    # Logic: Skill = CS Ability + xGC Ability + Team Structure
                    skill_score = (cs_per_90 * 20) + (max(0, 2.5 - xgc_per_90) * 3) + (team_def_strength[tid] * 0.5)
                    
                    # Add Attacking Threat (xGI) for Defenders if weighted
                    # We use the user's w_xgi slider here if it exists (for DEF)
                    attack_boost = 0
                    if pos_category == "DEF":
                        attack_boost = (xgi_val * 10) * weights['xgi']

                    # CONTEXT: Fixture Multiplier
                    avg_opp_att = sum(opp_att_strengths) / len(opp_att_strengths)
                    
                    # Raw Ratio
                    base_ratio = league_avg_att / avg_opp_att # 0.8 for Hard, 1.1 for Easy
                    
                    # Power Law adjusted by USER SLIDER (w_fix)
                    # If w_fix is 1.0 -> Power 4 (Full Wipeout)
                    # If w_fix is 0.0 -> Power 0 (Multiplier = 1.0, No Impact)
                    power_factor = 4.0 * weights['fix']
                    fixture_mult = base_ratio ** power_factor
                    
                    # Final Score Construction
                    # (Skill * Weight * Fixture) + (Attack * Weight) + (Form * Weight)
                    # Note: Attack boost is less affected by opponent attack strength (CS wipeout doesn't kill goal threat)
                    context_score = (skill_score * weights['cs'] * fixture_mult) + \
                                    attack_boost + \
                                    (ppm * weights['ppm'])
                    
                    stat_disp = cs_per_90
                    
                else:
                    # --- ATTACKING LOGIC ---
                    xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    
                    # Capability
                    skill_score = xgi * 10 
                    
                    # Fixture Multiplier (Opponent Defense)
                    avg_opp_def = sum(opp_def_strengths) / len(opp_def_strengths)
                    base_ratio = league_avg_def / avg_opp_def
                    
                    # Power Law for Attackers (Usually less sensitive than Defenders)
                    # Max Power 2.0 adjusted by User Slider
                    power_factor = 2.0 * weights['fix']
                    fixture_mult = base_ratio ** power_factor
                    
                    # Final Score
                    context_score = (skill_score * weights['xgi'] * fixture_mult) + \
                                    (ppm * weights['ppm'])
                    
                    stat_disp = xgi

                # 3. ROI INDEX
                roi_index = context_score / price
                
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
        
        # Normalize ROI
        df['ROI Index'] = min_max_scale(df['ROI Index'])
        
        return df.sort_values(by="ROI Index", ascending=False)

    # --- RENDER TABS ---
    def render_tab(p_ids, pos_cat, weights):
        df = run_analysis(p_ids, pos_cat, weights)
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
                "Context Score": st.column_config.NumberColumn("Ctx Score", format="%.1f", help="Combined Score before Price adjustment."),
            }
        )
        c1, _, c3 = st.columns([1, 2, 1])
        if c1.button("⬅️ Previous", key=f"p_{pos_cat}"): st.session_state.page -= 1; st.rerun()
        if c3.button("Next ➡️", key=f"n_{pos_cat}"): st.session_state.page += 1; st.rerun()

    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs(["🧤 GOALKEEPERS", "🛡️ DEFENDERS", "⚔️ MIDFIELDERS", "⚽ FORWARDS"])
    with tab_gk: render_tab([1], "GK", gk_weights)
    with tab_def: render_tab([2], "DEF", def_weights)
    with tab_mid: render_tab([3], "MID", mid_weights)
    with tab_fwd: render_tab([4], "FWD", fwd_weights)

if __name__ == "__main__":
    main()
