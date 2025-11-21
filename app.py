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

# --- 1. FIXTURE & CREDIBILITY ENGINE ---
def process_fixtures_and_credibility(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    
    # Map Raw Strengths
    t_stats = {}
    for t in teams_data:
        t_stats[t['id']] = {
            'att_h': t['strength_attack_home'],
            'att_a': t['strength_attack_away'],
            'def_h': t['strength_defence_home'],
            'def_a': t['strength_defence_away']
        }
    
    # Schedule Containers
    # We store lists of strengths to calculate averages later
    team_sched = {t['id']: {
        'past_opp_att': [], 'past_opp_def': [], 
        'fut_opp': [], # List of dicts: {'id': opp_id, 'is_home': bool}
        'display': []
    } for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue
        h, a = f['team_h'], f['team_a']
        
        # Opponent Strengths
        h_opp_att = t_stats[a]['att_a']
        h_opp_def = t_stats[a]['def_a']
        a_opp_att = t_stats[h]['att_h']
        a_opp_def = t_stats[h]['def_h']

        if f['finished']:
            # PAST: Used to calculate Credibility
            team_sched[h]['past_opp_att'].append(h_opp_att)
            team_sched[h]['past_opp_def'].append(h_opp_def)
            team_sched[a]['past_opp_att'].append(a_opp_att)
            team_sched[a]['past_opp_def'].append(a_opp_def)
        else:
            # FUTURE: Used to calculate ROI Opportunity
            # We store the Opponent ID so we can look up their Credibility later
            team_sched[h]['fut_opp'].append({'id': a, 'is_home': True})
            team_sched[h]['display'].append(f"{team_map[a]}(H)")
            
            team_sched[a]['fut_opp'].append({'id': h, 'is_home': False})
            team_sched[a]['display'].append(f"{team_map[h]}(A)")

    # --- CALCULATE TEAM CREDIBILITY (THE "FRAUD CHECK") ---
    # Did this team perform against strong or weak opponents?
    team_credibility = {}
    avg_league_strength = 1080.0
    
    for tid in team_sched:
        past_att = team_sched[tid]['past_opp_att']
        past_def = team_sched[tid]['past_opp_def']
        
        # Attack Credibility: based on Opponent Defense Strength faced
        if past_def:
            avg_opp_def = sum(past_def) / len(past_def)
            # If faced Strong Def (1300), Credibility = 1.2 (Proven)
            # If faced Weak Def (1000), Credibility = 0.9 (Stat-padder)
            att_cred = avg_opp_def / avg_league_strength
        else:
            att_cred = 1.0
            
        # Defense Credibility: based on Opponent Attack Strength faced
        if past_att:
            avg_opp_att = sum(past_att) / len(past_att)
            def_cred = avg_opp_att / avg_league_strength
        else:
            def_cred = 1.0
            
        team_credibility[tid] = {'att_cred': att_cred, 'def_cred': def_cred}

    return team_sched, team_credibility, t_stats

def get_smart_fixture_score(schedule_list, limit, opponent_type, team_credibility, t_stats):
    """
    Calculates fixture ease (0-10) adjusting for Opponent Credibility.
    opponent_type: "att" (We are defending against them) or "def" (We are attacking them)
    """
    if not schedule_list: return 5.0
    
    subset = schedule_list[:limit]
    adjusted_strengths = []
    
    for match in subset:
        opp_id = match['id']
        is_home = match['is_home'] # If we are home, opp is away
        
        if opponent_type == "att":
            # We are Defender/GK. Evaluating Opponent Attack.
            raw_strength = t_stats[opp_id]['att_a'] if is_home else t_stats[opp_id]['att_h']
            credibility = team_credibility[opp_id]['att_cred']
            # Adjusted Threat = Raw * Credibility
            # High Raw + High Cred = Very High Threat.
            # High Raw + Low Cred = Moderate Threat.
            adj = raw_strength * credibility
            adjusted_strengths.append(adj)
            
        else: # opponent_type == "def"
            # We are Attacker. Evaluating Opponent Defense.
            raw_strength = t_stats[opp_id]['def_a'] if is_home else t_stats[opp_id]['def_h']
            credibility = team_credibility[opp_id]['def_cred']
            # Adjusted Solidity = Raw * Credibility
            adj = raw_strength * credibility
            adjusted_strengths.append(adj)
            
    # Calculate Score
    avg_adj_strength = sum(adjusted_strengths) / len(adjusted_strengths)
    # Normalize: 1400 (Hardest) -> 0, 900 (Easiest) -> 10
    score = 10 - ((avg_adj_strength - 900) / 500 * 10)
    return max(0, min(10, score))

def get_resistance_factor(past_strengths):
    if not past_strengths: return 1.0
    avg_strength = sum(past_strengths) / len(past_strengths)
    return avg_strength / 1080.0

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### Context-Aware Model (Fraud Detection Enabled)")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    
    # Run Engine
    team_schedule, team_credibility, t_stats = process_fixtures_and_credibility(fixtures, teams)
    
    # Team Structure
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    min_str = min(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: ((v - min_str) / (max_str - min_str)) * 10 for k,v in team_conceded.items()}

    # Normalization Maxima
    all_elements = pd.DataFrame(data['elements'])
    MAX_PPM = all_elements['points_per_game'].astype(float).max()
    MAX_CS = all_elements['clean_sheets_per_90'].astype(float).max()
    MAX_XGI = all_elements['expected_goal_involvements_per_90'].astype(float).max()
    
    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Analyze next:", [1, 5, 10], 
        format_func=lambda x: f"{x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Impact Weights")
    
    w_budget = st.sidebar.slider("Price Impact", 0.0, 1.0, 0.5, key="price_weight", on_change=reset_page)
    
    st.sidebar.divider()
    st.sidebar.subheader("Stat Importance")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Clean Sheet Ability", 0.0, 1.0, 1.0, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Clean Sheet Ability", 0.0, 1.0, 1.0, key="def_cs", on_change=reset_page)
        w_xgi_def = st.slider("Attacking Threat", 0.0, 1.0, 1.0, key="def_xgi", on_change=reset_page)
        w_ppm_def = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="def_ppm", on_change=reset_page)
        w_fix_def = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="def_fix", on_change=reset_page)
        def_weights = {'cs': w_cs_def, 'xgi': w_xgi_def, 'ppm': w_ppm_def, 'fix': w_fix_def}

    # 3. MIDFIELDERS
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_xgi_mid = st.slider("Goal/Assist Threat", 0.0, 1.0, 1.0, key="mid_xgi", on_change=reset_page)
        w_ppm_mid = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="mid_ppm", on_change=reset_page)
        w_fix_mid = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="mid_fix", on_change=reset_page)
        mid_weights = {'xgi': w_xgi_mid, 'ppm': w_ppm_mid, 'fix': w_fix_mid}

    # 4. FORWARDS
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_xgi_fwd = st.slider("Goal/Assist Threat", 0.0, 1.0, 1.0, key="fwd_xgi", on_change=reset_page)
        w_ppm_fwd = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="fwd_ppm", on_change=reset_page)
        w_fix_fwd = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="fwd_fix", on_change=reset_page)
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
            
            # --- SMART FIXTURE SCORING ---
            # Determine if we are defending (vs Opp Attack) or Attacking (vs Opp Defense)
            if pos_category in ["GK", "DEF"]:
                opp_type = "att" # We face opponent attack
                # Resistance: Did we face strong attacks in past?
                past_sched = team_schedule[tid]['past_opp_att'] 
            else:
                opp_type = "def" # We face opponent defense
                # Resistance: Did we face strong defenses in past?
                past_sched = team_schedule[tid]['past_opp_def']
            
            # Calculate Score with Credibility Check
            fut_sched = team_schedule[tid]['fut_opp']
            fixture_score = get_smart_fixture_score(fut_sched, horizon_option, opp_type, team_credibility, t_stats)
            
            # Past Context (Resistance)
            resistance_mult = get_resistance_factor(past_sched)
            
            display_fixtures = ", ".join(team_schedule[tid]['display'][:horizon_option])

            try:
                price = p['now_cost'] / 10.0
                price = 4.0 if price <= 0 else price
                
                raw_ppm = float(p['points_per_game'])
                
                # --- COMPOSITE THREAT CALCULATION ---
                minutes = float(p['minutes'])
                raw_xgi = float(p.get('expected_goal_involvements_per_90', 0))
                
                goals = float(p.get('goals_scored', 0))
                assists = float(p.get('assists', 0))
                actual_gi_per_90 = ((goals + assists) / minutes) * 90
                
                # Composite Threat: Blend xGI and Actual
                comp_threat = (raw_xgi * 0.6) + (actual_gi_per_90 * 0.4)
                adj_threat = comp_threat * resistance_mult
                score_threat = (adj_threat / MAX_XGI) * 10 if MAX_XGI > 0 else 0
                
                # --- DEFENSIVE CALCULATION ---
                if pos_category in ["GK", "DEF"]:
                    raw_cs = float(p['clean_sheets_per_90'])
                    raw_xgc = float(p.get('expected_goals_conceded_per_90', 0))
                    inv_xgc = max(0, 2.5 - raw_xgc)
                    
                    comp_cs = (raw_cs * 20) + (inv_xgc * 3) + (team_def_strength[tid] * 0.5)
                    
                    # Apply Past Context (Did they keep CS vs Elite attacks?)
                    adj_def = comp_cs * resistance_mult
                    score_def = min(10, (adj_def / 25.0) * 10)
                    stat_disp = raw_cs
                else:
                    score_def = 0
                    stat_disp = comp_threat

                # Common
                adj_ppm = raw_ppm * resistance_mult
                score_ppm = (adj_ppm / MAX_PPM) * 10
                score_fix = fixture_score

                # WEIGHTED SUM
                total_score = 0
                if pos_category == "GK":
                    total_score = (score_def * weights['cs']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                elif pos_category == "DEF":
                    total_score = (score_def * weights['cs']) + \
                                  (score_threat * weights['xgi']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                else: # MID/FWD
                    total_score = (score_threat * weights['xgi']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                
                # Price Adjustment
                price_divisor = price ** w_budget
                roi_index = total_score / price_divisor

                candidates.append({
                    "Name": p['web_name'],
                    "Team": team_names[tid],
                    "Price": price,
                    "Key Stat": stat_disp,
                    "Upcoming Fixtures": display_fixtures,
                    "PPM": raw_ppm,
                    "Fix Score": round(fixture_score, 1),
                    "Past Diff": round(resistance_mult, 2),
                    "Raw Score": roi_index
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df
        
        df['ROI Index'] = min_max_scale(df['Raw Score'])
        df = df.sort_values(by="ROI Index", ascending=False)
        
        return df[["ROI Index", "Name", "Team", "Price", "Key Stat", "Upcoming Fixtures", "PPM", "Fix Score", "Past Diff"]]

    # --- RENDER ---
    def render_tab(p_ids, pos_cat, weights):
        df = run_analysis(p_ids, pos_cat, weights)
        if df.empty: st.warning("No players found."); return

        items_per_page = 50
        total_pages = max(1, (len(df) + items_per_page - 1) // items_per_page)
        if st.session_state.page >= total_pages: st.session_state.page = total_pages - 1
        start, end = st.session_state.page * items_per_page, (st.session_state.page + 1) * items_per_page
        
        if pos_cat in ["GK", "DEF"]:
            stat_label = "CS/90"
            stat_help = "Clean Sheets per 90"
        else:
            stat_label = "Threat/90"
            stat_help = "Composite: xGI (60%) + Actual G/A (40%) per 90"
        
        st.dataframe(
            df.iloc[start:end], hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Key Stat": st.column_config.NumberColumn(stat_label, format="%.2f", help=stat_help),
                "Upcoming Fixtures": st.column_config.TextColumn("Opponents", width="medium"),
                "PPM": st.column_config.NumberColumn("Pts/G", format="%.1f"),
                "Fix Score": st.column_config.NumberColumn("Fix Rating", help="Adjusted for Opponent Credibility. (10=Easy, 0=Hard)"),
                "Past Diff": st.column_config.NumberColumn("Past Ctx", help=">1.0: Performed vs Strong Teams (Boost). <1.0: Performed vs Weak Teams (Penalty)."),
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
