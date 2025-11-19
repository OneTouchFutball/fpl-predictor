import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL Explosion Model v7", page_icon="🧨", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stDataFrame { width: 100%; }
    /* Custom Progress Bar Colors based on 1-10 scale */
    .stProgress > div > div > div > div { 
        background-image: linear-gradient(to right, #a83232, #d19630, #37a849); 
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data(ttl=600)
def load_data():
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(base_url + "bootstrap-static/").json()
        fixtures = requests.get(base_url + "fixtures/").json()
        return static, fixtures
    except:
        return None, None

def get_team_leakiness(static_data):
    """Calculates Team xGC per 90. Used for Clean Sheet Potential."""
    team_xgc = {t['id']: [] for t in static_data['teams']}
    for p in static_data['elements']:
        if p['element_type'] in [1, 2] and p['minutes'] > 450:
            try:
                xgc = float(p.get('expected_goals_conceded_per_90', 0))
                team_xgc[p['team']].append(xgc)
            except: continue
    
    team_strength = {}
    for t_id, values in team_xgc.items():
        team_strength[t_id] = sum(values) / len(values) if values else 1.5
    return team_strength

def get_10_game_schedule(static_data, fixture_data):
    """Maps Team ID to list of next 10 opponents."""
    teams = {t['id']: t['short_name'] for t in static_data['teams']}
    next_gw = next((e['id'] for e in static_data['events'] if e['is_next']), 1)
    schedule = {t_id: [] for t_id in teams}
    for f in fixture_data:
        if f['event'] and f['event'] >= next_gw:
            h, a = f['team_h'], f['team_a']
            if len(schedule[h]) < 10: schedule[h].append({"opp": teams[a], "diff": f['team_h_difficulty']})
            if len(schedule[a]) < 10: schedule[a].append({"opp": teams[h], "diff": f['team_a_difficulty']})
    return schedule

# --- 2. SCORING ENGINES (EQUIVALENCE TUNED) ---

def calc_attacker_raw_score(p, schedule_map, w_ppm, w_xgi, w_form, w_fix):
    """
    Raw Score Calculation for MID/FWD.
    """
    try:
        ppm = float(p['points_per_game'])
        xgi = float(p.get('expected_goal_involvements', 0)) # Top players ~12-15
        form = float(p['form'])
        
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # SCORING (Targeting raw max of ~25-30)
        s_ppm = ppm * 2.0          # 9.0 PPM -> 18.0
        s_xgi = xgi * 0.8          # 15 xGI -> 12.0
        s_fix = (5.0 - avg_diff) * 3.0 # Diff 2 -> 9.0
        s_form = form * 1.0        # Form 8 -> 8.0
        
        raw_score = (s_ppm * w_ppm) + (s_xgi * w_xgi) + (s_form * w_form) + (s_fix * w_fix)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return raw_score, sched_str, avg_diff
    except: return 0, "", 0

def calc_defender_raw_score(p, schedule_map, leakiness_map, w_ppm, w_cs, w_fix, w_att):
    """
    Raw Score Calculation for DEF/GKP.
    Tuned to output similar magnitude to Attackers.
    """
    try:
        ppm = float(p['points_per_game'])
        xgi = float(p.get('expected_goal_involvements', 0)) # Defenders usually < 3.0
        team_xgc = leakiness_map.get(p['team'], 1.5) # Range 0.8 to 2.5
        
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # SCORING
        s_ppm = ppm * 2.0 # 6.0 PPM -> 12.0
        
        # EQUIVALENCE LOGIC:
        # Attacker xGI max is ~15. 
        # Best Team xGC is ~0.8. (3.0 - 0.8) = 2.2.
        # To match Attacker xGI impact, we multiply Clean Sheet score by ~5.5
        s_cs = (3.0 - team_xgc) * 5.5  # 2.2 * 5.5 -> 12.1 (Comparable to Attacker xGI)
        
        s_fix = (5.0 - avg_diff) * 3.5 
        s_att = xgi * 2.5 # Bonus for Trent/Porro
        
        raw_score = (s_ppm * w_ppm) + (s_cs * w_cs) + (s_fix * w_fix) + (s_att * w_att)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return raw_score, sched_str, avg_diff
    except: return 0, "", 0

# --- 3. MAIN APP ---
def main():
    st.title("🧨 FPL Explosion Model v7")
    st.markdown("**Unified ROI Index (1-10 Scale)** | **Attackers vs Defenders Equivalence**")
    
    with st.spinner("Calibrating model data..."):
        static, fixtures = load_data()
        if not static: st.error("API Error"); return
        schedules = get_10_game_schedule(static, fixtures)
        leakiness = get_team_leakiness(static)
        teams = {t['id']: t['short_name'] for t in static['teams']}

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("⚙️ Model Weights")
    
    with st.sidebar.expander("⚔️ Attacker Weights", expanded=True):
        aw_ppm = st.slider("Points Per Match", 0.1, 1.0, 0.9)
        aw_xgi = st.slider("Total xGI", 0.1, 1.0, 0.7)
        aw_fix = st.slider("Fixture Ease (10 Gms)", 0.1, 1.0, 0.3)
        aw_form = st.slider("Form", 0.1, 1.0, 0.3)
    
    with st.sidebar.expander("🛡️ Defender Weights", expanded=True):
        dw_ppm = st.slider("Points Per Match", 0.1, 1.0, 0.8, key="dw1")
        dw_cs = st.slider("Team CS Potential", 0.1, 1.0, 0.7, help="Equivalent to Attacker xGI in this model", key="dw2")
        dw_fix = st.slider("Fixture Ease", 0.1, 1.0, 0.4, key="dw3")
        dw_att = st.slider("Attacking Threat", 0.1, 1.0, 0.3, key="dw4")
        
    min_mins = st.sidebar.number_input("Min Minutes Played", 0, 3000, 500)

    # --- PROCESS DATA & NORMALIZE SCORES ---
    all_players = []
    
    # 1. Calculate Raw Scores
    for p in static['elements']:
        if p['minutes'] < min_mins: continue
        
        price = p['now_cost'] / 10.0
        pos = p['element_type']
        raw_score = 0
        sched = ""
        category = ""
        
        if pos in [3, 4]: # ATTACK
            raw_score, sched, avg_diff = calc_attacker_raw_score(p, schedules, aw_ppm, aw_xgi, aw_form, aw_fix)
            category = "Attack"
        elif pos in [1, 2]: # DEFENSE
            raw_score, sched, avg_diff = calc_defender_raw_score(p, schedules, leakiness, dw_ppm, dw_cs, dw_fix, dw_att)
            category = "Defense"
            
        if raw_score > 0:
            all_players.append({
                "Name": p['web_name'],
                "Team": teams[p['team']],
                "Pos": {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}[pos],
                "Category": category,
                "Price": price,
                "PPM": float(p['points_per_game']),
                "Raw Score": raw_score,
                "Schedule": sched
            })

    df_all = pd.DataFrame(all_players)

    # 2. Normalize to 1-10 Scale (The "ROI Index")
    if not df_all.empty:
        max_raw = df_all['Raw Score'].max()
        # Formula: 1 + (Raw / Max * 9) -> Ensures scale is 1.0 to 10.0
        df_all['ROI Index'] = 1.0 + ((df_all['Raw Score'] / max_raw) * 9.0)
    else:
        st.error("No players found. Check API connection.")
        return

    # --- TABS ---
    tab_exp, tab_att, tab_def, tab_val = st.tabs([
        "💥 Explosion Potential (All)", 
        "⚔️ Attackers", 
        "🛡️ Defenders", 
        "💎 Best Value"
    ])

    # --- TAB 1: UNIFIED EXPLOSION POTENTIAL ---
    with tab_exp:
        st.markdown("### 💥 The Unified Leaderboard")
        st.info("Ranking Goalkeepers, Defenders, Midfielders, and Forwards on one single **Explosion Scale (1-10)**.")
        
        # Sort by ROI Index
        df_exp = df_all.sort_values("ROI Index", ascending=False).head(50)
        
        st.dataframe(df_exp, hide_index=True, use_container_width=True, column_config={
            "ROI Index": st.column_config.ProgressColumn(
                "Expected ROI Index", 
                help="Scale 1-10. 10 = Elite Explosion Potential.",
                format="%.1f", 
                min_value=1, 
                max_value=10
            ),
            "Pos": st.column_config.TextColumn("Pos"),
            "PPM": st.column_config.NumberColumn("Pts/Match", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 2: ATTACKERS ---
    with tab_att:
        df_att = df_all[df_all['Category'] == "Attack"].copy()
        df_att = df_att.sort_values("ROI Index", ascending=False).head(50)
        
        st.dataframe(df_att, hide_index=True, use_container_width=True, column_config={
            "ROI Index": st.column_config.ProgressColumn("Expected ROI Index", format="%.1f", min_value=1, max_value=10),
            "PPM": st.column_config.NumberColumn("Pts/Match", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 3: DEFENDERS ---
    with tab_def:
        df_def = df_all[df_all['Category'] == "Defense"].copy()
        df_def = df_def.sort_values("ROI Index", ascending=False).head(50)
        
        st.dataframe(df_def, hide_index=True, use_container_width=True, column_config={
            "ROI Index": st.column_config.ProgressColumn("Expected ROI Index", format="%.1f", min_value=1, max_value=10),
            "PPM": st.column_config.NumberColumn("Pts/Match", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 4: VALUE ENGINE ---
    with tab_val:
        st.markdown("### ⚖️ Weighted Decision Engine")
        st.info("Use the sliders to find players based on **Performance (ROI Index)** vs **Cost (Price)**.")
        
        c1, c2 = st.columns(2)
        w_roi_imp = c1.slider("Importance: Expected ROI Index", 0, 100, 70)
        w_price_imp = c2.slider("Importance: Low Price", 0, 100, 30)
        
        df_val = df_all.copy()
        
        # 1. Normalize ROI Index (0 to 1) for calculation
        # It's already 1-10, so (Val - 1) / 9
        df_val['n_roi'] = (df_val['ROI Index'] - 1) / 9
        
        # 2. Normalize Price (0 to 1) - Inverted
        min_p, max_p = df_val['Price'].min(), df_val['Price'].max()
        df_val['n_price'] = 1 - ((df_val['Price'] - min_p) / (max_p - min_p))
        
        # 3. Weighted Sum
        df_val['Value Score'] = (df_val['n_roi'] * w_roi_imp) + (df_val['n_price'] * w_price_imp)
        
        df_val = df_val.sort_values("Value Score", ascending=False).head(50)
        
        cols = ["Name", "Team", "Pos", "Price", "PPM", "Value Score", "ROI Index", "Schedule"]
        st.dataframe(df_val[cols], hide_index=True, use_container_width=True, column_config={
            "Value Score": st.column_config.ProgressColumn("Algorithm Score", format="%.0f"),
            "ROI Index": st.column_config.NumberColumn("Exp. ROI (1-10)", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

if __name__ == "__main__":
    main()
