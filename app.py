import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Hybrid 25/26", page_icon="🧬", layout="wide")

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

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# =========================================
# 1. AI DATA INFRASTRUCTURE
# =========================================

def download_training_data():
    """Downloads 5 years of history for AI Training if missing."""
    if os.path.exists("fpl_5_year_history.csv"):
        return pd.read_csv("fpl_5_year_history.csv")
        
    status = st.empty()
    status.info("⏳ Initializing: Downloading historical data (One-time setup)...")
    
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_data = []
    
    for season in seasons:
        try:
            url = f"{base_url}/{season}/gws/merged_gw.csv"
            r = requests.get(url)
            if r.status_code == 200:
                df = pd.read_csv(io.BytesIO(r.content), on_bad_lines='skip', low_memory=False)
                
                cols = ['minutes', 'total_points', 'was_home', 'clean_sheets', 
                        'goals_conceded', 'expected_goals', 'expected_assists', 
                        'influence', 'creativity', 'threat', 'value', 'element_type']
                existing = [c for c in cols if c in df.columns]
                df = df[existing]
                all_data.append(df)
        except: pass
        
    if all_data:
        master = pd.concat(all_data)
        master.fillna(0, inplace=True)
        master.to_csv("fpl_5_year_history.csv", index=False)
        status.empty()
        return master
    return None

@st.cache_resource
def train_ai_model():
    df = download_training_data()
    if df is None: return None, None, None
    
    # Filter for meaningful minutes to teach AI what "Starters" look like
    df = df[df['minutes'] > 60].copy() 
    
    features = ['value', 'element_type', 'was_home', 'expected_goals', 'expected_assists', 
                'clean_sheets', 'goals_conceded', 'influence', 'creativity', 'threat']
    
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    y = df['total_points']
    
    # HistGradientBoosting is robust and handles non-linear patterns well
    model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
    model.fit(X, y)
    
    # Calculate Max Points for Normalization (99th percentile to ignore wild outliers)
    max_ai_pts = df['total_points'].quantile(0.99)
    
    return model, valid_features, max_ai_pts

# =========================================
# 2. LIVE DATA & FIXTURE ENGINE
# =========================================

@st.cache_data(ttl=1800)
def load_live_data():
    static = requests.get(f"{API_BASE}/bootstrap-static/").json()
    fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    return static, fixtures

def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    # Map Team Strengths
    t_stats = {t['id']: {'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                         'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']} 
               for t in teams_data}
    
    # Setup Containers
    team_sched = {t['id']: {'fut_opp_att': [], 'fut_opp_def': [], 'display': []} for t in teams_data}
    
    for f in fixtures:
        if not f['kickoff_time'] or f['finished']: continue
        h, a = f['team_h'], f['team_a']
        
        # Home Team faces Away Stats
        team_sched[h]['fut_opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['fut_opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        # Away Team faces Home Stats
        team_sched[a]['fut_opp_att'].append(t_stats[h]['att_h'])
        team_sched[a]['fut_opp_def'].append(t_stats[h]['def_h'])
        team_sched[a]['display'].append(f"{team_map[h]}(A)")
        
    return team_sched, 1080.0 # Approximate League Avg Strength

def get_fixture_multiplier(schedule_list, league_avg, limit, mode="def"):
    # Calculates Multiplier. > 1.0 is Good. < 1.0 is Bad.
    if not schedule_list: return 1.0
    subset = schedule_list[:limit]
    avg_strength = sum(subset) / len(subset)
    
    # Ratio: League / Opponent
    # If Opponent is Hard (1300), Ratio is Low (0.8)
    ratio = league_avg / avg_strength
    
    # Defenders are more sensitive to hard games (CS wipeout)
    power = 4.0 if mode == "def" else 2.0 
    return ratio ** power

def get_display_score(schedule_list, limit):
    # 0-10 Visual Score
    if not schedule_list: return 5.0
    avg = sum(schedule_list[:limit]) / len(schedule_list[:limit])
    # Normalize: 1350 (Hard) -> 0, 950 (Easy) -> 10
    return max(0, min(10, 10 - ((avg - 950)/400 * 10)))

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# =========================================
# 3. HYBRID APP LOGIC
# =========================================

def main():
    st.title("🧬 FPL Pro: Hybrid Intelligence")
    st.markdown("### AI Baseline + Form + Contextual Fixtures")

    # 1. Train AI
    with st.spinner("Consulting AI & Downloading Data..."):
        model, ai_cols, max_ai_pts = train_ai_model()
    
    if model is None:
        st.error("Data Error. Please refresh or check connection.")
        return

    # 2. Load Live
    static, fixtures = load_live_data()
    teams = static['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_sched, avg_str = process_fixtures(fixtures, teams)
    
    # 3. Prepare Player Data
    df = pd.DataFrame(static['elements'])
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 2.0] # Must be active
    
    # AI Input Prep (Normalize to per-match average for prediction)
    ai_input = pd.DataFrame()
    ai_input['value'] = df['now_cost']
    ai_input['element_type'] = df['element_type']
    ai_input['was_home'] = 0.5 # Neutral assumption
    
    stat_map = {
        'expected_goals': 'expected_goals_per_90',
        'expected_assists': 'expected_assists_per_90',
        'clean_sheets': 'clean_sheets_per_90',
        'goals_conceded': 'goals_conceded_per_90',
        'influence': 'influence', 'creativity': 'creativity', 'threat': 'threat'
    }
    for train_col, api_col in stat_map.items():
        if 'per_90' in api_col:
            ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0)
        else:
            # Divide totals by games played
            ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']
            
    # --- AI PREDICTION ---
    # The AI predicts "Expected Points" based on stats
    df['AI_Points'] = model.predict(ai_input[ai_cols])
    
    # --- USER INTERFACE ---
    
    st.sidebar.header("🔮 Horizon")
    horizon = st.sidebar.selectbox("Lookahead", [1, 5, 10], format_func=lambda x: f"Next {x} Matches")
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ The Balance")
    st.sidebar.info("Defaults optimized to balance AI Accuracy (60%) with Current Form (40%).")
    
    w_budget = st.sidebar.slider("Price Sensitivity", 0.0, 1.0, 0.5, help="0=Ignore Price, 1=Value Only")
    
    st.sidebar.subheader("Position Strategy")
    
    # 1. GK
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_ai_gk = st.slider("Trust AI (Stats)", 0.0, 1.0, 0.6, key="gk_ai")
        w_form_gk = st.slider("Trust Form (PPM)", 0.0, 1.0, 0.4, key="gk_form")
        w_fix_gk = st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="gk_fix")
        w_gk = {'ai': w_ai_gk, 'form': w_form_gk, 'fix': w_fix_gk}

    # 2. DEF
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_ai_def = st.slider("Trust AI (Stats)", 0.0, 1.0, 0.6, key="def_ai")
        w_form_def = st.slider("Trust Form (PPM)", 0.0, 1.0, 0.4, key="def_form")
        w_xgi_def = st.slider("Attacking Bonus", 0.0, 1.0, 0.3, key="def_xgi")
        w_fix_def = st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="def_fix")
        w_def = {'ai': w_ai_def, 'form': w_form_def, 'xgi': w_xgi_def, 'fix': w_fix_def}

    # 3. MID
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_ai_mid = st.slider("Trust AI (Stats)", 0.0, 1.0, 0.6, key="mid_ai")
        w_form_mid = st.slider("Trust Form (PPM)", 0.0, 1.0, 0.4, key="mid_form")
        w_fix_mid = st.slider("Fixture Impact", 0.0, 1.0, 0.8, key="mid_fix")
        w_mid = {'ai': w_ai_mid, 'form': w_form_mid, 'fix': w_fix_mid}

    # 4. FWD
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_ai_fwd = st.slider("Trust AI (Stats)", 0.0, 1.0, 0.6, key="fwd_ai")
        w_form_fwd = st.slider("Trust Form (PPM)", 0.0, 1.0, 0.4, key="fwd_form")
        w_fix_fwd = st.slider("Fixture Impact", 0.0, 1.0, 0.8, key="fwd_fix")
        w_fwd = {'ai': w_ai_fwd, 'form': w_form_fwd, 'fix': w_fix_fwd}

    st.sidebar.divider()
    min_mins = st.sidebar.slider("Min Minutes", 0, 2500, 400)

    # --- HYBRID ENGINE ---
    def run_hybrid_engine(p_ids, cat, w):
        cands = []
        subset = df[df['element_type'].isin(p_ids) & (df['minutes'] >= min_mins)]
        
        # Dynamic Max for Normalization (Adapts to current season form)
        MAX_PPM = subset['points_per_game'].astype(float).max()
        
        for _, row in subset.iterrows():
            tid = row['team']
            
            # 1. GET CONTEXT
            if cat in ["GK", "DEF"]:
                sched = team_sched[tid]['fut_opp_att']
                mode = "def"
            else:
                sched = team_sched[tid]['fut_opp_def']
                mode = "att"
                
            fix_mult = get_fixture_multiplier(sched, avg_str, horizon, mode)
            fix_score_display = get_display_score(sched, horizon)
            fix_display = ", ".join(team_sched[tid]['display'][:horizon])
            
            # 2. HYBRID CALCULATION
            
            # A. AI Component (0-10)
            # "Statistical Ability"
            score_ai = (row['AI_Points'] / max_ai_pts) * 10
            
            # B. Form Component (0-10)
            # "Current Output" - This captures the Mbeumo Factor
            raw_ppm = float(row['points_per_game'])
            score_form = (raw_ppm / MAX_PPM) * 10
            
            # C. Manual Bonus
            score_bonus = 0
            if cat == "DEF":
                xgi = float(row['expected_goal_involvements_per_90'])
                score_bonus = (xgi * 10) * w['xgi']
            
            # D. The Blend
            # Combine AI + Form + Bonus
            base_score = (score_ai * w['ai']) + (score_form * w['form']) + score_bonus
            
            # E. Apply Context
            # Blend the fixture multiplier based on user preference
            eff_mult = 1.0 + (fix_mult - 1.0) * w['fix']
            final_score = base_score * eff_mult
            
            # 3. ROI
            price = row['now_cost'] / 10.0
            price_div = price ** w_budget
            roi = final_score / price_div
            
            stat_disp = float(row['clean_sheets_per_90']) if cat in ["GK", "DEF"] else float(row['expected_goal_involvements_per_90'])

            cands.append({
                "Name": row['web_name'],
                "Team": team_names[tid],
                "Price": price,
                "Key Stat": stat_disp,
                "Upcoming": fix_display,
                "AI Base": round(row['AI_Points'], 2),
                "PPM": raw_ppm,
                "Fix Rate": round(fix_score_display, 1),
                "Raw ROI": roi
            })
            
        res = pd.DataFrame(cands)
        if res.empty: return res
        
        res['ROI Index'] = min_max_scale(res['Raw ROI'])
        return res.sort_values(by="ROI Index", ascending=False)

    # --- RENDER ---
    def render(p_ids, cat, w):
        d = run_hybrid_engine(p_ids, cat, w)
        if d.empty: st.write("No players."); return
        
        stat_lbl = "CS/90" if cat in ["GK", "DEF"] else "xGI/90"
        
        st.dataframe(
            d.head(50), hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Upcoming": st.column_config.TextColumn("Opponents", width="medium"),
                "AI Base": st.column_config.NumberColumn("AI Exp", help="Projected points by AI based on 5 years of data"),
                "PPM": st.column_config.NumberColumn("Form", help="Actual Points Per Match this season"),
                "Fix Rate": st.column_config.NumberColumn("Fix Rating", help="10=Easy, 0=Hard"),
                "Key Stat": st.column_config.NumberColumn(stat_lbl, format="%.2f")
            }
        )

    t1, t2, t3, t4 = st.tabs(["🧤 GK", "🛡️ DEF", "⚔️ MID", "⚽ FWD"])
    with t1: render([1], "GK", w_gk)
    with t2: render([2], "DEF", w_def)
    with t3: render([3], "MID", w_mid)
    with t4: render([4], "FWD", w_fwd)

if __name__ == "__main__":
    main()
