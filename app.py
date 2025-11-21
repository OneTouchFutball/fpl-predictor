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
    .stTabs [data-baseweb="tab"] { height: 60px; font-weight: 700; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { border-top: 3px solid #00cc00; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# =========================================
# 1. LIGHTWEIGHT AI ENGINE
# =========================================

@st.cache_resource
def get_trained_model():
    """
    Trains the AI. Optimized for speed and low memory.
    """
    # 1. FAST LOAD: Check if your GitHub file exists
    if os.path.exists("fpl_5_year_history.csv"):
        # Load specific columns only to save RAM
        cols = ['minutes', 'total_points', 'was_home', 'clean_sheets', 
                'goals_conceded', 'expected_goals', 'expected_assists', 
                'influence', 'creativity', 'threat', 'value', 'element_type',
                'goals_scored', 'assists', 'saves', 'bps', 'yellow_cards']
        
        try:
            df = pd.read_csv("fpl_5_year_history.csv", usecols=cols, dtype={'total_points': 'float32'})
        except ValueError:
            # Fallback if columns don't match perfectly
            df = pd.read_csv("fpl_5_year_history.csv")
    else:
        return None, None, "FILE_NOT_FOUND"

    # 2. FAST PROCESSING: Filter Starters Only
    # We only train on players who played > 60 mins. This reduces rows significantly.
    df = df[df['minutes'] > 60].copy()
    
    # Features
    features = ['element_type', 'was_home', 'expected_goals', 'expected_assists', 
                'clean_sheets', 'goals_conceded', 'influence', 'creativity', 'threat',
                'goals_scored', 'assists', 'saves', 'bps', 'yellow_cards']
    
    # Validate columns exist
    valid_features = [f for f in features if f in df.columns]
    
    X = df[valid_features]
    y = df['total_points']
    
    # 3. FAST TRAINING: Reduced complexity for Cloud compatibility
    # max_iter=30 is enough for FPL patterns (diminishing returns after that)
    model = HistGradientBoostingRegressor(max_iter=30, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Calculate max points for normalization
    max_ai_pts = df['total_points'].quantile(0.99)
    
    return model, valid_features, max_ai_pts

# =========================================
# 2. LIVE DATA & FIXTURES
# =========================================

@st.cache_data(ttl=1800)
def load_live_data():
    static = requests.get(f"{API_BASE}/bootstrap-static/").json()
    fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    return static, fixtures

def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    t_stats = {t['id']: {'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                         'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']} 
               for t in teams_data}
    
    team_sched = {t['id']: {'fut_opp_att': [], 'fut_opp_def': [], 'display': []} for t in teams_data}
    
    for f in fixtures:
        if not f['kickoff_time'] or f['finished']: continue
        h, a = f['team_h'], f['team_a']
        
        team_sched[h]['fut_opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['fut_opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        team_sched[a]['fut_opp_att'].append(t_stats[h]['att_h'])
        team_sched[a]['fut_opp_def'].append(t_stats[h]['def_h'])
        team_sched[a]['display'].append(f"{team_map[h]}(A)")
        
    return team_sched, 1080.0

def get_fixture_multiplier(schedule_list, league_avg, limit, mode="def"):
    if not schedule_list: return 1.0
    subset = schedule_list[:limit]
    avg_strength = sum(subset) / len(subset)
    ratio = league_avg / avg_strength
    power = 4.0 if mode == "def" else 2.0
    return ratio ** power

def get_display_score(schedule_list, limit):
    if not schedule_list: return 5.0
    avg = sum(schedule_list[:limit]) / len(schedule_list[:limit])
    return max(0, min(10, 10 - ((avg - 950)/400 * 10)))

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# =========================================
# 3. MAIN APP
# =========================================

def main():
    st.title("🧬 FPL Pro: Hybrid Intelligence")
    
    # 1. Load & Train (Fast)
    model_status = st.empty()
    
    # Check if model is already cached
    model, ai_cols, max_ai_pts = get_trained_model()
    
    if max_ai_pts == "FILE_NOT_FOUND":
        st.error("❌ `fpl_5_year_history.csv` not found in repository.")
        st.info("Please upload the CSV file to your GitHub folder.")
        return
    
    # 2. Load Live Data
    static, fixtures = load_live_data()
    teams = static['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_sched, avg_str = process_fixtures(fixtures, teams)
    
    # 3. Prepare Player Data
    df = pd.DataFrame(static['elements'])
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 2.0]
    
    ai_input = pd.DataFrame()
    ai_input['element_type'] = df['element_type']
    ai_input['was_home'] = 0.5
    
    stat_map = {
        'expected_goals': 'expected_goals_per_90',
        'expected_assists': 'expected_assists_per_90',
        'clean_sheets': 'clean_sheets_per_90',
        'goals_conceded': 'goals_conceded_per_90',
        'influence': 'influence', 'creativity': 'creativity', 'threat': 'threat',
        'goals_scored': 'goals_scored', 'assists': 'assists', 'saves': 'saves', 'bps': 'bps',
        'yellow_cards': 'yellow_cards'
    }
    
    for train_col, api_col in stat_map.items():
        if train_col in ai_cols:
            if 'per_90' in api_col:
                ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0)
            else:
                ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']
            
    # AI Prediction
    df['AI_Points'] = model.predict(ai_input[ai_cols])
    
    # --- UI ---
    st.sidebar.header("🔮 Horizon")
    horizon = st.sidebar.selectbox("Lookahead", [1, 5, 10], format_func=lambda x: f"Next {x} Matches")
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Weights")
    
    w_budget = st.sidebar.slider("Price Sensitivity", 0.0, 1.0, 0.5)
    
    with st.sidebar.expander("🧤 GK Settings", expanded=False):
        w_gk = {'ai': st.slider("AI", 0.0, 1.0, 0.6, key="g1"), 'form': st.slider("Form", 0.0, 1.0, 0.4, key="g2"), 'fix': st.slider("Fix", 0.0, 1.0, 1.0, key="g3")}
    with st.sidebar.expander("🛡️ DEF Settings", expanded=False):
        w_def = {'ai': st.slider("AI", 0.0, 1.0, 0.6, key="d1"), 'form': st.slider("Form", 0.0, 1.0, 0.4, key="d2"), 'xgi': st.slider("Att", 0.0, 1.0, 0.3, key="d3"), 'fix': st.slider("Fix", 0.0, 1.0, 1.0, key="d4")}
    with st.sidebar.expander("⚔️ MID Settings", expanded=False):
        w_mid = {'ai': st.slider("AI", 0.0, 1.0, 0.6, key="m1"), 'form': st.slider("Form", 0.0, 1.0, 0.4, key="m2"), 'fix': st.slider("Fix", 0.0, 1.0, 0.8, key="m3")}
    with st.sidebar.expander("⚽ FWD Settings", expanded=False):
        w_fwd = {'ai': st.slider("AI", 0.0, 1.0, 0.6, key="f1"), 'form': st.slider("Form", 0.0, 1.0, 0.4, key="f2"), 'fix': st.slider("Fix", 0.0, 1.0, 0.8, key="f3")}

    st.sidebar.divider()
    min_mins = st.sidebar.slider("Min Minutes", 0, 2500, 400)

    # --- HYBRID ENGINE ---
    def run_engine(p_ids, cat, w):
        cands = []
        subset = df[df['element_type'].isin(p_ids) & (df['minutes'] >= min_mins)]
        MAX_PPM = subset['points_per_game'].astype(float).max()
        
        for _, row in subset.iterrows():
            tid = row['team']
            
            if cat in ["GK", "DEF"]:
                sched = team_sched[tid]['fut_opp_att']
                mode = "def"
            else:
                sched = team_sched[tid]['fut_opp_def']
                mode = "att"
                
            fix_mult = get_fixture_multiplier(sched, avg_str, horizon, mode)
            fix_score_display = get_display_score(sched, horizon)
            fix_display = ", ".join(team_sched[tid]['display'][:horizon])
            
            # Scores
            score_ai = (row['AI_Points'] / max_ai_pts) * 10
            raw_ppm = float(row['points_per_game'])
            score_form = (raw_ppm / MAX_PPM) * 10
            
            score_bonus = 0
            if cat == "DEF":
                xgi = float(row['expected_goal_involvements_per_90'])
                score_bonus = (xgi * 10) * w['xgi']
            
            # Blend
            base_score = (score_ai * w['ai']) + (score_form * w['form']) + score_bonus
            
            # Context
            eff_mult = 1.0 + (fix_mult - 1.0) * w['fix']
            final_score = base_score * eff_mult
            
            # ROI
            price = row['now_cost'] / 10.0
            price_div = price ** w_budget
            roi = final_score / price_div
            
            stat_disp = float(row['clean_sheets_per_90']) if cat in ["GK", "DEF"] else float(row['expected_goal_involvements_per_90'])

            cands.append({
                "Name": row['web_name'],
                "Team": team_names.get(tid, "Unknown"),
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
        d = run_engine(p_ids, cat, w)
        if d.empty: st.write("No players."); return
        
        stat_lbl = "CS/90" if cat in ["GK", "DEF"] else "xGI/90"
        
        st.dataframe(
            d.head(50), hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Upcoming": st.column_config.TextColumn("Opponents", width="medium"),
                "AI Base": st.column_config.NumberColumn("AI Exp", help="Points predicted by AI stats"),
                "PPM": st.column_config.NumberColumn("Form", help="Actual Points Per Match"),
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
