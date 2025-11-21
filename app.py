import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Hybrid Intelligence", page_icon="🧬", layout="wide")

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
# PART 1: AI INFRASTRUCTURE (The Brain)
# =========================================

def download_training_data():
    """Downloads 5 years of data if missing."""
    status = st.empty()
    if os.path.exists("fpl_5_year_history.csv"):
        return pd.read_csv("fpl_5_year_history.csv")
        
    status.info("⏳ Initializing Hybrid Engine: Downloading historical data (One-time)...")
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_data = []
    
    bar = st.progress(0)
    for i, season in enumerate(seasons):
        try:
            url = f"{base_url}/{season}/gws/merged_gw.csv"
            r = requests.get(url)
            if r.status_code == 200:
                df = pd.read_csv(io.BytesIO(r.content), on_bad_lines='skip', low_memory=False)
                
                # Standardize
                cols = ['minutes', 'total_points', 'was_home', 'clean_sheets', 
                        'goals_conceded', 'expected_goals', 'expected_assists', 
                        'influence', 'creativity', 'threat', 'value', 'element_type']
                existing = [c for c in cols if c in df.columns]
                df = df[existing]
                all_data.append(df)
        except: pass
        bar.progress((i+1)/len(seasons))
        
    if all_data:
        master = pd.concat(all_data)
        master.fillna(0, inplace=True)
        master.to_csv("fpl_5_year_history.csv", index=False)
        status.success("✅ AI Memory Loaded.")
        bar.empty()
        return master
    return None

@st.cache_resource
def train_ai_model():
    df = download_training_data()
    if df is None: return None, None
    
    # Filter Starters (>60 mins)
    df = df[df['minutes'] > 60].copy()
    
    # Features
    features = ['value', 'element_type', 'was_home', 'expected_goals', 'expected_assists', 
                'clean_sheets', 'goals_conceded', 'influence', 'creativity', 'threat']
    
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    y = df['total_points']
    
    model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
    model.fit(X, y)
    
    return model, valid_features

# =========================================
# PART 2: MANUAL CONTEXT ENGINE (The Logic)
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
        
    return team_sched, 1080.0, 1080.0 # Return avgs

def get_fixture_multiplier(schedule_list, league_avg, limit, mode="def"):
    if not schedule_list: return 1.0
    subset = schedule_list[:limit]
    avg_strength = sum(subset) / len(subset)
    ratio = league_avg / avg_strength
    # Power Law: Defenders (4.0) get punished harder than Attackers (2.0) for hard games
    power = 4.0 if mode == "def" else 2.0 
    return ratio ** power

def get_display_score(schedule_list, limit):
    if not schedule_list: return 5.0
    avg = sum(schedule_list[:limit]) / len(schedule_list[:limit])
    return max(0, min(10, 10 - ((avg - 1000)/350 * 10)))

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# =========================================
# PART 3: THE HYBRID APP
# =========================================

def main():
    st.title("🧬 FPL Hybrid Intelligence")
    st.markdown("### AI Stats Engine + Manual Context Logic")

    # Load AI
    model, ai_cols = train_ai_model()
    
    # Load Live Data
    static, fixtures = load_live_data()
    if not static: return

    teams = static['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_sched, avg_att, avg_def = process_fixtures(fixtures, teams)
    
    # Prepare Live Data for AI
    df = pd.DataFrame(static['elements'])
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 1.5] # Filter new players
    
    # Normalize API data to match Training Data columns
    # (This maps "season totals" to "per match" for the AI)
    ai_input = pd.DataFrame()
    ai_input['value'] = df['now_cost']
    ai_input['element_type'] = df['element_type']
    ai_input['was_home'] = 0.5
    
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
            ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']
            
    # --- STEP 1: AI PREDICTION (The Base Truth) ---
    df['AI_Base_Points'] = model.predict(ai_input[ai_cols])
    
    # Normalize AI Score to 0-10 for mixing with User Weights
    MAX_AI = df['AI_Base_Points'].max()
    
    # --- UI CONTROLS ---
    st.sidebar.header("🔮 Horizon")
    horizon = st.sidebar.selectbox("Lookahead", [1, 5, 10], format_func=lambda x: f"Next {x} Matches")
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Hybrid Weights")
    st.sidebar.info("Mix the AI's prediction with your own preferences.")
    
    w_budget = st.sidebar.slider("Price Importance", 0.0, 1.0, 0.5, key="price")
    
    st.sidebar.subheader("Position Weights")
    
    # 1. GK
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_ai_gk = st.slider("AI Prediction Trust", 0.0, 1.0, 0.7, key="gk_ai")
        w_fix_gk = st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="gk_fix")
        weights_gk = {'ai': w_ai_gk, 'fix': w_fix_gk}
        
    # 2. DEF
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_ai_def = st.slider("AI Prediction Trust", 0.0, 1.0, 0.7, key="def_ai")
        w_xgi_def = st.slider("Attacking Bonus (User Pref)", 0.0, 1.0, 0.3, key="def_xgi")
        w_fix_def = st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="def_fix")
        weights_def = {'ai': w_ai_def, 'xgi': w_xgi_def, 'fix': w_fix_def}
        
    # 3. MID
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_ai_mid = st.slider("AI Prediction Trust", 0.0, 1.0, 0.8, key="mid_ai")
        w_fix_mid = st.slider("Fixture Impact", 0.0, 1.0, 0.7, key="mid_fix")
        weights_mid = {'ai': w_ai_mid, 'fix': w_fix_mid}
        
    # 4. FWD
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_ai_fwd = st.slider("AI Prediction Trust", 0.0, 1.0, 0.8, key="fwd_ai")
        w_fix_fwd = st.slider("Fixture Impact", 0.0, 1.0, 0.7, key="fwd_fix")
        weights_fwd = {'ai': w_ai_fwd, 'fix': w_fix_fwd}

    st.sidebar.divider()
    min_mins = st.sidebar.slider("Min Minutes", 0, 2500, 300)

    # --- ENGINE ---
    def run_hybrid_engine(p_ids, cat, w):
        cands = []
        
        # Filter DF
        subset = df[df['element_type'].isin(p_ids) & (df['minutes'] >= min_mins)]
        
        for _, row in subset.iterrows():
            tid = row['team']
            
            # 1. GET CONTEXT
            if cat in ["GK", "DEF"]:
                # Defending against Attack
                sched = team_sched[tid]['fut_opp_att']
                league_avg = avg_att
                mode = "def"
            else:
                # Attacking against Defense
                sched = team_sched[tid]['fut_opp_def']
                league_avg = avg_def
                mode = "att"
                
            # Calculate Multiplier & Score
            fix_mult = get_fixture_multiplier(sched, league_avg, horizon, mode)
            fix_score_display = get_display_score(sched, horizon)
            fix_display = ", ".join(team_sched[tid]['display'][:horizon])
            
            # 2. HYBRID SCORE CALCULATION
            
            # A. AI Component (Normalized 0-10)
            ai_score = (row['AI_Base_Points'] / MAX_AI) * 10
            
            # B. Manual Boosts (e.g. Attacking Defenders)
            manual_boost = 0
            if cat == "DEF":
                xgi = float(row['expected_goal_involvements_per_90'])
                manual_boost = (xgi * 10) * w['xgi']
            
            # C. Combine (Weighted Average)
            # (AI * Trust) + Manual_Boost
            combined_base = (ai_score * w['ai']) + manual_boost
            
            # D. Apply Context (Fixture Wipeout)
            # We blend the multiplier based on User's "Fixture Impact" Slider
            # If w_fix is 1.0, we apply full multiplier. If 0.5, we dampen it.
            effective_mult = 1.0 + (fix_mult - 1.0) * w['fix']
            
            final_score = combined_base * effective_mult
            
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
                "AI Base": round(row['AI_Base_Points'], 2),
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
                "ROI Index": st.column_config.ProgressColumn("Hybrid ROI", format="%.1f", min_value=0, max_value=10),
                "AI Base": st.column_config.NumberColumn("AI Pts", help="Projected points based purely on stats (No fixtures)"),
                "Fix Rate": st.column_config.NumberColumn("Fix Ease", help="10=Easy, 0=Hard"),
                "Key Stat": st.column_config.NumberColumn(stat_lbl, format="%.2f"),
                "Price": st.column_config.NumberColumn("£", format="£%.1f")
            }
        )

    t1, t2, t3, t4 = st.tabs(["🧤 GK", "🛡️ DEF", "⚔️ MID", "⚽ FWD"])
    with t1: render([1], "GK", weights_gk)
    with t2: render([2], "DEF", weights_def)
    with t3: render([3], "MID", weights_mid)
    with t4: render([4], "FWD", weights_fwd)

if __name__ == "__main__":
    main()
