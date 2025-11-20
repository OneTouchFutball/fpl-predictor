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

# --- FIXTURE ENGINE (DUAL MODE) ---
def process_fixtures(fixtures, teams_data, context_mode=False):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    
    # Strength Map for Context Mode
    strengths = {}
    if context_mode:
        for t in teams_data:
            strengths[t['id']] = {
                'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']
            }

    team_sched = {t['id']: {'past': [], 'future': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue
        h_id, a_id = f['team_h'], f['team_a']
        
        # --- MODE SELECTION ---
        if not context_mode:
            # DEFAULT MODE: Standard FDR (1-5)
            # Favourability = 6 - Difficulty + Home Bonus
            h_score = (6 - f['team_h_difficulty']) + 0.5
            a_score = (6 - f['team_a_difficulty'])
            
            h_obj = {'score': h_score, 'display': f"{team_map[a_id]}(H)"}
            a_obj = {'score': a_score, 'display': f"{team_map[h_id]}(A)"}
            
        else:
            # CONTEXT MODE: Team Strength Ratios
            h_str, a_str = strengths[h_id], strengths[a_id]
            
            # Ratio > 1.0 = Good Fixture
            # We store 'att' (Attack vs Def) and 'def' (Def vs Att) separately
            
            # Home Team Ratios
            h_att_ratio = h_str['att_h'] / a_str['def_a']
            h_def_ratio = h_str['def_h'] / a_str['att_a']
            
            # Away Team Ratios
            a_att_ratio = a_str['att_a'] / h_str['def_h']
            a_def_ratio = a_str['def_a'] / h_str['att_h']
            
            h_obj = {'att': h_att_ratio, 'def': h_def_ratio, 'display': f"{team_map[a_id]}(H)"}
            a_obj = {'att': a_att_ratio, 'def': a_def_ratio, 'display': f"{team_map[h_id]}(A)"}
        
        # Append to lists
        if f['finished']:
            team_sched[h_id]['past'].append(h_obj)
            team_sched[a_id]['past'].append(a_obj)
        else:
            team_sched[h_id]['future'].append(h_obj)
            team_sched[a_id]['future'].append(a_obj)
            
    return team_sched

def get_aggregated_data(schedule_list, limit=None, context_mode=False, mode='att'):
    if not schedule_list: return (1.0 if context_mode else 3.0), "-"
    
    subset = schedule_list[:limit] if limit else schedule_list
    
    if not context_mode:
        # Default: Average of simple scores
        avg_score = sum(item['score'] for item in subset) / len(subset)
    else:
        # Context: Average of ratios * 5 (to match scale)
        avg_score = (sum(item[mode] for item in subset) / len(subset)) * 5.0
        
    display_str = ", ".join([item['display'] for item in subset])
    return avg_score, display_str

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### Advanced Value Identification System")

    # --- SIDEBAR CONFIG ---
    st.sidebar.header("⚙️ System Logic")
    
    # THE CONTEXT TOGGLE
    enable_context = st.sidebar.toggle(
        "Enable Context-Aware Logic", 
        value=False, 
        help="ON: Uses Advanced Home/Away Team Strength Ratios. OFF: Uses Standard Fixture Difficulty."
    )

    st.sidebar.divider()
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Predict for upcoming:", [1, 5, 10], 
        format_func=lambda x: f"Next {x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )

    st.sidebar.divider()
    st.sidebar.header("⚖️ Model Weights")
    w_budget = st.sidebar.slider("Price Importance", 0.0, 1.0, 0.5, key="price_weight", on_change=reset_page)
    
    st.sidebar.divider()
    st.sidebar.subheader("Position Settings")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Clean Sheet / xGC", 0.1, 1.0, 0.5, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Clean Sheet / xGC", 0.1, 1.0, 0.5, key="def_cs", on_change=reset_page)
        w_xgi_def = st.slider("Attacking Threat (xGI)", 0.1, 1.0, 0.5, key="def_xgi", on_change=reset_page)
        w_ppm_def = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="def_ppm", on_change=reset_page)
        w_fix_def = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="def_fix", on_change=reset_page)
        def_weights = {'cs': w_cs_def, 'xgi': w_xgi_def, 'ppm': w_ppm_def, 'fix': w_fix_def}

    # 3. MIDFIELDERS
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_xgi_mid = st.slider("Total xGI Threat", 0.1, 1.0, 0.5, key="mid_xgi", on_change=reset_page)
        w_ppm_mid = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="mid_ppm", on_change=reset_page)
        w_fix_mid = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="mid_fix", on_change=reset_page)
        mid_weights = {'xgi': w_xgi_mid, 'ppm': w_ppm_mid, 'fix': w_fix_mid}

    # 4. FORWARDS
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_xgi_fwd = st.slider("Total xGI Threat", 0.1, 1.0, 0.5, key="fwd_xgi", on_change=reset_page)
        w_ppm_fwd = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="fwd_ppm", on_change=reset_page)
        w_fix_fwd = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="fwd_fix", on_change=reset_page)
        fwd_weights = {'xgi': w_xgi_fwd, 'ppm': w_ppm_fwd, 'fix': w_fix_fwd}

    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 250, key="min_mins", on_change=reset_page)

    # --- LOAD DATA ---
    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    
    # Calculate Schedule based on Toggle State
    team_schedule = process_fixtures(fixtures, teams, context_mode=enable_context)
    
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items() if (max_str := max(team_conceded.values()))}

    # --- ANALYSIS ENGINE ---
    def run_analysis(player_type_ids, pos_category, weights):
        candidates = []
        
        # FPL Points for weighting xStats
        if pos_category in ["GK", "DEF"]:
            pts_goal, pts_cs, pts_assist = 6, 4, 3
            context_mode_str = 'def'
        elif pos_category == "MID":
            pts_goal, pts_cs, pts_assist = 5, 1, 3
            context_mode_str = 'att'
        else: 
            pts_goal, pts_cs, pts_assist = 4, 0, 3
            context_mode_str = 'att'

        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue
            tid = p['team']
            
            # Retrieve Schedule Data (Context or Standard)
            past_score, _ = get_aggregated_data(
                team_schedule[tid]['past'], 
                context_mode=enable_context, 
                mode=context_mode_str
            )
            future_score, future_display = get_aggregated_data(
                team_schedule[tid]['future'], 
                limit=horizon_option, 
                context_mode=enable_context, 
                mode=context_mode_str
            )

            try:
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                price = 4.0 if price <= 0 else price
                
                # Stats
                xG = float(p.get('expected_goals_per_90', 0))
                xA = float(p.get('expected_assists_per_90', 0))
                xGI = float(p.get('expected_goal_involvements_per_90', 0))
                xGC = float(p.get('expected_goals_conceded_per_90', 0))
                CS_rate = float(p.get('clean_sheets_per_90', 0))
                
                # --- COMPONENT SCORES (0-10 Scale) ---
                
                # 1. ATTACK SCORE
                raw_att_points = (xG * pts_goal) + (xA * pts_assist)
                attack_score = min(10, raw_att_points * 8.0)

                # 2. DEFENSE SCORE
                inv_xgc_score = max(0, (2.5 - xGC) * 2.0)
                team_factor = team_def_strength[tid] / 10.0 
                def_raw = ((CS_rate * pts_cs) * 1.5) + inv_xgc_score
                def_score = min(10, def_raw * team_factor)

                # --- BASE SCORE CONSTRUCTION ---
                if pos_category == "GK":
                    base_score = (def_score * weights['cs']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix'])
                elif pos_category == "DEF":
                    base_score = (def_score * weights['cs']) + \
                                 (attack_score * weights['xgi']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix'])
                else: # MID/FWD
                    def_component = def_score if pos_category == "MID" else 0
                    base_score = (attack_score * weights['xgi']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix']) + \
                                 (def_component * 0.1)

                # --- THE DIVISOR / MULTIPLIER LOGIC (Toggle Switch) ---
                
                if enable_context:
                    # ADVANCED MODE: Ratio Logic (Future / Past)
                    # If High Future (Easy) & Low Past (Hard) -> Ratio > 1 -> Boost
                    past_res = max(2.0, min(past_score, 8.0))
                    fut_opp = max(2.0, min(future_score, 8.0))
                    
                    context_multiplier = fut_opp / past_res
                    # Dampen slightly so it's not too volatile
                    weighted_multiplier = context_multiplier ** 1.2
                    
                    raw_perf_metric = base_score * weighted_multiplier
                    
                    # Dynamic help text for table
                    fix_help = "Higher = Easier (Ratio based on Home/Away Strength)"
                    
                else:
                    # STANDARD MODE (Previous Default): Divisor Logic
                    # Raw Resistance Factor (Higher = Easier Past)
                    # If Past Score is High (Easy), we Divide base_score -> Lower ROI
                    resistance_factor = max(2.0, min(past_score, 5.0))
                    raw_perf_metric = base_score / resistance_factor
                    
                    fix_help = "Higher = Easier (Standard FDR)"

                status_icon = "✅" if p['status'] == 'a' else ("⚠️" if p['status'] == 'd' else "❌")
                stat_disp = CS_rate if pos_category in ["GK", "DEF"] else xGI

                candidates.append({
                    "Name": f"{status_icon} {p['web_name']}",
                    "Team": team_names[tid],
                    "Price": price,
                    "Key Stat": stat_disp,
                    "Upcoming Fixtures": future_display,
                    "PPM": ppm,
                    "Future Fix": round(future_score, 2),
                    "Past Fix": round(past_score, 2),
                    "Raw_Metric": raw_perf_metric,
                    "FixHelp": fix_help
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df

        # Normalize & Calculate ROI
        df['Norm_Perf'] = min_max_scale(df['Raw_Metric'])
        df['Value_Metric'] = df['Raw_Metric'] / df['Price']
        df['Norm_Value'] = min_max_scale(df['Value_Metric'])
        df['ROI Index'] = (df['Norm_Perf'] * (1 - w_budget)) + (df['Norm_Value'] * w_budget)
        
        df = df.sort_values(by="ROI Index", ascending=False)
        
        return df, candidates[0]['FixHelp'] if candidates else ""

    # --- RENDER TABS ---
    def render_tab(p_ids, pos_cat, weights):
        df, fix_help = run_analysis(p_ids, pos_cat, weights)
        if df.empty: st.warning("No players found."); return

        # Pagination
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
                "Future Fix": st.column_config.NumberColumn("Fut Fix", help=fix_help),
                "Past Fix": st.column_config.NumberColumn("Past Fix", help=fix_help),
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
