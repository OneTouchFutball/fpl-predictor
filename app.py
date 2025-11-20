import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Predictor 25/26", page_icon="⚽", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Increase Tab Size and Font */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-size: 18px; /* Bigger Text */
        font-weight: 700; /* Bold */
        color: #4a4a4a;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #00cc00;
        color: #00cc00;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    /* Button Styling */
    .stButton button { width: 100%; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 0

def reset_page():
    st.session_state.page = 0

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

# --- LOGIC ENGINE ---
def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    team_sched = {t['id']: {'past': [], 'future': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue

        h = f['team_h']
        a = f['team_a']
        h_diff = f['team_h_difficulty']
        a_diff = f['team_a_difficulty']

        # Favourability: 6 - Difficulty + 0.5 for Home
        h_fav = (6 - h_diff) + 0.5
        a_fav = (6 - a_diff)

        h_display = f"{team_map[a]}(H)"
        a_display = f"{team_map[h]}(A)"

        h_obj = {'score': h_fav, 'display': h_display}
        a_obj = {'score': a_fav, 'display': a_display}

        if f['finished']:
            team_sched[h]['past'].append(h_obj)
            team_sched[a]['past'].append(a_obj)
        else:
            team_sched[h]['future'].append(h_obj)
            team_sched[a]['future'].append(a_obj)

    return team_sched

def get_aggregated_data(schedule_list, limit=None):
    if not schedule_list:
        return 3.0, "-"
    subset = schedule_list[:limit] if limit else schedule_list
    avg_score = sum(item['score'] for item in subset) / len(subset)
    display_str = ", ".join([item['display'] for item in subset])
    return avg_score, display_str

def min_max_scale(series):
    """Scales a pandas series to 0-10 range"""
    if series.empty: return series
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v: return pd.Series([5.0] * len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### Weighted Model with Context-Aware Projections")

    data, fixtures = load_data()
    if not data or not fixtures:
        return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule = process_fixtures(fixtures, teams)
    
    # Team Defense Strength
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Predict for upcoming:",
        options=[1, 5, 10],
        format_func=lambda x: f"Next {x} Fixture{'s' if x > 1 else ''}",
        on_change=reset_page
    )

    st.sidebar.divider()
    st.sidebar.header("⚖️ Model Weights")
    
    # PRICE SENSITIVITY
    w_budget = st.sidebar.slider(
        "Price Importance (Value vs Raw Points)", 
        0.0, 1.0, 0.5,
        help="0.0 = Best Players Only. 1.0 = Best Value Only.",
        on_change=reset_page,
        key="price_weight"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Position Settings")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.5, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.5, key="def_cs", on_change=reset_page)
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
    min_minutes = st.sidebar.slider(
        "Min. Minutes Played", 0, 2000, 250, 
        help="Set to 0 to analyze ALL players.",
        on_change=reset_page,
        key="min_mins"
    )

    # --- ANALYSIS ---
    def run_analysis(player_type_ids, pos_category, weights):
        candidates = []
        
        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue

            tid = p['team']
            
            # 1. Fixture Metrics (Scores include Home/Away weighting)
            past_score, _ = get_aggregated_data(team_schedule[tid]['past'])
            future_score, future_display = get_aggregated_data(team_schedule[tid]['future'], limit=horizon_option)

            # 2. Base Metrics
            try:
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                if price <= 0: price = 4.0
                
                # --- LOGIC BLOCK ---
                if pos_category == "GK":
                    # GK Logic: Clean Sheet + Form + Fixture (No Attacking Threat)
                    stat_val = float(p['clean_sheets_per_90'])
                    stat_label = stat_val 
                    
                    # A. ROI SCORE (0-10 Index)
                    cs_potential = (stat_val * 10) + (team_def_strength[tid] / 2)
                    
                    base_score = (cs_potential * weights['cs']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix'])
                    
                    # B. BASE ABILITY (Intrinsic Points Potential)
                    base_strength = (ppm * 0.6) + (stat_val * 6 * 0.4)

                elif pos_category == "DEF":
                    # DEF Logic: Clean Sheet + xGI + Form + Fixture
                    cs_val = float(p['clean_sheets_per_90'])
                    xgi_val = float(p.get('expected_goal_involvements_per_90', 0))
                    stat_label = cs_val 
                    
                    cs_potential = (cs_val * 10) + (team_def_strength[tid] / 2)
                    
                    base_score = (cs_potential * weights['cs']) + \
                                 ((xgi_val * 10) * weights['xgi']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix'])
                    
                    # B. BASE ABILITY
                    base_strength = (ppm * 0.5) + (cs_val * 5 * 0.3) + (xgi_val * 5 * 0.2)

                else:
                    # MID/FWD Logic: xGI + Form + Fixture
                    stat_val = float(p.get('expected_goal_involvements_per_90', 0)) # xGI
                    stat_label = stat_val
                    
                    base_score = ((stat_val * 10) * weights['xgi']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix'])
                    
                    # B. BASE ABILITY
                    base_strength = (ppm * 0.55) + (stat_val * 7 * 0.45)

                # 3. PREDICTED POINTS (Context Aware)
                # Logic: Base Ability × Past Context Discount × Future Opportunity Boost
                
                # Past Context: If past games were easy, discount ability.
                past_context_ratio = (3.5 / max(1.5, past_score)) ** 0.7
                
                # Future Context: If future games are easy, boost ability.
                future_context_ratio = future_score / 3.5
                
                # Final Projection
                proj_points = base_strength * past_context_ratio * future_context_ratio

                # 4. ROI Resistance
                resistance_factor = max(2.0, min(past_score, 5.0))
                raw_perf_metric = base_score / resistance_factor
                
                status_icon = "✅" if p['status'] == 'a' else ("⚠️" if p['status'] == 'd' else "❌")

                candidates.append({
                    "Name": f"{status_icon} {p['web_name']}",
                    "Team": team_names[tid],
                    "Price": price,
                    "Stat_Display": stat_label,
                    "Upcoming Fixtures": future_display,
                    "PPM": ppm,
                    "Exp. Pts": proj_points,
                    "Future Fix": round(future_score, 2),
                    "Past Fix": round(past_score, 2),
                    "Raw_Metric": raw_perf_metric,
                })

            except Exception:
                continue

        # --- DATAFRAME ---
        df = pd.DataFrame(candidates)
        if df.empty: return df

        # Normalize Performance (0-10)
        df['Norm_Perf'] = min_max_scale(df['Raw_Metric'])
        
        # Normalize Value (0-10)
        df['Value_Metric'] = df['Raw_Metric'] / df['Price']
        df['Norm_Value'] = min_max_scale(df['Value_Metric'])
        
        # ROI Calculation
        df['ROI Index'] = (df['Norm_Perf'] * (1 - w_budget)) + (df['Norm_Value'] * w_budget)
        
        df = df.sort_values(by="ROI Index", ascending=False)
        
        cols = ["ROI Index", "Name", "Team", "Exp. Pts", "Price", "Stat_Display", "Upcoming Fixtures", "PPM", "Future Fix", "Past Fix"]
        return df[cols]

    # --- DISPLAY ---
    def render_tab(p_ids, pos_category, weights):
        df = run_analysis(p_ids, pos_category, weights)
        
        if df.empty:
            st.warning("No players found.")
            return

        # Pagination
        items_per_page = 50
        total_items = len(df)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        if st.session_state.page >= total_pages:
            st.session_state.page = total_pages - 1
        if st.session_state.page < 0:
            st.session_state.page = 0
            
        start_idx = st.session_state.page * items_per_page
        end_idx = start_idx + items_per_page
        
        df_display = df.iloc[start_idx:end_idx]
        
        st.caption(f"Showing **{start_idx + 1}-{min(end_idx, total_items)}** of **{total_items}** players")
        
        stat_col_name = "CS/90" if pos_category in ["GK", "DEF"] else "xGI/90"
        
        st.dataframe(
            df_display, 
            hide_index=True, 
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Exp. Pts": st.column_config.NumberColumn(
                    "Exp. Pts", 
                    format="%.1f", 
                    help="Predicted Points: Base Ability × Past Resistance Adjustment × Future Fixture Difficulty"
                ),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Stat_Display": st.column_config.NumberColumn(stat_col_name, format="%.2f", help=f"{stat_col_name} stats from FPL API"),
                "Upcoming Fixtures": st.column_config.TextColumn("Opponents", width="medium"),
                "PPM": st.column_config.NumberColumn("Pts/G", format="%.1f"),
                "Future Fix": st.column_config.NumberColumn("Fut Fix", help="Higher = Easier Upcoming Fixtures (Includes Home/Away)"),
                "Past Fix": st.column_config.NumberColumn("Past Fix", help="Higher = Easier Past Fixtures (Includes Home/Away)"),
            },
            use_container_width=True
        )
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ Previous 50", disabled=(st.session_state.page == 0), key=f"prev_{pos_category}"):
                st.session_state.page -= 1
                st.rerun()
        with c3:
            if st.button("Next 50 ➡️", disabled=(st.session_state.page == total_pages - 1), key=f"next_{pos_category}"):
                st.session_state.page += 1
                st.rerun()

    # --- RENDER TABS ---
    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs([
        "🧤 GOALKEEPERS", 
        "🛡️ DEFENDERS", 
        "⚔️ MIDFIELDERS", 
        "⚽ FORWARDS"
    ])

    with tab_gk: render_tab([1], "GK", gk_weights)
    with tab_def: render_tab([2], "DEF", def_weights)
    with tab_mid: render_tab([3], "MID", mid_weights)
    with tab_fwd: render_tab([4], "FWD", fwd_weights)

if __name__ == "__main__":
    main()
