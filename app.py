import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL Ultimate Predictor 25/26", page_icon="🛡️", layout="wide")

# --- 1. DATA LOADING & PROCESSING ---
@st.cache_data(ttl=600)
def load_data():
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(base_url + "bootstrap-static/").json()
        fixtures = requests.get(base_url + "fixtures/").json()
        return static, fixtures
    except:
        return None, None

def calculate_team_defensive_strength(static_data):
    """
    Calculates a 'Leakiness Score' for every team based on xGC.
    Lower score = Better Defense (More Clean Sheets).
    """
    team_xgc = {t['id']: [] for t in static_data['teams']}
    
    for p in static_data['elements']:
        # Only look at Defenders (2) and GKs (1) who play regularly (>400 mins)
        if p['element_type'] in [1, 2] and p['minutes'] > 400:
            try:
                xgc_per_90 = float(p.get('expected_goals_conceded_per_90', 0))
                team_xgc[p['team']].append(xgc_per_90)
            except:
                continue
    
    # Average xGC per 90 for the team's defensive unit
    team_strength = {}
    for t_id, values in team_xgc.items():
        if values:
            avg_xgc = sum(values) / len(values)
            team_strength[t_id] = avg_xgc
        else:
            team_strength[t_id] = 1.5 # Default average if no data
            
    return team_strength

def get_fixture_schedule(static_data, fixture_data):
    """Maps Team ID to list of next 5 opponents + difficulties."""
    teams = {t['id']: t['short_name'] for t in static_data['teams']}
    next_gw = next((e['id'] for e in static_data['events'] if e['is_next']), 1)
    
    schedule = {t_id: [] for t_id in teams}
    
    for f in fixture_data:
        if f['event'] and f['event'] >= next_gw:
            h, a = f['team_h'], f['team_a']
            schedule[h].append({"opp": teams[a], "diff": f['team_h_difficulty']})
            schedule[a].append({"opp": teams[h], "diff": f['team_a_difficulty']})
            
    return schedule

# --- 2. MODEL LOGIC ---

def calculate_attacker_score(p, schedule, w_xgi, w_form, w_fix):
    """Model for MIDs and FWDs (Focus: Goals/Assists)"""
    try:
        # Metrics
        xgi = float(p.get('expected_goal_involvements', 0))
        form = float(p['form'])
        
        # Schedule
        my_fix = schedule.get(p['team'], [])[:5] # Next 5 games
        if not my_fix: return 0, ""
        avg_diff = sum(m['diff'] for m in my_fix) / len(my_fix)
        
        # Normalization & Weighting
        # xGI: High is ~12.0 -> Score 9.0
        s_xgi = xgi * 1.2 
        # Form: High is ~8.0 -> Score 8.0
        s_form = form 
        # Fixtures: Low Diff (2) is Good. (5 - 2) * 3 = 9.0
        s_fix = (5.5 - avg_diff) * 2.5
        
        total_score = (s_xgi * w_xgi) + (s_form * w_form) + (s_fix * w_fix)
        
        fix_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fix])
        return total_score, fix_str, avg_diff
    except:
        return 0, "", 0

def calculate_defender_score(p, schedule, team_leakiness, w_cs, w_xgi, w_fix):
    """Model for DEFs and GKPs (Focus: Clean Sheets + Attacking Bonus)"""
    try:
        # Metrics
        xgi = float(p.get('expected_goal_involvements', 0)) # Attacking threat
        team_xgc = team_leakiness.get(p['team'], 1.5) # Team defensive strength
        
        # Schedule
        my_fix = schedule.get(p['team'], [])[:5]
        if not my_fix: return 0, ""
        avg_diff = sum(m['diff'] for m in my_fix) / len(my_fix)
        
        # --- SCORING ---
        
        # 1. Clean Sheet Potential (The Core Stat)
        # Team xGC usually ranges 0.8 (Best) to 2.5 (Worst).
        # We invert it: Lower xGC = Higher Score.
        s_clean_sheet = (3.0 - team_xgc) * 4.0 
        
        # 2. Fixture Ease (Crucial for Defenders)
        s_fix = (5.5 - avg_diff) * 3.0
        
        # 3. Attacking Bonus (The "Trent/Porro" Factor)
        # Defenders have lower xGI, so we boost the multiplier
        s_attack = xgi * 3.0 
        
        total_score = (s_clean_sheet * w_cs) + (s_fix * w_fix) + (s_attack * w_xgi)
        
        fix_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fix])
        return total_score, fix_str, avg_diff
    except:
        return 0, "", 0

# --- 3. MAIN APP ---
def main():
    st.title("🛡️ FPL 25/26 Multi-Model Predictor")
    st.markdown("### Attackers • Defenders • Best Value")
    
    # Load Data
    with st.spinner("Crunching stats..."):
        static, fixtures = load_data()
        if not static:
            st.error("API Error")
            return
        
        team_leakiness = calculate_team_defensive_strength(static)
        schedules = get_fixture_schedule(static, fixtures)
        teams = {t['id']: t['short_name'] for t in static['teams']}

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["⚔️ Attackers (MID/FWD)", "🛡️ Defenders (DEF/GKP)", "💰 Best Value Gems"])

    # --- COMMON FILTERS ---
    with st.sidebar:
        st.header("Filters")
        min_mins = st.number_input("Min Minutes Played", 0, 3000, 500)
        price_filter = st.slider("Price Range", 3.5, 15.0, (4.0, 14.0))
        st.divider()
        st.info("Tip: Use the tabs above to switch between Attacking, Defensive, and Value models.")

    # ==========================================
    # TAB 1: ATTACKERS
    # ==========================================
    with tab1:
        col1, col2, col3 = st.columns(3)
        w_xgi_att = col1.slider("Weight: xGI", 0.1, 1.0, 0.6, key="att1")
        w_form_att = col2.slider("Weight: Form", 0.1, 1.0, 0.2, key="att2")
        w_fix_att = col3.slider("Weight: Fixtures", 0.1, 1.0, 0.2, key="att3")
        
        attackers = []
        for p in static['elements']:
            if p['element_type'] in [3, 4] and p['minutes'] >= min_mins:
                price = p['now_cost'] / 10.0
                if not (price_filter[0] <= price <= price_filter[1]): continue
                
                score, schedule_str, avg_diff = calculate_attacker_score(
                    p, schedules, w_xgi_att, w_form_att, w_fix_att
                )
                
                attackers.append({
                    "Name": p['web_name'],
                    "Team": teams[p['team']],
                    "Pos": "MID" if p['element_type'] == 3 else "FWD",
                    "Price": price,
                    "xGI": float(p['expected_goal_involvements']),
                    "Next 5": schedule_str,
                    "Score": round(score, 2)
                })
        
        df_att = pd.DataFrame(attackers).sort_values("Score", ascending=False).head(20)
        st.dataframe(df_att, use_container_width=True, hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn("Predicted Points", format="%.2f", min_value=0, max_value=max(df_att['Score'])),
            "Price": st.column_config.NumberColumn("£", format="£%.1f")
        })

    # ==========================================
    # TAB 2: DEFENDERS
    # ==========================================
    with tab2:
        st.markdown("**Model Logic:** Combines **Team Clean Sheet Prob** (based on xGC) with **Individual xGI**.")
        col1, col2, col3 = st.columns(3)
        w_cs_def = col1.slider("Weight: Clean Sheet Prob", 0.1, 1.0, 0.5, key="def1")
        w_fix_def = col2.slider("Weight: Fixtures", 0.1, 1.0, 0.3, key="def2")
        w_xgi_def = col3.slider("Weight: Attacking Bonus", 0.1, 1.0, 0.2, key="def3")

        defenders = []
        for p in static['elements']:
            if p['element_type'] in [1, 2] and p['minutes'] >= min_mins:
                price = p['now_cost'] / 10.0
                if not (price_filter[0] <= price <= price_filter[1]): continue

                score, schedule_str, avg_diff = calculate_defender_score(
                    p, schedules, team_leakiness, w_cs_def, w_xgi_def, w_fix_def
                )
                
                # Add defensive stats for display
                xgc_per_90 = float(p.get('expected_goals_conceded_per_90', 0))
                
                defenders.append({
                    "Name": p['web_name'],
                    "Team": teams[p['team']],
                    "Pos": "DEF" if p['element_type'] == 2 else "GKP",
                    "Price": price,
                    "Team xGC/90": round(team_leakiness.get(p['team'], 0), 2),
                    "xGI": float(p['expected_goal_involvements']),
                    "Next 5": schedule_str,
                    "Score": round(score, 2)
                })

        df_def = pd.DataFrame(defenders).sort_values("Score", ascending=False).head(20)
        st.dataframe(df_def, use_container_width=True, hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn("Predicted Points", format="%.2f", min_value=0, max_value=max(df_def['Score'])),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Team xGC/90": st.column_config.NumberColumn("Team Leakiness", help="Lower is better defense")
        })

    # ==========================================
    # TAB 3: BEST VALUE (ROI)
    # ==========================================
    with tab3:
        st.markdown("### 💰 Value Kings (Points per Million £)")
        st.markdown("This calculates the **Predicted Score / Price**. Use this to find budget enablers.")
        
        # Combine both lists
        all_players = attackers + defenders
        
        # Calculate Value Score
        value_picks = []
        for p in all_players:
            # Avoid division by zero or super cheap bench fodder causing outliers
            if p['Price'] > 3.8:
                value_ratio = p['Score'] / p['Price']
                p['Value Score'] = round(value_ratio, 2)
                value_picks.append(p)
        
        df_val = pd.DataFrame(value_picks).sort_values("Value Score", ascending=False).head(25)
        
        st.dataframe(df_val, use_container_width=True, hide_index=True, column_config={
            "Value Score": st.column_config.ProgressColumn(
                "Value (Score per £m)", 
                format="%.2f", 
                min_value=0, 
                max_value=max(df_val['Value Score']),
                help="How many predicted points you get for every £1m spent."
            ),
            "Score": st.column_config.NumberColumn("Pred. Points", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f")
        })

if __name__ == "__main__":
    main()
