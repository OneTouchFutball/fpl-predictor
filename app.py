import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FPL Long-Term Predictor", page_icon="📉", layout="wide")

# --- CSS FOR VISUALS ---
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #f63366, #fffd80, #0068c9);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING (CACHED) ---
@st.cache_data(ttl=600)
def load_fpl_data():
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        # Fetch 1: General Data (Players, Stats)
        static = requests.get(base_url + "bootstrap-static/").json()
        # Fetch 2: Master Schedule (All Fixtures)
        fixtures = requests.get(base_url + "fixtures/").json()
        return static, fixtures
    except Exception as e:
        return None, None

# --- 2. PROCESS FIXTURES (NEXT 10 GAMES) ---
def process_fixture_difficulty(static_data, fixture_data):
    """
    Maps every Team ID to a list of their upcoming 10 opponents and difficulty ratings.
    """
    teams = {t['id']: t['short_name'] for t in static_data['teams']}
    
    # Identify the next Gameweek
    events = static_data['events']
    next_gw = next((e['id'] for e in events if e['is_next']), 1)
    
    # Initialize schedule dictionary
    team_schedule = {t_id: [] for t_id in teams}
    
    # Loop through fixture list
    for f in fixture_data:
        # Only look at future games
        if f['event'] and f['event'] >= next_gw:
            
            h_team = f['team_h']
            a_team = f['team_a']
            h_diff = f['team_h_difficulty']
            a_diff = f['team_a_difficulty']
            
            # Add to Home Team's schedule
            if len(team_schedule[h_team]) < 10:
                team_schedule[h_team].append({
                    "opp": teams[a_team],
                    "diff": h_diff,
                    "loc": "(H)"
                })
            
            # Add to Away Team's schedule
            if len(team_schedule[a_team]) < 10:
                team_schedule[a_team].append({
                    "opp": teams[h_team],
                    "diff": a_diff,
                    "loc": "(A)"
                })
                
    return team_schedule

# --- 3. MAIN APP LOGIC ---
def main():
    st.title("🔭 FPL Horizon Model (Total xGI Version)")
    st.markdown("""
    **Logic:**  
    1. **Total xGI**: Uses cumulative `expected_goal_involvements` (Reliability > Flashiness).  
    2. **10-Game Horizon**: Analyzes the difficulty of the next 10 matches.  
    3. **Explosion Probability**: Weighted score of Stats + Form + Fixture Ease.
    """)

    # Load Data
    with st.spinner("Analyzing Season Data & Fixture Lists..."):
        static, fixtures = load_fpl_data()
        
    if not static:
        st.error("Could not connect to FPL API.")
        return

    # Process Fixtures
    team_schedules = process_fixture_difficulty(static, fixtures)

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("⚙️ Model Weights")
    w_xgi = st.sidebar.slider("Weight: Total xGI", 0.1, 1.0, 0.6, help="Importance of Expected Goal Involvement")
    w_form = st.sidebar.slider("Weight: Current Form", 0.1, 1.0, 0.2, help="Importance of short-term form")
    w_fix = st.sidebar.slider("Weight: Fixture Ease", 0.1, 1.0, 0.3, help="Importance of having easy games")

    st.sidebar.divider()
    st.sidebar.header("🔎 Filters")
    position = st.sidebar.multiselect("Position", ["DEF", "MID", "FWD"], ["MID", "FWD"])
    price_min, price_max = st.sidebar.slider("Price (£m)", 3.5, 15.0, (5.0, 14.0))
    min_minutes = st.sidebar.number_input("Min Minutes Played", 0, 3000, 600)

    # --- CALCULATION LOOP ---
    candidates = []
    
    for p in static['elements']:
        # Filter: Position
        p_type_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        pos_str = p_type_map[p['element_type']]
        if pos_str not in position: continue
        
        # Filter: Minutes & Status
        if p['minutes'] < min_minutes: continue
        if p['status'] != 'a': continue # Only available players
        
        # Filter: Price
        price = p['now_cost'] / 10.0
        if not (price_min <= price <= price_max): continue

        # --- GET STATS ---
        try:
            # NOW USING TOTAL xGI (Not per 90)
            total_xgi = float(p.get('expected_goal_involvements', 0))
            form = float(p['form'])
            points = p['total_points']
        except:
            continue

        # --- GET FIXTURES ---
        t_id = p['team']
        schedule = team_schedules.get(t_id, [])
        
        if not schedule: continue
        
        # Calculate Average Difficulty (Next 10 Games)
        avg_diff = sum(m['diff'] for m in schedule) / len(schedule)
        
        # Create readable string for table
        # Format: "ARS(4), LIV(5), SOU(2)..."
        fixture_visual = ", ".join([f"{m['opp']}({m['diff']})" for m in schedule])

        # --- PREDICTION ALGORITHM ---
        # Normalize xGI (Top players have ~10-15 xGI, so we don't multiply by 10 like we did for xGI/90)
        score_xgi = total_xgi * 1.5 
        
        # Normalize Form (0-10 scale usually)
        score_form = form
        
        # Normalize Fixtures 
        # Difficulty 2 (Easy) -> Should give high points (e.g., 8)
        # Difficulty 4 (Hard) -> Should give low points (e.g., 2)
        score_fix = (5.5 - avg_diff) * 3.0
        
        # Weighted Sum
        final_score = (score_xgi * w_xgi) + (score_form * w_form) + (score_fix * w_fix)

        candidates.append({
            "Player": p['web_name'],
            "Team": static['teams'][p['team']-1]['short_name'],
            "Pos": pos_str,
            "Price": f"£{price}m",
            "Total xGI": total_xgi,
            "Form": form,
            "Avg Diff (10 Gms)": round(avg_diff, 2),
            "Next 10 Fixtures": fixture_visual,
            "Explosion Score": round(final_score, 2)
        })

    # --- DISPLAY ---
    if candidates:
        df = pd.DataFrame(candidates)
        # Sort by Score
        df = df.sort_values(by="Explosion Score", ascending=False).head(30)
        
        st.success(f"Analysis complete. Top {len(df)} picks based on Total xGI and Schedule.")
        
        st.dataframe(
            df,
            column_config={
                "Explosion Score": st.column_config.ProgressColumn(
                    "Model Probability",
                    help="Based on xGI, Form, and Fixtures",
                    format="%.2f",
                    min_value=0,
                    max_value=max(df["Explosion Score"])
                ),
                "Total xGI": st.column_config.NumberColumn(
                    "Total xGI",
                    help="Cumulative Expected Goal Involvement for the season",
                    format="%.2f"
                ),
                "Avg Diff (10 Gms)": st.column_config.NumberColumn(
                    "10-Game Difficulty",
                    help="Lower is easier. 1=Easy, 5=Hard",
                    format="%.2f"
                ),
                "Next 10 Fixtures": st.column_config.TextColumn(
                    "Upcoming Schedule (Diff)",
                    width="large"
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No players found. Try lowering the price or minutes filter.")

if __name__ == "__main__":
    main()
