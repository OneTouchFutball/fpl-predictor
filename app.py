import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL AI Pro 25/26", page_icon="🧠", layout="wide")

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# --- 1. ROBUST DATA DOWNLOADER (Internal) ---
def download_historical_data_internal():
    """
    Downloads training data if CSV is missing.
    """
    status_placeholder = st.empty()
    status_placeholder.info("⏳ First run detected: Downloading 5 years of history to train AI...")
    
    # Including 2025-26 to get latest trends if available
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_data = []

    progress_bar = st.progress(0)
    
    for i, season in enumerate(seasons):
        try:
            url = f"{base_url}/{season}/gws/merged_gw.csv"
            response = requests.get(url)
            
            if response.status_code == 200:
                # robust reading
                df = pd.read_csv(io.BytesIO(response.content), encoding='utf-8', on_bad_lines='skip', low_memory=False)
                df['season_id'] = season
                
                cols_to_keep = [
                    'name', 'position', 'team', 'minutes', 'total_points', 'was_home', 
                    'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                    'expected_goals', 'expected_assists', 'influence', 'creativity', 'threat',
                    'value', 'element_type'
                ]
                existing = [c for c in cols_to_keep if c in df.columns]
                df = df[existing]
                
                if 'value' in df.columns:
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                all_data.append(df)
        except Exception:
            pass # Skip broken seasons seamlessly
        
        progress_bar.progress((i + 1) / len(seasons))

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.fillna(0, inplace=True)
        if 'was_home' in master_df.columns:
            master_df['was_home'] = master_df['was_home'].astype(bool).astype(int)
            
        master_df.to_csv("fpl_5_year_history.csv", index=False)
        status_placeholder.success("✅ Historical Data Secured.")
        progress_bar.empty()
        return master_df
    else:
        status_placeholder.error("❌ Failed to download training data.")
        return None

# --- 2. AI ENGINE (TRAINING) ---
@st.cache_resource
def train_brain():
    if not os.path.exists("fpl_5_year_history.csv"):
        df = download_historical_data_internal()
    else:
        df = pd.read_csv("fpl_5_year_history.csv")
        
    if df is None: return None, None, "ERROR"

    # Filter for significant minutes to reduce noise
    df_starters = df[df['minutes'] > 60].copy()
    
    # Handle Position Encoding
    if 'position' not in df_starters.columns and 'element_type' in df_starters.columns:
        df_starters['pos_code'] = df_starters['element_type']
    elif 'position' in df_starters.columns:
        le = LabelEncoder()
        df_starters['pos_code'] = le.fit_transform(df_starters['position'].astype(str))
    else:
        df_starters['pos_code'] = 0 

    # Features: Value, Home/Away, and Underlying Stats
    features = [
        'value', 'pos_code', 'was_home',
        'expected_goals', 'expected_assists', 
        'clean_sheets', 'goals_conceded', 
        'influence', 'creativity', 'threat'
    ]
    
    valid_features = [f for f in features if f in df_starters.columns]
    
    X = df_starters[valid_features]
    y = df_starters['total_points']
    
    # Train Gradient Boosting (The Brain)
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X, y)
    
    return model, valid_features, "OK"

# --- 3. LIVE DATA & FIXTURES ---
@st.cache_data(ttl=600)
def get_live_data_and_fixtures():
    # 1. Static Data
    static = requests.get(f"{API_BASE}/bootstrap-static/").json()
    
    # 2. Fixtures
    fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    
    return static, fixtures

# --- 4. FIXTURE PROCESSING ---
def calculate_fixture_difficulty(static, fixtures, horizon):
    """
    Returns a map of Team ID -> {multiplier: float, display_text: str}
    based on the next N fixtures.
    """
    team_map = {t['id']: t['short_name'] for t in static['teams']}
    team_data = {t['id']: {'diff_sum': 0, 'count': 0, 'opponents': []} for t in static['teams']}
    
    # Filter future fixtures
    future_fixtures = [f for f in fixtures if not f['finished'] and f['kickoff_time']]
    
    for f in future_fixtures:
        h = f['team_h']
        a = f['team_a']
        
        # Process Home Team
        if team_data[h]['count'] < horizon:
            diff = f['team_h_difficulty']
            team_data[h]['diff_sum'] += diff
            team_data[h]['count'] += 1
            team_data[h]['opponents'].append(f"{team_map[a]}(H)")
            
        # Process Away Team
        if team_data[a]['count'] < horizon:
            diff = f['team_a_difficulty']
            team_data[a]['diff_sum'] += diff
            team_data[a]['count'] += 1
            team_data[a]['opponents'].append(f"{team_map[h]}(A)")
            
    # Calculate Multipliers
    results = {}
    for tid, data in team_data.items():
        if data['count'] > 0:
            avg_diff = data['diff_sum'] / data['count']
            # Logic: Difficulty 2 (Easy) -> Multiplier 1.2
            #        Difficulty 4 (Hard) -> Multiplier 0.8
            # Formula: 1.4 - (AvgDiff * 0.2)
            mult = 1.4 - (avg_diff * 0.15) 
            display = ", ".join(data['opponents'])
        else:
            mult = 1.0
            display = "-"
            
        results[tid] = {'mult': mult, 'display': display}
        
    return results

# --- MAIN APP ---
def main():
    st.title("🧠 FPL AI: Hybrid Prediction Engine")
    st.markdown("""
    **Logic:** 
    1. **Deep Learning:** Predicts "Base Performance" using 5 years of player stat correlations.
    2. **Fixture Engine:** Adjusts that prediction based on the specific next opponents you select.
    """)
    
    # --- LOAD ---
    with st.spinner("Training AI & Loading Data..."):
        model, feature_cols, status = train_brain()
        static, fixtures = get_live_data_and_fixtures()
        
    if status == "ERROR":
        st.error("Critical Data Error.")
        return

    # --- CONTROLS ---
    st.sidebar.header("⚙️ Configuration")
    
    # Horizon
    horizon = st.sidebar.selectbox("Prediction Horizon", [1, 5, 10], format_func=lambda x: f"Next {x} Fixture{'s' if x>1 else ''}")
    
    # Price Weight
    w_budget = st.sidebar.slider("Price Importance", 0.0, 1.0, 0.5, help="0.0 = Pick Best Players. 1.0 = Pick Best Value.")
    
    # Filters
    st.sidebar.divider()
    min_price = st.sidebar.number_input("Min Price (£m)", 3.5, 15.0, 4.0)
    pos_filter = st.sidebar.selectbox("Position", ["All", "GK", "DEF", "MID", "FWD"])

    # --- PROCESSING ---
    
    # 1. Process Fixtures based on Horizon
    fix_data = calculate_fixture_difficulty(static, fixtures, horizon)
    
    # 2. Prepare Player Data for AI
    df = pd.DataFrame(static['elements'])
    
    # Filter players with minutes
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 1] # Must have played a bit
    
    # Fix Price (API is int, we need float)
    df['price_real'] = df['now_cost'] / 10.0
    
    # Normalize Stats for AI Input
    # Map API columns to Model Training columns
    stats_map = {
        'expected_goals': 'expected_goals_per_90',
        'expected_assists': 'expected_assists_per_90',
        'clean_sheets': 'clean_sheets_per_90',
        'goals_conceded': 'goals_conceded_per_90',
        'influence': 'influence', 
        'creativity': 'creativity',
        'threat': 'threat'
    }
    
    # Create Input DataFrame
    X_pred = pd.DataFrame()
    X_pred['value'] = df['now_cost'] # AI trained on raw integer cost
    X_pred['pos_code'] = df['element_type']
    X_pred['was_home'] = 0.5 # Neutral for base prediction
    
    for model_col, api_col in stats_map.items():
        if 'per_90' in api_col:
            X_pred[model_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0)
        else:
            # Divide totals by games played
            X_pred[model_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']
            
    # 3. AI Prediction (Base Points)
    base_points = model.predict(X_pred[feature_cols])
    
    # 4. Apply Fixture & Price Logic
    results = []
    team_names = {t['id']: t['name'] for t in static['teams']}
    
    for idx, (pid, pred) in enumerate(zip(df['id'], base_points)):
        row = df.iloc[idx]
        tid = row['team']
        
        # Get Fixture Multiplier
        f_info = fix_data.get(tid, {'mult': 1.0, 'display': '-'})
        
        # Adjusted Prediction
        final_pts = pred * f_info['mult']
        
        # ROI Calculation: Points / (Price ^ Weight)
        # If Weight 0: Div by 1. If Weight 1: Div by Price.
        price_factor = row['price_real'] ** w_budget
        roi = final_pts / price_factor if price_factor > 0 else 0
        
        results.append({
            "Player": row['web_name'],
            "Team": team_names.get(tid, "Unknown"),
            "Pos": row['element_type'],
            "Price": row['price_real'],
            "Upcoming": f_info['display'],
            "AI Base": pred,
            "Fix Mult": f_info['mult'],
            "Exp. Pts": final_pts,
            "ROI Index": roi * 10 # Scale up for visuals
        })
        
    # --- DISPLAY ---
    res_df = pd.DataFrame(results)
    
    # Filter Position
    if pos_filter != "All":
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        res_df = res_df[res_df['Pos'] == pos_map[pos_filter]]
        
    # Filter Price
    res_df = res_df[res_df['Price'] >= min_price]
    
    # Sort
    res_df = res_df.sort_values(by="ROI Index", ascending=False).head(50)
    
    # Render
    st.dataframe(
        res_df[['ROI Index', 'Player', 'Team', 'Price', 'Exp. Pts', 'Upcoming']],
        hide_index=True,
        use_container_width=True,
        column_config={
            "ROI Index": st.column_config.ProgressColumn("AI Value Score", format="%.1f", min_value=0, max_value=15),
            "Exp. Pts": st.column_config.NumberColumn("Exp. Pts", format="%.2f", help=f"AI Prediction adjusted for next {horizon} games."),
            "Price": st.column_config.NumberColumn("Price", format="£%.1f"),
            "Upcoming": st.column_config.TextColumn("Next Opponents", width="medium"),
        }
    )

if __name__ == "__main__":
    main()
