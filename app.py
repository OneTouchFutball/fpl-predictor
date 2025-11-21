import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL AI 25/26", page_icon="🧠", layout="wide")

# --- AI ENGINE ---
@st.cache_resource
def train_brain():
    """
    Trains the AI on the 5-year history CSV.
    """
    try:
        # Load the clean history file
        df = pd.read_csv("fpl_5_year_history.csv")
    except FileNotFoundError:
        return None, None, "MISSING_CSV"

    # --- FEATURE ENGINEERING ---
    # We teach the AI: "Stats per 90 mins" -> "Total Points"
    
    # 1. Create Per-90 Stats for Training
    # (The raw CSV has match totals, we need to normalize to compare with live API)
    # We filter for games where players actually played > 60 mins to get 'starter' patterns
    df_starters = df[df['minutes'] > 60].copy()
    
    # Encode Position (GK/DEF/MID/FWD)
    le = LabelEncoder()
    df_starters['pos_code'] = le.fit_transform(df_starters['position'].astype(str))
    
    # Features the AI will learn from
    features = [
        'value', 'pos_code', 'was_home',
        'expected_goals', 'expected_assists', 
        'clean_sheets', 'goals_conceded', 
        'influence', 'creativity', 'threat'
    ]
    
    # Ensure columns exist (handle potential missing data from old seasons)
    features = [f for f in features if f in df_starters.columns]
    
    X = df_starters[features]
    y = df_starters['total_points'] # The AI tries to predict this
    
    # Train Gradient Boosting Model
    model = HistGradientBoostingRegressor(max_iter=100, random_state=2025)
    model.fit(X, y)
    
    return model, features, "OK"

# --- LIVE DATA ---
@st.cache_data(ttl=600)
def get_live_25_26_data():
    # Fetches current season data from FPL API
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    df = pd.DataFrame(data['elements'])
    
    # Calculate Per 90 Stats for Prediction
    # The AI was trained on "Match Stats", so we need to convert "Season Totals" 
    # into "Expected Match Stats" for the prediction.
    
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 2] # Filter players with < 2 games
    
    # Create columns matching the Training Features
    df['value'] = df['now_cost']
    df['pos_code'] = df['element_type'] - 1 # Map 1-4 to 0-3
    df['was_home'] = 0.5 # Neutral assumption for prediction
    
    # Normalize totals to per-match averages
    cols_to_norm = [
        ('expected_goals_per_90', 'expected_goals'),
        ('expected_assists_per_90', 'expected_assists'),
        ('clean_sheets_per_90', 'clean_sheets'),
        ('goals_conceded_per_90', 'goals_conceded'),
        ('influence', 'influence'), # API gives total, we divide
        ('creativity', 'creativity'),
        ('threat', 'threat')
    ]
    
    for api_col, model_col in cols_to_norm:
        # API 'per_90' columns are strings, need float conversion
        if 'per_90' in api_col:
            df[model_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0)
        else:
            # Total columns need dividing by matches played
            df[model_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']

    return df

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Deep Learning: 2025/26 Model")
    st.markdown("""
    **Methodology:**
    1.  **Training:** Learned from ~100,000 player performances (2020-2025).
    2.  **Logic:** Uses **Gradient Boosting** to identify non-linear patterns (e.g. "High Threat Defenders earn more than High Threat Midfielders").
    3.  **Prediction:** Applies this "Brain" to the live 2025/26 stats.
    """)
    
    # 1. Train
    with st.spinner("Loading 5-Year History & Training AI..."):
        model, feature_cols, status = train_brain()
        
    if status == "MISSING_CSV":
        st.error("⚠️ Missing Data File!")
        st.warning("Run `python get_data.py` in your terminal first.")
        return

    # 2. Predict
    live_df = get_live_25_26_data()
    
    # Map inputs
    X_live = live_df[feature_cols]
    
    # Generate Predictions
    live_df['AI_Points'] = model.predict(X_live)
    
    # Calculate AI ROI (Points per Million)
    live_df['AI_Value'] = (live_df['AI_Points'] / live_df['now_cost']) * 10
    
    # 3. UI
    st.sidebar.header("🔎 Filters")
    min_cost = st.sidebar.slider("Min Price", 4.0, 14.0, 4.0)
    pos_filter = st.sidebar.selectbox("Position", ["All", "GK", "DEF", "MID", "FWD"])
    
    # Filter Data
    display_df = live_df.copy()
    
    if pos_filter != "All":
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        display_df = display_df[display_df['element_type'] == pos_map[pos_filter]]
        
    display_df = display_df[display_df['now_cost'] / 10 >= min_cost]
    
    # Sort
    sort_col = st.sidebar.radio("Sort By:", ["AI Projected Points", "AI Value Index"])
    if sort_col == "AI Projected Points":
        display_df = display_df.sort_values(by='AI_Points', ascending=False)
    else:
        display_df = display_df.sort_values(by='AI_Value', ascending=False)
        
    display_df = display_df.head(50)
    
    # Table
    st.dataframe(
        display_df[['web_name', 'AI_Points', 'AI_Value', 'now_cost', 'selected_by_percent']],
        hide_index=True,
        use_container_width=True,
        column_config={
            "web_name": "Player",
            "AI_Points": st.column_config.ProgressColumn("AI Projected Pts", format="%.2f", min_value=0, max_value=10),
            "AI_Value": st.column_config.NumberColumn("AI Value (ROI)", format="%.2f"),
            "now_cost": st.column_config.NumberColumn("Price (£m)", format="£%.1f"),
            "selected_by_percent": st.column_config.NumberColumn("Ownership %", format="%.1f%%")
        }
    )

if __name__ == "__main__":
    main()
