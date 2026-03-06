import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from music_query_parser.parser import MusicQueryParser

st.set_page_config(page_title="Bayesian Music Engine", page_icon="🎵", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('kaggle_dataset.csv')

@st.cache_resource
def load_parser():
    return MusicQueryParser()

df = load_data()
parser = load_parser()

with st.sidebar:
    st.header("⚙️ Bayesian Control Panel")
    all_genres = sorted(df['track_genre'].unique())
    selected_genres = st.multiselect("Genre Prior", all_genres, default=all_genres[:3])
    sigma = st.slider("Model Uncertainty (σ)", 0.01, 1.0, 0.2)

st.title("🎵 Bayesian Music Recommendation Engine")
user_vibe = st.text_input("What's the vibe?", placeholder="e.g., Chill indie pop or high energy party")

if st.button("Generate Bayesian Playlist") and user_vibe:
    # 1. Parsing (Step 2 & 3)
    query_spec = parser.parse(user_vibe)
    constraints = query_spec.constraints
    
    with st.expander("Show Bayesian Inference Logic"):
        st.json(query_spec.to_dict())

    # 2. Applying Priors
    candidates = df[df['track_genre'].isin(selected_genres)].copy()

    # 3. Dynamic Likelihood Calculation
    # We loop through whatever features the parser found (energy, danceability, etc.)
    candidates['likelihood'] = 1.0
    for feature, (min_val, max_val) in constraints.items():
        if feature in candidates.columns:
            # Calculate center of the range to use as our "Target"
            target = (min_val + max_val) / 2
            # Update Likelihood: P(Vibe | Song)
            dist = np.exp(-((candidates[feature] - target)**2) / (2 * sigma**2))
            candidates['likelihood'] *= dist

    # 4. Ranking
    results = candidates.sort_values('likelihood', ascending=False).head(15)

    # 5. Display Results
    st.success(f"Top matches for: {user_vibe}")
    
    st.subheader("✨ Bayesian Inference Results")
    
    # This loop creates the "Certainty" rows
    for idx, row in results.iterrows():
        confidence = row['likelihood'] * 100
        
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{row['track_name']}**")
                st.caption(f"{row['artists']} | {row['track_genre']}")
            with col2:
                # This bar shows the 'Posterior Probability' visually
                st.write(f"Match Certainty: {confidence:.1f}%")
                st.progress(row['likelihood']) 
            with col3:
                st.metric("Score", f"{row['likelihood']:.2f}")
            st.divider()

    # The Visualization (Posterior Distribution)
    st.subheader("📊 Posterior Probability Distribution")
    fig_prob = px.bar(results, x='track_name', y='likelihood', 
                      color='likelihood', title="Certainty per Track",
                      color_continuous_scale='Viridis')
    st.plotly_chart(fig_prob, use_container_width=True)
    
    st.balloons()
