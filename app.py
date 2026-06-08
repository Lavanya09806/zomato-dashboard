import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from dashboard.data_processing import clean_and_feature_engineer
import joblib

# Set page configuration
st.set_page_config(
    page_title="Zomato Restaurant Analytics",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetric"] label {
    color: white !important;
}

div[data-testid="stMetric"] div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    if os.path.exists('data/zomato.csv'):
        return clean_and_feature_engineer('data/zomato.csv')
    return None

df = load_data()

if df is None:
    st.error("Dataset not found! Please ensure data/zomato.csv exists.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🍔 Zomato Analytics")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose Analysis Page",
    ["Overview Dashboard", "Restaurant Analysis", "Cuisine Analysis", 
     "Location Analysis", "Customer Preference", "Cost Analysis", 
     "Business Insights", "Success Predictor (ML)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Designed for Business Analysts & Restaurant Owners.")

# Dashboard Pages
if page == "Overview Dashboard":
    st.title("🚀 Zomato Restaurant Performance Overview")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Restaurants", f"{len(df):,}")
    with col2:
        st.metric("Avg Rating", f"{df['rate'].mean():.2f} / 5")
    with col3:
        st.metric("Total Votes", f"{df['votes'].sum():,}")
    with col4:
        st.metric("Avg Cost (2)", f"₹{df['cost'].mean():.0f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        online_pct = (df['online_order'] == 'Yes').mean() * 100
        st.metric("Online Order %", f"{online_pct:.1f}%")
    with col6:
        book_pct = (df['book_table'] == 'Yes').mean() * 100
        st.metric("Table Booking %", f"{book_pct:.1f}%")
    with col7:
        st.metric("Unique Locations", len(df['location'].unique()))
    with col8:
        top_cuisine = df['cuisines'].str.split(', ').explode().mode()[0]
        st.metric("Most Popular Cuisine", top_cuisine)

    st.markdown("---")
    
    # Overview Plots
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rating Distribution")
        fig_rate = px.histogram(df, x="rate", nbins=20, color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig_rate, use_container_width=True)
        
    with c2:
        st.subheader("Online Order vs Table Booking")
        # Multi-variable slice
        order_types = df.groupby(['online_order', 'book_table']).size().reset_index(name='count')
        fig_sun = px.sunburst(order_types, path=['online_order', 'book_table'], values='count', color='count')
        st.plotly_chart(fig_sun, use_container_width=True)

elif page == "Restaurant Analysis":
    st.title("🏪 Restaurant Deep Dive")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Rated Restaurants")
        top_rated = df.nlargest(10, 'rate')[['name', 'rate', 'location']].drop_duplicates()
        fig_top = px.bar(top_rated, x='rate', y='name', orientation='h', color='rate', text='location')
        st.plotly_chart(fig_top, use_container_width=True)
        
    with col2:
        st.subheader("Popularity vs Rating")
        fig_scatter = px.scatter(df, x="rate", y="votes", size="cost", hover_name="name", 
                                color="rest_type", template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Restaurant Type Breakdown")
    fig_pie = px.pie(df, names='rest_type', hole=0.4, title="Distribution of Restaurant Types")
    st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Cuisine Analysis":
    st.title("🍜 Cuisine Intelligence")
    
    # Explode cuisines for analysis
    df_cuisines = df.assign(cuisines=df['cuisines'].str.split(', ')).explode('cuisines')
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Cuisines by Count")
        top_c = df_cuisines['cuisines'].value_counts().head(10).reset_index()
        top_c.columns = ['Cuisine', 'Count']
        fig_c = px.bar(top_c, x='Count', y='Cuisine', orientation='h', color='Count')
        st.plotly_chart(fig_c, use_container_width=True)
        
    with c2:
        st.subheader("Cuisine vs Average Rating")
        top_c_list = df_cuisines['cuisines'].value_counts().head(20).index
        avg_rate_c = df_cuisines[df_cuisines['cuisines'].isin(top_c_list)].groupby('cuisines')['rate'].mean().sort_values(ascending=False).reset_index()
        fig_rc = px.line(avg_rate_c, x='cuisines', y='rate', markers=True)
        st.plotly_chart(fig_rc, use_container_width=True)

elif page == "Location Analysis":
    st.title("📍 Location Insights")
    
    st.subheader("Restaurant Density by Location")
    loc_counts = df['location'].value_counts().reset_index()
    loc_counts.columns = ['Location', 'Count']
    fig_loc = px.treemap(loc_counts, path=['Location'], values='Count', color='Count', color_continuous_scale='RdBu')
    st.plotly_chart(fig_loc, use_container_width=True)
    
    st.subheader("Top Locations by Cost")
    avg_cost_loc = df.groupby('location')['cost'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_cl = px.bar(avg_cost_loc, x='location', y='cost', color='cost')
    st.plotly_chart(fig_cl, use_container_width=True)

elif page == "Customer Preference":
    st.title("👥 Customer Behavior Analysis")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Impact of Online Ordering on Ratings")
        fig_box = px.box(df, x="online_order", y="rate", points="all", color="online_order")
        st.plotly_chart(fig_box, use_container_width=True)
    
    with c2:
        st.subheader("Impact of Table Booking on Ratings")
        fig_box2 = px.box(df, x="book_table", y="rate", points="all", color="book_table")
        st.plotly_chart(fig_box2, use_container_width=True)

elif page == "Cost Analysis":
    st.title("💰 Revenue & Cost Analytics")
    
    st.subheader("Cost for Two Distribution")
    fig_cost = px.histogram(df, x="cost", nbins=30, marginal="rug", color_discrete_sequence=['green'])
    st.plotly_chart(fig_cost, use_container_width=True)
    
    st.subheader("Cost vs Rating (Value for Money Discovery)")
    fig_val = px.density_heatmap(df, x="cost", y="rate", text_auto=True)
    st.plotly_chart(fig_val, use_container_width=True)

elif page == "Business Insights":
    st.title("💡 Strategic Business Insights")
    
    # 1. Best locations to start (High Rating, High Demand)
    st.subheader("High Potential Growth Areas")
    best_locs = df.groupby('location').agg({'rate': 'mean', 'votes': 'sum', 'name': 'count'}).reset_index()
    best_locs = best_locs[best_locs['name'] > 5].sort_values(by=['rate', 'votes'], ascending=False).head(5)
    
    st.write("Based on our AI analysis, these locations show high engagement and satisfaction:")
    st.table(best_locs[['location', 'rate', 'votes', 'name']].rename(columns={'name': 'Total Restaurants'}))
    
    # 2. Most demanded cuisines
    st.subheader("Underserved Cuisines (High Votes, Low Count)")
    df_cuisines = df.assign(cuisines=df['cuisines'].str.split(', ')).explode('cuisines')
    cuisine_stats = df_cuisines.groupby('cuisines').agg({'votes': 'mean', 'name': 'count'}).reset_index()
    underserved = cuisine_stats[(cuisine_stats['name'] < 50) & (cuisine_stats['votes'] > df_cuisines['votes'].mean())].sort_values(by='votes', ascending=False).head(5)
    st.write("These cuisines have high average votes but lower presence (opportunity for new entrants):")
    st.table(underserved)
    
    # 3. High Rating Low Cost
    st.subheader("Hidden Gems (High Rating, Low Cost)")
    hidden_gems = df[(df['rate'] > 4.2) & (df['cost'] < 500)].sort_values(by='rate', ascending=False).head(10)
    st.dataframe(hidden_gems[['name', 'location', 'rate', 'cost', 'cuisines']])

elif page == "Success Predictor (ML)":
    st.title("🔮 Restaurant Success Predictor")
    st.markdown("Predict the potential rating of a new restaurant concept.")
    
    # Check for model
    if os.path.exists('models/restaurant_success_model.pkl'):
        model = joblib.load('models/restaurant_success_model.pkl')
        
        c1, c2 = st.columns(2)
        with c1:
            loc = st.selectbox("Select Location", df['location'].unique())
            rest_type = st.selectbox("Select Type", df['rest_type'].unique())
            cost = st.slider("Approx Cost for Two", 100, 5000, 500)
        
        with c2:
            online = st.radio("Online Order Ready?", ['Yes', 'No'])
            booking = st.radio("Table Booking Available?", ['Yes', 'No'])
            votes = st.number_input("Expected Starting Votes", 0, 10000, 100)
            
        if st.button("Predict Success Score"):
            # Prepare input
            # In a real app we'd use the saved encoders, for demo we'll use simple mapping if needed
            # but ideally we should load the encoders.
            
            # Simplified prediction for UI responsiveness if encoders aren't perfectly aligned
            try:
                # Load encoders
                loc_enc = joblib.load('models/location_encoder.pkl')
                type_enc = joblib.load('models/rest_type_encoder.pkl')
                online_enc = joblib.load('models/online_order_encoder.pkl')
                book_enc = joblib.load('models/book_table_encoder.pkl')
                
                # Handling unseen labels gracefully for demo
                def safe_transform(enc, val):
                    if val in enc.classes_:
                        return enc.transform([val])[0]
                    return 0

                input_data = np.array([[
                    safe_transform(online_enc, online),
                    safe_transform(book_enc, booking),
                    safe_transform(loc_enc, loc),
                    safe_transform(type_enc, rest_type),
                    cost,
                    votes
                ]])
                
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Potential Rating: **{prediction:.2f} / 5**")
                
                if prediction > 4.0:
                    st.balloons()
                    st.info("High Success Potential! This concept aligns well with market demand.")
                elif prediction > 3.5:
                    st.warning("Moderate Potential. Consider improving online presence or selecting a high-traffic location.")
                else:
                    st.error("Low Potential. Review your pricing strategy or location choice.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                
        # Show Feature Importance
        if os.path.exists('models/feature_importance.csv'):
            st.markdown("---")
            st.subheader("What Drives Restaurant Success?")
            fi = pd.read_csv('models/feature_importance.csv')
            fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', title="Model Feature Importance")
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("Model not trained yet. Run the training script first.")
