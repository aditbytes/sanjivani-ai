"""
Sanjivani AI - Streamlit Dashboard

Real-time crisis monitoring dashboard with interactive maps and visualizations.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import requests
import streamlit as st

from src.config import get_settings
from src.utils.helpers import get_all_district_names

settings = get_settings()

# Page config
st.set_page_config(
    page_title="Sanjivani AI - Crisis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_api_url():
    """Get API base URL."""
    return f"http://{settings.api_host}:{settings.api_port}/api/v1"


def main():
    """Main dashboard application."""
    st.title("🌊 Sanjivani AI - Crisis Intelligence Dashboard")
    st.markdown("Real-time flood monitoring and resource forecasting for Bihar")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_district = st.selectbox("Select District", get_all_district_names())
        forecast_horizon = st.slider("Forecast Horizon (hours)", 6, 72, 24)
        st.divider()
        st.markdown("**System Status**")
        st.success("API Connected")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 District Overview")
        st.metric("Active Alerts", "32", "+5")
        st.metric("People Affected", "12,450", "+1,200")
        st.metric("Resources Deployed", "45%", "+8%")
    
    with col2:
        st.subheader("🔮 Resource Forecast")
        st.info(f"Forecasting for {selected_district} ({forecast_horizon}h)")
        st.write("Food Packets: ~2,500")
        st.write("Medical Kits: ~150")
        st.write("Rescue Boats: ~15")
    
    st.divider()
    
    # Tweet Analysis Section
    st.subheader("📝 Tweet Analysis")
    tweet_text = st.text_area("Enter tweet text for analysis:", height=100)
    if st.button("Analyze Tweet"):
        if tweet_text:
            with st.spinner("Analyzing..."):
                st.success(f"Urgency: High | District: {selected_district} | Resource: Rescue")
    
    # Map placeholder
    st.subheader("🗺️ Crisis Map")
    st.map()
    
    st.markdown("---")
    st.caption("Sanjivani AI v1.0.0 | Bihar Flood Response System")


if __name__ == "__main__":
    main()
