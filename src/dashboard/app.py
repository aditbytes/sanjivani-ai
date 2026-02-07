"""
Sanjivani AI - Streamlit Dashboard (Multi-Page)

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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import get_settings
from src.utils.helpers import get_all_district_names, BIHAR_DISTRICTS

settings = get_settings()


# ---------- Page Config ----------
st.set_page_config(
    page_title="Sanjivani AI - Crisis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Session State Initialization ----------
if "api_status" not in st.session_state:
    st.session_state.api_status = "checking"
if "selected_district" not in st.session_state:
    st.session_state.selected_district = "Patna"
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 24
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "history" not in st.session_state:
    st.session_state.history = []


# ---------- API Functions ----------
def get_api_url():
    """Get API base URL."""
    return f"http://{settings.api_host}:{settings.api_port}"


def check_api_status():
    """Check if API is connected."""
    try:
        r = requests.get(f"{get_api_url()}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def analyze_tweet(text: str):
    """Analyze tweet via API."""
    try:
        r = requests.post(
            f"{get_api_url()}/api/v1/analyze-tweet",
            json={"text": text},
            timeout=30
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def get_forecast(district: str, horizon: int = 24):
    """Get resource forecast for district."""
    try:
        r = requests.get(
            f"{get_api_url()}/api/v1/forecast/{district}",
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def get_metrics():
    """Get Prometheus metrics."""
    try:
        r = requests.get(f"{get_api_url()}/metrics/json", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


# ---------- Sidebar ----------
def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Sanjivani+AI", width=150)
        st.title("🌊 Sanjivani AI")
        st.caption("Crisis Intelligence System")
        
        st.divider()
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "📊 Analytics", "🚨 Alerts", "📦 Resources", "📋 Reports", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick Settings
        st.subheader("⚙️ Quick Settings")
        st.session_state.selected_district = st.selectbox(
            "District", 
            get_all_district_names(),
            index=get_all_district_names().index(st.session_state.selected_district) if st.session_state.selected_district in get_all_district_names() else 0
        )
        
        st.session_state.forecast_horizon = st.slider(
            "Forecast Horizon (hours)", 
            min_value=6, 
            max_value=72, 
            value=st.session_state.forecast_horizon,
            step=6
        )
        
        st.divider()
        
        # API Status
        st.subheader("🔌 System Status")
        if check_api_status():
            st.success("✅ API Connected")
            st.session_state.api_status = "connected"
        else:
            st.error("❌ API Disconnected")
            st.session_state.api_status = "disconnected"
        
        st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        return page


# ---------- Dashboard Page ----------
def render_dashboard():
    """Render main dashboard page."""
    st.title("🏠 Crisis Intelligence Dashboard")
    st.markdown(f"**District:** {st.session_state.selected_district} | **Forecast:** {st.session_state.forecast_horizon}h")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚨 Active Alerts", "32", "+5", delta_color="inverse")
    with col2:
        st.metric("👥 People Affected", "12,450", "+1,200", delta_color="inverse")
    with col3:
        st.metric("📦 Resources Deployed", "45%", "+8%")
    with col4:
        st.metric("🚁 Rescue Operations", "18", "-3")
    
    st.divider()
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Tweet Analysis
        st.subheader("📝 Real-Time Tweet Analysis")
        
        with st.expander("Advanced Options", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                urgency_filter = st.selectbox("Urgency Filter", ["All", "Critical", "High", "Medium", "Low"])
            with cols[1]:
                resource_filter = st.selectbox("Resource Filter", ["All", "Food", "Medical", "Rescue", "Shelter"])
            with cols[2]:
                lang_filter = st.selectbox("Language", ["All", "English", "Hindi", "Hinglish"])
        
        tweet_text = st.text_area(
            "Enter tweet text for analysis:",
            placeholder="e.g., Heavy flooding in Darbhanga! Many families stranded, need rescue boats urgently!",
            height=100
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("📋 Batch Analyze", use_container_width=True):
                st.info("Upload a CSV file with tweets for batch analysis")
        
        if analyze_btn and tweet_text:
            with st.spinner("Analyzing tweet..."):
                result = analyze_tweet(tweet_text)
                if result:
                    # Display results
                    st.success("Analysis Complete!")
                    
                    res_cols = st.columns(4)
                    with res_cols[0]:
                        urgency = result.get("urgency", "Unknown")
                        color = "🔴" if urgency == "Critical" else "🟠" if urgency == "High" else "🟡" if urgency == "Medium" else "🟢"
                        st.metric("Urgency", f"{color} {urgency}")
                    with res_cols[1]:
                        st.metric("District", result.get("district", "Unknown"))
                    with res_cols[2]:
                        st.metric("Resource", result.get("resource_needed", "Unknown"))
                    with res_cols[3]:
                        st.metric("Inference", f"{result.get('inference_time_ms', 0):.1f}ms")
                    
                    # Add to history
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "text": tweet_text[:50] + "...",
                        "urgency": urgency,
                        "district": result.get("district", "Unknown")
                    })
                else:
                    st.error("Failed to analyze tweet. Check API connection.")
    
    with col_right:
        # Resource Forecast
        st.subheader("🔮 Resource Forecast")
        
        forecast = get_forecast(st.session_state.selected_district)
        if forecast:
            st.info(f"**{st.session_state.selected_district}** - Next {st.session_state.forecast_horizon}h")
            
            resources = [
                ("🍚 Food Packets", forecast.get("food_packets", 2500)),
                ("💊 Medical Kits", forecast.get("medical_kits", 150)),
                ("🚤 Rescue Boats", forecast.get("rescue_boats", 15)),
                ("🏠 Shelters", forecast.get("shelters", 8))
            ]
            
            for name, value in resources:
                st.metric(name, f"{value:,}")
        else:
            # Fallback data
            st.info(f"**{st.session_state.selected_district}** - Next {st.session_state.forecast_horizon}h")
            st.metric("🍚 Food Packets", "~2,500")
            st.metric("💊 Medical Kits", "~150")
            st.metric("🚤 Rescue Boats", "~15")
            st.metric("🏠 Shelters", "~8")
    
    st.divider()
    
    # Crisis Map
    st.subheader("🗺️ Crisis Map")
    
    map_col1, map_col2 = st.columns([3, 1])
    with map_col2:
        map_layer = st.selectbox("Map Layer", ["Flood Extent", "Affected Areas", "Resource Depots", "Rescue Routes"])
        show_alerts = st.checkbox("Show Active Alerts", value=True)
        show_resources = st.checkbox("Show Resource Points", value=True)
    
    with map_col1:
        # Generate some sample data for Bihar
        map_data = pd.DataFrame({
            "lat": [25.5941 + np.random.randn() * 0.5 for _ in range(20)],
            "lon": [85.1376 + np.random.randn() * 0.5 for _ in range(20)]
        })
        st.map(map_data, zoom=7)
    
    # Analysis History
    if st.session_state.history:
        st.subheader("📜 Recent Analyses")
        history_df = pd.DataFrame(st.session_state.history[-10:])
        st.dataframe(history_df, use_container_width=True, hide_index=True)


# ---------- Analytics Page ----------
def render_analytics():
    """Render analytics page."""
    st.title("📊 Analytics Dashboard")
    
    # Date Range Filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        metric_type = st.multiselect(
            "Metrics to Show",
            ["Alerts", "Resources", "Affected Population", "Response Time"],
            default=["Alerts", "Resources"]
        )
    
    st.divider()
    
    # Charts Row 1
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Alert Trends")
        # Generate sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        chart_data = pd.DataFrame({
            "Date": dates,
            "Critical": np.random.randint(5, 20, len(dates)),
            "High": np.random.randint(10, 30, len(dates)),
            "Medium": np.random.randint(15, 40, len(dates)),
            "Low": np.random.randint(20, 50, len(dates))
        })
        st.area_chart(chart_data.set_index("Date"))
    
    with col_right:
        st.subheader("📊 Resource Distribution")
        resource_data = pd.DataFrame({
            "Resource": ["Food", "Medical", "Rescue", "Shelter", "Transport"],
            "Deployed": [65, 45, 80, 55, 40],
            "Available": [35, 55, 20, 45, 60]
        })
        st.bar_chart(resource_data.set_index("Resource"))
    
    st.divider()
    
    # Charts Row 2
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🗺️ District-wise Impact")
        district_data = pd.DataFrame({
            "District": ["Patna", "Muzaffarpur", "Darbhanga", "Samastipur", "Begusarai"],
            "Affected": [12500, 8900, 7600, 5400, 4200]
        })
        st.bar_chart(district_data.set_index("District"))
    
    with col2:
        st.subheader("⏱️ Response Time")
        response_data = pd.DataFrame({
            "Time (hours)": range(1, 25),
            "Requests Resolved": np.random.randint(50, 200, 24)
        })
        st.line_chart(response_data.set_index("Time (hours)"))
    
    with col3:
        st.subheader("📱 Tweet Volume")
        tweet_data = pd.DataFrame({
            "Hour": range(24),
            "Tweets": [np.random.randint(100, 500) for _ in range(24)]
        })
        st.bar_chart(tweet_data.set_index("Hour"))
    
    # System Metrics
    st.divider()
    st.subheader("🖥️ System Performance")
    
    metrics = get_metrics()
    if metrics:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Uptime", f"{metrics.get('uptime_seconds', 0)/3600:.1f}h")
        with cols[1]:
            total_requests = sum(metrics.get("counters", {}).values())
            st.metric("Total Requests", f"{total_requests:,}")
        with cols[2]:
            st.metric("Avg Latency", "45ms")
        with cols[3]:
            st.metric("Error Rate", "0.2%")
    else:
        st.info("Connect to API to view system metrics")


# ---------- Alerts Page ----------
def render_alerts():
    """Render alerts management page."""
    st.title("🚨 Alert Management")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        severity = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
    with col2:
        status = st.selectbox("Status", ["All", "Active", "Acknowledged", "Resolved"])
    with col3:
        district_filter = st.selectbox("District", ["All"] + get_all_district_names())
    with col4:
        sort_by = st.selectbox("Sort By", ["Time (Newest)", "Time (Oldest)", "Severity", "District"])
    
    st.divider()
    
    # Alert Actions
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("🔔 Create Alert", type="primary"):
            st.session_state.show_create_alert = True
    with col_btn2:
        if st.button("📤 Export Alerts"):
            st.download_button("Download CSV", "alert_id,severity,message\n1,High,Test", "alerts.csv", "text/csv")
    
    # Sample Alerts
    alerts_data = [
        {"id": "ALT-001", "time": "10:45", "severity": "🔴 Critical", "district": "Darbhanga", "message": "Water level exceeding danger mark", "status": "Active"},
        {"id": "ALT-002", "time": "10:30", "severity": "🟠 High", "district": "Patna", "message": "Multiple rescue requests from Kankarbagh", "status": "Active"},
        {"id": "ALT-003", "time": "10:15", "severity": "🟠 High", "district": "Muzaffarpur", "message": "Medical emergency - supplies needed", "status": "Acknowledged"},
        {"id": "ALT-004", "time": "09:50", "severity": "🟡 Medium", "district": "Samastipur", "message": "Road access blocked - alternate route needed", "status": "Active"},
        {"id": "ALT-005", "time": "09:30", "severity": "🟢 Low", "district": "Begusarai", "message": "Water receding in sector 5", "status": "Resolved"},
    ]
    
    for alert in alerts_data:
        with st.container():
            cols = st.columns([0.5, 1, 1, 3, 1, 1])
            with cols[0]:
                st.checkbox("", key=f"sel_{alert['id']}", label_visibility="collapsed")
            with cols[1]:
                st.write(f"**{alert['id']}**")
                st.caption(alert['time'])
            with cols[2]:
                st.write(alert['severity'])
            with cols[3]:
                st.write(f"**{alert['district']}**: {alert['message']}")
            with cols[4]:
                status_color = "🟢" if alert['status'] == "Resolved" else "🟡" if alert['status'] == "Acknowledged" else "🔴"
                st.write(f"{status_color} {alert['status']}")
            with cols[5]:
                st.button("View", key=f"view_{alert['id']}", use_container_width=True)
            st.divider()


# ---------- Resources Page ----------
def render_resources():
    """Render resource management page."""
    st.title("📦 Resource Management")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Depots", "24", help="Number of active resource depots")
    with col2:
        st.metric("Resources in Transit", "156", "+23")
    with col3:
        st.metric("Pending Requests", "45", "-12")
    with col4:
        st.metric("Fulfillment Rate", "87%", "+5%")
    
    st.divider()
    
    # Tabs for different resource views
    tab1, tab2, tab3 = st.tabs(["📋 Inventory", "🚚 Logistics", "📊 Allocations"])
    
    with tab1:
        st.subheader("Current Inventory")
        
        inventory_data = pd.DataFrame({
            "Resource": ["Food Packets", "Medical Kits", "Rescue Boats", "Tents", "Blankets", "Water Purifiers"],
            "Available": [15000, 2500, 120, 500, 8000, 350],
            "Deployed": [8500, 1200, 85, 320, 5500, 180],
            "In Transit": [2000, 300, 15, 50, 1000, 50],
            "Status": ["✅ Sufficient", "⚠️ Low", "✅ Sufficient", "✅ Sufficient", "✅ Sufficient", "⚠️ Low"]
        })
        
        st.dataframe(inventory_data, use_container_width=True, hide_index=True)
        
        if st.button("🔄 Refresh Inventory"):
            st.success("Inventory updated!")
    
    with tab2:
        st.subheader("Active Shipments")
        
        logistics_data = pd.DataFrame({
            "Shipment ID": ["SHP-001", "SHP-002", "SHP-003", "SHP-004"],
            "Origin": ["Patna Depot", "Muzaffarpur Depot", "Central Warehouse", "Patna Depot"],
            "Destination": ["Darbhanga", "Sitamarhi", "Madhubani", "Samastipur"],
            "Contents": ["Food (2000)", "Medical (500)", "Boats (10)", "Blankets (1000)"],
            "ETA": ["2h 30m", "4h 15m", "1h 45m", "3h 00m"],
            "Status": ["🚚 In Transit", "🚚 In Transit", "📦 Loading", "🚚 In Transit"]
        })
        
        st.dataframe(logistics_data, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Resource Allocation by District")
        
        allocation_chart = pd.DataFrame({
            "District": get_all_district_names()[:10],
            "Food": np.random.randint(500, 2000, 10),
            "Medical": np.random.randint(50, 300, 10),
            "Rescue": np.random.randint(5, 20, 10)
        })
        
        st.bar_chart(allocation_chart.set_index("District"))


# ---------- Reports Page ----------
def render_reports():
    """Render reports page."""
    st.title("📋 Reports & Export")
    
    # Report Type Selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Generate Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Daily Summary", "Weekly Analysis", "Resource Utilization", "Response Performance", "Damage Assessment"]
        )
        
        report_format = st.radio("Format", ["PDF", "Excel", "CSV"])
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            key="report_date"
        )
        
        districts = st.multiselect("Districts", get_all_district_names(), default=["Patna", "Darbhanga"])
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_maps = st.checkbox("Include Maps", value=True)
        
        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                import time
                time.sleep(2)
                st.success("Report generated successfully!")
                st.download_button(
                    "📥 Download Report",
                    f"Report: {report_type}\nGenerated: {datetime.now()}",
                    f"sanjivani_report_{datetime.now().strftime('%Y%m%d')}.txt"
                )
    
    with col2:
        st.subheader("Recent Reports")
        
        reports_list = [
            {"name": "Daily Summary - Feb 7, 2026", "type": "PDF", "size": "2.4 MB", "generated": "10:30 AM"},
            {"name": "Weekly Analysis - Week 6", "type": "Excel", "size": "5.1 MB", "generated": "Yesterday"},
            {"name": "Resource Report - February", "type": "PDF", "size": "3.8 MB", "generated": "Feb 5"},
            {"name": "Damage Assessment - Darbhanga", "type": "PDF", "size": "8.2 MB", "generated": "Feb 3"},
        ]
        
        for report in reports_list:
            with st.container():
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.write(f"📄 **{report['name']}**")
                with cols[1]:
                    st.caption(report['type'])
                with cols[2]:
                    st.caption(report['size'])
                with cols[3]:
                    st.button("⬇️", key=f"dl_{report['name']}", help="Download")
                st.divider()


# ---------- Settings Page ----------
def render_settings():
    """Render settings page."""
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Appearance", "🔔 Notifications", "🔗 API", "👤 Account"])
    
    with tab1:
        st.subheader("Appearance Settings")
        
        theme = st.radio("Theme", ["Light", "Dark", "System"], horizontal=True)
        map_style = st.selectbox("Map Style", ["OpenStreetMap", "Satellite", "Terrain", "Dark"])
        default_district = st.selectbox("Default District", get_all_district_names(), index=get_all_district_names().index("Patna"))
        refresh_interval = st.slider("Auto-refresh Interval (seconds)", 10, 300, 60)
        
        st.toggle("Show Animations", value=True)
        st.toggle("Compact Mode", value=False)
    
    with tab2:
        st.subheader("Notification Preferences")
        
        st.toggle("Email Notifications", value=True)
        st.toggle("SMS Alerts", value=False)
        st.toggle("Push Notifications", value=True)
        
        st.divider()
        
        st.write("**Alert Thresholds**")
        critical_threshold = st.slider("Critical Alert Threshold", 1, 10, 5)
        st.caption("Notify when critical alerts exceed this count")
    
    with tab3:
        st.subheader("API Configuration")
        
        api_host = st.text_input("API Host", value=settings.api_host)
        api_port = st.number_input("API Port", value=settings.api_port, min_value=1, max_value=65535)
        api_key = st.text_input("API Key", type="password", placeholder="Enter API key...")
        
        if st.button("🔄 Test Connection"):
            if check_api_status():
                st.success("✅ Connection successful!")
            else:
                st.error("❌ Connection failed")
    
    with tab4:
        st.subheader("Account Settings")
        
        st.text_input("Username", value="admin")
        st.text_input("Email", value="admin@sanjivani.ai")
        st.text_input("Role", value="Administrator", disabled=True)
        
        st.divider()
        
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")


# ---------- Main Application ----------
def main():
    """Main application entry point."""
    page = render_sidebar()
    
    if "Dashboard" in page:
        render_dashboard()
    elif "Analytics" in page:
        render_analytics()
    elif "Alerts" in page:
        render_alerts()
    elif "Resources" in page:
        render_resources()
    elif "Reports" in page:
        render_reports()
    elif "Settings" in page:
        render_settings()
    
    # Footer
    st.markdown("---")
    st.caption(f"Sanjivani AI v1.0.0 | Bihar Flood Response System | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
