"""
Sanjivani AI - Map Component

Interactive Folium maps for crisis visualization.
"""

from typing import Dict, List, Optional

import folium
from folium import plugins

from src.utils.helpers import BIHAR_DISTRICTS
from src.utils.logger import get_logger

logger = get_logger(__name__)

BIHAR_CENTER = [25.6, 85.1]


def create_base_map(zoom: int = 7) -> folium.Map:
    """Create base map centered on Bihar."""
    m = folium.Map(location=BIHAR_CENTER, zoom_start=zoom, tiles="CartoDB positron")
    plugins.Fullscreen().add_to(m)
    return m


def add_alert_markers(m: folium.Map, alerts: List[Dict]) -> folium.Map:
    """Add crisis alert markers to map."""
    colors = {"Critical": "red", "High": "orange", "Medium": "yellow", "Low": "green"}
    
    for alert in alerts:
        lat, lon = alert.get("latitude"), alert.get("longitude")
        if lat and lon:
            color = colors.get(alert.get("urgency", "Medium"), "blue")
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=f"{alert.get('text', '')[:100]}...",
                color=color,
                fill=True,
            ).add_to(m)
    
    return m


def add_district_overlay(m: folium.Map, district: str, data: Dict = None) -> folium.Map:
    """Add district boundary and data overlay."""
    coords = BIHAR_DISTRICTS.get(district, {"lat": 25.5, "lon": 85.1})
    
    folium.Marker(
        location=[coords["lat"], coords["lon"]],
        popup=district,
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    
    return m


def render_map_html(m: folium.Map) -> str:
    """Render map to HTML for Streamlit embedding."""
    return m._repr_html_()
