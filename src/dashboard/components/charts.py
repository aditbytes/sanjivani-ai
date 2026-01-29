"""
Sanjivani AI - Chart Components

Plotly charts for dashboard visualizations.
"""

from typing import Dict, List

import plotly.express as px
import plotly.graph_objects as go

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_urgency_pie(alerts: List[Dict]) -> go.Figure:
    """Create pie chart of alert urgency distribution."""
    urgency_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Non-Urgent": 0}
    for alert in alerts:
        level = alert.get("urgency", "Medium")
        urgency_counts[level] = urgency_counts.get(level, 0) + 1
    
    fig = px.pie(
        names=list(urgency_counts.keys()),
        values=list(urgency_counts.values()),
        title="Alert Urgency Distribution",
        color_discrete_sequence=["#dc3545", "#fd7e14", "#ffc107", "#28a745", "#6c757d"],
    )
    return fig


def create_resource_bar(predictions: Dict[str, int]) -> go.Figure:
    """Create bar chart for resource predictions."""
    fig = px.bar(
        x=list(predictions.keys()),
        y=list(predictions.values()),
        title="Predicted Resource Requirements",
        labels={"x": "Resource", "y": "Quantity"},
        color_discrete_sequence=["#007bff"],
    )
    return fig


def create_timeline_chart(data: List[Dict]) -> go.Figure:
    """Create timeline of alerts over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d.get("timestamp") for d in data],
        y=[d.get("count", 0) for d in data],
        mode="lines+markers",
        name="Alerts",
    ))
    fig.update_layout(title="Alerts Over Time", xaxis_title="Time", yaxis_title="Count")
    return fig


def create_district_heatmap(district_data: Dict[str, float]) -> go.Figure:
    """Create heatmap of district severity."""
    fig = px.bar(
        x=list(district_data.keys()),
        y=list(district_data.values()),
        title="District Alert Severity",
        color=list(district_data.values()),
        color_continuous_scale="Reds",
    )
    return fig
