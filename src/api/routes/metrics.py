"""
Sanjivani AI - Prometheus Metrics Endpoint

Exposes application metrics for monitoring and alerting.
"""

import time
from typing import Dict, Any
from collections import defaultdict

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ---------- In-Memory Metrics Storage ----------
# For production, use prometheus_client library

class MetricsCollector:
    """Simple in-memory metrics collector."""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, list] = defaultdict(list)
        self.gauges: Dict[str, float] = {}
        self.start_time = time.time()
    
    def inc_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self.counters[key] += value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        # Keep only last 1000 observations to limit memory
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"{key} {value}")
        
        # Histograms (simplified - just export sum and count)
        for key, values in self.histograms.items():
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values):.4f}")
                lines.append(f"{key}_avg {sum(values)/len(values):.4f}")
        
        # Uptime
        uptime = time.time() - self.start_time
        lines.append(f'sanjivani_uptime_seconds {uptime:.2f}')
        
        return "\n".join(lines)


# Global metrics collector
metrics = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        path = request.url.path
        method = request.method
        status = response.status_code
        
        # Increment request counter
        metrics.inc_counter(
            "sanjivani_http_requests_total",
            labels={"method": method, "path": path, "status": str(status)}
        )
        
        # Record latency
        metrics.observe_histogram(
            "sanjivani_http_request_duration_seconds",
            duration,
            labels={"method": method, "path": path}
        )
        
        return response


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    """
    content = metrics.get_prometheus_format()
    return Response(content=content, media_type="text/plain")


@router.get("/metrics/json")
async def json_metrics() -> Dict[str, Any]:
    """Return metrics in JSON format for debugging."""
    return {
        "counters": dict(metrics.counters),
        "gauges": dict(metrics.gauges),
        "histograms": {k: {"count": len(v), "sum": sum(v)} for k, v in metrics.histograms.items()},
        "uptime_seconds": time.time() - metrics.start_time
    }


def record_model_inference(model_name: str, duration: float, success: bool):
    """Helper to record model inference metrics."""
    metrics.inc_counter(
        "sanjivani_model_inference_total",
        labels={"model": model_name, "success": str(success).lower()}
    )
    if success:
        metrics.observe_histogram(
            "sanjivani_model_inference_duration_seconds",
            duration,
            labels={"model": model_name}
        )
