"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_readiness(self):
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True
    
    def test_liveness(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["alive"] is True


class TestNLPEndpoints:
    """Test NLP API endpoints."""
    
    def test_analyze_tweet(self):
        response = client.post(
            "/api/v1/analyze-tweet",
            json={"text": "Flooding in Patna, need rescue"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "urgency" in data
        assert "processed_text" in data
    
    def test_analyze_tweet_empty(self):
        response = client.post("/api/v1/analyze-tweet", json={"text": ""})
        assert response.status_code == 422  # Validation error


class TestForecastingEndpoints:
    """Test forecasting API endpoints."""
    
    def test_districts_list(self):
        response = client.get("/api/v1/districts")
        assert response.status_code == 200
        districts = response.json()
        assert isinstance(districts, list)
        assert "Patna" in districts
    
    def test_forecast_district(self):
        response = client.get("/api/v1/forecast/Patna")
        assert response.status_code == 200
        data = response.json()
        assert data["district"] == "Patna"
        assert "predictions" in data
