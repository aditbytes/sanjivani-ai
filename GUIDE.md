# 📖 Sanjivani AI - User Guide

> Complete guide for using the Sanjivani AI Crisis Intelligence System

---

## Table of Contents

1. [Overview](#overview)
2. [Dashboard](#dashboard)
3. [API Usage](#api-usage)
4. [Model Training](#model-training)
5. [Production Deployment](#production-deployment)

---

## Overview

Sanjivani AI is a multimodal crisis intelligence system for flood disaster response in Bihar, India. It provides:

- **NLP Module**: Tweet classification for urgency, resources, and vulnerability
- **Forecasting Module**: Resource prediction using XGBoost
- **Vision Module**: Satellite imagery analysis (requires data)
- **Dashboard**: 6-page Streamlit interface
- **API**: Production-ready FastAPI backend

---

## Dashboard

Access: **http://localhost:8501**

### Pages

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | Real-time metrics, tweet analysis, crisis map |
| 📊 **Analytics** | Charts, trends, district impact visualization |
| 🚨 **Alerts** | Alert management with severity filtering |
| 📦 **Resources** | Inventory, logistics, allocation tracking |
| 📋 **Reports** | Generate PDF/Excel/CSV reports |
| ⚙️ **Settings** | Theme, notifications, API config |

### Tweet Analysis

1. Go to Dashboard page
2. Enter tweet in text area
3. Click "Analyze" button
4. View results: Urgency, District, Resource

### Settings

- **District**: Select focus district
- **Forecast Horizon**: 6-72 hours
- **Theme**: Light/Dark mode
- **Notifications**: Email, SMS, Push toggles

---

## API Usage

Base URL: **http://localhost:8000**

### Health Check

```bash
curl http://localhost:8000/health
```

### Analyze Tweet

```bash
curl -X POST http://localhost:8000/api/v1/analyze-tweet \
  -H "Content-Type: application/json" \
  -d '{"text": "Flooding in Patna, need rescue boats!"}'
```

Response:
```json
{
  "urgency": "High",
  "district": "Patna",
  "resource_needed": "Rescue",
  "inference_time_ms": 17.5
}
```

### Get Forecast

```bash
curl http://localhost:8000/api/v1/forecast/Patna
```

### List Districts

```bash
curl http://localhost:8000/api/v1/districts
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

---

## API Authentication

For production, use API keys:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/districts
```

Set API keys via environment:
```bash
export API_KEYS=key1,key2,key3
```

---

## Model Training

### NLP (DistilBERT)

```bash
PYTHONPATH=. python src/nlp/train.py
```

Output:
- `models/nlp/best_model.pth` (265 MB)
- `models/nlp/training_history.json`

### Forecasting (XGBoost)

```bash
PYTHONPATH=. python src/forecasting/train.py
```

Output:
- `models/forecasting/xgboost_food_packets.pkl`
- `models/forecasting/xgboost_medical_kits.pkl`
- `models/forecasting/xgboost_rescue_boats.pkl`
- `models/forecasting/xgboost_shelters.pkl`

---

## Production Deployment

### Using Docker Compose

```bash
# Copy production env
cp .env.production.example .env.production

# Edit .env.production with real values
# Set API_KEYS, CORS_ORIGINS, etc.

# Start services
docker compose -f docker-compose.prod.yml up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Nginx | 80/443 | Reverse proxy |
| API | 8000 | FastAPI backend |
| Dashboard | 8501 | Streamlit UI |
| Redis | 6379 | Caching |

### Health Checks

```bash
# API
curl http://localhost:8000/health

# Dashboard
curl http://localhost:8501/_stcore/health
```

---

## Monitoring

### Prometheus Metrics

Available at `/metrics`:

- `sanjivani_http_requests_total` - Request count
- `sanjivani_http_request_duration_seconds` - Latency
- `sanjivani_uptime_seconds` - Uptime

### Request Tracing

Every response includes `X-Request-ID` header for debugging.

---

## Testing

```bash
# All tests
PYTHONPATH=. pytest tests/ -v

# Specific test file
PYTHONPATH=. pytest tests/test_api.py -v

# With coverage
PYTHONPATH=. pytest tests/ --cov=src
```

---

## Project Structure

```
sanjivani-ai/
├── src/
│   ├── api/              # FastAPI backend
│   │   ├── main.py
│   │   ├── routes/       # API endpoints
│   │   └── middleware/   # Auth, rate limiting
│   ├── dashboard/        # Streamlit UI
│   │   └── app.py
│   ├── nlp/              # NLP models
│   ├── forecasting/      # XGBoost models
│   └── vision/           # Image models
├── models/               # Trained models
├── data/                 # Datasets
├── tests/                # Test suite
├── docker/               # Docker configs
└── .github/workflows/    # CI/CD
```

---

## Support

- **Issues**: https://github.com/username/sanjivani-ai/issues
- **Docs**: http://localhost:8000/docs
- **API Status**: http://localhost:8000/health
