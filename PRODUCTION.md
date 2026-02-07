# 🔄 Demo to Production Migration Guide

> Step-by-step guide to transition Sanjivani AI from demo to production with real data

---

## Overview

| Component | Demo Mode | Production Mode |
|-----------|-----------|-----------------|
| **Data** | Synthetic tweets | Real Twitter/X data |
| **NLP Model** | Low accuracy (31%) | High accuracy (80%+) |
| **Forecasting** | Sample flood events | Historical flood data |
| **Vision** | Not available | Satellite imagery |
| **API** | No auth, debug=true | API keys, rate limiting |
| **Dashboard** | Mock values | Live data |

---

## Phase 1: Data Acquisition

### 1.1 Twitter/X Crisis Data

**Option A: Twitter API**
```bash
# Get Twitter Developer Account
# https://developer.twitter.com/

# Set credentials in .env
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_BEARER_TOKEN=your_bearer_token
```

**Option B: Existing Datasets**
- [CrisisNLP](https://crisisnlp.qcri.org/) - Crisis tweet datasets
- [CrisisLex](https://crisislex.org/) - Labeled crisis data
- [HumAID](https://huggingface.co/datasets/nlp-thedeep/humaid) - Humanitarian tweets

**Data Format**:
```json
{
  "text": "Heavy flooding in Patna, need rescue!",
  "urgency": "Critical",
  "resource_needed": "Rescue",
  "vulnerability": "Elderly",
  "district": "Patna",
  "timestamp": "2024-07-15T10:30:00Z"
}
```

Save to: `data/processed/train.json`, `val.json`, `test.json`

### 1.2 Historical Flood Data

**Sources**:
- [India Meteorological Department](https://mausam.imd.gov.in/)
- [Central Water Commission](https://cwc.gov.in/)
- [Bihar State Disaster Management Authority](http://bsdma.org/)

**Data Format**:
```json
{
  "year": 2023,
  "district": "Darbhanga",
  "flood_area_sq_km": 450.5,
  "water_level_m": 8.2,
  "rainfall_mm": 320,
  "duration_days": 14,
  "population_affected": 125000,
  "food_packets_needed": 50000,
  "medical_kits_needed": 2500,
  "rescue_boats_needed": 120,
  "shelters_needed": 45
}
```

Save to: `data/processed/historical_floods.json`

### 1.3 Satellite Imagery

**Sources**:
- [Copernicus Sentinel Hub](https://www.sentinel-hub.com/)
- [NASA Earthdata](https://earthdata.nasa.gov/)
- [Google Earth Engine](https://earthengine.google.com/)

**Requirements**:
- Pre-flood and post-flood images
- Minimum 10m resolution
- RGB + NIR bands preferred

Save to: `data/satellite/`

---

## Phase 2: Model Retraining

### 2.1 Prepare Real Data

```bash
# Validate data format
PYTHONPATH=. python scripts/validate_data.py

# Split into train/val/test (70/15/15)
PYTHONPATH=. python scripts/split_data.py
```

### 2.2 Retrain NLP Model

```bash
# Update config for more epochs
export NLP_EPOCHS=10
export NLP_BATCH_SIZE=32

# Train on real data
PYTHONPATH=. python src/nlp/train.py
```

**Expected Improvements**:
| Metric | Demo | Production |
|--------|------|------------|
| Accuracy | 31% | 80-90% |
| F1 Score | 0.28 | 0.82+ |

### 2.3 Retrain Forecasting Models

```bash
# Ensure historical_floods.json has real data
PYTHONPATH=. python src/forecasting/train.py
```

### 2.4 Train Vision Models (Optional)

```bash
# Requires satellite imagery in data/satellite/
PYTHONPATH=. python src/vision/train_segmentation.py
PYTHONPATH=. python src/vision/train_detection.py
```

---

## Phase 3: Configuration Changes

### 3.1 Environment Variables

```bash
# Copy production template
cp .env.production.example .env.production
```

**Update these values**:

```bash
# .env.production

# Disable debug mode
DEBUG=false
LOG_LEVEL=INFO

# Enable security
ENABLE_RATE_LIMIT=true
RATE_LIMIT_RPM=60
API_KEYS=prod-key-1,prod-key-2

# Set allowed origins
CORS_ORIGINS=https://yourdomain.com

# External APIs (for real data)
TWITTER_BEARER_TOKEN=your_real_token
SENTINEL_API_KEY=your_sentinel_key

# Production inference
INFERENCE_DEVICE=cuda  # if GPU available
```

### 3.2 API Configuration

Edit `src/api/main.py` (already production-ready):
- Docs disabled in production (`DEBUG=false`)
- Rate limiting enabled
- Exception handlers active

### 3.3 Dashboard Configuration

The dashboard auto-detects production mode via API settings.

---

## Phase 4: Database Setup (Optional)

### 4.1 PostgreSQL for Persistence

```bash
# Start PostgreSQL
docker run -d \
  --name sanjivani-db \
  -e POSTGRES_PASSWORD=securepassword \
  -e POSTGRES_DB=sanjivani \
  -p 5432:5432 \
  postgres:15

# Add to .env.production
DATABASE_URL=postgresql://postgres:securepassword@localhost:5432/sanjivani
```

### 4.2 Redis for Caching

```bash
# Start Redis
docker run -d \
  --name sanjivani-redis \
  -p 6379:6379 \
  redis:7-alpine

# Add to .env.production
REDIS_URL=redis://localhost:6379
```

---

## Phase 5: Deployment

### 5.1 Docker Production Build

```bash
# Build production images
docker compose -f docker-compose.prod.yml build

# Start services
docker compose -f docker-compose.prod.yml up -d
```

### 5.2 SSL/HTTPS Setup

Edit `docker/nginx.conf`:
```nginx
# Uncomment SSL section
listen 443 ssl http2;
ssl_certificate /etc/nginx/ssl/cert.pem;
ssl_certificate_key /etc/nginx/ssl/key.pem;
```

### 5.3 Cloud Deployment

**AWS**:
```bash
# ECS/Fargate or EC2
aws ecr create-repository --repository-name sanjivani-api
docker tag sanjivani-api:latest <account>.dkr.ecr.<region>.amazonaws.com/sanjivani-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/sanjivani-api:latest
```

**GCP**:
```bash
# Cloud Run
gcloud run deploy sanjivani-api \
  --image gcr.io/<project>/sanjivani-api \
  --platform managed
```

---

## Phase 6: Verification Checklist

### Pre-Production
- [ ] Real training data loaded
- [ ] Models retrained with real data
- [ ] Accuracy > 80%
- [ ] API keys configured
- [ ] Rate limiting enabled
- [ ] CORS restricted
- [ ] DEBUG=false
- [ ] SSL certificates installed

### Post-Deployment
- [ ] Health check passing
- [ ] API responding
- [ ] Dashboard loading
- [ ] Metrics collecting
- [ ] Logs streaming
- [ ] Alerts configured

---

## Quick Command Reference

| Action | Command |
|--------|---------|
| Retrain NLP | `PYTHONPATH=. python src/nlp/train.py` |
| Retrain Forecasting | `PYTHONPATH=. python src/forecasting/train.py` |
| Test production | `pytest tests/ -v` |
| Deploy | `docker compose -f docker-compose.prod.yml up -d` |
| Check health | `curl https://yourdomain.com/health` |

---

## Support

For issues during migration:
1. Check logs: `docker compose logs -f api`
2. Verify data format in `data/processed/`
3. Ensure models exist in `models/`
4. Test API locally before deploying
