# 📖 Sanjivani AI - Complete Setup & Usage Guide

> **Comprehensive guide for setting up, configuring, and using the Sanjivani AI Crisis Intelligence System**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Installation Methods](#installation-methods)
   - [Local Development Setup](#local-development-setup)
   - [Docker Deployment](#docker-deployment)
   - [Production Deployment](#production-deployment)
4. [Configuration Guide](#configuration-guide)
5. [Running the Application](#running-the-application)
6. [API Usage Guide](#api-usage-guide)
7. [Dashboard Guide](#dashboard-guide)
8. [Training Models](#training-models)
9. [Data Management](#data-management)
10. [Development Workflow](#development-workflow)
11. [Testing Guide](#testing-guide)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 10+ | Ubuntu 22.04 LTS |
| **Python** | 3.10 | 3.11 |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 20 GB | 50 GB+ (for models & data) |
| **GPU** | Not required | NVIDIA GPU with CUDA 11.8+ |

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| Git | 2.30+ | Version control |
| pip | 22.0+ | Package management |
| Docker | 24.0+ | Containerization (optional) |
| Docker Compose | 2.20+ | Container orchestration (optional) |
| PostgreSQL | 15+ | Database (if not using Docker) |
| Redis | 7+ | Caching (if not using Docker) |

### External API Keys (Optional)

| Service | Purpose | Required For |
|---------|---------|--------------|
| **Twitter API v2** | Real-time tweet streaming | Live tweet analysis |
| **Sentinel Hub** | Satellite imagery download | Vision module training |
| **Sentry** | Error monitoring | Production monitoring |

---

## Quick Start

### 🚀 5-Minute Setup (Docker)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/sanjivani-ai.git
cd sanjivani-ai

# 2. Create environment file
cp .env.example .env

# 3. Start all services
docker-compose up -d

# 4. Access the application
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### 🐍 5-Minute Setup (Local Python)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/sanjivani-ai.git
cd sanjivani-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
cp .env.example .env

# 5. Start the API
uvicorn src.api.main:app --reload

# 6. Start the Dashboard (new terminal)
streamlit run src/dashboard/app.py
```

---

## Installation Methods

### Local Development Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/sanjivani-ai.git
cd sanjivani-ai
```

#### Step 2: Create Python Virtual Environment

**Using venv (recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate - Linux/macOS
source venv/bin/activate

# Activate - Windows Command Prompt
venv\Scripts\activate.bat

# Activate - Windows PowerShell
venv\Scripts\Activate.ps1
```

**Using conda:**
```bash
# Create conda environment
conda create -n sanjivani python=3.11 -y

# Activate environment
conda activate sanjivani
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# If you have GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Installing individual components:**

```bash
# Core only (for API without ML)
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv

# NLP module
pip install torch transformers spacy nltk

# Vision module
pip install torch torchvision opencv-python segmentation-models-pytorch ultralytics

# Forecasting module
pip install xgboost tensorflow scikit-learn

# Dashboard
pip install streamlit plotly folium streamlit-folium

# Database
pip install sqlalchemy psycopg2-binary redis alembic geoalchemy2
```

#### Step 4: Download NLP Models

```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Step 5: Setup PostgreSQL Database

**Option A: Using Docker (recommended)**
```bash
# Start PostgreSQL with PostGIS
docker run -d \
  --name sanjivani-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=sanjivani \
  -p 5432:5432 \
  postgis/postgis:15-3.3

# Start Redis
docker run -d \
  --name sanjivani-redis \
  -p 6379:6379 \
  redis:7-alpine
```

**Option B: Local PostgreSQL Installation**

```bash
# macOS (Homebrew)
brew install postgresql postgis
brew services start postgresql
createdb sanjivani

# Ubuntu
sudo apt update
sudo apt install postgresql postgresql-contrib postgis
sudo -u postgres createdb sanjivani

# Enable PostGIS extension
psql -d sanjivani -c "CREATE EXTENSION IF NOT EXISTS postgis;"
```

#### Step 6: Initialize Database Tables

```bash
# Run database initialization script
python -c "
from src.data.database import DatabaseManager
manager = DatabaseManager()
manager.init_db()
print('Database tables created successfully!')
"
```

---

### Docker Deployment

#### Step 1: Prerequisites

Ensure Docker and Docker Compose are installed:

```bash
# Check Docker version
docker --version  # Should be 24.0+

# Check Docker Compose version
docker compose version  # Should be 2.20+
```

#### Step 2: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration Guide section)
nano .env
```

#### Step 3: Build and Start Services

```bash
# Build all images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f dashboard
```

#### Step 4: Verify Deployment

```bash
# Check running containers
docker-compose ps

# Expected output:
# NAME                  STATUS
# sanjivani-api         running
# sanjivani-dashboard   running
# sanjivani-db          running
# sanjivani-redis       running

# Test API health
curl http://localhost:8000/health
```

#### Docker Commands Reference

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Restart a specific service
docker-compose restart api

# Scale a service (e.g., multiple API instances)
docker-compose up -d --scale api=3

# View resource usage
docker stats

# Access container shell
docker exec -it sanjivani-api bash

# View container logs
docker logs sanjivani-api --tail 100 -f
```

---

### Production Deployment

#### Using Docker Compose with Production Settings

```bash
# Create production environment file
cp .env.example .env.production

# Edit production settings
nano .env.production
```

**Production .env settings:**
```ini
# Core
DEBUG=false
LOG_LEVEL=WARNING

# Security
SECRET_KEY=your-secure-random-key-here
CORS_ORIGINS=https://yourdomain.com

# Database
DATABASE_URL=postgresql://user:securepassword@db-host:5432/sanjivani

# Redis  
REDIS_URL=redis://:password@redis-host:6379

# API
API_HOST=0.0.0.0
API_PORT=8000
```

**Start with production compose file:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### Using Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sanjivani-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sanjivani-api
  template:
    metadata:
      labels:
        app: sanjivani-api
    spec:
      containers:
      - name: api
        image: sanjivani-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sanjivani-secrets
              key: database-url
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "250m"
```

---

## Configuration Guide

### Environment Variables

Create a `.env` file in the project root with the following settings:

#### Core Application Settings

```ini
# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Application name and version
APP_NAME=Sanjivani AI
APP_VERSION=1.0.0

# Debug mode (set to false in production)
DEBUG=true

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
```

#### API Configuration

```ini
# =============================================================================
# API SETTINGS
# =============================================================================

# API server host and port
API_HOST=0.0.0.0
API_PORT=8000

# CORS allowed origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:8501

# API rate limiting (requests per minute)
RATE_LIMIT=100
```

#### Database Configuration

```ini
# =============================================================================
# DATABASE SETTINGS
# =============================================================================

# PostgreSQL connection URL
# Format: postgresql://user:password@host:port/database
DATABASE_URL=postgresql://postgres:password@localhost:5432/sanjivani

# Connection pool settings
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# Redis connection URL
REDIS_URL=redis://localhost:6379
```

#### ML Model Settings

```ini
# =============================================================================
# ML MODEL SETTINGS
# =============================================================================

# Inference device: cpu, cuda, mps (Apple Silicon)
INFERENCE_DEVICE=cpu

# Model paths (relative to project root)
NLP_MODEL_PATH=models/nlp/distilbert_crisis.pt
VISION_SEGMENTATION_MODEL_PATH=models/vision/unet_flood.pt
VISION_DETECTION_MODEL_PATH=models/vision/yolov8_flood.pt
FORECASTING_XGBOOST_PATH=models/forecasting/xgboost_resource.json
FORECASTING_LSTM_PATH=models/forecasting/lstm_resource.h5

# Batch sizes
NLP_BATCH_SIZE=32
VISION_BATCH_SIZE=8
```

#### External API Keys

```ini
# =============================================================================
# EXTERNAL API KEYS
# =============================================================================

# Twitter API v2 (for real-time streaming)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Sentinel Hub (for satellite imagery)
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret

# Sentry (for error monitoring)
SENTRY_DSN=https://your-sentry-dsn
```

#### Geographic Settings

```ini
# =============================================================================
# GEOGRAPHIC SETTINGS (BIHAR)
# =============================================================================

# Bihar center coordinates
BIHAR_CENTER_LAT=25.0961
BIHAR_CENTER_LON=85.1376

# Default map zoom level
BIHAR_DEFAULT_ZOOM=7
```

### Configuration Validation

Run this script to validate your configuration:

```bash
python -c "
from src.config import get_settings
settings = get_settings()
print('✓ Configuration loaded successfully!')
print(f'  App: {settings.app_name} v{settings.app_version}')
print(f'  Debug: {settings.debug}')
print(f'  API: {settings.api_host}:{settings.api_port}')
print(f'  Database: {settings.database_url[:20]}...')
"
```

---

## Running the Application

### Starting the API Server

#### Development Mode (with hot reload)

```bash
# Using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Using Python module
python -m src.api.main
```

#### Production Mode

```bash
# Using gunicorn with uvicorn workers
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# With increased timeout and workers
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile -
```

### Starting the Dashboard

```bash
# Development mode
streamlit run src/dashboard/app.py

# With custom port
streamlit run src/dashboard/app.py --server.port 8502

# Production mode
streamlit run src/dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
```

### Starting All Services (Development)

Create a `scripts/start_dev.sh` script:

```bash
#!/bin/bash

# Start API in background
uvicorn src.api.main:app --reload --port 8000 &
API_PID=$!

# Start Dashboard in background
streamlit run src/dashboard/app.py --server.port 8501 &
DASH_PID=$!

echo "Started API (PID: $API_PID) on http://localhost:8000"
echo "Started Dashboard (PID: $DASH_PID) on http://localhost:8501"

# Wait for Ctrl+C
trap "kill $API_PID $DASH_PID" EXIT
wait
```

Run with: `bash scripts/start_dev.sh`

---

## API Usage Guide

### API Documentation

Once the API is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Authentication

Currently, the API is open. For production, add authentication:

```python
# Example: Add API key authentication
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

### API Endpoints

#### Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "redis": "connected"
}
```

#### Tweet Analysis

```bash
# Analyze a crisis tweet
curl -X POST "http://localhost:8000/api/v1/analyze-tweet" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Urgent! Flood in Patna, need rescue for elderly people. Water level rising fast!"
  }'

# Response:
{
  "text": "Urgent! Flood in Patna...",
  "urgency": {
    "label": "Critical",
    "confidence": 0.92
  },
  "resources": [
    {"label": "Rescue", "confidence": 0.88}
  ],
  "vulnerability": {
    "label": "Elderly",
    "confidence": 0.85
  },
  "location": {
    "district": "Patna",
    "latitude": 25.5941,
    "longitude": 85.1376
  },
  "inference_time_ms": 45.2
}
```

#### Image Analysis

```bash
# Analyze satellite image (base64 encoded)
curl -X POST "http://localhost:8000/api/v1/analyze-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "/9j/4AAQSkZJRg...",
    "include_detections": true
  }'

# Response:
{
  "flood_percentage": 34.5,
  "flood_mask": [[0, 1, 1, ...], ...],
  "detections": [
    {"class": "person", "confidence": 0.87, "bbox": [100, 200, 150, 280]},
    {"class": "vehicle", "confidence": 0.92, "bbox": [300, 150, 380, 220]}
  ],
  "object_counts": {
    "person": 12,
    "vehicle": 5
  },
  "inference_time_ms": 234.5
}
```

#### Resource Forecast

```bash
# Get resource forecast for a district
curl "http://localhost:8000/api/v1/forecast/Patna?horizon_hours=48"

# Response:
{
  "district": "Patna",
  "horizon_hours": 48,
  "predictions": {
    "food_packets": 2500,
    "medical_kits": 150,
    "rescue_boats": 25,
    "shelters": 8
  },
  "confidence": 0.78,
  "generated_at": "2024-01-29T20:00:00Z"
}
```

#### List Districts

```bash
# Get all Bihar districts
curl "http://localhost:8000/api/v1/districts"

# Response:
{
  "districts": [
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Gaya", "lat": 24.7955, "lon": 85.0002},
    ...
  ],
  "count": 38
}
```

### Python Client Example

```python
import requests

class SanjivaniClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_tweet(self, text: str) -> dict:
        response = requests.post(
            f"{self.base_url}/api/v1/analyze-tweet",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    
    def get_forecast(self, district: str, horizon_hours: int = 24) -> dict:
        response = requests.get(
            f"{self.base_url}/api/v1/forecast/{district}",
            params={"horizon_hours": horizon_hours}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = SanjivaniClient()
result = client.analyze_tweet("Help needed in Muzaffarpur, children stranded!")
print(f"Urgency: {result['urgency']['label']}")
```

---

## Dashboard Guide

### Accessing the Dashboard

1. Start the dashboard: `streamlit run src/dashboard/app.py`
2. Open in browser: http://localhost:8501

### Dashboard Features

#### 1. District Overview Panel

- **Active Alerts**: Real-time count of active crisis alerts
- **People Affected**: Estimated affected population
- **Resources Deployed**: Deployment progress percentage

#### 2. Resource Forecasting

1. Select a district from the sidebar dropdown
2. Adjust the forecast horizon slider (6-72 hours)
3. View predicted resource requirements

#### 3. Tweet Analysis Tool

1. Enter tweet text in the input area
2. Click "Analyze Tweet"
3. View classification results (urgency, resources, vulnerability, location)

#### 4. Crisis Map

- Interactive map of Bihar with alert markers
- Color-coded by urgency level
- Click markers for alert details

### Customizing the Dashboard

#### Adding New Visualizations

```python
# src/dashboard/components/custom_chart.py
import streamlit as st
import plotly.express as px

def render_alert_timeline(alerts_df):
    """Render alert timeline chart."""
    fig = px.line(
        alerts_df,
        x="created_at",
        y="count",
        color="urgency",
        title="Alerts Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
```

#### Adding New Pages

Create a new file in `src/dashboard/pages/`:

```python
# src/dashboard/pages/analytics.py
import streamlit as st

st.title("📊 Analytics Dashboard")
st.write("Detailed analytics and insights")
# Add your visualizations here
```

---

## Training Models

### NLP Model Training

#### Step 1: Prepare Training Data

```bash
# Create data directory
mkdir -p data/nlp

# Expected format: CSV with columns
# text, urgency, resources, vulnerability
```

**Sample data format (`data/nlp/train.csv`):**
```csv
text,urgency,resources,vulnerability
"Help! Water rising in our village",Critical,Rescue,None
"Need food and medicine for elderly",High,"Food,Medical",Elderly
...
```

#### Step 2: Train the Model

```bash
# Run training script
python -m src.nlp.train \
  --data_path data/nlp/train.csv \
  --val_path data/nlp/val.csv \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir models/nlp
```

#### Step 3: Evaluate the Model

```bash
# Run evaluation
python -m src.nlp.evaluate \
  --model_path models/nlp/distilbert_crisis.pt \
  --test_path data/nlp/test.csv
```

### Vision Model Training

#### Step 1: Prepare Training Data

```bash
# Create data directories
mkdir -p data/vision/{images,masks}

# Download sample data (if available)
python scripts/download_sample_data.py --type vision
```

**Expected structure:**
```
data/vision/
├── images/
│   ├── train/
│   │   ├── image_001.png
│   │   └── ...
│   └── val/
├── masks/
│   ├── train/
│   │   ├── image_001.png  # Same name as image
│   │   └── ...
│   └── val/
└── annotations/  # For detection
    ├── train/
    └── val/
```

#### Step 2: Train Segmentation Model

```bash
# Train U-Net segmentation model
python -m src.vision.train_segmentation \
  --data_dir data/vision \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --encoder resnet50 \
  --output_dir models/vision

# With GPU
CUDA_VISIBLE_DEVICES=0 python -m src.vision.train_segmentation ...
```

#### Step 3: Train Detection Model

```bash
# Train YOLOv8 detection model
python -m src.vision.train_detection \
  --data_dir data/vision/annotations \
  --epochs 100 \
  --batch_size 16 \
  --model yolov8n  # nano, small, medium, large, xlarge
  --output_dir models/vision
```

### Forecasting Model Training

#### Step 1: Prepare Historical Data

```bash
# Expected format: CSV with time series data
# district, date, rainfall, water_level, affected_population, resources_needed
```

#### Step 2: Train Ensemble Model

```bash
# Train XGBoost and LSTM ensemble
python -m src.forecasting.train \
  --data_path data/forecasting/historical.csv \
  --xgboost_params '{"n_estimators": 100, "max_depth": 6}' \
  --lstm_params '{"units": 64, "dropout": 0.2}' \
  --output_dir models/forecasting
```

### Training with GPU

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# For Apple Silicon (M1/M2)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# Will use 'mps' device automatically
```

---

## Data Management

### Data Directory Structure

```
data/
├── raw/                    # Raw, unprocessed data
│   ├── tweets/
│   ├── satellite/
│   └── historical/
├── processed/              # Cleaned and processed data
│   ├── nlp/
│   ├── vision/
│   └── forecasting/
├── splits/                 # Train/val/test splits
└── external/               # External data sources
```

### Database Operations

#### Creating Database Backup

```bash
# Backup PostgreSQL database
pg_dump -h localhost -U postgres -d sanjivani > backup_$(date +%Y%m%d).sql

# Compressed backup
pg_dump -h localhost -U postgres -d sanjivani | gzip > backup_$(date +%Y%m%d).sql.gz
```

#### Restoring Database

```bash
# Restore from backup
psql -h localhost -U postgres -d sanjivani < backup_20240129.sql

# From compressed backup
gunzip -c backup_20240129.sql.gz | psql -h localhost -U postgres -d sanjivani
```

#### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1

# View migration history
alembic history
```

### Data Loading Scripts

```bash
# Load sample data
python scripts/load_sample_data.py

# Load tweets from CSV
python -c "
from src.data.loaders import load_tweets_from_csv
load_tweets_from_csv('data/raw/tweets/sample.csv')
"

# Load satellite images
python -c "
from src.data.loaders import load_satellite_images
load_satellite_images('data/raw/satellite/')
"
```

---

## Development Workflow

### Code Style and Formatting

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run all checks
pre-commit run --all-files
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**`.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/new-feature

# Commit message format:
# feat: new feature
# fix: bug fix
# docs: documentation
# refactor: code refactoring
# test: adding tests
# chore: maintenance
```

### Adding New Endpoints

1. **Create route file:**
```python
# src/api/routes/new_feature.py
from fastapi import APIRouter, Depends
from src.api.schemas.new_feature import NewFeatureRequest, NewFeatureResponse

router = APIRouter()

@router.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature(request: NewFeatureRequest):
    # Implementation
    return NewFeatureResponse(...)
```

2. **Create schemas:**
```python
# src/api/schemas/new_feature.py
from pydantic import BaseModel

class NewFeatureRequest(BaseModel):
    input_data: str

class NewFeatureResponse(BaseModel):
    result: str
```

3. **Register in main.py:**
```python
from src.api.routes import new_feature
app.include_router(new_feature.router, prefix="/api/v1", tags=["New Feature"])
```

---

## Testing Guide

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_nlp.py

# Run specific test function
pytest tests/test_nlp.py::test_tweet_classification

# Run tests matching pattern
pytest -k "nlp"
```

### Test Coverage

```bash
# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html

# Coverage with specific modules
pytest --cov=src/nlp --cov=src/api tests/
```

### Writing Tests

```python
# tests/test_new_feature.py
import pytest
from src.new_module import new_function

class TestNewFeature:
    def test_basic_functionality(self):
        result = new_function("input")
        assert result == "expected_output"
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            new_function("")
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        result = await async_new_function("input")
        assert result is not None
```

### API Testing

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_analyze_tweet():
    response = client.post(
        "/api/v1/analyze-tweet",
        json={"text": "Flood in Patna, need help!"}
    )
    assert response.status_code == 200
    assert "urgency" in response.json()
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Option 1: Install package in editable mode
pip install -e .

# Option 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Option 3: Run from project root
cd /path/to/sanjivani-ai
python -m src.api.main
```

#### 2. Database Connection Failed

**Problem:** `psycopg2.OperationalError: could not connect to server`

**Solution:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Restart PostgreSQL container
docker-compose restart db

# Check connection string in .env
cat .env | grep DATABASE_URL

# Test connection
python -c "
from src.data.database import get_engine
engine = get_engine()
with engine.connect() as conn:
    print('Connected successfully!')
"
```

#### 3. CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size in .env
NLP_BATCH_SIZE=16
VISION_BATCH_SIZE=4

# Or use CPU
INFERENCE_DEVICE=cpu

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 4. Model File Not Found

**Problem:** `FileNotFoundError: Model not found at models/nlp/...`

**Solution:**
```bash
# Create models directory
mkdir -p models/nlp models/vision models/forecasting

# Download pre-trained models (if available)
python scripts/download_models.py

# Or train new models
python -m src.nlp.train --data_path data/nlp/train.csv
```

#### 5. Port Already in Use

**Problem:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
uvicorn src.api.main:app --port 8001
```

#### 6. Streamlit Errors

**Problem:** `StreamlitAPIException` or blank page

**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Check for syntax errors
python -m py_compile src/dashboard/app.py

# Run with debug info
streamlit run src/dashboard/app.py --logger.level debug
```

### Checking System Health

```bash
# Run health check script
python -c "
from src.config import get_settings
from src.data.database import DatabaseManager

settings = get_settings()
print('Configuration: OK')

manager = DatabaseManager()
if manager.check_health():
    print('Database: OK')
else:
    print('Database: FAILED')

import redis
try:
    r = redis.from_url(settings.redis_url)
    r.ping()
    print('Redis: OK')
except:
    print('Redis: FAILED')
"
```

### Log Files

```bash
# View application logs
tail -f logs/sanjivani.log

# Docker logs
docker-compose logs -f --tail 100

# Filter by level
grep ERROR logs/sanjivani.log
grep WARNING logs/sanjivani.log
```

---

## FAQ

### General Questions

**Q: What Python version is supported?**
A: Python 3.10 or higher is required. Python 3.11 is recommended for best performance.

**Q: Can I run without GPU?**
A: Yes, all models can run on CPU. Set `INFERENCE_DEVICE=cpu` in your `.env` file. GPU is recommended for faster inference.

**Q: How do I update dependencies?**
A: Run `pip install -r requirements.txt --upgrade` to update all dependencies.

### API Questions

**Q: How do I enable authentication?**
A: Implement API key or JWT authentication in `src/api/middleware/`. See the API Usage Guide section.

**Q: What's the rate limit?**
A: Default is 100 requests/minute. Configure with `RATE_LIMIT` environment variable.

### Model Questions

**Q: How accurate are the models?**
A: See README.md for model performance metrics. DistilBERT F1: 0.89, U-Net IoU: 0.85, XGBoost MAPE: 15%.

**Q: Can I use my own models?**
A: Yes, update the model paths in `.env` to point to your trained models.

**Q: How often should I retrain?**
A: Recommended: NLP monthly, Vision quarterly, Forecasting weekly with new data.

### Deployment Questions

**Q: How do I scale the API?**
A: Use multiple workers with gunicorn, or scale containers with `docker-compose up --scale api=4`.

**Q: What's the recommended server spec for production?**
A: Minimum: 4 CPU cores, 16GB RAM, 100GB SSD. With GPU: Add NVIDIA GPU with 8GB+ VRAM.

---

## Getting Help

- **Documentation**: This guide and SYSTEM_ARCHITECTURE.md
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Email**: support@sanjivani-ai.org

---

<p align="center">
  <strong>🌊 Sanjivani AI - Saving Lives Through AI 🌊</strong>
</p>
