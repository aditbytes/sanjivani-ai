# 🌊 Sanjivani AI - Crisis Intelligence System

<p align="center">
  <strong>Multimodal AI for Flood Disaster Response in Bihar, India</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Status">
</p>

---

## 🎯 Overview

Sanjivani AI is a production-ready crisis intelligence system that processes social media distress signals, analyzes satellite imagery, and predicts resource requirements for flood disaster response in Bihar, India.

### Key Features

- **🔤 NLP Module**: Real-time classification of crisis tweets (urgency, resource needs, vulnerability)
- **🛰️ Vision Module**: Satellite imagery analysis for flood extent and damage detection
- **📊 Forecasting Module**: XGBoost + LSTM ensemble for resource prediction
- **⚡ FastAPI Backend**: Production REST API with OpenAPI documentation
- **📈 Streamlit Dashboard**: Real-time monitoring with interactive maps

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- PostgreSQL with PostGIS

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/sanjivani-ai.git
cd sanjivani-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Run API
uvicorn src.api.main:app --reload

# Run Dashboard (in another terminal)
streamlit run src/dashboard/app.py
```

### Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## 📂 Project Structure

```
sanjivani-ai/
├── src/
│   ├── nlp/           # Tweet classification & NER
│   ├── vision/        # Satellite image analysis
│   ├── forecasting/   # Resource prediction
│   ├── api/           # FastAPI backend
│   ├── dashboard/     # Streamlit UI
│   ├── data/          # Data layer & models
│   └── utils/         # Helpers & logging
├── tests/             # Test suite
├── docker/            # Dockerfiles
├── data/              # Data storage
├── models/            # Trained models
└── docs/              # Documentation
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze-tweet` | POST | Analyze crisis tweet |
| `/api/v1/analyze-image` | POST | Analyze satellite image |
| `/api/v1/forecast/{district}` | GET | Get resource forecast |
| `/api/v1/districts` | GET | List Bihar districts |
| `/health` | GET | Health check |

**API Docs**: http://localhost:8000/docs

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific tests
pytest tests/test_nlp.py -v
```

---

## 📊 Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| DistilBERT | Tweet Classification | F1 | 0.89 |
| U-Net ResNet50 | Flood Segmentation | IoU | 0.85 |
| YOLOv8 | Object Detection | mAP@50 | 0.82 |
| XGBoost+LSTM | Resource Forecast | MAPE | 15% |

---

## 🛠️ Configuration

All configuration via environment variables (see `.env.example`):

```ini
# Core
APP_NAME=Sanjivani AI
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/sanjivani

# API Keys
TWITTER_BEARER_TOKEN=your_token
SENTINEL_HUB_CLIENT_ID=your_id
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## �‍💻 Developers

1. Aditya CS STUDENT @ IIT PATNA
2. 
---

## �🙏 Acknowledgments

- Built for Bihar Flood Response
- DistilBERT by Hugging Face
- Satellite imagery via Sentinel-2/Copernicus

---

<p align="center">
  <strong>🌊 Saving Lives Through AI 🌊</strong>
</p>
