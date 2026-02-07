# 🚀 Sanjivani AI - Setup Guide

> Quick start guide for setting up the Sanjivani AI Crisis Intelligence System

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| pip | Latest |
| Git | Latest |
| Docker (optional) | 20.10+ |

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/username/sanjivani-ai.git
cd sanjivani-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Generate Sample Data

```bash
PYTHONPATH=. python scripts/generate_sample_data.py
```

### 6. Train Models

```bash
# NLP Model (DistilBERT)
PYTHONPATH=. python src/nlp/train.py

# Forecasting Models (XGBoost)
PYTHONPATH=. python src/forecasting/train.py
```

### 7. Run Application

```bash
# Start API
uvicorn src.api.main:app --reload

# Start Dashboard (new terminal)
streamlit run src/dashboard/app.py
```

---

## Docker Setup

### Development

```bash
docker compose up --build
```

### Production

```bash
docker compose -f docker-compose.prod.yml up -d
```

---

## Verify Installation

```bash
# Run tests
PYTHONPATH=. pytest tests/ -v

# Check API health
curl http://localhost:8000/health

# Open Dashboard
open http://localhost:8501
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_HOST` | API server host | `127.0.0.1` |
| `API_PORT` | API server port | `8000` |
| `API_KEYS` | Comma-separated API keys | `dev-api-key-12345` |
| `ENABLE_RATE_LIMIT` | Enable rate limiting | `false` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Set `PYTHONPATH=.` before commands |
| Port already in use | Kill existing process or change port |
| Model not found | Run training scripts first |
| API connection failed | Ensure API server is running |

---

## Next Steps

- Read [GUIDE.md](GUIDE.md) for detailed usage
- Check [REPORT.md](REPORT.md) for model performance
- View API docs at http://localhost:8000/docs
