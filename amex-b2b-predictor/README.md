# B2B Customer Sentiment & Growth Predictor
### American Express — AI-Powered Acquisition Targeting Engine

A production-ready GenAI + ML system that analyzes business reviews, news, and financial data to identify and score B2B acquisition targets for American Express.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React Dashboard (Port 3000)                            │
│  • Real-time business analysis UI                       │
│  • Pipeline visualization + KPIs                        │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────────────┐
│  Flask Backend (Port 5000)                              │
│  • /api/analyze       → Single business scoring        │
│  • /api/batch-analyze → CSV batch processing           │
│  • /api/dashboard-stats → Aggregated KPIs              │
└──────┬────────────────────┬───────────────────────────┘
       │                    │
┌──────▼──────┐    ┌────────▼────────┐    ┌─────────────┐
│ ML Pipeline │    │ Sentiment Engine│    │ Claude API  │
│ GBM + RF    │    │ TextBlob + NLP  │    │ Recommend.  │
│ Scikit-learn│    │ Online reviews  │    │ Generator   │
└─────────────┘    └─────────────────┘    └─────────────┘
```

## Features

- **AI Acquisition Scoring** — Gradient Boosting model scores 0-100
- **Sentiment Analysis** — NLP on reviews + news using TextBlob
- **LLM Recommendations** — Claude API generates strategic acquisition advice
- **Batch Processing** — Analyze 100s of businesses from CSV
- **4-Tier Classification** — Platinum / Gold / Silver / Bronze
- **LTV Estimation** — Predicted lifetime value per business
- **Product Matching** — Recommends best Amex products per business profile

---

## Quick Start (Local)

### 1. Clone & Setup

```bash
git clone https://github.com/yourname/amex-b2b-predictor.git
cd amex-b2b-predictor
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download TextBlob corpora
python -c "import textblob; textblob.download_corpora()"

# Set your API key
cp ../.env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here

# Run the server
python app.py
# → Backend running at http://localhost:5000
```

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Analyze a business
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "business_name": "TechNova Solutions",
    "sector": "Technology",
    "annual_revenue": 8500000,
    "employee_count": 45,
    "years_in_business": 7,
    "location": "San Francisco, CA",
    "monthly_spend_estimate": 95000,
    "review_text": "Excellent software company with amazing support",
    "news_text": "TechNova raises $12M Series B, expanding to enterprise"
  }'
```

### 4. Batch Process CSV

```bash
python scripts/batch_process.py --input data/sample_businesses.csv --output results/scored_leads.csv
```

---

## Deployment Options

### Option A: Deploy to Render.com (RECOMMENDED — Free tier available)

1. Push code to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt && python -c "import textblob; textblob.download_corpora()"`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Add Environment Variable: `ANTHROPIC_API_KEY` = your key
6. Deploy → Your API is live at `https://your-app.onrender.com`

**Cost**: Free (spins down after 15 min inactivity) or $7/month for always-on

---

### Option B: Deploy to Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# From project root
cd backend
railway init
railway add
railway up

# Set env var
railway variables set ANTHROPIC_API_KEY=your_key_here
```

**Cost**: $5/month with $5 free credit

---

### Option C: Docker + Any Cloud VPS

```bash
# Build and run locally with Docker
cd amex-b2b-predictor

# Create .env file first
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Start everything
docker-compose up --build -d

# Check logs
docker-compose logs -f backend
```

To deploy to DigitalOcean / AWS EC2 / Google Cloud:
1. Create a $6/month Droplet (DigitalOcean) or t2.micro EC2
2. SSH in, install Docker: `curl -fsSL https://get.docker.com | sh`
3. Clone your repo, create .env, run `docker-compose up -d`
4. Point your domain to the server IP

---

### Option D: Heroku

```bash
cd backend
heroku create amex-b2b-predictor
heroku config:set ANTHROPIC_API_KEY=your_key_here

# Create Procfile
echo "web: gunicorn app:app --bind 0.0.0.0:\$PORT" > Procfile

git add . && git commit -m "deploy"
git push heroku main
heroku open
```

---

## API Reference

### POST /api/analyze

Analyze a single business and return acquisition score.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| business_name | string | Yes | Company name |
| sector | string | Yes | Industry sector |
| annual_revenue | number | Yes | Annual revenue in USD |
| employee_count | number | No | Number of employees |
| years_in_business | number | No | Years operating |
| location | string | No | City, State |
| monthly_spend_estimate | number | No | Monthly card spend estimate |
| review_text | string | No | Customer reviews text |
| news_text | string | No | Recent news/press text |

**Response:**
```json
{
  "business_name": "TechNova Solutions",
  "acquisition_score": 87.3,
  "tier": "Platinum",
  "risk_level": "Low",
  "review_sentiment": {"score": 0.82, "label": "Positive"},
  "news_sentiment": {"score": 0.71, "label": "Positive"},
  "top_features": [
    {"feature": "Annual Revenue", "importance": 24.1},
    {"feature": "Customer Reviews", "importance": 18.7}
  ],
  "recommendation": "TechNova Solutions presents an exceptionally strong...",
  "recommended_products": ["Amex Business Platinum", "Corporate Gold Card"],
  "estimated_ltv": 285000,
  "confidence": 74.6
}
```

### POST /api/batch-analyze

Analyze multiple businesses at once.

**Request Body:** `{ "businesses": [ ...array of business objects... ] }`

### GET /api/dashboard-stats

Returns aggregated pipeline statistics for dashboard.

---

## Project Structure

```
amex-b2b-predictor/
├── backend/
│   ├── app.py              # Flask API entry point
│   ├── predictor.py        # ML model (Gradient Boosting)
│   ├── sentiment_analyzer.py # TextBlob + Claude LLM
│   ├── data_processor.py   # Feature engineering
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── sample_businesses.csv
├── models/                 # Auto-created, stores trained model
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Resume / Portfolio Claims

This project directly demonstrates:

✅ **"AI-Powered Targeting"** — ML model scores acquisition probability using 18 features including NLP sentiment

✅ **"Identify Customer Trends"** — Sector-level analytics, monthly trend tracking, LTV estimation

✅ **"Complex Data → Clear Recommendations"** — Raw reviews + financials → actionable Platinum/Gold/Silver/Bronze tiers

✅ **"GenAI Integration"** — Claude API generates strategic per-business recommendations

✅ **"Full Stack Deployment"** — Flask REST API + React dashboard, Docker-ready, cloud deployable

**Interview talking points:**
- "Built an end-to-end ML pipeline using Gradient Boosting with 18 engineered features including NLP sentiment scores"
- "Integrated Claude API to generate human-readable strategic recommendations from structured + unstructured data"
- "Achieved batch processing capability for 1000+ businesses with automatic tier classification and LTV estimation"
- "System deployed via Docker on cloud infrastructure with REST API serving real-time predictions"

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| ANTHROPIC_API_KEY | Yes (for LLM recommendations) | Get from console.anthropic.com |
| FLASK_ENV | No | `production` or `development` |
| PORT | No | Server port (default 5000) |

**Note:** Without ANTHROPIC_API_KEY, the system still works — it falls back to rule-based recommendations. All ML scoring and sentiment analysis works without the API key.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Model | Scikit-learn GradientBoostingClassifier | Acquisition probability scoring |
| NLP | TextBlob | Sentiment analysis on reviews/news |
| LLM | Anthropic Claude API | Strategic recommendation generation |
| API | Flask + Flask-CORS | REST API backend |
| Feature Engineering | Pandas + NumPy | Data processing pipeline |
| Serving | Gunicorn | Production WSGI server |
| Container | Docker + Docker Compose | Deployment packaging |
