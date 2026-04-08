"""
B2B Customer Sentiment & Growth Predictor
American Express - AI-Powered Customer Targeting Engine
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import io
from datetime import datetime
import logging

from predictor import B2BPredictor
from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Frontend se connection allow karne ke liye CORS
CORS(app)

# Initialize components
predictor = B2BPredictor()
sentiment_analyzer = SentimentAnalyzer()
data_processor = DataProcessor()

# Load & train model on startup
logger.info("Training model on startup...")
predictor.train()
logger.info("Model ready.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

@app.route("/api/analyze", methods=["POST"])
def analyze_business():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        review_sentiment = sentiment_analyzer.analyze(data.get("review_text", ""))
        news_sentiment = sentiment_analyzer.analyze(data.get("news_text", ""))
        features = data_processor.build_features(data, review_sentiment, news_sentiment)
        result = predictor.predict(features)

        recommendation = sentiment_analyzer.generate_recommendation(
            business_name=data.get("business_name", "Unknown"),
            sector=data.get("sector", "Unknown"),
            score=result["acquisition_score"],
            review_text=data.get("review_text", ""),
            news_text=data.get("news_text", ""),
            structured_data=data
        )

        response = {
            "business_name": data.get("business_name"),
            "acquisition_score": round(result["acquisition_score"] * 100, 1),
            "tier": result["tier"],
            "risk_level": result["risk_level"],
            "review_sentiment": review_sentiment,
            "news_sentiment": news_sentiment,
            "top_features": result["top_features"],
            "recommendation": recommendation,
            "recommended_products": result["recommended_products"],
            "estimated_ltv": result["estimated_ltv"],
            "confidence": result["confidence"],
            "analyzed_at": datetime.utcnow().isoformat()
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    """
    Analyze multiple businesses from uploaded CSV or JSON.
    """
    businesses = []

    # Case 1: Handle CSV File Upload (Form Data)
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",")
            # Convert DataFrame to list of dictionaries
            businesses = df.to_dict(orient='records')
        else:
            return jsonify({"error": "Invalid file format. Please upload a CSV."}), 400
    
    # Case 2: Handle JSON Data (Old method)
    else:
        data = request.json
        if data:
            businesses = data.get("businesses", [])

    if not businesses:
        return jsonify({"error": "No business data provided"}), 400

    results = []
    for biz in businesses:
        try:
            review_sentiment = sentiment_analyzer.analyze(str(biz.get("review_text", "")))
            news_sentiment = sentiment_analyzer.analyze(str(biz.get("news_text", "")))
            features = data_processor.build_features(biz, review_sentiment, news_sentiment)
            pred = predictor.predict(features)

            results.append({
                "business_name": biz.get("business_name"),
                "sector": biz.get("sector"),
                "acquisition_score": round(pred["acquisition_score"] * 100, 1),
                "tier": pred["tier"],
                "risk_level": pred["risk_level"],
                "estimated_ltv": pred["estimated_ltv"]
            })
        except Exception as e:
            logger.error(f"Batch item error: {e}")
            results.append({"business_name": biz.get("business_name"), "error": "Processing failed"})

    results.sort(key=lambda x: x.get("acquisition_score", 0), reverse=True)

    return jsonify({
        "status": "success",
        "processed_count": len(results),
        "high_priority": sum(1 for r in results if r.get("tier") == "Platinum"),
        "results": results
    })

@app.route("/api/dashboard-stats", methods=["GET"])
def dashboard_stats():
    sample = data_processor.get_sample_pipeline_data()
    return jsonify(sample)

@app.route("/api/sectors", methods=["GET"])
def get_sectors():
    sectors = ["Technology", "Retail", "Healthcare", "Finance", "Manufacturing", "Logistics"]
    return jsonify(sectors)

@app.route('/')
def home():
    return {"status": "online", "message": "Amex B2B Predictor API is running"}

if __name__ == "__main__":
    # Render $PORT environment variable use karta hai
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
