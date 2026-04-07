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
from datetime import datetime
import logging

from predictor import B2BPredictor
from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
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
    """
    Analyze a single business and return Amex acquisition score + insights.
    Body: { business_name, sector, location, annual_revenue, employee_count,
            years_in_business, monthly_spend_estimate, review_text, news_text }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Step 1: Sentiment analysis on unstructured text
        review_sentiment = sentiment_analyzer.analyze(data.get("review_text", ""))
        news_sentiment = sentiment_analyzer.analyze(data.get("news_text", ""))

        # Step 2: Build feature vector
        features = data_processor.build_features(data, review_sentiment, news_sentiment)

        # Step 3: Predict acquisition probability
        result = predictor.predict(features)

        # Step 4: Generate LLM-based recommendation
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
    Analyze multiple businesses at once from uploaded CSV data.
    Body: { businesses: [...] }
    """
    data = request.json
    businesses = data.get("businesses", [])

    if not businesses:
        return jsonify({"error": "No businesses provided"}), 400

    results = []
    for biz in businesses:
        try:
            review_sentiment = sentiment_analyzer.analyze(biz.get("review_text", ""))
            news_sentiment = sentiment_analyzer.analyze(biz.get("news_text", ""))
            features = data_processor.build_features(biz, review_sentiment, news_sentiment)
            pred = predictor.predict(features)

            results.append({
                "business_name": biz.get("business_name"),
                "sector": biz.get("sector"),
                "location": biz.get("location"),
                "acquisition_score": round(pred["acquisition_score"] * 100, 1),
                "tier": pred["tier"],
                "risk_level": pred["risk_level"],
                "estimated_ltv": pred["estimated_ltv"],
                "recommended_products": pred["recommended_products"],
                "review_sentiment_score": review_sentiment["score"],
                "news_sentiment_score": news_sentiment["score"]
            })
        except Exception as e:
            results.append({"business_name": biz.get("business_name"), "error": str(e)})

    # Sort by acquisition score descending
    results.sort(key=lambda x: x.get("acquisition_score", 0), reverse=True)

    return jsonify({
        "total": len(results),
        "high_priority": sum(1 for r in results if r.get("tier") == "Platinum"),
        "results": results
    })


@app.route("/api/dashboard-stats", methods=["GET"])
def dashboard_stats():
    """Returns aggregated stats for the dashboard overview."""
    sample = data_processor.get_sample_pipeline_data()
    return jsonify(sample)


@app.route("/api/sectors", methods=["GET"])
def get_sectors():
    sectors = [
        "Technology", "Retail", "Healthcare", "Finance",
        "Manufacturing", "Hospitality", "Real Estate",
        "Professional Services", "Education", "Logistics"
    ]
    return jsonify(sectors)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # ... baaki saara purana code ...

@app.route('/')
def home():
    return {"status": "online", "message": "Amex B2B Predictor API is running"}

if __name__ == "__main__":
    app.run()
