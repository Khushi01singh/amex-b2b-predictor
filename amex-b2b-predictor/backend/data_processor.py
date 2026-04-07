"""
Data Processor - Feature Engineering for B2B Predictor
Converts raw business data + sentiment into ML-ready feature vectors
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

SECTOR_MAP = {
    "technology": "tech", "software": "tech", "it": "tech", "saas": "tech",
    "retail": "retail", "ecommerce": "retail", "wholesale": "retail",
    "healthcare": "healthcare", "medical": "healthcare", "pharma": "healthcare",
    "finance": "finance", "banking": "finance", "insurance": "finance", "fintech": "finance",
    "manufacturing": "manufacturing", "industrial": "manufacturing",
    "hospitality": "hospitality", "restaurant": "hospitality", "hotel": "hospitality",
    "real estate": "other", "education": "other", "logistics": "other",
    "professional services": "other", "consulting": "other"
}

LOCATION_TIER = {
    # Tier 1: Top metros
    "new york": 1, "los angeles": 1, "chicago": 1, "san francisco": 1,
    "seattle": 1, "boston": 1, "miami": 1, "washington": 1,
    # Tier 2: Large cities
    "dallas": 2, "houston": 2, "phoenix": 2, "atlanta": 2,
    "denver": 2, "austin": 2, "portland": 2, "nashville": 2,
    # Default
    "other": 3
}


class DataProcessor:
    def build_features(self, data: dict, review_sentiment: dict, news_sentiment: dict) -> list:
        """
        Convert raw business data + sentiments into feature vector.
        Matches FEATURE_NAMES order in predictor.py
        """
        # Revenue
        revenue = float(data.get("annual_revenue", 500000) or 500000)
        revenue_log = np.log1p(max(revenue, 1))

        # Employees
        employees = float(data.get("employee_count", 10) or 10)
        employees_log = np.log1p(max(employees, 1))

        # Years
        years = float(data.get("years_in_business", 3) or 3)
        years = np.clip(years, 0, 50)

        # Monthly spend
        monthly_spend = float(data.get("monthly_spend_estimate", revenue / 12 * 0.05) or revenue / 12 * 0.05)
        spend_log = np.log1p(max(monthly_spend, 1))

        # Sentiment scores
        rev_score = float(review_sentiment.get("score", 0.5))
        news_score = float(news_sentiment.get("score", 0.5))

        # Review volume (optional field)
        review_volume = float(data.get("review_count", 50) or 50)

        # Sector encoding
        raw_sector = str(data.get("sector", "other")).lower()
        sector_key = SECTOR_MAP.get(raw_sector, "other")
        s_tech = 1 if sector_key == "tech" else 0
        s_retail = 1 if sector_key == "retail" else 0
        s_health = 1 if sector_key == "healthcare" else 0
        s_finance = 1 if sector_key == "finance" else 0
        s_manuf = 1 if sector_key == "manufacturing" else 0
        s_hosp = 1 if sector_key == "hospitality" else 0
        s_other = 1 if sector_key == "other" else 0

        # Location tier
        location = str(data.get("location", "other")).lower()
        tier = 3
        for city, t in LOCATION_TIER.items():
            if city in location:
                tier = t
                break

        # Derived features
        revenue_growth = float(data.get("revenue_growth_pct", 0.10) or 0.10)
        revenue_growth = np.clip(revenue_growth, -0.5, 1.0)

        digital_presence = self._estimate_digital_presence(data)
        creditworthiness = self._estimate_creditworthiness(data, rev_score)

        return [
            revenue_log, employees_log, years, spend_log,
            rev_score, news_score, review_volume,
            s_tech, s_retail, s_health, s_finance, s_manuf, s_hosp, s_other,
            tier, revenue_growth, digital_presence, creditworthiness
        ]

    def _estimate_digital_presence(self, data: dict) -> float:
        """Heuristic digital presence score 0-1."""
        score = 0.3  # base
        if data.get("website_url"):
            score += 0.2
        if data.get("social_followers", 0) > 1000:
            score += 0.2
        if data.get("google_rating", 0) >= 4.0:
            score += 0.15
        if data.get("review_count", 0) > 50:
            score += 0.15
        return min(score, 1.0)

    def _estimate_creditworthiness(self, data: dict, sentiment_score: float) -> float:
        """Heuristic creditworthiness score 0-1."""
        score = 0.5
        revenue = float(data.get("annual_revenue", 0) or 0)
        years = float(data.get("years_in_business", 0) or 0)

        if revenue > 1_000_000:
            score += 0.15
        if revenue > 5_000_000:
            score += 0.10
        if years > 5:
            score += 0.10
        if years > 15:
            score += 0.05

        score += (sentiment_score - 0.5) * 0.2
        return np.clip(score, 0, 1)

    def get_sample_pipeline_data(self) -> dict:
        """Returns sample dashboard stats for demo."""
        return {
            "total_businesses_analyzed": 1247,
            "high_priority_leads": 312,
            "avg_acquisition_score": 61.4,
            "pipeline_value_estimate": 45_600_000,
            "sector_breakdown": [
                {"sector": "Technology", "count": 287, "avg_score": 72.1},
                {"sector": "Finance", "count": 198, "avg_score": 68.4},
                {"sector": "Healthcare", "count": 156, "avg_score": 65.2},
                {"sector": "Retail", "count": 234, "avg_score": 58.7},
                {"sector": "Manufacturing", "count": 145, "avg_score": 54.3},
                {"sector": "Hospitality", "count": 122, "avg_score": 49.8},
                {"sector": "Other", "count": 105, "avg_score": 45.2}
            ],
            "score_distribution": [
                {"range": "80-100 (Platinum)", "count": 89},
                {"range": "60-79 (Gold)", "count": 223},
                {"range": "40-59 (Silver)", "count": 445},
                {"range": "0-39 (Bronze)", "count": 490}
            ],
            "monthly_trend": [
                {"month": "Oct", "analyzed": 145, "converted": 23},
                {"month": "Nov", "analyzed": 167, "converted": 31},
                {"month": "Dec", "analyzed": 134, "converted": 19},
                {"month": "Jan", "analyzed": 189, "converted": 42},
                {"month": "Feb", "analyzed": 213, "converted": 55},
                {"month": "Mar", "analyzed": 256, "converted": 71}
            ],
            "top_leads": [
                {"name": "TechNova Solutions", "sector": "Technology", "score": 94.2, "tier": "Platinum", "ltv": 285000},
                {"name": "MedCore Systems", "sector": "Healthcare", "score": 91.7, "tier": "Platinum", "ltv": 198000},
                {"name": "FinBridge Capital", "sector": "Finance", "score": 89.3, "tier": "Platinum", "ltv": 412000},
                {"name": "RetailEdge Inc", "sector": "Retail", "score": 86.1, "tier": "Platinum", "ltv": 156000},
                {"name": "CloudAxis Corp", "sector": "Technology", "score": 83.5, "tier": "Platinum", "ltv": 234000}
            ]
        }
