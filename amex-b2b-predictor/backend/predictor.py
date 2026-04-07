"""
B2B Acquisition Predictor - ML Model
Uses Gradient Boosting + Logistic Regression ensemble
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "annual_revenue_log",
    "employee_count_log",
    "years_in_business",
    "monthly_spend_estimate_log",
    "review_sentiment_score",
    "news_sentiment_score",
    "review_volume",
    "sector_tech",
    "sector_retail",
    "sector_healthcare",
    "sector_finance",
    "sector_manufacturing",
    "sector_hospitality",
    "sector_other",
    "location_tier",        # 1=Metro, 2=Urban, 3=Suburban, 4=Rural
    "revenue_growth_proxy",
    "digital_presence_score",
    "creditworthiness_score"
]

PRODUCTS_MAP = {
    "Platinum": ["Amex Business Platinum", "Corporate Gold Card", "Working Capital Terms"],
    "Gold": ["Amex Business Gold", "Business Green Rewards", "Pay Over Time"],
    "Silver": ["Business Blue Cash", "SimplyCash Plus", "Kabbage Line of Credit"],
    "Bronze": ["SimplyCash", "Amex Basic Business"]
}

MODEL_PATH = "models/b2b_predictor.pkl"


class B2BPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        os.makedirs("models", exist_ok=True)

    def _generate_training_data(self, n_samples=2000):
        """Generate realistic synthetic training data."""
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            sector = np.random.choice(
                ["tech", "retail", "healthcare", "finance",
                 "manufacturing", "hospitality", "other"],
                p=[0.20, 0.18, 0.12, 0.15, 0.10, 0.10, 0.15]
            )
            revenue = np.random.lognormal(mean=13.5, sigma=1.8)  # ~$1M median
            employees = int(np.clip(revenue / 80000 * np.random.uniform(0.5, 1.5), 1, 10000))
            years = np.random.randint(1, 40)
            monthly_spend = revenue / 12 * np.random.uniform(0.02, 0.15)
            review_score = np.random.beta(6, 3)         # skewed positive
            news_score = np.random.normal(0.55, 0.2)
            news_score = np.clip(news_score, 0, 1)
            digital_presence = np.random.beta(4, 3)
            creditworthiness = np.random.beta(5, 2)
            location_tier = np.random.choice([1, 2, 3, 4], p=[0.35, 0.30, 0.25, 0.10])

            # Target: 1 = likely Amex customer (acquisition candidate)
            score = (
                0.25 * (np.log(revenue) / 20) +
                0.15 * review_score +
                0.15 * news_score +
                0.15 * creditworthiness +
                0.10 * digital_presence +
                0.10 * (1 / location_tier) +
                0.10 * min(years / 20, 1.0) +
                (0.10 if sector in ["tech", "finance"] else 0.05)
            )
            target = 1 if score + np.random.normal(0, 0.05) > 0.55 else 0

            row = [
                np.log1p(revenue),
                np.log1p(employees),
                years,
                np.log1p(monthly_spend),
                review_score,
                news_score,
                np.random.randint(0, 500),              # review_volume
                1 if sector == "tech" else 0,
                1 if sector == "retail" else 0,
                1 if sector == "healthcare" else 0,
                1 if sector == "finance" else 0,
                1 if sector == "manufacturing" else 0,
                1 if sector == "hospitality" else 0,
                1 if sector == "other" else 0,
                location_tier,
                np.random.uniform(0, 0.3),              # revenue_growth_proxy
                digital_presence,
                creditworthiness,
                target
            ]
            data.append(row)

        cols = FEATURE_NAMES + ["target"]
        return pd.DataFrame(data, columns=cols)

    def train(self):
        """Train the ensemble model."""
        if os.path.exists(MODEL_PATH):
            logger.info("Loading cached model...")
            self.model = joblib.load(MODEL_PATH)
            self.is_trained = True
            return

        logger.info("Generating training data & fitting model...")
        df = self._generate_training_data(2000)

        X = df[FEATURE_NAMES].values
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Gradient Boosting (primary)
        gb = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08,
            max_depth=4, random_state=42
        )
        gb.fit(X_train_scaled, y_train)

        self.model = gb
        self.is_trained = True

        preds = gb.predict(X_test_scaled)
        logger.info("\n" + classification_report(y_test, preds))
        joblib.dump(gb, MODEL_PATH)
        joblib.dump(self.scaler, "models/scaler.pkl")
        logger.info("Model saved.")

    def predict(self, feature_vector: list) -> dict:
        """Predict acquisition probability and return enriched result."""
        if not self.is_trained:
            self.train()

        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0][1]
        feature_importance = self.model.feature_importances_

        # Top 3 contributing features
        feat_scores = sorted(
            zip(FEATURE_NAMES, feature_importance),
            key=lambda x: x[1], reverse=True
        )[:3]
        top_features = [
            {"feature": self._friendly_name(f), "importance": round(v * 100, 1)}
            for f, v in feat_scores
        ]

        # Tier classification
        if prob >= 0.80:
            tier = "Platinum"
        elif prob >= 0.60:
            tier = "Gold"
        elif prob >= 0.40:
            tier = "Silver"
        else:
            tier = "Bronze"

        # Risk
        if prob >= 0.70:
            risk = "Low"
        elif prob >= 0.45:
            risk = "Medium"
        else:
            risk = "High"

        # Estimated LTV (simplified model)
        base_spend = np.expm1(feature_vector[3]) * 12  # annual from monthly log
        ltv = int(base_spend * (1 + prob) * np.random.uniform(1.5, 3.0))

        return {
            "acquisition_score": prob,
            "tier": tier,
            "risk_level": risk,
            "top_features": top_features,
            "recommended_products": PRODUCTS_MAP[tier],
            "estimated_ltv": ltv,
            "confidence": round(abs(prob - 0.5) * 2 * 100, 1)  # 0-100
        }

    def _friendly_name(self, feature: str) -> str:
        mapping = {
            "annual_revenue_log": "Annual Revenue",
            "monthly_spend_estimate_log": "Monthly Spend",
            "review_sentiment_score": "Customer Reviews",
            "news_sentiment_score": "News Sentiment",
            "creditworthiness_score": "Creditworthiness",
            "digital_presence_score": "Digital Presence",
            "years_in_business": "Business Maturity",
            "employee_count_log": "Company Size",
            "location_tier": "Market Location",
            "revenue_growth_proxy": "Growth Rate"
        }
        return mapping.get(feature, feature.replace("_", " ").title())
