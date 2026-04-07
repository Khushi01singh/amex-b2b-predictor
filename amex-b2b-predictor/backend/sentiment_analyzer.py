"""
Sentiment Analyzer + LLM Recommendation Engine
Uses Anthropic Claude API for deep text analysis & strategic recommendations
"""

import os
import re
import json
import logging
import anthropic
from textblob import TextBlob

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


class SentimentAnalyzer:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text using TextBlob (fast, offline).
        Returns score (0-1), label, and key themes.
        """
        if not text or len(text.strip()) < 10:
            return {"score": 0.5, "label": "Neutral", "themes": [], "confidence": 0.0}

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity      # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # Normalize to 0-1
        score = (polarity + 1) / 2

        if score >= 0.65:
            label = "Positive"
        elif score <= 0.35:
            label = "Negative"
        else:
            label = "Neutral"

        # Extract key noun phrases as themes
        themes = list(set([str(np).title() for np in blob.noun_phrases[:5]]))

        return {
            "score": round(score, 3),
            "label": label,
            "subjectivity": round(subjectivity, 3),
            "themes": themes,
            "confidence": round(abs(polarity), 3)
        }

    def generate_recommendation(
        self,
        business_name: str,
        sector: str,
        score: float,
        review_text: str,
        news_text: str,
        structured_data: dict
    ) -> str:
        """
        Use Claude API to generate a strategic acquisition recommendation.
        Falls back to rule-based recommendation if API key not available.
        """
        if self.client:
            return self._llm_recommendation(
                business_name, sector, score, review_text, news_text, structured_data
            )
        else:
            return self._rule_based_recommendation(business_name, sector, score, structured_data)

    def _llm_recommendation(self, business_name, sector, score, review_text, news_text, data):
        """Call Claude API for strategic recommendation."""
        try:
            prompt = f"""You are an American Express B2B acquisition strategist. Analyze this business and provide a concise strategic recommendation for Amex targeting.

Business: {business_name}
Sector: {sector}
Annual Revenue: ${data.get('annual_revenue', 'N/A'):,}
Employees: {data.get('employee_count', 'N/A')}
Years in Business: {data.get('years_in_business', 'N/A')}
Monthly Spend Estimate: ${data.get('monthly_spend_estimate', 'N/A'):,}
AI Acquisition Score: {score*100:.1f}/100

Customer Reviews Sample: "{review_text[:300] if review_text else 'No reviews available'}"
Recent News: "{news_text[:300] if news_text else 'No news available'}"

Provide a 3-4 sentence strategic recommendation covering:
1. Why this business is (or isn't) a strong Amex acquisition target
2. Which Amex product best fits their profile
3. The key risk or opportunity
Keep it concise, data-driven, and professional."""

            message = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        except Exception as e:
            logger.warning(f"LLM API call failed: {e}. Using rule-based fallback.")
            return self._rule_based_recommendation(business_name, sector, score, data)

    def _rule_based_recommendation(self, business_name, sector, score, data):
        """Fallback rule-based recommendation."""
        revenue = data.get("annual_revenue", 0)
        years = data.get("years_in_business", 0)

        if score >= 0.80:
            strength = "exceptionally strong"
            action = "Prioritize immediate outreach"
            product = "Amex Business Platinum or Corporate Card"
        elif score >= 0.60:
            strength = "strong"
            action = "Schedule targeted acquisition campaign"
            product = "Amex Business Gold"
        elif score >= 0.40:
            strength = "moderate"
            action = "Add to nurture pipeline"
            product = "Business Blue Cash"
        else:
            strength = "low"
            action = "Monitor for future opportunities"
            product = "SimplyCash Basic"

        return (
            f"{business_name} presents a {strength} acquisition opportunity for Amex "
            f"with a score of {score*100:.1f}/100. Operating in the {sector} sector "
            f"with {years} years of business history, this company aligns well with Amex's "
            f"B2B targeting criteria. {action} with the {product} as the primary offer. "
            f"Annual revenue of ${revenue:,.0f} suggests significant spending potential "
            f"for rewards-based card products."
        )
