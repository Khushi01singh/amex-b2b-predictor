"""
Batch Processor Script
Usage: python batch_process.py --input data/sample_businesses.csv --output results/scored.csv
"""

import argparse
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/backend")

from predictor import B2BPredictor
from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor

def process_csv(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    predictor = B2BPredictor()
    sentiment_analyzer = SentimentAnalyzer()
    data_processor = DataProcessor()

    print("Training model...")
    predictor.train()

    results = []
    for idx, row in df.iterrows():
        data = row.to_dict()
        print(f"  [{idx+1}/{len(df)}] Analyzing: {data.get('business_name', 'Unknown')}...")

        try:
            review_sentiment = sentiment_analyzer.analyze(str(data.get("review_text", "")))
            news_sentiment = sentiment_analyzer.analyze(str(data.get("news_text", "")))
            features = data_processor.build_features(data, review_sentiment, news_sentiment)
            pred = predictor.predict(features)

            results.append({
                **{k: data.get(k) for k in ["business_name","sector","location","annual_revenue","employee_count","years_in_business"]},
                "acquisition_score": round(pred["acquisition_score"] * 100, 1),
                "tier": pred["tier"],
                "risk_level": pred["risk_level"],
                "estimated_ltv": pred["estimated_ltv"],
                "recommended_products": ", ".join(pred["recommended_products"]),
                "review_sentiment": review_sentiment["label"],
                "review_sentiment_score": review_sentiment["score"],
                "news_sentiment": news_sentiment["label"],
                "news_sentiment_score": news_sentiment["score"]
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"business_name": data.get("business_name"), "error": str(e)})

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("acquisition_score", ascending=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\n✅ Done! Results saved to {output_path}")
    print(f"   Total businesses: {len(out_df)}")
    print(f"   Platinum tier: {len(out_df[out_df['tier']=='Platinum'])}")
    print(f"   Gold tier: {len(out_df[out_df['tier']=='Gold'])}")
    print(f"\nTop 5 Acquisition Targets:")
    print(out_df[["business_name","sector","acquisition_score","tier"]].head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch B2B Acquisition Scorer")
    parser.add_argument("--input", default="data/sample_businesses.csv")
    parser.add_argument("--output", default="results/scored_leads.csv")
    args = parser.parse_args()
    process_csv(args.input, args.output)
