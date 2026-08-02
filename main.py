# main.py
from src.scraper import MarketDataExtractor
from src.nlp_engine import FinancialSentimentAnalyzer
from src.forecaster import PriceForecaster
import os

def run_pipeline():
    TICKER = "AAPL"
    print(f"=== Starting Automated Market Pipeline for {TICKER} ===")
    
    # 1. Scrape Data (Pulling 3 months of history to give Prophet enough data to learn trends)
    extractor = MarketDataExtractor(TICKER)
    prices_df = extractor.fetch_pricing_data(period="3mo") 
    news_df = extractor.scrape_news_headlines()
    
    # 2. Analyze NLP Sentiment
    nlp = FinancialSentimentAnalyzer()
    sentiment_df = nlp.process_daily_news(news_df)
    
    # 3. Merge & Forecast
    if prices_df is not None and sentiment_df is not None:
        forecaster = PriceForecaster()
        prophet_df = forecaster.prepare_and_merge(prices_df, sentiment_df)
        
        predictions = forecaster.train_and_predict(prophet_df, forecast_days=7)
        
        print("\n=== Pipeline Complete! ===")
        print("Predicted Prices for the Next 7 Days (yhat):")
        print(predictions.to_string(index=False))
        
        # Optional: Save locally to verify output
        if not os.path.exists('data'):
            os.makedirs('data')
        predictions.to_csv('data/latest_predictions.csv', index=False)
        print("\nPredictions saved to data/latest_predictions.csv")
    else:
        print("Pipeline failed due to missing data.")

if __name__ == "__main__":
    run_pipeline()