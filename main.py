# main.py
from src.scraper import MarketDataExtractor
from src.nlp_engine import FinancialSentimentAnalyzer
from src.forecaster import PriceForecaster
from src.db_logger import DatabaseLogger  # NEW IMPORT
import os

def run_pipeline():
    TICKER = "AAPL"
    print(f"=== Starting Automated Market Pipeline for {TICKER} ===")
    
    extractor = MarketDataExtractor(TICKER)
    prices_df = extractor.fetch_pricing_data(period="3mo") 
    news_df = extractor.scrape_news_headlines()
    
    nlp = FinancialSentimentAnalyzer()
    sentiment_df = nlp.process_daily_news(news_df)
    
    if prices_df is not None and sentiment_df is not None:
        forecaster = PriceForecaster()
        prophet_df = forecaster.prepare_and_merge(prices_df, sentiment_df)
        
        predictions = forecaster.train_and_predict(prophet_df, forecast_days=7)
        
        print("\n=== Pipeline Complete! ===")
        print(predictions.to_string(index=False))
        
        # 1. Save to local CSV (for GitHub backup)
        if not os.path.exists('data'):
            os.makedirs('data')
        predictions.to_csv('data/latest_predictions.csv', index=False)
        
        # 2. Log to PostgreSQL (Production database)
        db = DatabaseLogger()
        db.log_predictions(predictions)
        
    else:
        print("Pipeline failed due to missing data.")

if __name__ == "__main__":
    run_pipeline()