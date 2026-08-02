# src/forecaster.py
import pandas as pd
from prophet import Prophet

class PriceForecaster:
    def __init__(self):
        # Initialize Prophet with daily seasonality (crucial for daily stock data)
        self.model = Prophet(daily_seasonality=True)
        
        # Add our engineered NLP feature as an external regressor
        self.model.add_regressor('Daily_Avg_Sentiment')

    def prepare_and_merge(self, prices_df, sentiment_df):
        """
        Merges numerical prices with text sentiment and formats for Prophet.
        """
        print("Merging pricing data with NLP sentiment features...")
        
        # FIX: Merge ONLY on 'Date' (prices_df doesn't have a 'Ticker' column)
        merged_df = pd.merge(prices_df, sentiment_df, on='Date', how='left')
        
        # Fill missing sentiment days with 0.0 (Neutral)
        merged_df['Daily_Avg_Sentiment'] = merged_df['Daily_Avg_Sentiment'].fillna(0.0)
        
        # Prophet STRICTLY requires columns to be named 'ds' (datestamp) and 'y' (target)
        prophet_df = merged_df[['Date', 'Close', 'Daily_Avg_Sentiment']].rename(
            columns={'Date': 'ds', 'Close': 'y'}
        )
        return prophet_df
    
    def train_and_predict(self, prophet_df, forecast_days=7):
        """
        Trains the Prophet model and generates future predictions.
        """
        print("Training Prophet time-series model...")
        self.model.fit(prophet_df)
        
        # Create a dataframe outlining the next 7 days
        future = self.model.make_future_dataframe(periods=forecast_days)
        
        # Prevent Data Leakage: 
        # Map historical sentiment to past dates, and assume 0.0 (Neutral) for future dates.
        sentiment_map = dict(zip(prophet_df['ds'], prophet_df['Daily_Avg_Sentiment']))
        future['Daily_Avg_Sentiment'] = future['ds'].map(sentiment_map).fillna(0.0)
        
        print(f"Generating price forecast for the next {forecast_days} days...")
        forecast = self.model.predict(future)
        
        # Return the dates (ds), predicted price (yhat), and the confidence intervals
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        return result