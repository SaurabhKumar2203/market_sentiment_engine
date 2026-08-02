# src/nlp_engine.py
import pandas as pd
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

class FinancialSentimentAnalyzer:
    def __init__(self):
        print("Loading FinBERT model... (This may take a minute)")
        self.model_name = "ProsusAI/finbert"
        self.analyzer = pipeline("sentiment-analysis", model=self.model_name)

    def compute_sentiment_score(self, text):
        try:
            result = self.analyzer(text)[0]
            label = result['label']
            confidence = result['score']
            
            if label == 'positive':
                return confidence      
            elif label == 'negative':
                return -confidence     
            else:
                return 0.0             
                
        except Exception as e:
            print(f"Error processing text: {text[:30]}... | Error: {e}")
            return 0.0

    def process_daily_news(self, news_df):
        if news_df is None or news_df.empty:
            print("No news data to process.")
            return None
            
        print(f"Running NLP inference on {len(news_df)} headlines...")
        news_df['Sentiment_Score'] = news_df['Headline'].apply(self.compute_sentiment_score)
        
        # Aggregate to daily level
        daily_sentiment = news_df.groupby(['Date', 'Ticker'])['Sentiment_Score'].mean().reset_index()
        daily_sentiment.rename(columns={'Sentiment_Score': 'Daily_Avg_Sentiment'}, inplace=True)
        
        return daily_sentiment