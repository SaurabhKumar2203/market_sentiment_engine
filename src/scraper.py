import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

class MarketDataExtractor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

    def fetch_pricing_data(self, period="1mo", interval="1d"):
        print(f"Fetching pricing data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        hist_data = stock.history(period=period, interval=interval)
        
        if hist_data.empty:
            print("Warning: No pricing data found.")
            return None
            
        hist_data.reset_index(inplace=True)
        hist_data['Date'] = hist_data['Date'].dt.tz_localize(None)
        
        return hist_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def scrape_news_headlines(self):
        """
        Scrapes recent news headlines for the ticker using BeautifulSoup from Finviz.
        """
        print(f"Scraping news headlines for {self.ticker} from Finviz...")
        # Pivot to Finviz, a highly reliable source for stock news
        url = f"https://finviz.com/quote.ashx?t={self.ticker}"
        time.sleep(2) 
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status() 
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Finviz stores news in an HTML table with the id 'news-table'
        news_table = soup.find(id='news-table')
        
        if news_table:
            # Find all table rows (tr) within the news table
            for row in news_table.findAll('tr'):
                a_tag = row.a
                if a_tag:
                    title = a_tag.get_text(strip=True)
                    headlines.append({
                        # We map today's date so it cleanly merges with Prophet's daily prices
                        'Date': pd.Timestamp.today().normalize(), 
                        'Ticker': self.ticker,
                        'Headline': title
                    })
                    
        df_news = pd.DataFrame(headlines)
        print(f"Successfully scraped {len(df_news)} headlines.")
        return df_news