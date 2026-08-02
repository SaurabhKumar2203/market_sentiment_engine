# src/db_logger.py
from sqlalchemy import create_engine
import os
import pandas as pd
from datetime import datetime

class DatabaseLogger:
    def __init__(self):
        # We fetch the connection string from environment variables for security
        self.db_url = os.getenv("DATABASE_URL")
        
        if not self.db_url:
            print("Warning: DATABASE_URL not found. Skipping database logging.")
            self.engine = None
        else:
            # SQLAlchemy requires the prefix 'postgresql://' instead of 'postgres://'
            if self.db_url.startswith("postgres://"):
                self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            self.engine = create_engine(self.db_url)

    def log_predictions(self, forecast_df, table_name="market_predictions"):
        if self.engine is None:
            return
            
        print(f"Logging {len(forecast_df)} rows to PostgreSQL table: '{table_name}'...")
        
        # Create a copy so we don't modify the original dataframe
        df_to_log = forecast_df.copy()
        
        # Add metadata: When was this prediction generated? 
        df_to_log['inference_date'] = datetime.today().date()
        df_to_log['ticker'] = 'AAPL' # Hardcoded for now, but can be passed dynamically
        
        try:
            # if_exists='append' ensures we just add today's run to the historical log
            df_to_log.to_sql(table_name, self.engine, if_exists='append', index=False)
            print("Successfully logged to PostgreSQL database.")
        except Exception as e:
            print(f"Failed to log to database: {e}")