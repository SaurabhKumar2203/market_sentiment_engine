# app.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="AI Market Forecaster", layout="wide")
st.title("📈 Automated Market Sentiment & Pricing Engine")
st.markdown("""
This dashboard displays real-time 7-day price forecasts for AAPL. 
The backend pipeline automatically scrapes daily financial news, processes sentiment using a fine-tuned **FinBERT Transformer**, and generates future price predictions using **Meta's Prophet** time-series model.
""")

# 2. Database Connection (Cached so it doesn't reconnect on every click)
@st.cache_resource
def init_connection():
    db_url = st.secrets["DATABASE_URL"]
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return create_engine(db_url)

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_data():
    engine = init_connection()
    # Pull the latest inference run from the database
    query = """
        SELECT * FROM market_predictions 
        WHERE inference_date = (SELECT MAX(inference_date) FROM market_predictions)
        ORDER BY ds ASC
    """
    df = pd.read_sql(query, engine)
    return df

# 3. Fetch and Display Data
try:
    df = fetch_data()
    
    if df.empty:
        st.warning("No predictions found in the database. Has the GitHub Action run yet?")
    else:
        st.subheader(f"Latest 7-Day Forecast (Generated on {df['inference_date'].iloc[0]})")
        
        # 4. Create an Interactive Plotly Chart
        fig = go.Figure()
        
        # Add the prediction line
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['yhat'], 
            mode='lines+markers', 
            name='Predicted Price',
            line=dict(color='#00FF00', width=3)
        ))
        
        # Add the upper and lower confidence bounds
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['yhat_upper'], 
            mode='lines', line=dict(width=0), 
            showlegend=False, name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['yhat_lower'], 
            mode='lines', line=dict(width=0), 
            fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', 
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title="AAPL Price Forecast (Next 7 Days)",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            template="plotly_dark",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Show Raw Data
        with st.expander("View Raw Data"):
            st.dataframe(df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            
except Exception as e:
    st.error(f"Failed to connect to the database or fetch data: {e}")