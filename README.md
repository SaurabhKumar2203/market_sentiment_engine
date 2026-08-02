# 📈 Automated Market Sentiment & Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/NLP-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![Prophet](https://img.shields.io/badge/Forecasting-Prophet-orange.svg)](https://facebook.github.io/prophet/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)](https://neon.tech/)
[![GitHub Actions](https://img.shields.io/badge/MLOps-GitHub_Actions-2088FF.svg)](https://github.com/features/actions)

## 📌 Project Overview
An automated, end-to-end Machine Learning pipeline that predicts 7-day stock price movements (currently targeting AAPL). The engine scrapes daily financial news, processes unstructured text into quantitative sentiment scores using a fine-tuned Transformer model, and feeds these features into a time-series forecasting model.

**[View the Live Streamlit Dashboard Here] (INSERT_YOUR_STREAMLIT_URL_HERE)**

## 🏗️ System Architecture & MLOps Pipeline

This project is fully automated to run daily at midnight via GitHub Actions without human intervention.

1. **Data Extraction:** Pulls historical OHLCV pricing data via `yfinance` and scrapes real-time news headlines using `BeautifulSoup`.
2. **NLP Engine:** Passes raw text through **FinBERT** (`ProsusAI/finbert`), converting qualitative news into a daily aggregated numerical sentiment score (-1.0 to +1.0).
3. **Time-Series Forecasting:** Merges price momentum with NLP sentiment features to train a **Meta Prophet** model, predicting the next 7 days of price action while strictly avoiding data leakage.
4. **Data Engineering:** Ingests the forecasted data into a cloud-hosted **PostgreSQL** database (Neon.tech).
5. **Frontend Application:** A real-time **Streamlit** dashboard queries the database to visualize the latest predictions with Plotly.

## 🛠️ Technology Stack

* **Data Scraping & APIs:** `yfinance`, `BeautifulSoup`, `requests`
* **Machine Learning / NLP:** `Transformers` (HuggingFace), `PyTorch`, `Prophet`
* **Data Processing:** `Pandas`, `NumPy`
* **Database & ORM:** `PostgreSQL` (Neon), `SQLAlchemy`
* **DevOps / MLOps:** `GitHub Actions` (CI/CD Pipeline)
* **Frontend:** `Streamlit`, `Plotly`

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/market_sentiment_engine.git](https://github.com/YOUR_USERNAME/market_sentiment_engine.git)
cd market_sentiment_engine
