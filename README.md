# 📈 Automated Market Sentiment & Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/NLP-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![Prophet](https://img.shields.io/badge/Forecasting-Prophet-orange.svg)](https://facebook.github.io/prophet/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)](https://neon.tech/)
[![GitHub Actions](https://img.shields.io/badge/MLOps-GitHub_Actions-2088FF.svg)](https://github.com/features/actions)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

## 📌 Project Overview
An automated, end-to-end Machine Learning pipeline that predicts 7-day stock price movements (currently targeting AAPL). The engine scrapes daily financial news, processes unstructured text into quantitative sentiment scores using a fine-tuned Transformer model, and feeds these features into a time-series forecasting model.

**[🔴 View the Live Interactive Dashboard Here](https://marketsentimentengine-hbd95htju9yqh72f5hcuws.streamlit.app/)**

---

## 📸 Dashboard Preview

![Streamlit Dashboard Screenshot](<img width="1881" height="881" alt="Screenshot 2026-08-02 082044" src="https://github.com/user-attachments/assets/db92514b-71cf-432b-b546-c12fef481d33" />
)


---

## 🏗️ System Architecture & MLOps Pipeline

This project is fully automated to run daily at midnight UTC via GitHub Actions without human intervention. The architecture ensures strict separation of concerns between data extraction, inference, logging, and visualization.

1. **Data Extraction (ETL):** Pulls historical OHLCV pricing data via `yfinance` and scrapes real-time news headlines from Finviz using `BeautifulSoup`.
2. **NLP Sentiment Engine:** Passes raw headline text through **FinBERT** (`ProsusAI/finbert`), converting qualitative financial news into a daily aggregated numerical sentiment score (-1.0 to +1.0).
3. **Time-Series Forecasting:** Merges price momentum with NLP sentiment features to train a **Meta Prophet** model. To prevent **data leakage**, future sentiment is neutralized for the 7-day forecasting horizon.
4. **Database Logging (Cold Start & Drift Handling):** Ingests the forecasted data into a cloud-hosted **PostgreSQL** database (Neon.tech), tagging each prediction with an `inference_date` to allow for future model drift monitoring.
5. **Frontend Application:** A real-time **Streamlit** dashboard queries the database to visualize the latest predictions and confidence intervals using Plotly.

---

## 🛠️ Technology Stack

* **Data Extraction:** `yfinance`, `BeautifulSoup`, `requests`
* **Machine Learning & NLP:** `Transformers` (HuggingFace), `PyTorch`, `Prophet`
* **Data Processing:** `Pandas`, `NumPy`
* **Database & ORM:** `PostgreSQL` (Neon.tech), `SQLAlchemy`, `psycopg2`
* **DevOps & MLOps:** `GitHub Actions` (CI/CD Cron Scheduling)
* **Frontend Visualization:** `Streamlit`, `Plotly`

---

## 📂 Project Structure

```text
market_sentiment_engine/
├── .github/workflows/
│   └── daily_pipeline.yml     # GitHub Actions cron job script
├── src/
│   ├── scraper.py             # Data extraction (yfinance & Finviz)
│   ├── nlp_engine.py          # HuggingFace FinBERT inference
│   ├── forecaster.py          # Prophet time-series modeling
│   └── db_logger.py           # PostgreSQL database connection & insertion
├── app.py                     # Streamlit frontend dashboard
├── main.py                    # Core pipeline execution script
├── requirements.txt           # Dependency management
└── README.md                  # Project documentation

```

---

## 🚀 How to Run Locally

If you wish to clone and run this architecture on your local machine:

### 1. Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/market_sentiment_engine.git](https://github.com/YOUR_USERNAME/market_sentiment_engine.git)
cd market_sentiment_engine

```

### 2. Set up the virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

```bash
pip install -r requirements.txt

```

### 3. Configure Database Secrets

You will need a PostgreSQL database (e.g., Neon or Supabase). Create a `.streamlit/secrets.toml` file in the root directory and add your connection string:

```toml
DATABASE_URL = "postgresql://username:password@hostname/dbname?sslmode=require"

```

Also, set the environment variable for the backend pipeline:

* **Windows (PowerShell):** `$env:DATABASE_URL="your_url_here"`
* **Mac/Linux:** `export DATABASE_URL="your_url_here"`

### 4. Run the Backend & Frontend

First, run the backend pipeline to generate predictions and seed your database:

```bash
python main.py

```

Next, launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
