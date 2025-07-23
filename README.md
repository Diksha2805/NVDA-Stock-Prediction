# 📘 NVIDIA Stock Price Prediction using ARIMA & LSTM

This project forecasts NVIDIA (NVDA) stock prices using time series techniques: a traditional statistical model (**ARIMA**) and a deep learning model (**LSTM**) for comparison. Historical data is fetched directly from Yahoo Finance.

---

## ✨ Features
- 📥 Automatically downloads NVDA historical stock data via `yfinance`
- 📊 Visualizes stock closing prices over time
- 🔥 Implements ARIMA for statistical forecasting
- 🤖 Uses LSTM (Long Short-Term Memory) for deep learning prediction
- 📈 Compares actual vs predicted stock prices
- 🧮 Calculates Mean Squared Error (MSE) for performance evaluation

---

## ⚙️ Tech Stack
- **Language:** Python 3.x
- **Libraries:**
  - `yfinance`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `statsmodels`
  - `scikit-learn`
  - `tensorflow`

---

## 🚀 How to Run
- Install dependencies:
  ```bash
  pip install -r requirements.txt

Run the script:
python Main.py
View visualizations and prediction results in the console or pop-up plots.

📁 NVIDIA_Stock_Prediction/
├── Main.py                         # Script for data download, modeling, and visualization
├── NVDA_stock_data.csv             # Historical stock data (auto-generated)
├── README.md                       # Project documentation
├── README_NVDA_Stock_Prediction.pdf # PDF version of README
└── requirements.txt                # Python dependencies

