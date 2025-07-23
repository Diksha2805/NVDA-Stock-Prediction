# 📘 NVIDIA Stock Price Prediction using ARIMA & LSTM

This project predicts NVIDIA (NVDA) stock prices using **time series forecasting techniques**. It applies both a traditional statistical model (ARIMA) and a deep learning model (LSTM) to forecast and compare performance. The dataset is downloaded automatically from Yahoo Finance.

---

## ✨ Features
- 📥 Downloads historical NVDA stock data via `yfinance`
- 📊 Visualizes stock closing prices over time
- 🔥 Implements ARIMA for statistical forecasting
- 🤖 Uses LSTM (Long Short-Term Memory) networks for deep learning-based prediction
- 📈 Compares actual vs predicted prices for both models
- 🧮 Calculates Mean Squared Error (MSE) to evaluate performance

---

## ⚙️ Tech Stack
- Python 3.x
- Libraries: 
  - `yfinance`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `statsmodels`
  - `scikit-learn`
  - `tensorflow`

---

## 🚀 How to Run
1. Clone this repository or download the ZIP.
2. Install dependencies:
pip install -r requirements.txt

markdown
Copy
Edit
3. Run the script:
python Main.py

yaml
Copy
Edit
4. View visualizations and prediction results.

---

## 📂 Project Structure
├── Main.py # Main script for downloading data, training models, and plotting results
├── NVDA_stock_data.csv # Historical NVIDIA stock dataset
├── README.md # Project documentation
├── README_NVDA_Stock_Prediction.pdf # PDF version of README
└── requirements.txt # List of Python dependencies

yaml
Copy
Edit

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
