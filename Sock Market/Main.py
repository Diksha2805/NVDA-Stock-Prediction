import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# ------------------ Download NVDA STOCK DATA ------------------

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "NVDA_stock_data.csv")

# Download stock data if not already downloaded
if not os.path.exists(file_path):
    print("Downloading NVDA stock data...")
    df = yf.download('NVDA', start='2013-06-01', end='2025-06-01', interval='1d')
    df.to_csv(file_path)
else:
    print("Loading existing NVDA stock data...")
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# Clean data
print("Checking for missing values:")
print(df.isnull().sum())
df.dropna(inplace=True)
df['Daily_Return'] = df['Close'].pct_change()

# ------------------ Visualization ------------------
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], color="blue")
plt.title("NVDA Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ------------------ ARIMA Model ------------------
closing_price = df['Close']
train_size = int(len(closing_price) * 0.8)
train, test = closing_price[:train_size], closing_price[train_size:]

print("Training ARIMA model...")
model_arima = ARIMA(train, order=(5, 1, 0))
model_fit_arima = model_arima.fit()
predictions_arima = model_fit_arima.forecast(steps=len(test))

mse_arima = mean_squared_error(test, predictions_arima)
print(f'ARIMA Mean Squared Error: {mse_arima:.4f}')

# Plot ARIMA predictions
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, color='blue', label='Actual')
plt.plot(test.index, predictions_arima, color='red', label='ARIMA Predicted')
plt.title('Stock Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# ------------------ LSTM Model ------------------
print("Preparing data for LSTM model...")
closing_price_values = closing_price.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
closing_price_scaled = scaler.fit_transform(closing_price_values)

train_scaled, test_scaled = closing_price_scaled[:train_size], closing_price_scaled[train_size:]

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_scaled)
X_test, y_test = create_dataset(test_scaled)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train LSTM model
print("Training LSTM model...")
model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predict and inverse transform
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual')
plt.plot(predictions_lstm, color='red', label='LSTM Predicted')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

print("Data Info:")
print(df.info())
print(df.describe())
