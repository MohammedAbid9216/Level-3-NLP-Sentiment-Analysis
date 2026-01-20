# LEVEL 3 – TASK 1: TIME SERIES ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# 1. Load dataset
df = pd.read_csv("sales_timeseries.csv")
print("Dataset loaded successfully")
print(df.head())

# 2. Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 3. Plot original time series
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], marker='o')
plt.title("Sales Time Series")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()

# 4. Decompose time series (trend, seasonality, residual)
decomposition = seasonal_decompose(df['Sales'], model='additive', period=12)
decomposition.plot()
plt.show()

# 5. Moving Average
df['Moving_Avg'] = df['Sales'].rolling(window=3).mean()

plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Original Sales')
plt.plot(df['Moving_Avg'], label='Moving Average', color='red')
plt.legend()
plt.title("Sales vs Moving Average")
plt.show()

# 6. ARIMA Model
model = ARIMA(df['Sales'], order=(1, 1, 1))
model_fit = model.fit()

# 7. Forecast future values
forecast = model_fit.forecast(steps=6)
print("\nForecast for next 6 months:")
print(forecast)

# 8. Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Original Data')
plt.plot(forecast, label='Forecast', color='green')
plt.legend()
plt.title("Sales Forecast using ARIMA")
plt.show()
