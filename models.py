import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load data
df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/^NSEI?period1=1538764800&period2=1646649600&interval=1d&events=history&includeAdjustedClose=true')
df = df.set_index('Date')

# Convert data to stationary by differencing
df_diff = df.diff().dropna()

# Split data into training and testing sets
train_data = df_diff[:int(len(df_diff)*0.8)]
test_data = df_diff[int(len(df_diff)*0.8):]

# Plot the training and testing data
plt.figure(figsize=(10, 5))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Testing Data')
plt.legend()
plt.title('NSEI Daily Returns')
plt.show()


# Fit the ARMA model on training data
# model = ARMA(train_data, order=(2, 1))
model = sm.tsa.arima.ARIMA(train_data, order=(2,1))
# result = model.fit()
model_fit = model.fit(disp=False)

# Make predictions on testing data
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
predictions = predictions.cumsum() + df['Adj Close'].iloc[-1]

# Calculate the mean squared error
mse = mean_squared_error(df['Adj Close'].iloc[int(len(df)*0.8):], predictions)
print('MSE:', mse)

# Plot the predictions and the actual data
plt.figure(figsize=(10, 5))
plt.plot(df['Adj Close'].iloc[int(len(df)*0.8):], label='Actual Data')
plt.plot(predictions, label='ARMA Predictions')
plt.legend()
plt.title('NSEI Stock Price Predictions')
plt.show()

# Calculate the profit/loss based on the ARMA predictions
buy_price = df['Adj Close'].iloc[int(len(df)*0.8)]
sell_price = predictions[-1]
profit_loss = sell_price - buy_price
print('Profit/Loss:', profit_loss)

def predict_next_day_price(data):
    # Convert data to stationary by differencing
    diff_data = data.diff().dropna()

    # Fit the ARMA model on the data
    model = ARMA(diff_data, order=(2, 1))
    model_fit = model.fit(disp=False)

    # Make the prediction for the next day
    last_price = data.iloc[-1]
    next_day_diff = model_fit.forecast()[0][0]
    next_day_price = last_price + next_day_diff

    return next_day_price

next_day_price = predict_next_day_price(df['Close'])

print("The predicted price for the next day is:", next_day_price)
