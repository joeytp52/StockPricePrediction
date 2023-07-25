import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf  # Add this import for fetching data from Yahoo Finance
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from alpha_vantage.timeseries import TimeSeries
from matplotlib import dates as mdates
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import ReduceLROnPlateau

# Fetch Stock Price Data from Yahoo Finance using yfinance
company = 'META'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = yf.download(company, start=start, end=end)

# Prep Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Calculate RSI for the closing prices
def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    
    return rsi

# Calculate RSI for the closing prices
closing_prices = data['Close']
rsi = calculate_rsi(closing_prices) / 100

# Scale the RSI data along with the closing price and volume data
closing_prices_scaled = scaler.transform(closing_prices.values.reshape(-1, 1))
volume_scaled = scaler.transform(data['Volume'].values.reshape(-1, 1))
rsi_scaled = scaler.transform(rsi.values.reshape(-1, 1))

# Merge the scaled data into a single dataset
scaled_data_with_rsi = np.hstack((closing_prices_scaled, volume_scaled, rsi_scaled))

# Handle NaN values in scaled_data_with_rsi
scaled_data_with_rsi = np.nan_to_num(scaled_data_with_rsi, nan=np.nanmean(scaled_data_with_rsi))

# Modify the prediction_days to include the number of features
prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data_with_rsi)):
    x_train.append(scaled_data_with_rsi[x - prediction_days:x, :])
    y_train.append(scaled_data_with_rsi[x, 0])  # Use the closing price as the target

x_train, y_train = np.array(x_train), np.array(y_train)

# Handle NaN values in x_train
x_train = np.nan_to_num(x_train, nan=np.nanmean(x_train))

# Modify the model architecture
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(prediction_days, 3)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Add EarlyStopping callback
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Define the learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

# Compile the model with the optimizer and loss function
model.compile(optimizer='adam', loss='mean_squared_error')

'''Test The Model Accuracy on Existing Data'''

# Load Test Data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

# Fetch test data from Yahoo Finance
test_data = yf.download(company, start=test_start, end=test_end)

# Extract actual closing prices from test_data
actual_prices = test_data['Close'].values

# Calculate RSI for the test closing prices
test_closing_prices = test_data['Close']
test_rsi = calculate_rsi(test_closing_prices)

# Scale the test data along with the existing volume data
test_volume_scaled = scaler.transform(test_data['Volume'].values.reshape(-1, 1))
test_rsi_scaled = scaler.transform(test_rsi.values.reshape(-1, 1))

# Merge the scaled test data (excluding the closing price) into the existing dataset
test_data_with_rsi = np.hstack((test_closing_prices.values.reshape(-1, 1), test_volume_scaled, test_rsi_scaled))

# Prepare x_test with all three features
x_test = []
for x in range(prediction_days, len(test_data_with_rsi)):
    x_test.append(test_data_with_rsi[x - prediction_days:x, :])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))  # Ensure the correct shape

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Implement walk-forward validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

fold = 1
for train_index, test_index in tscv.split(x_train):
    print(f"Training Fold {fold}")
    
    # Get the indices for training and validation data
    x_train_fold, x_val_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

    # Train the model with the learning rate scheduler and early stopping callbacks
    model.fit(x_train_fold, y_train_fold, epochs=100, batch_size=32, validation_data=(x_val_fold, y_val_fold), callbacks=[reduce_lr, early_stopping])

    # Make predictions for the validation set
    predictions_fold = model.predict(x_val_fold)

    # Store the predictions for this fold
    predictions_fold = np.array(predictions_fold)
    predictions_fold = scaler.inverse_transform(predictions_fold.reshape(-1, 1)).flatten()

    # Evaluate the performance on this fold
    mse = np.mean((predictions_fold - scaler.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()) ** 2)
    print(f"Mean Squared Error for Fold {fold}: {mse}")

    print(f"Completed Fold {fold} out of {n_splits}")
    fold += 1


# Convert dates to numerical format for the x-axis
dts_actual = pd.to_datetime(test_data.index)
dts_predicted = pd.to_datetime(test_data.index[-len(predicted_prices):])
dts_actual_num = mdates.date2num(dts_actual)
dts_predicted_num = mdates.date2num(dts_predicted)

# Plot Test Predictions
plt.plot(dts_actual_num, actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(dts_predicted_num, predicted_prices, color="red", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()

years = mdates.YearLocator()
yearsFmt = mdates.DateFormatter('\n%Y')

plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yearsFmt)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.5, which='both', color='lightgrey')
plt.tight_layout()
plt.show()

# Predicting into the future
days_to_predict = 14
real_data = x_test[-1]  # Use the entire last sequence in x_test as the initial real_data
predictions = []

# Create a scaler for the closing price
scaler_price = MinMaxScaler(feature_range=(0, 1))

for _ in range(days_to_predict):
    # Extract the closing price for scaling
    closing_price = real_data[:, 0]

    # Scale the closing price
    closing_price_scaled = scaler_price.fit_transform(closing_price.reshape(-1, 1))

    real_data_reshaped = np.reshape(real_data, (1, prediction_days, 3))

    # Predict the next value
    prediction = model.predict(real_data_reshaped)
    predictions.append(scaler_price.inverse_transform(prediction)[0, 0])

    # Update real_data with the predicted value for the next iteration
    real_data = np.concatenate([real_data[1:], np.array([[prediction[0, 0], 0, 0]])], axis=0)

print("Predictions for the next 14 days:")
for i, pred in enumerate(predictions, start=1):
    print(f"Day {i}: {pred:.2f}")