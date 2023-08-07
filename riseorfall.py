import joblib
import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from datetime import datetime
import joblib

# Load the saved model
model_filename = 'random_forest_model.pkl'
loaded_model = joblib.load(model_filename)

def prepare_data(stock_data, look_back=5):
    data = stock_data.copy()
    data['Close_Shifted'] = data['Close'].shift(-1)
    data['Up_Down'] = (data['Close_Shifted'] > data['Close']).astype(int)
    for i in range(1, look_back + 1):
        data[f'Close_Shifted_{i}'] = data['Close'].shift(-i)
    
    # Perform data imputation
    imputer = SimpleImputer(strategy='mean')  # Choose an imputation strategy 
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    
    return data_imputed

# Function to get historical stock price data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Set the stock symbol and date range
stock_symbol = 'META'  # Replace with the desired stock symbol
start_date = '2012-01-01'  # Replace with the desired start date
end_date = datetime.today().strftime('%Y-%m-%d')

# Get historical stock price data
stock_data = get_stock_data(stock_symbol, start_date, end_date)

# Prepare the data for training
look_back_days = 5
data = prepare_data(stock_data, look_back=look_back_days)

def calculate_rsi(prices, window=14):
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()

    return macd, signal_line

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(prices, window=20):
    rolling_mean = prices.rolling(window=window, min_periods=1).mean()
    rolling_std = prices.rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    return upper_band, lower_band

# Function to calculate Price Rate of Change (ROC)
def calculate_roc(prices, window=10):
    roc = prices.pct_change(periods=window)
    return roc

# Function to calculate Price Volume Trend (PVT)
def calculate_pvt(prices, volume):
    pvt = ((prices.diff() / prices.shift(1)) * volume).cumsum()
    return pvt

# Function to calculate Average True Range (ATR)
def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=1).mean()
    return atr

# Function to calculate Historical Volatility
def calculate_historical_volatility(prices, window=21):
    returns = prices.pct_change().dropna()
    historical_volatility = returns.rolling(window=window).std() * (252 ** 0.5)  # Annualized volatility
    return historical_volatility

# Collect additional features
additional_features = pd.DataFrame({
    'Volume': stock_data['Volume'],
    'MA_10': stock_data['Close'].rolling(window=10).mean(),
    'MA_50': stock_data['Close'].rolling(window=50).mean(),
    'RSI': calculate_rsi(stock_data['Close']),
    'MACD': calculate_macd(stock_data['Close'])[0],
    'Upper_Band': calculate_bollinger_bands(stock_data['Close'])[0],
    'Lower_Band': calculate_bollinger_bands(stock_data['Close'])[1],
    'ROC': calculate_roc(stock_data['Close']),
    'PVT': calculate_pvt(stock_data['Close'], stock_data['Volume']),
    'ATR': calculate_atr(stock_data['High'], stock_data['Low'], stock_data['Close']),
    'Historical_Volatility': calculate_historical_volatility(stock_data['Close']),
})

# Merge additional features with the existing dataset
data_imputed = pd.concat([data, additional_features], axis=1)

# Step 2: Prepare the Features
desired_columns = ['Close', 'Adj Close', 'Close_Shifted', 'Up_Down']
current_features = data_imputed[desired_columns].iloc[-1, :].values.reshape(1, -1)  # Use the most recent row

# Print input data
print("Input Features:")
print(current_features)

# Step 3: Make Predictions
predicted_label = loaded_model.predict(current_features)

# Print loaded model
print("Loaded Model:")
print(loaded_model)

# Print predicted label
print("Predicted Label:", predicted_label)

# Step 4: Print Prediction with Stock Symbol and Date
if predicted_label == 1:
    prediction_text = "rise"
else:
    prediction_text = "fall"

print(f"Prediction for {stock_symbol} on {end_date}: The stock price is predicted to {prediction_text}.")