import joblib
import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from datetime import datetime

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
stock_symbol = 'FNMAJ'  # Replace with the desired stock symbol
start_date = '2012-01-01'  # Replace with the desired start date
end_date = datetime.today().strftime('%Y-%m-%d')

# Get historical stock price data
stock_data = get_stock_data(stock_symbol, start_date, end_date)

# Prepare the data for training
look_back_days = 5
data = prepare_data(stock_data, look_back=look_back_days)

# Save the prepared data to a CSV file
prepared_data_filename = 'prepared_data.csv'
data.to_csv(prepared_data_filename, index=False)

# Function to calculate Stochastic Oscillator
def calculate_stochastic(prices, high, low, window=14):
    stoch_k = (prices - low.rolling(window=window, min_periods=1).min()) / \
              (high.rolling(window=window, min_periods=1).max() - low.rolling(window=window, min_periods=1).min())
    stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()  # You can adjust the window for stoch_d if needed
    return stoch_k, stoch_d

# Function to calculate Simple Moving Averages (SMA)
def calculate_sma(prices, window=10, name='SMA'):
    sma = prices.rolling(window=window, min_periods=1).mean()
    return sma.rename(f'{name}_{window}')

# Function to calculate Exponential Moving Averages (EMA)
def calculate_ema(prices, window=10, name='EMA'):
    ema = prices.ewm(span=window, min_periods=1, adjust=False).mean()
    return ema.rename(f'{name}_{window}')

# Function to calculate Fibonacci Retracement Levels
def calculate_fibonacci_levels(high, low, window=20):
    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()

    range_percent = (highest_high - lowest_low) / highest_high * 100

    fib_0 = highest_high
    fib_23_6 = highest_high - (range_percent * 0.236)
    fib_38_2 = highest_high - (range_percent * 0.382)
    fib_50 = highest_high - (range_percent * 0.5)
    fib_61_8 = highest_high - (range_percent * 0.618)
    fib_100 = lowest_low

    return fib_0, fib_23_6, fib_38_2, fib_50, fib_61_8, fib_100

# Function to calculate RSI Divergence
def calculate_rsi_divergence(rsi):
    rsi_shifted = rsi.shift()
    rsi_divergence = rsi - rsi_shifted
    return rsi_divergence

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

# Calculate Stochastic Oscillator
stoch_k, stoch_d = calculate_stochastic(stock_data['Close'], stock_data['High'], stock_data['Low'])
fib_0, fib_23_6, fib_38_2, fib_50, fib_61_8, fib_100 = calculate_fibonacci_levels(
    stock_data['Close'], stock_data['High'], stock_data['Low']
    )

# Calculate RSI Divergence
rsi_values = calculate_rsi(stock_data['Close'])
rsi_divergence = calculate_rsi_divergence(rsi_values)
    
# Calculate MACD
macd, signal_line = calculate_macd(stock_data['Close'])
    
# Calculate Bollinger Bands
upper_band, lower_band = calculate_bollinger_bands(stock_data['Close'])
    
# Calculate ROC
roc = calculate_roc(stock_data['Close'])
    
# Calculate PVT
pvt = calculate_pvt(stock_data['Close'], stock_data['Volume'])
    
# Calculate ATR
atr = calculate_atr(stock_data['High'], stock_data['Low'], stock_data['Close'])
    
# Calculate Historical Volatility
historical_volatility = calculate_historical_volatility(stock_data['Close'])
    
data = prepare_data(stock_data, look_back=look_back_days)

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
        'Stoch_K':  stoch_k,
        'Stoch_D':  stoch_d,
        'SMA_20': calculate_sma(stock_data['Close'], window=20),
        'EMA_20': calculate_ema(stock_data['Close'], window=20),
        'SMA_50': calculate_sma(stock_data['Close'], window=50),
        'EMA_50':  calculate_ema(stock_data['Close'], window=50),
        'Fib_0': fib_0,
        'Fib_23_6': fib_23_6,
        'Fib_38_2': fib_38_2,
        'Fib_50': fib_50,
        'Fib_61_8': fib_61_8,
        'Fib_100': fib_100,
        'RSI_Divergence': rsi_divergence,
        'SMA_100': calculate_sma(stock_data['Close'], window=100, name='SMA'),
        'EMA_100': calculate_ema(stock_data['Close'], window=100, name='EMA'),
})

# Merge additional features with the existing dataset
data_imputed = pd.concat([data, additional_features], axis=1)

# Step 2: Prepare the Features
desired_features = ['Low', 'Close', 'Up_Down', 'High']  # List your desired features here
current_features = data_imputed[desired_features].iloc[-1, :].values.reshape(1, -1)  # Use the most recent row

# Print input data
print("Input Features:")
print(current_features)

# Step 3: Make Predictions
predicted_probabilities = loaded_model.predict_proba(current_features)
predicted_label = loaded_model.predict(current_features)

# Print predicted probabilities
print("Predicted Probabilities:", predicted_probabilities)

# Print predicted label
print("Predicted Label:", predicted_label)

# Step 4: Print Prediction with Stock Symbol and Date
if predicted_label == 1:
    prediction_text = "rise"
else:
    prediction_text = "fall"

print(f"Prediction for {stock_symbol} on {end_date}: The stock price is predicted to {prediction_text}.")