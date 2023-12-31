import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

def prepare_data(stock_data, look_back=5):
    data = stock_data.copy()
    data['Close_Shifted'] = data['Close'].shift(-1)
    data['Up_Down'] = (data['Close_Shifted'] > data['Close']).astype(int)

    # Calculate shifted prices using numpy.shift()
    for i in range(1, look_back + 1):
        data[f'Close_Shifted_{i}'] = data['Close'].shift(-i)

    # Fill missing values using pandas.fillna()
    data.fillna(method='ffill', inplace=True)

    return data

# Function to get historical stock price data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# List of stock symbols
stock_symbols = ['AAPL', 'PFE', 'HCA', 'JCI', 'TAP', 'UBER', 'WBD', 'PLD', 'TRGP', 'ALB']  # Add more stock symbols as needed

# Date range
start_date = '2012-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Prepare data for all stocks
look_back_days = 5
all_data = []

# Function to calculate Stochastic Oscillator
def calculate_stochastic(prices, high, low):
    stoch_k = (prices - low.rolling(window=14, min_periods=1).min()) / \
              (high.rolling(window=14, min_periods=1).max() - low.rolling(window=14, min_periods=1).min())
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
def calculate_fibonacci_levels(prices, high, low, window=20):
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

    # Calculate upper and lower bands using numpy.std()
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

for stock_symbol in stock_symbols:
    stock_data = get_stock_data(stock_symbol, start_date, end_date)
    
    # Calculate daily percentage change
    daily_percentage_change = stock_data['Close'].pct_change() * 100

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
    
    # Merge additional features and calculated technical indicators
    additional_features = pd.DataFrame({
        'Volume': stock_data['Volume'],
        'MA_10': stock_data['Close'].rolling(window=10).mean(),
        'MA_50': stock_data['Close'].rolling(window=50).mean(),
        'RSI': calculate_rsi(stock_data['Close']),
        'MACD': calculate_macd(stock_data['Close'])[0],
        'Daily_Pct_Change': daily_percentage_change,
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
    
    data = pd.concat([data, additional_features], axis=1)
    all_data.append(data)

# Merge data for all stocks
merged_data = pd.concat(all_data)
merged_data.to_csv('merged_data.csv', index=False)

# Merge additional features with the existing merged dataset
merged_data_reset = merged_data.reset_index(drop=True)
additional_features_reset = additional_features.reset_index(drop=True)

merged_data_test = pd.concat([merged_data_reset, additional_features_reset], axis=1)
merged_data_test.to_csv('merged_data_with_features.csv', index=False)

# Split the data into features (X) and target (y)
X = merged_data_test.drop(['Up_Down'], axis=1).values
y = merged_data_test['Up_Down'].values

# Handle missing values in X with SimpleImputer
imputer = SimpleImputer(strategy='mean') 
X_imputed = imputer.fit_transform(X)

# Drop rows with missing values in y
X_imputed = X_imputed[~pd.isna(y)]
y = y[~pd.isna(y)]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Create a parameter grid to search for the best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 150],           # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],          # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]             # Minimum number of samples required to be at a leaf node
}
similar_features = ["Close", "Adj Close", "Close_Shifted", "Close_Shifted_1", "Close_Shifted_2", "Close_Shifted_3", "Close_Shifted_4", "Close_Shifted_5",]

# Initialize the RFE with the Random Forest classifier
min_selected_features = 5  # Minimum number of features you want to select
rfe = RFE(estimator=rf_classifier, n_features_to_select=min_selected_features)  # Start with the minimum number of features

# Best Hyperparameters
best_hyperparameters = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

# Define a threshold for improvement in performance
performance_threshold = 0.005  # You can adjust this threshold as needed

best_performance = 0
best_features = None

# Initialize the tqdm progress bar for the RFE loop
rfe_bar = tqdm(total=len(X_train[0]), desc="RFE Progress", ncols=100, leave=False)

# Start the timer
start_time = time.time()

# Loop until no significant performance improvement
while True:
    # Fit RFE and evaluate performance on validation data
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    # Train the Random Forest classifier using the best hyperparameters
    rf_classifier_best = RandomForestClassifier(random_state=42, **best_hyperparameters)
    rf_classifier_best.fit(X_train_rfe, y_train)

    # Make predictions on the validation set using the best model
    predictions_val = rf_classifier_best.predict(X_test_rfe)

    # Calculate the accuracy of the model
    current_performance = accuracy_score(y_test, predictions_val)
    
    # Check if performance improvement is significant
    if current_performance - best_performance > performance_threshold:
        best_performance = current_performance
        best_features = rfe.support_
        
        # Update X_train and X_test with the selected features
        X_train = X_train_rfe
        X_test = X_test_rfe
    else:
        break

    # Update the progress bar
    rfe_bar.update(1)

# Stop the timer
end_time = time.time()

# Close the progress bar
rfe_bar.close()

# Calculate the duration in minutes and seconds
duration_seconds = end_time - start_time
minutes = int(duration_seconds // 60)
seconds = int(duration_seconds % 60)
print(f"Training duration: {minutes} minutes and {seconds} seconds")

print("Selected Features after Dynamic RFE:")
print([feature for feature, selected in zip(data.columns[:-1], best_features) if selected])

# Train the Random Forest classifier using the best hyperparameters
rf_classifier_best = RandomForestClassifier(random_state=42, **best_hyperparameters)
rf_classifier_best.fit(X_train, y_train)

# Calculate the predicted labels using the best model
predictions_best = rf_classifier_best.predict(X_test)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'True_Labels': y_test,
    'Predicted_Labels': predictions_best
})

# Add the 'Close_Shifted' column from the test data to the results DataFrame
results_df['Close_Shifted'] = X_test[:, similar_features.index('Close_Shifted')]

# Calculate the absolute percentage change in Close_Shifted for correct predictions
correct_predictions = results_df[results_df['True_Labels'] == results_df['Predicted_Labels']].copy()
correct_predictions['Absolute_Percent_Change'] = (
    abs(correct_predictions['Close_Shifted'] - correct_predictions['Close_Shifted'].shift()) /
    correct_predictions['Close_Shifted'].shift()
) * 100

# Calculate the average magnitude of price change for correct predictions
average_magnitude_change = correct_predictions['Absolute_Percent_Change'].mean()

print(f"Average Magnitude of Price Change for Correct Predictions: {average_magnitude_change:.2f}%")

# Calculate the accuracy and F1 of the best model
accuracy_best = accuracy_score(y_test, predictions_best)
print(f"Best Model Accuracy: {accuracy_best:.2f}")
f1_best = f1_score(y_test, predictions_best)
print(f"Best Model F1: {f1_best:.2f}")

# Plotting the confusion matrix for the best model
conf_matrix_best = confusion_matrix(y_test, predictions_best)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_best, annot=True, fmt='', cmap='Blues', xticklabels=['Rise', 'Fall'], yticklabels=['Rise', 'Fall'])
plt.title('Confusion Matrix for Best Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

model_filename = 'random_forest_model.pkl'
joblib.dump(rf_classifier_best, model_filename)