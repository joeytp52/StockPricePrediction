import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def prepare_data(stock_data, look_back=5):
    data = stock_data.copy()
    data['Close_Shifted'] = data['Close'].shift(-1)
    data['Up_Down'] = (data['Close_Shifted'] > data['Close']).astype(int)
    for i in range(1, look_back + 1):
        data[f'Close_Shifted_{i}'] = data['Close'].shift(-i)
    data.dropna(inplace=True)
    return data

# Function to get historical stock price data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Set the stock symbol and date range
stock_symbol = 'AAPL'  # Replace with the desired stock symbol
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
data = pd.concat([data, additional_features], axis=1)

# Split the data into features (X) and target (y)
X = data.drop(['Up_Down'], axis=1).values
y = data['Up_Down'].values

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

# Initialize the RFE with the Random Forest classifier and desired number of features to select
num_features_to_select = 4  # Change this to the desired number of features
rfe = RFE(estimator=rf_classifier, n_features_to_select=num_features_to_select)

# Fit the RFE to the training data to identify the most important features
rfe.fit(X_train, y_train)

# Get the best features selected by RFE
selected_features = [feature for feature, rank in zip(data.columns[:-1], rfe.support_) if rank]
print(selected_features)

# Get the feature data after RFE selection
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Best Hyperparameters
best_hyperparameters = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

# Train the Random Forest classifier using the best hyperparameters
rf_classifier_best = RandomForestClassifier(random_state=42, **best_hyperparameters)
rf_classifier_best.fit(X_train[:, rfe.support_], y_train)

# Make predictions on the test set using the best model
predictions_best = rf_classifier_best.predict(X_test[:, rfe.support_])

# Calculate the accuracy and F1 of the best model
accuracy_best = accuracy_score(y_test, predictions_best)
print(f"Best Model Accuracy: {accuracy_best:.2f}")
f1_best = f1_score(y_test, predictions_best)
print(f"Best Model F1: {f1_best:.2f}")

# Plotting the confusion matrix for the best model
conf_matrix_best = confusion_matrix(y_test, predictions_best)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_best, annot=True, fmt='', cmap='Blues', xticklabels=['True', 'False'], yticklabels=['True', 'False'])
plt.title('Confusion Matrix for Best Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

model_filename = 'random_forest_model.pkl'
joblib.dump(rf_classifier_best, model_filename)