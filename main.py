import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from alpha_vantage.timeseries import TimeSeries
from matplotlib import dates as mdates

api_key = 'G8TI1VY50NJ6W74T'
ts = TimeSeries(key=api_key, output_format='pandas')

#Load Data
company = 'TSLA'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data, meta_data = ts.get_daily(symbol=company, outputsize='full')

#Prep Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['4. close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build The Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

'''Test The Model Accuracy on Existing Data'''

#Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data, test_meta_data = ts.get_daily(symbol=company, outputsize='full')
actual_prices = test_data['4. close'].values

total_dataset = pd.concat((data['4. close'], test_data['4. close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Make Predictions on Test Data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

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

#Predicting Next Day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"prediction: {prediction}")