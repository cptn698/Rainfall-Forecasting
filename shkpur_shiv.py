

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

from math import sqrt


from statsmodels.tsa.seasonal import seasonal_decompose


from tensorflow.keras.layers import LSTM, Dense

# %matplotlib inline

df = pd.read_csv(r"D:\ayyan_11\CODE\ARIMA\shkpur\SHEKHPURA.csv")

df = df[['Date', 'Value']]

df

plt.figure(figsize=(18, 6))
sns.lineplot(x='Date', y='Value', data=df)
plt.xlabel('Date')

plt.ylabel('Precipitation')
plt.title('Precipitation Forecast')
plt.show()

df.drop(df.tail(1).index, inplace=True)

train_dates=pd.to_datetime(df['Date'])



df['Date'] = pd.to_datetime(df['Date'])

df

# Perform seasonal decomposition
result = seasonal_decompose(df['Value'], model='additive', period=180)

# Plot the components
plt.figure(figsize=(10, 6))
plt.subplot(411)
plt.plot(df['Value'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(result.seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(result.resid, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

cols=list(df)[1:2]
print(cols)

training_df=df[cols].astype(float)
plot_df=training_df
plot_df.plot.line()



from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar=scalar.fit(training_df)
training_df_scaled=scalar.transform(training_df)

"""#trainX for storing training data
#trainY for storing target values(rainfall)
"""

trainX=[]
trainY=[]

n_past=70       # 100 last month learned
n_fut=100


#Adding the values to trainX and trainY lists
for i in range(n_past,len(training_df_scaled)-n_fut+1):
    trainX.append(training_df_scaled[i-n_past:i,0:training_df_scaled.shape[1]])
    trainY.append(training_df_scaled[i+n_fut-1:i+n_fut,0])

#Converting into numpy arrays
trainX,trainY=np.array(trainX),np.array(trainY)

print('trainX shape == {}'.format(trainX.shape))#Knowing the shape
#(364,20,4) means there are 364windows(groups) of 20*4


print('trainY shape == {}'.format(trainY.shape))
#There are 364 values from previous values

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.initializers import HeNormal

model=Sequential()
model.add(LSTM(32,activation='relu',input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(32,activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(32,activation='relu',return_sequences=False))
model.add(Dropout(0.1))

model.add(Dense(trainY.shape[1]))



optimizer  = Adam( )  # Adjust the learning rate  , learning_rate=0.001 ,clipvalue=1.0
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.summary()



history=model.fit(trainX,trainY,epochs=15,batch_size=4,validation_split=0.2,verbose=1)

"""## Performance of model

# Training RMSE: 0.5935
# Testing RMSE: 0.5600
"""

# Access the training and validation loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Calculate RMSE
train_rmse = sqrt(train_loss[-1])
val_rmse = sqrt(val_loss[-1])

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testimg RMSE: {val_rmse:.4f}")

plt.plot(history.history['loss'],label='Training loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.legend()

#Predicting...
#Libraries that will help us extract only business days in the US.
#Otherwise our dates would be wrong when we look back (or forward).
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

df

n_past = 10              #overlap months
n_days_for_prediction=120 #3.1 month


predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='M').tolist()


prediction = model.predict(trainX[-n_days_for_prediction:])


prediction_copies = np.repeat(prediction, training_df.shape[1], axis=-1)



y_pred_future = scalar.inverse_transform(prediction_copies)[:,0]

forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Value':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

df_forecast['Value'] = df_forecast['Value'].shift(3)



plt.figure(figsize=(17, 8))

# Plot the original data
sns.lineplot(x='Date', y='Value', data=df)

# Plot the forecast data

df_forecast['Value'] = df_forecast['Value'].clip(lower=0)

sns.lineplot(x='Date', y='Value', data=df_forecast)

# Show the plot
plt.show()

# n_past=70       # 100 last month learned
# n_fut=100

df_forecast.head(12)

# pred = model.predict(trainX)

import seaborn as sns

original = df[['Date', 'Value']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '1982-1-1']

sns.set(rc = {'figure.figsize':(15,8)})
# sns.lineplot(original['Date'], original['Precipitation'])
plt.figure(figsize=(18, 6))
sns.lineplot(x='Date', y='Value', data=original)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Original Precipitation Data')
plt.show()


# Precipitation

# sns.lineplot(df_forecast['Date'], df_forecast['Precipitation'])

plt.figure(figsize=(18, 6))
sns.lineplot(x='Date', y='Value', data=df_forecast)
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.title('Precipitation Forecast')
plt.show()







