import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock_data = pd.read_csv('./data/google/google24m.csv')
stock_data.head()

import math

stock_data['average'] = (stock_data['high'] + stock_data['low']) / 2
stock_data.head()

input_feature= stock_data.iloc[:,[2,6]].values
input_data = input_feature

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
input_data = sc.fit_transform(input_feature)

lookback= 30
print(stock_data.shape, 'stock')


X=[]
y=[]
for i in range(len(stock_data)-lookback-1):
    t=[]
    for j in range(0,lookback):
        
        t.append(input_data[[(i+j)], :])
    X.append(t)
    y.append(input_data[i+ lookback,0])

test_size=int(1/7 * len(X))

X, y= np.array(X), np.array(y)
X_test = X[:test_size]
X_train = X[test_size+1:]
X = X.reshape(X.shape[0],lookback, 2)
X_test = X_test.reshape(X_test.shape[0],lookback, 2)
X_train = X_train.reshape(X_train.shape[0],lookback, 2)
y_train = y[test_size+1:]
print(y_train.shape)
print(X.shape)
print(X_train.shape)

from keras import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=64, return_sequences= True, input_shape=(X.shape[1],2)))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=200, batch_size=64)

predicted_value= model.predict(X_test)

plt.plot(predicted_value, color= 'red')
plt.plot(input_data[lookback:test_size+(lookback),0], color='green')
# plt.plot(input_data[:,1], color='green')
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()