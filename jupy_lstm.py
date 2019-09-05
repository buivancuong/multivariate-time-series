# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# 

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock_data = pd.read_csv('./data/google/google18m.csv')
stock_data.head()

#%% [markdown]
# 

#%%
import math

stock_data['average'] = (stock_data['high'] + stock_data['low']) / 2
stock_data.head()

#%% [markdown]
# 

#%%
input_feature = stock_data.iloc[:,[2,6]].values

# plt.plot(input_feature[:,0])
# plt.title("Volume of stocks sold")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Volume of stocks traded")
# plt.show()

# #%% [markdown]
# # 

# #%%
# plt.plot(input_feature[:,1], color='blue')
# plt.title("Google Stock Prices")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Stock Opening Price")
# plt.show()

#%% [markdown]
# 

#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
input_data = sc.fit_transform(input_feature)

#%% [markdown]
# 

#%%
lookback = 30
print(stock_data.shape, 'stock')
X = []
y_volume = []
y_average = []

for i in range(len(stock_data) - lookback - 1):
    t = []
    for j in range(lookback):
        t.append(input_data[i + j, :])
    X.append(t)
    y_volume.append(input_data[i + lookback, 0])
    y_average.append(input_data[i + lookback, 1])

#%% [markdown]
# 

#%%
X = np.array(X)
y_volume = np.array(y_volume)
y_average = np.array(y_average)
test_size = int(1/7 * len(X))

X_train = X[test_size + 1:]
X_test = X[:test_size]
y_volume_train = y_volume[test_size + 1:]
y_volume_test = y_volume[:test_size]
y_average_train = y_average[test_size + 1:]
y_average_test = y_average[:test_size]

print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(y_volume_train.shape)

#%% [markdown]
# 

#%%
from keras import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=30, return_sequences=True, input_shape=(X.shape[1], 2)))
model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=30))
model.add(Dense(units=1))
model.summary()

#%% [markdown]
# 

#%%
model.compile(optimizer='adam', loss='mean_squared_error')

#%% [markdown]
# 

#%%
model.fit(X_train, y_average_train, epochs=200, batch_size=64)

#%% [markdown]
# 

#%%
predicted_value = model.predict(X_test)
# print(predicted_value)

#%% [markdown]
# 

#%%
plt.plot(predicted_value, color= 'red')
# plt.plot(input_data[lookback:test_size+(lookback),0], color='green')
plt.plot(input_data[lookback:,1], color='green')
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()


#%%



#%%



