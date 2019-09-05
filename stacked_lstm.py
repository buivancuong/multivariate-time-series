from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

from prepare_data import *


class Stacked_LSTM:

    def __init__(self, memory_step, num_features):
        self.memory_step = memory_step
        self.num_features = num_features

    def define_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=64, activation='relu', return_sequences=True,
                            input_shape=(self.memory_step, self.num_features)))
        self.model.add(LSTM(units=64, activation='relu'))
        self.model.add(Dense(units=self.num_features))
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())

    def training_model(self, X_train, y_train, epochs=200):
        self.model.fit(X_train, y_train, epochs=200)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def show_plot(self, y_hat, y_test):
        plt.plot(y_hat, color='red')
        plt.plot(y_test, color='green')
        plt.show()
