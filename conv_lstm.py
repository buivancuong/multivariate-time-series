from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
import matplotlib.pyplot as plt
import math


class Conv_LSTM:
    '''
    Convert Sequence Time Series data to 2D Image input

    Output of each CNN will be sent directly into correspond LSTM

    Rarely exact
    '''

    def __init__(self, memory_step, num_features):
        self.memory_step = memory_step
        self.sub_sequences = int(math.sqrt(int(memory_step)))
        self.num_features = num_features

    def define_model(self):
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(
            self.sub_sequences, 1, self.sub_sequences, self.num_features)))
        self.model.add(Flatten())
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())

    def training_model(self, X_train, y_train, epochs=500):
        self.model.fit(X_train, y_train, epochs=epochs)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def show_plot(self, y_hat, y_test):
        plt.plot(y_hat, color='red')
        plt.plot(y_test, color='green')
        plt.show()
