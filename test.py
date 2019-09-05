from prepare_data import *
from vanilla_lstm import *
from stacked_lstm import *
import math

from prepare_data import *

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
print(in_seq1)

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

print(in_seq1)

dataset = hstack((in_seq1, in_seq2, out_seq))

print(dataset)

n_step = 3

X, y = parallel_split_sequence(dataset, n_step)

print(X)
print(y)

n_features = X.shape[2]

vlstm = Stacked_LSTM(memory_step=n_step, num_features=n_features)
vlstm.define_model()
vlstm.training_model(X_train=X, y_train=y)

x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_step, n_features))
yhat = vlstm.prediction(X_test=x_input)
print(yhat)