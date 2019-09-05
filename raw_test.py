# multivariate data preparation
from numpy import array
from numpy import hstack
import math
from prepare_data import *
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
print(in_seq1)
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
print(in_seq2)
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
print(out_seq)
print('*****************************')
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
print(in_seq1)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
print(in_seq2)
out_seq = out_seq.reshape((len(out_seq), 1))
print(out_seq)
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)

n_steps = 3
# convert into input/output
X, y = parallel_split_sequence(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])