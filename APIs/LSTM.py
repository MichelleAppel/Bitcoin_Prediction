import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm_helper, time

import numpy as np
import random

from Blockchain_test import return_data

# Load data from Blockchain_test.py
matrix = return_data()
matrix = matrix[0] # Bitcoin price only

# N_FEATURES = len(matrix)

BATCH_START = 0
BATCH_SIZE = 30
TIME_STEPS = 5

INPUT_SIZE = len(matrix)-1 # Amount of features
OUTPUT_SIZE = 1 # Bitcoin price

TRAIN_TEST_RATIO = 0.7
CELL_SIZE = 10
LR = 10
EPOCHS = 1

train_X, train_y, test_X, test_y = lstm_helper.load_data(matrix, TIME_STEPS, False)

N_BATCHES = int(np.size(train_X)/TIME_STEPS)
N_FEATURES = 1

# Build model
model = Sequential()

model.add(LSTM(
    input_dim = 1,
    output_dim = TIME_STEPS,
    return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences = False,
    ))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim = 1
    ))
model.add(Activation('linear'))

start = time.time()
model.compile(loss = 'mse', optimizer = 'rmsprop')

print('compilation time : ', time.time() - start)

print(train_X.reshape((N_FEATURES, N_BATCHES, TIME_STEPS)))

# Train the model
model.fit(
    train_X,
    train_y,
    batch_size = 512,
    epochs = EPOCHS,
    validation_split = 0.05
)

print(test_X)

# Plot
predictions = lstm_helper.predict_sequences_multiple(model, test_X, 50, TIME_STEPS)
print(predictions)
lstm_helper.plot_results_multiple(predictions, test_y, TIME_STEPS)