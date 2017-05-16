import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm_helper, time
import numpy as np
from Blockchain_test import return_data

# Load data from Blockchain_test.py
matrix = return_data()
matrix = matrix[0] # Bitcoin price only

NORMALISATION = True
DENORMALISATION = True

BATCH_SIZE = 256
SEQ_LEN = 10
PREDICTION_LEN = 4
EPOCHS = 1

INPUT_DIM = len(matrix) # Amount of features
# N_FEATURES = len(matrix)
N_FEATURES = 1

OUTPUT_SIZE = 1 # Bitcoin price

train_X, train_y, test_X, test_y, i_list = lstm_helper.load_data(matrix, SEQ_LEN, NORMALISATION)

N_BATCHES = int(np.size(train_X) / SEQ_LEN)


# Build model
model = Sequential()

model.add(LSTM(
    input_dim = 1,
    output_dim = SEQ_LEN,
    return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(
    units = BATCH_SIZE,
    return_sequences = False,
    ))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim = 1,
    ))
model.add(Activation('linear'))

start = time.time()
model.compile(loss = 'mse', optimizer = 'rmsprop')

print('compilation time : ', time.time() - start)

# Train the model
model.fit(
    train_X,
    train_y,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = 0.05
)

# Plot
# Denormalised
predictions = lstm_helper.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN, DENORMALISATION, i_list)
print("test", lstm_helper.denormalise_windows(train_X, i_list)[0])
print("pred", predictions[0])
print("real", lstm_helper.denormalise_windows(test_y.reshape(len(test_y),1), i_list)[0:PREDICTION_LEN])
lstm_helper.plot_results_multiple(predictions, test_y, PREDICTION_LEN, DENORMALISATION, i_list)

# Normalised
predictions = lstm_helper.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN, False, i_list)
print("test_norm", train_X[0])
print("pred_norm", predictions[0])
print("real_norm", test_y[0:PREDICTION_LEN])
lstm_helper.plot_results_multiple(predictions, test_y, PREDICTION_LEN, False, i_list)