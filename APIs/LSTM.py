from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# lstm_helper.py
import lstm_helper
import numpy as np
from Blockchain_test import return_data

# Load data from Blockchain_test.py
matrix = return_data()
matrix = matrix[0] # Bitcoin price only

# Whether the data should be normalised
NORMALISATION = True
DENORMALISATION = True

SEQ_LEN = 20 # The length of the sequence
PREDICTION_LEN = 5 # The amount of predicted values

INPUT_DIM = 1 # Amount of features
UNITS = 32 # The amount of units in the LSTM
OUTPUT_DIM = 1 # Bitcoin price

BATCH_SIZE = 64 # The batch size
EPOCHS = 1 # The amount of epochs

# Get the training and test test
train_X, train_y, test_X, test_y, i_list = lstm_helper.load_data(matrix, SEQ_LEN, NORMALISATION)

# Build model
model = Sequential()

model.add(LSTM(
    input_dim = INPUT_DIM,
    units = UNITS
    ))

model.add(Dense(
    output_dim = OUTPUT_DIM,
    ))

model.compile(loss = 'mse', optimizer = 'rmsprop')

# Train the model
model.fit(
    train_X,
    train_y,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = 0.05
)

# Compute mean error
def error(predicted, real):
    errors = []
    predicted = np.array(predicted)
    predicted = predicted.reshape(np.size(predicted), 1)
    for p, r in zip(predicted, real):
        errors.append(np.abs(p-r))
    mean_error = np.array(errors).mean()
    return mean_error

# Plot
# Denormalised
predictions = lstm_helper.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN, DENORMALISATION, i_list)
denorm_test_x = lstm_helper.denormalise_windows(test_X, i_list)
denorm_test_y = lstm_helper.denormalise_windows(test_y.reshape(len(test_y),1), i_list)
denorm_test_y = np.array(denorm_test_y).reshape(np.array(test_y).shape)

print("Denormalised error: ", error(predictions, denorm_test_y))

lstm_helper.plot_results_multiple(predictions, test_y, PREDICTION_LEN, DENORMALISATION, i_list)

# Normalised
predictions = lstm_helper.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN, False, i_list)
# print("test_norm", test_X[0])
# print("pred_norm", predictions[0])
# print("real_norm", test_y[0:PREDICTION_LEN])
lstm_helper.plot_results_multiple(predictions, test_y, PREDICTION_LEN, False, i_list)

print("Normalised error: ", error(predictions, test_y))