from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import optimizers, layers
from keras.models import Sequential

import matplotlib.pyplot as plt

# lstm.py
import lstm
import numpy as np
from Blockchain_test import return_data

# Load data from Blockchain_test.py
matrix = return_data()
matrix = matrix[0][300:] # Bitcoin price only, before 300 only zeros


# ----------------------------------------------------- Parameters --------------------------------------------------- #

# Parameters
NORMALISATION = False # Whether the data should be normalised

TRAIN_TEST_RATIO = 0.9 # The train / test ratio

SEQ_LEN = 15 # The length of the sequence
PREDICTION_LEN = 1 # The amount of predicted values
PREDICTION_DELAY = 0 # Amount of time between sequence and prediction, 0 is next timestep after the sequence

INPUT_DIM = 1 # Amount of features (?)
UNITS = 16 # The amount of units in the LSTM
OUTPUT_DIM = 1 # Bitcoin price

LEARNING_RATE = 0.01 # Learning rate

BATCH_SIZE = 16 # The batch size
EPOCHS = 150 # The amount of epochs

VALIDATION_SPLIT = 0.1


# --------------------------------------------------- Retrieve data -------------------------------------------------- #

# Get the training and test test
train_X, train_y, test_X, test_y = lstm.load_data(
    matrix, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NORMALISATION, TRAIN_TEST_RATIO
    )

# Plot the train and test data
time_step_of_seq = 0
lstm.plot_train_test_set(train_X, train_y, test_X, test_y, time_step_of_seq)


# ---------------------------------------------------- Build model --------------------------------------------------- #

model = Sequential()

model.add(LSTM(
    input_dim = INPUT_DIM,
    units = UNITS, # the size of the LSTM’s hidden state
    ))

model.add(Dense(
    output_dim = OUTPUT_DIM,
    ))

rms = optimizers.RMSprop(lr = LEARNING_RATE) # Optimizer
model.compile(loss = 'mse', optimizer = rms) # Compile

# Train the model
model_fit = model.fit(
    train_X,
    train_y,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = VALIDATION_SPLIT,
)

lstm.plot_loss(model_fit) # Plot the training and validation loss


# -------------------------------------------------- Predict prices -------------------------------------------------- #

# Predictions Train Set
predictions = lstm.predict_sequences_multiple(model, train_X, SEQ_LEN, PREDICTION_LEN) # Get predictions
lstm.plot_results_multiple(predictions, train_y, PREDICTION_LEN, PREDICTION_DELAY, "Train Set") # Plot predictions
print("Error: ", lstm.error(predictions, train_y, PREDICTION_DELAY)) # Print the error

# Predictions Test Set
predictions = lstm.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN) # Get predictions
lstm.plot_results_multiple(predictions, test_y, PREDICTION_LEN, PREDICTION_DELAY, "Test Set") # Plot predictions
print("Error: ", lstm.error(predictions, test_y, PREDICTION_DELAY)) # Print the error