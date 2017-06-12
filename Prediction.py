
# TODO:
# Scale Y-axis of loss V
# Add Early stop V
# Plot resultaten, wegschrijven V
# Heeft het toevoegen van meer features een positief effect op het voorspellen van de Bitcoin prijs? V

# Experimenteer vooral met L2 regularization en dropout, beide dingen zitten standaard in Keras  http://cs231n.github.io/neural-networks-2/
# Recurrent Dropout/normale dropout
#
# per 1-5 epochs de plots met voorspelde Bitcoin prices op train/validatie set naar disk wegschrijven
# wijzigen van je feature vectors.

# Gradient clipping (2)
# Andere papers evaluatie mse
# Hoeveelheid training data verhogen en plotten
# Baseline

import numpy as np
import matplotlib.pyplot as plt

# import methods from lstm.py
import time

from lstm import train_and_predict_with_plots, train_and_predict, run_and_plot_write_to_disk, plot_train_and_test_data
from Blockchain import return_data

# Load data from Blockchain_test.py
matrix = return_data()

# Seems like all the Bitcoin prices are 0.0 before timestep = 294
matrix = matrix[:, 294:]

# ----------------------------------------------------- Parameters --------------------------------------------------- #

# Parameters
FEATURE_NORMALISATION = True  # Whether the data should be normalised
Y_NORMALISATION = True

TRAIN_TEST_RATIO = 0.9 # The train / test ratio

SEQ_LEN = 2 # The length of the sequence
PREDICTION_LEN = 1  # The amount of predicted values
PREDICTION_DELAY = 0  # Amount of time between sequence and prediction, 0 is next timestep after the sequence

NO_FEATURES = 1
UNITS = 256  # The amount of units in the LSTM
OUTPUT_DIM = 1  # Bitcoin price

INITIAL_LEARNING_RATE = 5e-3  # Learning rate

BATCH_SIZE = 32  # The batch size
EPOCHS = 200  # The amount of epochs

DROPOUT_RATIO = 0
VALIDATION_SPLIT = 0.1

ES_MIN_DELTA = 0

# ----------------------------------------------------- Run model ---------------------------------------------------- #

TEST_NAME = "NO_FEATURES_0-19_SEQ_LEN_2"
TEST_OBJECT = "NO_FEATURES"
initial_value = 1
final_value = len(matrix)

run_and_plot_write_to_disk(TEST_NAME, TEST_OBJECT, initial_value, final_value,
                matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NO_FEATURES, UNITS,
                OUTPUT_DIM, INITIAL_LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO, VALIDATION_SPLIT, ES_MIN_DELTA)



