
# TODO:
# Scale Y-axis of loss V
# Add Early stop V
# Plot resultaten, wegschrijven V
# Experimenteer vooral met L2 regularization en dropout, beide dingen zitten standaard in Keras  http://cs231n.github.io/neural-networks-2/
# per 1-5 epochs de plots met voorspelde Bitcoin prices op train/validatie set naar disk wegschrijven
# wijzigen van je feature vectors.
# Heeft het toevoegen van meer features een positief effect op het voorspellen van de Bitcoin prijs? V

# Gradient clipping (2)
# Andere papers evaluatie mse
# Hoeveelheid training data verhogen en plotten
# Baseline

# lstm.py
import lstm
import numpy as np
from Blockchain import return_data

# Load data from Blockchain_test.py
matrix = return_data()

matrix = np.array(matrix) # Bitcoin price only

# ----------------------------------------------------- Parameters --------------------------------------------------- #

# Parameters
NORMALISATION = True  # Whether the data should be normalised

TRAIN_TEST_RATIO = 0.9  # The train / test ratio

SEQ_LEN = 1  # The length of the sequence
PREDICTION_LEN = 1  # The amount of predicted values
PREDICTION_DELAY = 0  # Amount of time between sequence and prediction, 0 is next timestep after the sequence

NO_FEATURES = len(matrix)
UNITS = 256  # The amount of units in the LSTM
OUTPUT_DIM = 1  # Bitcoin price

LEARNING_RATE = 0.001  # Learning rate

BATCH_SIZE = 32  # The batch size
EPOCHS = 1000  # The amount of epochs

DROPOUT_RATIO = 0
VALIDATION_SPLIT = 0.1

# --------------------------------------------------- Iterate model -------------------------------------------------- #

TEST_NAME = "BATCH_SIZE_EARLY_STOPPING"
TEST_OBJECT = "BATCH_SIZE"
initial_value = 1
final_value = 164

lstm.run_and_plot(TEST_NAME, TEST_OBJECT, initial_value, final_value,
                matrix, NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY,
                NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO, VALIDATION_SPLIT)