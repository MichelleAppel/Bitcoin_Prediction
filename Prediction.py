from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.models import Sequential
from keras import callbacks
import os

import matplotlib.pyplot as plt

# TODO:
# Scale Y-axis of loss V
# Add Early stop V
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
matrix = matrix[:, 300:]  # Bitcoin price only, before 300 only zeros

# ----------------------------------------------------- Parameters --------------------------------------------------- #

# Parameters
NORMALISATION = True  # Whether the data should be normalised

TRAIN_TEST_RATIO = 0.9  # The train / test ratio

SEQ_LEN = 1  # The length of the sequence
PREDICTION_LEN = 1  # The amount of predicted values
PREDICTION_DELAY = 0  # Amount of time between sequence and prediction, 0 is next timestep after the sequence

NO_FEATURES = len(matrix)
UNITS = 64  # The amount of units in the LSTM
OUTPUT_DIM = 1  # Bitcoin price

LEARNING_RATE = 0.001  # Learning rate

BATCH_SIZE = 32  # The batch size
EPOCHS = 500  # The amount of epochs

DROPOUT_RATIO = 0.2
VALIDATION_SPLIT = 0.1


def train_and_predict(matrix, NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NO_FEATURES,
                      UNITS,
                      OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO, VALIDATION_SPLIT):
    # ------------------------------------------------- Retrieve data ------------------------------------------------ #

    matrix = matrix[:NO_FEATURES]

    # Get the training and test test
    train_X, train_y, test_X, test_y = lstm.load_data(
        matrix, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NORMALISATION, TRAIN_TEST_RATIO
    )

    # Plot the train and test data
    time_step_of_seq = 0  # 0 for first step, -1 for last step
    # lstm.plot_train_test_set(train_X, train_y, test_X, test_y, time_step_of_seq)


    # -------------------------------------------------- Build model ------------------------------------------------- #

    model = Sequential()

    model.add(LSTM(
        input_dim=NO_FEATURES,  # input dim is number of features
        units=UNITS  # the size of the LSTM’s hidden state
    ))

    # model.add(Dropout(DROPOUT_RATIO))

    model.add(Dense(
        output_dim=OUTPUT_DIM,
    ))

    rms = optimizers.RMSprop(lr=LEARNING_RATE)  # Optimizer
    model.compile(loss='mse', optimizer=rms)  # Compile

    callback = [callbacks.EarlyStopping(monitor='val_loss', min_delta=80000, patience=40, mode='auto')]
    # Plot predictions callback
    # Plot gradients

    # Train the model
    model_fit = model.fit(
        train_X,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        # callbacks=callback
    )

    loss_plot = lstm.plot_loss(model_fit)  # Plot the training and validation loss

    # ------------------------------------------------ Predict prices ------------------------------------------------ #

    # Predictions Train Set
    predictions_train = lstm.predict_sequences_multiple(model, train_X, SEQ_LEN, PREDICTION_LEN)  # Get predictions
    train_plot = lstm.plot_results_multiple(predictions_train, train_y, PREDICTION_LEN, PREDICTION_DELAY, "Train Set")  # Plot predictions

    train_error = lstm.error(predictions_train, train_y, PREDICTION_DELAY)
    print("Error: ", train_error)  # Print the error

    # Predictions Test Set
    predictions = lstm.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN)  # Get predictions
    test_plot = lstm.plot_results_multiple(predictions, test_y, PREDICTION_LEN, PREDICTION_DELAY,
                                           "Test Set")  # Plot predictions

    test_error = lstm.error(predictions, test_y, PREDICTION_DELAY)
    print("Error: ", test_error)  # Print the error

    return [loss_plot, train_error, train_plot, test_error, test_plot, model]

# --------------------------------------------------- Iterate model -------------------------------------------------- #

TESTING_OBJECT = "no_features" # The object of testing, title of directories

newpath_loss = 'plt/' + TESTING_OBJECT + '/loss/' # Create a new path
if not os.path.exists(newpath_loss): # If not already exist
    os.makedirs(newpath_loss) # Create

newpath_predictions = 'plt/' + TESTING_OBJECT + '/predictions/' # Create a new path
if not os.path.exists(newpath_predictions): # If not already exist
    os.makedirs(newpath_predictions) # Create

train_errors = []
test_errors = []

for no in range(1, NO_FEATURES): # Iterate over testing object

    path_loss_plot = newpath_loss + TESTING_OBJECT + str(no)
    path_train_plot = newpath_predictions + 'train_' + TESTING_OBJECT + str(no)  # Create name for plot file
    path_test_plot = newpath_predictions + 'test_' + TESTING_OBJECT + str(no) # Create name for plot file


    # Run training and predictions with parameters
    loss_plot, train_error, train_plot, test_error, test_plot, model = \
            train_and_predict(matrix, NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY,
                              no, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO, VALIDATION_SPLIT)

    # Save plot
    # Keep this order, apparently it is important
    test_plot.savefig(path_test_plot)
    test_plot.close()

    train_plot.savefig(path_train_plot)
    train_plot.close()

    loss_plot.savefig(path_loss_plot)
    loss_plot.close()








    # Append errors to list
    train_errors.append(train_error)
    test_errors.append(test_errors)

print(test_errors)
print(train_errors)

# # Plot errors and save
# plt.plot(train_errors)
# # plt.savefig(newpath_loss + 'train')
# plt.show()
#
# plt.plot(test_errors)
# # plt.savefig(newpath_loss + 'test')
# plt.show()