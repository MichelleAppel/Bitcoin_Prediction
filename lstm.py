import warnings
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from keras.layers.core import Dense, ActivityRegularization
from keras.layers.recurrent import LSTM
from keras import optimizers, regularizers
from keras.models import Sequential
from keras import metrics
from keras import callbacks
import os

import time

from Blockchain_test import return_data

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras import metrics
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ReduceLROnPlateau

warnings.filterwarnings("ignore")  # Ignore warnings

# ----------------------------------------------------- Plot data ---------------------------------------------------- #

def plot_BTC_data(matrix):
    plt.plot(matrix[0])
    plt.title("BTC price in USD from 01-03-2009 to " + time.strftime("%d-%m-%Y"))
    plt.xlabel("time (days)")
    plt.ylabel("BTC/USD")
    plt.savefig('plt/current_BTC_graph')

def plot_train_and_test_data(matrix, TRAIN_TEST_RATIO):
    matrix = matrix[0]

    row = round(TRAIN_TEST_RATIO * matrix.shape[0])
    print(str(row))

    plt.plot(matrix[:row])
    plt.title("BTC price in USD train and validation target")
    plt.xlabel("time (days)")
    plt.ylabel("BTC/USD")
    plt.savefig('plt/train_y')
    plt.close()

    plt.plot(matrix[row:])
    plt.title("BTC price in USD test target")
    plt.xlabel("time (days)")
    plt.ylabel("BTC/USD")
    plt.savefig('plt/test_y')
    plt.close()

# ----------------------------------------------------- Load data ---------------------------------------------------- #

# Gets a matrix as input and divides it into training and test sets
def load_data(matrix, seq_len, pred_len, pred_delay, feature_normalisation, y_normalisation, ratio):
    sequence_length = seq_len + pred_len + pred_delay  # The length of the slice that is taken from the data

    result = []  # List that is going to contain the sequences
    for index in range(len(matrix[0]) - sequence_length):  # Take every possible sequence from beginning to end
        result.append(matrix[:, index: index + sequence_length])  # Append sequence to result list

    result = np.array(result)  # Convert result to numpy array

    row = round(ratio * result.shape[0])  # Up until this row the data is training data

    # Split in x and y
    x_train = result[:int(row), :, :seq_len]  # The sequence of the training data
    y_train = result[:int(row), 0, -pred_len]  # The to be predicted values of the training data
    # y_train = np.random.rand(len(y_train)) # Noise experiment

    x_test = result[int(row):, :, :seq_len]  # The sequence of the test data
    y_test = result[int(row):, 0, -pred_len]  # The to be predicted values of the test data
    # y_test = np.random.rand(len(y_test)) # Noise experiment

    mu = [0]
    sigma = [1]

    if feature_normalisation:  # Normalise
        mu = np.mean(matrix[:int(row)], axis=1)  # Mean
        # mu = np.min(matrix, axis=1) # Min

        sigma = np.abs(np.max(matrix[:int(row)], axis=1) - np.min(matrix[:int(row)], axis=1)) # Deviation
        # sigma = np.std(matrix, axis=1)

        matrix = matrix.transpose()  # Transpose
        matrix = (matrix - mu) / sigma  # Normalise
        matrix = matrix.transpose()  # Transpose back

        result = []  # List that is going to contain the sequences
        for index in range(len(matrix[0]) - sequence_length):  # Take every possible sequence from beginning to end
            result.append(matrix[:, index: index + sequence_length])  # Append sequence to result list

        result = np.array(result)  # Convert result to numpy array

        # Split in x and y
        x_train = result[:int(row), :, :seq_len]  # The sequence of the training data
        x_test = result[int(row):, :, :seq_len]  # The sequence of the test data
        if y_normalisation:
            y_train = result[:int(row), 0, -pred_len]  # The to be predicted values of the training data
            y_test = result[int(row):, 0, -pred_len]  # The to be predicted values of the test data


    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2],
                                   x_train.shape[1]))  # Reshape, because expected lstm_1_input to have 3 dimensions
    x_test = np.reshape(x_test, (
    x_test.shape[0], x_test.shape[2], x_test.shape[1]))  # Reshape, because expected lstm_1_input to have 3 dimensions

    return [x_train, y_train, x_test, y_test, mu, sigma]

# ----------------------------------------------------- Plot data ---------------------------------------------------- #

# Plots the train and test set
# time_step_of_seq is the nth day of the sequence that is to be plotted: 0 is the first day, -1 is the last day
def plot_train_test_set(train_X, train_y, test_X, test_y, time_step_of_seq):
    # Trainingset
    trainx = train_X.reshape((train_X.shape[0], train_X.shape[2], train_X.shape[1]))[:, :, time_step_of_seq]

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(trainx, label="train_X")
    # ax.plot(train_y, label="train_y")

    plt.legend()
    plt.show()

    # Testset
    testx = test_X.reshape((test_X.shape[0], test_X.shape[2], test_X.shape[1]))[:, :, time_step_of_seq]

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(testx, label="test_X")
    # ax.plot(test_y, label="test_y")

    plt.legend()
    plt.show()

# ------------------------------------------------- Predict sequences ------------------------------------------------ #

def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of window_size steps before shifting prediction run forward by prediction_len steps

    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):  # -1
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    return prediction_seqs

# --------------------------------------------------- Plot results --------------------------------------------------- #

# Plots the results
def plot_results_multiple(predicted_data, true_data, prediction_len, prediction_delay, set):
    true_data = true_data.reshape(len(true_data), 1)  # reshape true data from batches to one long sequence

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="True Data " + set)

    if prediction_len == 1:
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
    else:
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()

    plt.title('Bitcoin price prediction in USD from ' + set)
    plt.ylabel('BTC/USD')
    plt.xlabel('time')

    return plt

# ----------------------------------------------------- Plot loss ---------------------------------------------------- #

# Plot the train and validation loss
def plot_loss(model_fit):
    # summarize history for accuracy
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.yscale('symlog', linthreshy=0.000001)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    return [plt, model_fit.history['val_loss']]

# ---------------------------------------------------- Mean error ---------------------------------------------------- #

def mse(y_pred, y_true):
    # mean squared error
    return np.mean(np.square(y_pred - y_true))

def mae(y_pred, y_true):
    # mean absolute error
    return np.mean(np.abs(y_pred-y_true))

# Compute mean error
def errors(predicted, real, prediction_delay):

    predicted = np.array(predicted)
    predicted = predicted.reshape(np.size(predicted))

    mean_error_l1 = mae(predicted, real)
    mean_error_l2 = mse(predicted, real)

    return mean_error_l1, mean_error_l2

# ------------------------------------------------- Baseline error --------------------------------------------------- #

def baseline_error(data, prediction_delay):
    y_true = data[:-(1+prediction_delay)]
    y_pred = data[1+prediction_delay:]

    baseline_l1, baseline_l2 = errors(y_pred, y_true, prediction_delay)

    return baseline_l1, baseline_l2

# -------------------------------------------- Show predictions callback --------------------------------------------- #

class ShowPredictionCallback(Callback):

    def __init__(self, X, y, interval=10):
        self.X = X
        self.y = y
        self.interval = interval

    def on_train_end(self, logs=None):
        self.plot_model_predictions("final")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            self.plot_model_predictions(epoch)

    def plot_model_predictions(self, epoch):
        # Pass through network and get predictions
        y_pred = self.model.predict(self.X)
        y = self.y

        plt.clf()
        plt.title("Bitcoin Price Predictions (Epoch {})".format(epoch))
        plt.plot(y, c='r', lw=1, label="ground truth")
        plt.plot(y_pred, c='b', lw=1, label="predictions")
        plt.xlabel("Time")
        plt.ylabel("Bitcoint Price (USD)")
        plt.legend()
        plt.show()

# ------------------------------------------- Train and predict methods ---------------------------------------------- #

# Build a model
def build_model(train_X, train_y, NO_FEATURES, UNITS, DROPOUT_RATIO, OUTPUT_DIM, LEARNING_RATE, VALIDATION_SPLIT, EPOCHS, BATCH_SIZE):
    print('Build model...')

    model = Sequential()

    model.add(LSTM(
        input_dim=NO_FEATURES,  # input dim is number of features
        units=UNITS,  # the size of the LSTM’s hidden state
        # kernel_regularizer = regularizers.l2(0.5),


        # use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        # bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
        # bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
        # bias_constraint=None,

        # dropout = DROPOUT_RATIO,
        recurrent_dropout=DROPOUT_RATIO
    ))

    model.add(Dense(
        output_dim=OUTPUT_DIM,
    ))

    rms = optimizers.RMSprop(lr=LEARNING_RATE)  # RMS Optimizer
    # adam = optimizers.adam(lr=LEARNING_RATE) # Adam Optimizer
    # model.compile(loss='mse', optimizer=rms)  # Compile

    model.compile(
        loss='mse',
        optimizer=rms,
        # metrics=[metrics.mae, metrics.mse]
    )

    # callback = [callbacks.ModelCheckpoint('models/model', monitor='val_loss', verbose=0, save_best_only=False,
    #                                       save_weights_only=False, mode='auto', period=1)]
    # Plot predictions callback
    # Plot gradients

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=5e-15, verbose=1),
        # Reduce learning rate if no progress
        # EarlyStopping(monitor='val_loss', patience=100, verbose=1),  # early stopping on validation loss
        # TensorBoard(log_dir=log_dir, histogram_freq=20),  # dump results to TensorBoard
        # ShowPredictionCallback(test_X, test_y, 50)
    ]

    model_fit = model.fit(
        x=train_X,
        y=train_y,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    return model, model_fit

# Predict train and test set
def predict_train_test_set(model, train_X, train_y, test_X, test_y, PREDICTION_LEN, PREDICTION_DELAY, SEQ_LEN, Y_NORMALISATION, sigma, mu):
    if Y_NORMALISATION:
        train_y = train_y * sigma[0] + mu[0]
        test_y = test_y * sigma[0] + mu[0]

    # Predictions Train Set
    predictions_train = predict_sequences_multiple(model, train_X, SEQ_LEN, PREDICTION_LEN)  # Get predictions
    if Y_NORMALISATION:
        predictions_train = np.array(predictions_train) * sigma[0] + mu[0]

    # Predictions Test Set
    predictions_test = predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN)  # Get predictions
    if Y_NORMALISATION:
        predictions_test = np.array(predictions_test) * sigma[0] + mu[0]

    return predictions_train, predictions_test

# ------------------------------------------------- Train and predict ------------------------------------------------ #

# Method that runs a training and a prediction, returns predictions and plots
def train_and_predict(matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                      PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO,
                      VALIDATION_SPLIT):

    # ------------------------------------------------- Retrieve data ------------------------------------------------ #

    matrix = np.array(matrix[:NO_FEATURES])

    # Get the training and test test
    train_X, train_y, test_X, test_y, mu, sigma = load_data(
        matrix, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO
    )

    # Plot the train and test data
    time_step_of_seq = 0  # 0 for first step, -1 for last step
    # plot_train_test_set(train_X, train_y, test_X, test_y, time_step_of_seq)

    # -------------------------------------------------- Build model ------------------------------------------------- #

    # Build and fit the model
    model, model_fit = build_model(train_X, train_y, NO_FEATURES, UNITS, DROPOUT_RATIO, OUTPUT_DIM, LEARNING_RATE,
                                   VALIDATION_SPLIT, EPOCHS, BATCH_SIZE)

    # ------------------------------------------------ Predict prices ------------------------------------------------ #

    # Get predictions of train and test set
    predictions_train, predictions_test = predict_train_test_set(model, train_X, train_y, test_X, test_y, PREDICTION_LEN,
                                                                 PREDICTION_DELAY, SEQ_LEN, Y_NORMALISATION, sigma, mu)

    # Denormalise
    if Y_NORMALISATION:
        train_y = train_y * sigma + mu
        test_y = test_y * sigma + mu

    # Train errors
    train_error_l1, train_error_l2 = errors(predictions_train, train_y, PREDICTION_DELAY)
    baseline_train_l1, baseline_train_l2 = baseline_error(train_y, PREDICTION_DELAY)
    # Show errors and baseline of train set
    print("")
    print("--- Train ---")
    print("L1 error: ", train_error_l1, ", Baseline L1 error: ", baseline_train_l1)
    print("L2 error: ", train_error_l2, ", Baseline L1 error: ", baseline_train_l2)
    print("")

    # Test errors
    test_error_l1, test_error_l2 = errors(predictions_test, test_y, PREDICTION_DELAY)
    baseline_test_l1, baseline_test_l2 = baseline_error(test_y, PREDICTION_DELAY)
    # Show errors and baseline of test set
    print("--- Test ---")
    print("L1 error: ", test_error_l1, ", Baseline L1 error: ", baseline_test_l1)
    print("L2 error: ", test_error_l2, ", Baseline L1 error: ", baseline_test_l2)
    print("")


    return [train_y, predictions_train, train_error_l1, train_error_l2, baseline_train_l1, baseline_train_l2,
            test_y, predictions_test, test_error_l1, test_error_l2, baseline_test_l1, baseline_test_l2, sigma, mu]

# ------------------------------------------ Train and predict with plots -------------------------------------------- #

# Method that runs a training and a prediction, returns predictions and plots
def train_and_predict_with_plots(matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                                 PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO,
                                 VALIDATION_SPLIT):

    # ------------------------------------------------- Retrieve data ------------------------------------------------ #

    matrix = np.array(matrix[:NO_FEATURES])

    # Get the training and test test
    train_X, train_y, test_X, test_y, mu, sigma = load_data(
        matrix, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO
    )

    # Plot the train and test data
    time_step_of_seq = 0  # 0 for first step, -1 for last step
    # plot_train_test_set(train_X, train_y, test_X, test_y, time_step_of_seq)

    # -------------------------------------------------- Build model ------------------------------------------------- #

    # Build and fit the model
    model, model_fit = build_model(train_X, train_y, NO_FEATURES, UNITS, DROPOUT_RATIO, OUTPUT_DIM, LEARNING_RATE,
                                   VALIDATION_SPLIT, EPOCHS, BATCH_SIZE)
    loss_plot, val_error = plot_loss(model_fit)  # Plot the training and validation loss

    # ------------------------------------------------ Predict prices ------------------------------------------------ #

    # Get predictions of train and test set
    predictions_train, predictions_test = predict_train_test_set(model, train_X, train_y, test_X, test_y, PREDICTION_LEN,
                                                                 PREDICTION_DELAY, SEQ_LEN, Y_NORMALISATION, sigma, mu)

    # Denormalise
    if Y_NORMALISATION:
        train_y = train_y * sigma[0] + mu[0]
        test_y = test_y * sigma[0] + mu[0]

    # Train errors
    train_error_l1, train_error_l2 = errors(predictions_train, train_y, PREDICTION_DELAY)
    baseline_train_l1, baseline_train_l2 = baseline_error(train_y, PREDICTION_DELAY)
    # Show errors and baseline of train set
    print("")
    print("--- Train ---")
    print("L1 error: ", train_error_l1, ", Baseline L1 error: ", baseline_train_l1)
    print("L2 error: ", train_error_l2, ", Baseline L1 error: ", baseline_train_l2)
    print("")

    # Test errors
    test_error_l1, test_error_l2 = errors(predictions_test, test_y, PREDICTION_DELAY)
    baseline_test_l1, baseline_test_l2 = baseline_error(test_y, PREDICTION_DELAY)
    # Show errors and baseline of test set
    print("--- Test ---")
    print("L1 error: ", test_error_l1, ", Baseline L1 error: ", baseline_test_l1)
    print("L2 error: ", test_error_l2, ", Baseline L1 error: ", baseline_test_l2)
    print("")

    train_plot = plot_results_multiple(predictions_train, train_y, PREDICTION_LEN, PREDICTION_DELAY, "Train Set")  # Plot predictions
    test_plot = plot_results_multiple(predictions_test, test_y, PREDICTION_LEN, PREDICTION_DELAY, "Test Set")  # Plot predictions

    return [loss_plot, val_error, predictions_train, train_error_l1, train_error_l2, baseline_train_l1, baseline_train_l2,
            train_plot, predictions_test, test_error_l1, test_error_l2, baseline_test_l1, baseline_test_l2, test_plot, model, sigma, mu]

# -------------------------------------------------- Write to disk --------------------------------------------------- #

def run_and_plot_write_to_disk(TEST_NAME, TEST_OBJECT, initial_value, final_value,
                               matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                               PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS,
                               DROPOUT_RATIO, VALIDATION_SPLIT, ES_MIN_DELTA):

    global baseline_train_l1, baseline_test_l1, baseline_train_l2, baseline_test_l2

    newpath_loss = 'plt/' + TEST_NAME + '/loss/'  # Create a new path
    if not os.path.exists(newpath_loss):  # If not already exist
        os.makedirs(newpath_loss)  # Create

    newpath_predictions = 'plt/' + TEST_NAME + '/predictions/'  # Create a new path
    if not os.path.exists(newpath_predictions):  # If not already exist
        os.makedirs(newpath_predictions)  # Create

    train_L1_errors = []
    train_L1_baselines = []
    test_L1_errors = []
    test_L1_baselines = []

    train_L2_errors = []
    train_L2_baselines = []
    test_L2_errors = []
    test_L2_baselines = []

    for number in range(initial_value, final_value):  # Iterate over testing object

        if TEST_OBJECT == "TRAIN_TEST_RATIO":
            TRAIN_TEST_RATIO = float(number)/100
        elif TEST_OBJECT == "SEQ_LEN":
            SEQ_LEN = number
        elif TEST_OBJECT == "PREDICTION_DELAY":
            PREDICTION_DELAY = number
        elif TEST_OBJECT == "UNITS":
            UNITS = number
        elif TEST_OBJECT == "LEARNING_RATE":
            LEARNING_RATE = float(number)/1000
        elif TEST_OBJECT == "BATCH_SIZE":
            BATCH_SIZE = number
        elif TEST_OBJECT == "DROPOUT_RATIO":
            DROPOUT_RATIO = float(number)/100
        elif TEST_OBJECT == "VALIDATION_SPLIT":
            VALIDATION_SPLIT = float(number)/100
        elif TEST_OBJECT == "ES_MIN_DELTA":
            ES_MIN_DELTA = 2 ** number
        elif TEST_OBJECT == "NO_FEATURES":
            NO_FEATURES = number
            # EPOCHS = EPOCHS * number

        path_loss_plot = newpath_loss + TEST_NAME + str(number)
        path_train_plot = newpath_predictions + 'train_' + TEST_NAME + str(number)  # Create name for plot file
        path_test_plot = newpath_predictions + 'test_' + TEST_NAME + str(number)  # Create name for plot file

        # Run training and predictions with parameters
        loss_plot, val_error, predictions_train, train_error_l1, train_error_l2, baseline_train_l1, baseline_train_l2, train_plot, predictions, test_error_l1, test_error_l2, \
        baseline_test_l1, baseline_test_l2, test_plot, model, sigma, mu = \
            train_and_predict_with_plots(matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY,
                                         NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO,
                                         VALIDATION_SPLIT)

        # Save plot
        # Keep this order, apparently it is important
        test_plot.title("BTC price prediction in USD from test set, " + TEST_OBJECT + ": " + str(number))
        test_plot.text(0.2, int(predictions[0][0]) + 2,
         'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
         'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
         'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
                       'EPOCHS = ' + str(EPOCHS) + '\n' +
                       'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
         'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
         'ES_MIN_DELTA = ' + str(ES_MIN_DELTA)
                       , fontsize = 6, horizontalalignment='left',
                       verticalalignment='bottom')
        test_plot.savefig(path_test_plot)
        test_plot.close()

        train_plot.title("BTC price prediction in USD from train set, " + TEST_OBJECT + ": " + str(number))
        train_plot.text(0.2, int(predictions_train[0][0]) + 2,
         'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
         'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
         'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
                        'EPOCHS = ' + str(EPOCHS) + '\n' +
         'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
         'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
         'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize = 6, horizontalalignment='left',
                        verticalalignment='bottom')
        train_plot.savefig(path_train_plot)
        train_plot.close()

        test_plot.title("Train and validation loss from train set, " + TEST_OBJECT + ": " + str(number))
        loss_plot.text(0.2, int(val_error[0]),          'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
         'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
         'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
                       'EPOCHS = ' + str(EPOCHS) + '\n' +
                       'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
                       'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
                       'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize = 6, horizontalalignment='left',
                       verticalalignment='top')
        loss_plot.savefig(path_loss_plot)
        loss_plot.close()

        # Append errors to list
        train_L1_errors.append(train_error_l1)
        train_L1_baselines.append(baseline_train_l1)
        test_L1_errors.append(test_error_l1)
        test_L1_baselines.append(baseline_test_l1)

        train_L2_errors.append(train_error_l2)
        train_L2_baselines.append(baseline_train_l2)
        test_L2_errors.append(test_error_l2)
        test_L2_baselines.append(baseline_test_l2)

        print("progress: " + str(number) + " of " + str(final_value))

    # Plot errors and save
    plt.plot(train_L1_errors)
    plt.plot(train_L1_baselines)
    plt.title("Train L1 error + baseline " + TEST_NAME)
    plt.text(0.2, baseline_train_l1 + 1, 'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
             'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
             'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
             'EPOCHS = ' + str(EPOCHS) + '\n' +
             'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
             'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
             'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize=6, horizontalalignment='left',
             verticalalignment='bottom')
    plt.savefig(newpath_loss + 'l1train')
    plt.close()

    plt.plot(test_L1_errors)
    plt.plot(test_L1_baselines)
    plt.title("Test L1 error + baseline " + TEST_NAME)
    plt.text(0.2, baseline_test_l1 + 1, 'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
             'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
             'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
             'EPOCHS = ' + str(EPOCHS) + '\n' +
             'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
             'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
             'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize = 6, horizontalalignment='left',
             verticalalignment='bottom')
    plt.savefig(newpath_loss + 'l1test')
    plt.close()

    plt.plot(train_L2_errors)
    plt.plot(train_L2_baselines)
    plt.title("Train L2 error + baseline " + TEST_NAME)
    plt.text(0.2, baseline_train_l2 + 1, 'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
             'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
             'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
             'EPOCHS = ' + str(EPOCHS) + '\n' +
             'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
             'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
             'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize=6, horizontalalignment='left',
             verticalalignment='bottom')
    plt.savefig(newpath_loss + 'l2train')
    plt.close()

    plt.plot(test_L2_errors)
    plt.plot(test_L2_baselines)
    plt.title("Test L2 error + baseline " + TEST_NAME)
    plt.text(0.2, baseline_test_l2 + 1, 'FEATURE NORMALISATION = ' + str(FEATURE_NORMALISATION) + '\n' +
         'Y NORMALISATION = ' + str(Y_NORMALISATION) + '\n' +
         'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
         'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
         'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
         'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
         'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
         'UNITS = ' + str(UNITS) + '\n' +
         'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
             'LEARNING_RATE = ' + str(LEARNING_RATE) + '\n' +
             'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
             'EPOCHS = ' + str(EPOCHS) + '\n' +
             'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
             'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
             'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize = 6, horizontalalignment='left',
             verticalalignment='bottom')
    plt.savefig(newpath_loss + 'l2test')
    plt.close()

    print("train_L1_errors", train_L1_errors)
    print("test_L1_errors", test_L1_errors)
    print("train_L1_baselines", train_L1_baselines)
    print("test_L1_baselines", test_L1_baselines)
    print("")
    print("train_L2_errors", train_L2_errors)
    print("test_L2_errors", test_L2_errors)
    print("train_L2_baselines", train_L2_baselines)
    print("test_L2_baselines", test_L2_baselines)

    print("Result in folder:" + TEST_NAME)

# ------------------------------------------- Feature target normalisation test -------------------------------------- #
def feature_target_normalisation_test(matrix, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                                      PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, INITIAL_LEARNING_RATE,
                                      BATCH_SIZE, EPOCHS, DROPOUT_RATIO,
                                      VALIDATION_SPLIT):
    FEATURE_NORMALISATION = False  # Whether the data should be normalised
    Y_NORMALISATION = False

    train_y, predictions_train, train_error_l1, train_error_l2, baseline_train_l1, baseline_train_l2, \
    test_y, predictions_test, test_error_l1, test_error_l2, baseline_test_l1, baseline_test_l2 = \
        train_and_predict(matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                          PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, INITIAL_LEARNING_RATE, BATCH_SIZE, EPOCHS,
                          DROPOUT_RATIO,
                          VALIDATION_SPLIT)

    predictions_false_false = predictions_test
    l1_false_false = train_error_l1
    l2_false_false = train_error_l2

    FEATURE_NORMALISATION = True  # Whether the data should be normalised
    Y_NORMALISATION = False

    train_y, predictions_train, train_error_l1, train_error_l2, baseline_train_l1, baseline_train_l2, \
    test_y, predictions_test, test_error_l1, test_error_l2, baseline_test_l1, baseline_test_l2 = \
        train_and_predict(matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                          PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, INITIAL_LEARNING_RATE, BATCH_SIZE, EPOCHS,
                          DROPOUT_RATIO,
                          VALIDATION_SPLIT)

    predictions_true_false = predictions_test
    l1_true_false = train_error_l1
    l2_true_false = train_error_l2

    FEATURE_NORMALISATION = True  # Whether the data should be normalised
    Y_NORMALISATION = True

    train_y, predictions_train, train_error_l1, train_error_l2, baseline_train_l1, baseline_train_l2, \
    test_y, predictions_test, test_error_l1, test_error_l2, baseline_test_l1, baseline_test_l2 = \
        train_and_predict(matrix, FEATURE_NORMALISATION, Y_NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN,
                          PREDICTION_DELAY, NO_FEATURES, UNITS, OUTPUT_DIM, INITIAL_LEARNING_RATE, BATCH_SIZE, EPOCHS,
                          DROPOUT_RATIO,
                          VALIDATION_SPLIT)

    predictions_true_true = predictions_test
    l1_true_true = train_error_l1
    l2_true_true = train_error_l2

    # # Print maximum values of predictions of train and test set
    # print("max predictions train " + str(np.array(predictions_train).max()))
    # print("max predictions test " + str(np.array(predictions).max()))

    plt.plot(test_y, label="True data")
    plt.plot(predictions_false_false, label="F norm : false, T norm : false")
    plt.plot(predictions_true_false, label="F norm : true, T norm : false")
    plt.plot(predictions_true_true, label="F norm : true, T norm : true")
    plt.legend()

    plt.title("BTC price prediction in USD from test set")
    plt.ylabel('BTC/USD')
    plt.xlabel('time (days)')

    plt.text(0.2, 1000,
             'TRAIN_TEST_RATIO = ' + str(TRAIN_TEST_RATIO) + '\n' +
             'SEQ_LEN = ' + str(SEQ_LEN) + '\n' +
             'PREDICTION_LEN = ' + str(PREDICTION_LEN) + '\n' +
             'PREDICTION_DELAY = ' + str(PREDICTION_DELAY) + '\n' +
             'NO_FEATURES = ' + str(NO_FEATURES) + '\n' +
             'UNITS = ' + str(UNITS) + '\n' +
             'OUTPUT_DIM = ' + str(OUTPUT_DIM) + '\n' +
             'INITIAL_LEARNING_RATE = ' + str(INITIAL_LEARNING_RATE) + '\n' +
             'BATCH_SIZE = ' + str(BATCH_SIZE) + '\n' +
             'EPOCHS = ' + str(EPOCHS) + '\n' +
             'DROPOUT_RATIO = ' + str(DROPOUT_RATIO) + '\n' +
             'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT) + '\n' +
             'ES_MIN_DELTA = ' + str(ES_MIN_DELTA), fontsize=6, horizontalalignment='left',
             verticalalignment='bottom')

    plt.savefig('plt/feattargetNormalisation1')