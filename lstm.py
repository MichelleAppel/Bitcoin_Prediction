import warnings
import numpy as np
from numpy import newaxis

from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.models import Sequential
from keras import callbacks
import os

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")  # Ignore warnings


# Gets a matrix as input and divides it into training and test sets
def load_data(matrix, seq_len, pred_len, pred_delay, normalise_window, ratio):
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

    if normalise_window:  # Normalise
        mu = np.mean(matrix, axis=1)  # Mean

        sigma = np.abs(np.max(matrix, axis=1) - np.min(matrix, axis=1)) # Deviation
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

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2],
                                   x_train.shape[1]))  # Reshape, because expected lstm_1_input to have 3 dimensions
    x_test = np.reshape(x_test, (
    x_test.shape[0], x_test.shape[2], x_test.shape[1]))  # Reshape, because expected lstm_1_input to have 3 dimensions

    return [x_train, y_train, x_test, y_test]


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

# Plot the train and validation loss
def plot_loss(model_fit):
    # summarize history for accuracy
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.yscale('symlog', linthreshy=1)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    return [plt, model_fit.history['val_loss']]

# Compute mean error
def error(predicted, real, prediction_delay):
    errors = []
    predicted = np.array(predicted)
    predicted = predicted.reshape(np.size(predicted), 1)
    for p, r in zip(predicted, real):
        errors.append(np.abs(p - r))
    mean_error = np.array(errors).mean()
    return mean_error

def baseline_mse(data, prediction_delay):
    total = 0
    test_length = len(data) - 1 - prediction_delay

    for i in range(test_length):
        total += np.abs(data[i] - data[i + 1 + prediction_delay]) # ** 2

    baseline_test = total / test_length
    return baseline_test


# Method that runs a training and a prediction
def train_and_predict(matrix, NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NO_FEATURES,
                      UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO, VALIDATION_SPLIT, ES_MIN_DELTA):

    # ------------------------------------------------- Retrieve data ------------------------------------------------ #

    matrix = np.array(matrix[:NO_FEATURES])

    # Get the training and test test
    train_X, train_y, test_X, test_y = load_data(
        matrix, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NORMALISATION, TRAIN_TEST_RATIO
    )

    # Plot the train and test data
    time_step_of_seq = 0  # 0 for first step, -1 for last step
    # plot_train_test_set(train_X, train_y, test_X, test_y, time_step_of_seq)

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

    rms = optimizers.RMSprop(lr=LEARNING_RATE)  # RMS Optimizer
    adam = optimizers.adam(lr=LEARNING_RATE) # Adam Optimizer
    model.compile(loss='mse', optimizer=adam)  # Compile

    # callback = [callbacks.ModelCheckpoint('models/model', monitor='val_loss', verbose=0, save_best_only=False,
    #                                       save_weights_only=False, mode='auto', period=1)]
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

    loss_plot, val_error = plot_loss(model_fit)  # Plot the training and validation loss

    # ------------------------------------------------ Predict prices ------------------------------------------------ #

    # Predictions Train Set
    predictions_train = predict_sequences_multiple(model, train_X, SEQ_LEN, PREDICTION_LEN)  # Get predictions
    train_plot = plot_results_multiple(predictions_train, train_y, PREDICTION_LEN, PREDICTION_DELAY, "Train Set")  # Plot predictions

    train_error = error(predictions_train, train_y, PREDICTION_DELAY)
    baseline_train = baseline_mse(train_y, PREDICTION_DELAY)
    print("Error: ", train_error, " Baseline: ", baseline_train)  # Print the error

    # Predictions Test Set
    predictions = predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN)  # Get predictions
    test_plot = plot_results_multiple(predictions, test_y, PREDICTION_LEN, PREDICTION_DELAY, "Test Set")  # Plot predictions

    test_error = error(predictions, test_y, PREDICTION_DELAY)
    baseline_test = baseline_mse(test_y, PREDICTION_DELAY)
    print("Error: ", test_error, " Baseline: ", baseline_test)  # Print the error

    return [loss_plot, val_error, predictions_train, train_error, baseline_train, train_plot, predictions, test_error,
            baseline_test, test_plot, model]


def run_and_plot(TEST_NAME, TEST_OBJECT, initial_value, final_value,
                matrix, NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY,
                NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO,
                              VALIDATION_SPLIT, ES_MIN_DELTA):

    global baseline_train, baseline_test

    newpath_loss = 'plt/' + TEST_NAME + '/loss/'  # Create a new path
    if not os.path.exists(newpath_loss):  # If not already exist
        os.makedirs(newpath_loss)  # Create

    newpath_predictions = 'plt/' + TEST_NAME + '/predictions/'  # Create a new path
    if not os.path.exists(newpath_predictions):  # If not already exist
        os.makedirs(newpath_predictions)  # Create

    train_errors = []
    train_baselines = []
    test_errors = []
    test_baselines = []

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
        loss_plot, val_error, predictions_train, train_error, baseline_train, train_plot, predictions, test_error, baseline_test, \
        test_plot, model = \
            train_and_predict(matrix, NORMALISATION, TRAIN_TEST_RATIO, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY,
                              NO_FEATURES, UNITS, OUTPUT_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATIO,
                              VALIDATION_SPLIT, ES_MIN_DELTA)

        # Save plot
        # Keep this order, apparently it is important
        test_plot.title("BTC price prediction in USD from test set, " + TEST_OBJECT + ": " + str(number))
        test_plot.text(0.2, int(predictions[0][0])+2, 'NORMALISATION = ' + str(NORMALISATION) + '\n' +
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
        train_plot.text(0.2, int(predictions_train[0][0])+2, 'NORMALISATION = ' + str(NORMALISATION) + '\n' +
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
        loss_plot.text(0.2, int(val_error[0]), 'NORMALISATION = ' + str(NORMALISATION) + '\n' +
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
        train_errors.append(train_error)
        train_baselines.append(baseline_train)
        test_errors.append(test_error)
        test_baselines.append(baseline_test)

        print("progress: " + str(number) + " of " + str(final_value))

    # Plot errors and save
    plt.plot(train_errors)
    plt.plot(train_baselines)
    plt.title("Train error + baseline " + TEST_NAME)
    plt.text(0.2, baseline_train+1, 'NORMALISATION = ' + str(NORMALISATION) + '\n' +
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
    plt.savefig(newpath_loss + 'train')
    plt.close()

    plt.plot(test_errors)
    plt.plot(test_baselines)
    plt.title("Test error + baseline " + TEST_NAME)
    plt.text(0.2, baseline_test+1, 'NORMALISATION = ' + str(NORMALISATION) + '\n' +
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
    plt.savefig(newpath_loss + 'test')
    plt.close()

    print("Result in " + TEST_NAME)