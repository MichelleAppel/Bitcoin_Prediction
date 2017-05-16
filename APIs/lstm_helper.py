import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len, denormalise, i_list):
    true_data = true_data.reshape(len(true_data),1)

    if denormalise:
        true_data = denormalise_windows(true_data, i_list)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')

    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def load_data(matrix, seq_len, normalise_window):
    # f = open(filename, 'r').read()
    # data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(matrix) - sequence_length):
        result.append(matrix[index: index + sequence_length])

    i_list = []
    if normalise_window:
        result, i_list = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    i_list.append(row)

    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test, i_list]


def normalise_windows(window_data):
    normalised_data = []
    i_list = [] # contains the initial values

    for window in window_data:
        i_list.append(window[0])
        if not window[0] == 0:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        else:
            normalised_window = [(float(p) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data, i_list

def denormalise_windows(window_data, i_list):
    row = i_list[-1]
    i_list = i_list[row:-1]

    denormalised_data = []

    for window, initial_val in zip(window_data, i_list):
        if not initial_val == 0:
            denormalised_window = [(float(initial_val)*(float(p) + 1)) for p in window]
        else:
            denormalised_window = [(float(p)) for p in window]
        denormalised_data.append(denormalised_window)

    return denormalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print
    "Compilation Time : ", time.time() - start
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len, denormalise, i_list):
    # Predict sequence of window_size steps before shifting prediction run forward by prediction_len steps

    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)-1):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    if denormalise:
        prediction_seqs = denormalise_windows(prediction_seqs, i_list)

    return prediction_seqs