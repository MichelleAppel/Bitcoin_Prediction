import time

import numpy as np
import matplotlib.pyplot as plt
from Blockchain_test import return_data

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras import metrics
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ReduceLROnPlateau

################################################################################

def plot_data(data):
    fig, ax = plt.subplots(1,3, figsize=(16,6))
    ax[0].plot(data[0,:])
    ax[0].set_title("average_USD_price")
    ax[1].plot(data[1,:])
    ax[1].set_title("blockchain_size")
    ax[2].plot(data[2,:])
    ax[2].set_title("average_block_size")
    plt.show()

def mse(y_pred, y_true):
    # mean squared error
    return np.mean(np.square(y_pred - y_true))

def mae(y_pred, y_true):
    # mean absolute error
    return np.mean(np.abs(y_pred-y_true))

def predict_baselines(X, y):
    # Baseline 1: take last value of input
    y_pred = X[:, -1, 0]
    print(" Baseline 1.")
    print("    MSE = {:.3f}".format(mse(y_pred, y)))
    print("    MAE = {:.3f}".format(mae(y_pred, y)))
    # Baseline 2: take average value of input
    y_pred = np.mean(X[:, :, 0], axis=1)
    print("  Baseline 2.")
    print("    MSE = {:.3f}".format(mse(y_pred, y)))
    print("    MAE = {:.3f}".format(mae(y_pred, y)))

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

        plt.clf()
        plt.title("Bitcoin Price Predictions (Epoch {})".format(epoch))
        plt.plot(y, c='r', lw=1, label="ground truth")
        plt.plot(y_pred, c='b', lw=1, label="predictions")
        plt.xlabel("Time")
        plt.ylabel("Bitcoint Price (USD)")
        plt.legend()
        plt.show()


################################################################################

test_ratio = 0.025
val_ratio  = 0.10

num_history  = 5
batch_size   = 32
num_epochs   = 800
learn_rate   = 4e-2
lstm_units   = 256

# For TensorBoard logs
log_dir = "/home/trunia1/Dropbox/PhD Research/tmp/logs/{}".format(time.time())

################################################################################

# Load data from Blockchain_test.py
matrix = return_data()

# [N, features]
data = np.transpose(matrix)

# Seems like all the Bitcoin prices are 0.0 before timestep = 294
data = data[294:,:]

num_train_val = int((1.0-test_ratio-val_ratio)*len(data))
print("num_train_val = {}".format(num_train_val))

# Split dataset
train_val_data = data[0:num_train_val]
test_data      = data[num_train_val:]
print("num_test = {}".format(len(test_data)))

# Plot the train, validation and test data
# plt.plot(np.arange(0, num_train_val), train_val_data[:,0], c='r', label="train_val")
# plt.plot(np.arange(num_train_val, num_train_val+len(test_data)), test_data[:,0], c='b', label="test")
# plt.ylabel("Bitcoin price in USD")
# plt.xlabel("time")
# plt.show()

X, y = [], []
for i in range(0, len(train_val_data) - (num_history+1) ):
    X.append(train_val_data[i:i+num_history,0])
    y.append(train_val_data[i+num_history:i+num_history+1,0])

X = np.asarray(X, np.float32)
y = np.asarray(y, np.float32)
X = np.expand_dims(X, axis=1)

print("X.shape", X.shape)
print("y.shape", y.shape)

num_train = int((1.0-val_ratio)*len(X))
print("Baselines on Train Set:")
predict_baselines(X[0:num_train], y[0:num_train])
print("Baselines on Validation Set:")
predict_baselines(X[num_train:], y[num_train:])

################################################################################

print('Build model...')

model = Sequential()
model.add(LSTM(lstm_units, input_shape=(1, num_history)))
model.add(Dense(1))

optimizer = RMSprop(lr=learn_rate)

model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=[metrics.mae, metrics.mse]
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=5e-5, verbose=1),  # Reduce learning rate if no progress
    #EarlyStopping(monitor='val_loss', patience=100, verbose=1),  # early stopping on validation loss
    TensorBoard(log_dir=log_dir, histogram_freq=20),  # dump results to TensorBoard
    ShowPredictionCallback(X, y, 800)
]

model.fit(
    x=X, y=y, validation_split=val_ratio,
    epochs=num_epochs, batch_size=batch_size,
    callbacks=callbacks, verbose=2
)