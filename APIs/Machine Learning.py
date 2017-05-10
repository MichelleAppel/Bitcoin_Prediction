from test import return_data

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
# from sklearn.metrics import mean_squared_error
from lstm import generate_data, lstm_model, load_csvdata
import dateutil.parser
import datetime
import matplotlib.dates as mdates

# Get data from testfile
y, X = return_data()

def split_train_test(y, X, ratio_train, ratio_val):
    train_length = int(len(y)*ratio_train)
    val_length = int(len(y)*ratio_val)
    y = np.array(y)
    X = np.array(X)
    return y[:train_length], X[:, :train_length], y[train_length:train_length+val_length], X[:, train_length:train_length+val_length], y[train_length+val_length:], X[:, train_length+val_length:],

train_ratio = 0.6
val_ratio = 0.1

y_train, X_train, y_val, X_val, y_test, X_test = split_train_test(y, X, train_ratio, val_ratio)

LOG_DIR = './ops_logs/lstm_weather'
TIMESTEPS = len(X_train)
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),model_dir=LOG_DIR)

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X_val, y_val,
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)

regressor.fit(X_train.transpose(), y_train,
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(X_test)
print("test", y_test)
print("predicted", predicted)