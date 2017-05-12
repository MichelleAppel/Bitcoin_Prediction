from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# import lstm, time

import random

from Blockchain_test import return_data

# Load data from Blockchain_test.py
y, X = return_data()


BATCH_START = 0
BATCH_SIZE = 30
TIME_STEPS = 5

INPUT_SIZE = len(X) # Amount of features
OUTPUT_SIZE = 1 # Bitcoin price

TRAIN_TEST_RATIO = 0.7
CELL_SIZE = 10
LR = 10


# Split data into training and test sets
def split_train_test(y, X, TIME_STEPS, BATCH_START, TRAIN_TEST_RATIO):
    # The list that is going to contain the sequences
    seq_list_y = []
    seq_list_X = []

    # Devide data in batches of TIME_STEPS length starting at BATCH_START
    while BATCH_START + TIME_STEPS < len(y)-len(y)%TIME_STEPS:
        seq_list_y.append(y[BATCH_START:BATCH_START + TIME_STEPS])

        features = []
        for feat in X:
            features.append(feat[BATCH_START:BATCH_START + TIME_STEPS])
        seq_list_X.append(features)

        BATCH_START += TIME_STEPS

    amount_of_batches_train = int(len(seq_list_y)*TRAIN_TEST_RATIO)

    train_y = []
    train_X = []

    # Randomly devide data into training and test data
    for i in range(0, amount_of_batches_train):
        random_int = random.randint(0, len(seq_list_y)-1)

        train_y.append(seq_list_y[random_int])
        train_X.append(seq_list_X[random_int])

        del seq_list_y[random_int]
        del seq_list_X[random_int]

    test_Y = seq_list_y
    test_X = seq_list_X

    return train_y, train_X, test_Y, test_X

train_y, train_X, test_Y, test_X = split_train_test(y, X, TIME_STEPS, BATCH_START, TRAIN_TEST_RATIO)

