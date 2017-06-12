from Blockchain_test import return_data
from lstm import load_data

import numpy as np
from sklearn import svm

NO_FEATURES = 1

SEQ_LEN = 1
PREDICTION_LEN = 1
PREDICTION_DELAY = 0

NORMALISATION = True
TRAIN_TEST_RATIO = 0.9

matrix = return_data()

matrix = np.array(matrix[:NO_FEATURES])

# Get the training and test test
train_X, train_y, test_X, test_y = load_data(
    matrix, SEQ_LEN, PREDICTION_LEN, PREDICTION_DELAY, NORMALISATION, TRAIN_TEST_RATIO
)

clf = svm.SVR()
# clf = svm.SVR(C=1.0, cache_size=500, coef0=0.0, degree=1, epsilon=0.3, gamma='auto',
#     kernel='rbf', max_iter=500, shrinking=False, tol=0.01, verbose=False)

train_X = np.array(train_X).reshape((len(train_X), 1))
train_y = np.array(train_y).reshape((len(train_y), 1))
test_X = np.array(test_X).reshape((len(test_X), 1))
test_y = np.array(test_y).reshape((len(test_y), 1))

clf.fit(train_X, train_y)
print(clf.predict(test_X))
print(clf.score(train_X, train_y))
print(clf.score(test_X, test_y))
# print(test_y)