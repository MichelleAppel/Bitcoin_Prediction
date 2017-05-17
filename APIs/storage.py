# # def split_train_test(y, X, TRAIN_TEST_RATIO):
# #     amount_of_elements_train = int(len(y)*TRAIN_TEST_RATIO)
# #     return np.array(y)[:amount_of_elements_train], np.array(X)[:, :amount_of_elements_train],\
# #            np.array(y)[amount_of_elements_train:], np.array(X)[:, amount_of_elements_train:]
#
# # # Split data into training and test sets
# # def split_train_test_random_batches(y, X, TIME_STEPS, BATCH_START, TRAIN_TEST_RATIO):
# #     # The list that is going to contain the sequences
# #     seq_list_y = []
# #     seq_list_X = []
# #
# #     # Devide data in batches of TIME_STEPS length starting at BATCH_START
# #     while BATCH_START + TIME_STEPS < len(y)-len(y)%TIME_STEPS:
# #         seq_list_y.append(y[BATCH_START:BATCH_START + TIME_STEPS])
# #
# #         features = []
# #         for feat in X:
# #             features.append(feat[BATCH_START:BATCH_START + TIME_STEPS])
# #         seq_list_X.append(features)
# #
# #         BATCH_START += TIME_STEPS
# #
# #     amount_of_batches_train = int(len(seq_list_y)*TRAIN_TEST_RATIO)
# #
# #     train_y = []
# #     train_X = []
# #
# #     # Randomly devide data into training and test data
# #     for i in range(0, amount_of_batches_train):
# #         random_int = random.randint(0, len(seq_list_y)-1)
# #
# #         train_y.append(seq_list_y[random_int])
# #         train_X.append(seq_list_X[random_int])
# #
# #         del seq_list_y[random_int]
# #         del seq_list_X[random_int]
# #
# #     test_Y = seq_list_y
# #     test_X = seq_list_X
# #
# #     return np.array(train_y), np.array(train_X), np.array(test_Y), np.array(test_X)
#
# # Get the train and test sets
# # train_y, train_X, test_y, test_X = split_train_test_random_batches(y, X, TIME_STEPS, BATCH_START, TRAIN_TEST_RATIO)
#
# import time
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
# import lstm_helper, time
# import numpy as np
# from Blockchain_test import return_data
#
# # Load data from Blockchain_test.py
# matrix = return_data()
# matrix = matrix[0] # Bitcoin price only
#
# NORMALISATION = True
# DENORMALISATION = True
#
# BATCH_SIZE = 30
# SEQ_LEN = 15
# PREDICTION_LEN = 4
# EPOCHS = 1
#
# INPUT_DIM = len(matrix) # Amount of features
#
# OUTPUT_SIZE = 1 # Bitcoin price
#
# train_X, train_y, test_X, test_y, i_list = lstm_helper.load_data(matrix, SEQ_LEN, NORMALISATION)
#
# N_BATCHES = int(np.size(train_X) / SEQ_LEN)
#
#
# # Build model
# model = Sequential()
#
# model.add(LSTM(
#     input_dim = 1,
#     output_dim = 20,
#     return_sequences = False))
# # model.add(Dropout(0.2))
#
# # model.add(LSTM(
# #     units = BATCH_SIZE,
# #     return_sequences = False,
# #     ))
# # model.add(Dropout(0.2))
#
# model.add(Dense(
#     output_dim = 1,
#     ))
# model.add(Activation('linear'))
#
# start = time.time()
# model.compile(loss = 'mse', optimizer = 'rmsprop')
#
# print('compilation time : ', time.time() - start)
#
# # Train the model
# model.fit(
#     train_X,
#     train_y,
#     batch_size = BATCH_SIZE,
#     epochs = EPOCHS,
#     validation_split = 0.05
# )
#
# # Compute mean error
# def error(predicted, real):
#     errors = []
#     predicted = np.array(predicted)
#     predicted = predicted.reshape(np.size(predicted), 1)
#     for p, r in zip(predicted, real):
#         errors.append(np.abs(p-r))
#     mean_error = np.array(errors).mean()
#     return mean_error
#
# # Plot
# # Denormalised
# predictions = lstm_helper.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN, DENORMALISATION, i_list)
# denorm_test_x = lstm_helper.denormalise_windows(test_X, i_list)
# denorm_test_y = lstm_helper.denormalise_windows(test_y.reshape(len(test_y),1), i_list)
# denorm_test_y = np.array(denorm_test_y).reshape(np.array(test_y).shape)
#
# print("Denormalised error: ", error(predictions, denorm_test_y))
#
# lstm_helper.plot_results_multiple(predictions, test_y, PREDICTION_LEN, DENORMALISATION, i_list)
#
# # Normalised
# predictions = lstm_helper.predict_sequences_multiple(model, test_X, SEQ_LEN, PREDICTION_LEN, False, i_list)
# # print("test_norm", test_X[0])
# # print("pred_norm", predictions[0])
# # print("real_norm", test_y[0:PREDICTION_LEN])
# lstm_helper.plot_results_multiple(predictions, test_y, PREDICTION_LEN, False, i_list)
#
# print("Normalised error: ", error(predictions, test_y))