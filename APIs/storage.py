# def split_train_test(y, X, TRAIN_TEST_RATIO):
#     amount_of_elements_train = int(len(y)*TRAIN_TEST_RATIO)
#     return np.array(y)[:amount_of_elements_train], np.array(X)[:, :amount_of_elements_train],\
#            np.array(y)[amount_of_elements_train:], np.array(X)[:, amount_of_elements_train:]

# # Split data into training and test sets
# def split_train_test_random_batches(y, X, TIME_STEPS, BATCH_START, TRAIN_TEST_RATIO):
#     # The list that is going to contain the sequences
#     seq_list_y = []
#     seq_list_X = []
#
#     # Devide data in batches of TIME_STEPS length starting at BATCH_START
#     while BATCH_START + TIME_STEPS < len(y)-len(y)%TIME_STEPS:
#         seq_list_y.append(y[BATCH_START:BATCH_START + TIME_STEPS])
#
#         features = []
#         for feat in X:
#             features.append(feat[BATCH_START:BATCH_START + TIME_STEPS])
#         seq_list_X.append(features)
#
#         BATCH_START += TIME_STEPS
#
#     amount_of_batches_train = int(len(seq_list_y)*TRAIN_TEST_RATIO)
#
#     train_y = []
#     train_X = []
#
#     # Randomly devide data into training and test data
#     for i in range(0, amount_of_batches_train):
#         random_int = random.randint(0, len(seq_list_y)-1)
#
#         train_y.append(seq_list_y[random_int])
#         train_X.append(seq_list_X[random_int])
#
#         del seq_list_y[random_int]
#         del seq_list_X[random_int]
#
#     test_Y = seq_list_y
#     test_X = seq_list_X
#
#     return np.array(train_y), np.array(train_X), np.array(test_Y), np.array(test_X)

# Get the train and test sets
# train_y, train_X, test_y, test_X = split_train_test_random_batches(y, X, TIME_STEPS, BATCH_START, TRAIN_TEST_RATIO)