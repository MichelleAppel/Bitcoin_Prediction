import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

NORMALISATION = True  # Whether the data should be normalised

TRAIN_TEST_RATIO = 0.95  # The train / test ratio

SEQ_LEN = 1  # The length of the sequence
PREDICTION_LEN = 1  # The amount of predicted values
PREDICTION_DELAY = 0  # Amount of time between sequence and prediction, 0 is next timestep after the sequence

NO_FEATURES = 7
UNITS = 256  # The amount of units in the LSTM
OUTPUT_DIM = 1  # Bitcoin price

LEARNING_RATE = 0.001  # Learning rate

BATCH_SIZE = 1  # The batch size
EPOCHS = 18  # The amount of epochs

DROPOUT_RATIO = 0.2
VALIDATION_SPLIT = 0.1

list = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,10,16,100,150,300,5000]

plt.plot(list)
plt.yscale('symlog', linthreshy=10)


plt.legend("label")
plt.text(0.2, list[0]+2, 'NORMALISATION = ' + str(NORMALISATION) + '\n' +
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
         'VALIDATION_SPLIT = ' + str(VALIDATION_SPLIT), fontsize=6, horizontalalignment='left',
         verticalalignment='bottom')

plt.show()
