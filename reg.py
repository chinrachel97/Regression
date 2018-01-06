#fst_degree ok

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
import csv

np.random.seed(1671) #for reproducibility

#network and training
NB_EPOCH = 200
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 1 #number of outputs
OPTIMIZER = Adam() #optimizer
N_HIDDEN = 900
VALIDATION_SPLIT = 0.2 #how much TRAIN is reserved for validation
DROPOUT = 0.3

#read the file
filename = 'fst_degree.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
headers = next(reader)

#load dataset
dataset = np.loadtxt(raw_data, delimiter=",")

#split into input and output variables
X = dataset[:, 0]
Y = dataset[:, 1]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

#compile model
model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=['accuracy'])

#fit the model
history = model.fit(X, Y, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X, Y, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])

#calculate predictions
to_predict = np.array([101,120])
predictions = model.predict(to_predict)
print("Prediction:", predictions)