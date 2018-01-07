from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import csv

seed = 7
np.random.seed(seed) #for reproducibility

#network and training
NB_EPOCH = 50
EPOCHS = 20
LEARNING_RATE = 0.1
DECAY_RATE = LEARNING_RATE / EPOCHS
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 1 #number of outputs
OPTIMIZER = Adam(lr=0.05, decay=DECAY_RATE) #optimizer
N_HIDDEN = 900
VALIDATION_SPLIT = 0.2 #how much TRAIN is reserved for validation
DROPOUT = 0.3

#read the file for training
filename = 'cube_values.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
headers = next(reader)

#load dataset
dataset = np.loadtxt(raw_data, delimiter=",")

#split into input and output variables
X = dataset[:, 0]
Y = dataset[:, 1]

X = np.array(X)
Y = np.array(Y)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

#read the file for testing
filename = 'cube_values_test.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
headers = next(reader)

#load dataset
dataset = np.loadtxt(raw_data, delimiter=",")

#split into input and output variables
X_test = dataset[:, 0]
Y_test = dataset[:, 1]

X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = X_test.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

def baseline_model():
	model = Sequential()
	model.add(Dense(1, input_dim=1))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dense(1))
	
	#compile model
	model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, 
	metrics=['mean_squared_error'])
	
	return model
	
# evaluate model with standardized dataset
np.random.seed(seed)

kreg = KerasRegressor(build_fn=baseline_model, epochs=NB_EPOCH, 
batch_size=BATCH_SIZE, verbose=VERBOSE)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', kreg))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=2, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Error: %.4f (%.4f) MSE" % (results.mean(), results.std()))

#calculate predictions
kreg.fit(X, Y, batch_size=16, epochs=NB_EPOCH)
predictions = kreg.predict(X_test)
score = r2_score(Y_test, predictions)
print("Y_test: ", Y_test[:20])
print("Predictions: ", predictions[:20])
print("Score: ", score)