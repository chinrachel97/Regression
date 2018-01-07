#lin_values have difference of ~1

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
import csv

seed = 7
np.random.seed(seed) #for reproducibility

#network and training
NB_EPOCH = 100
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

#read the file
filename = 'lin_values.csv'
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
Y = X.reshape(-1,1)

def baseline_model():
	model = Sequential()
	model.add(Dense(1, input_dim=1))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(8))
	model.add(Activation('relu'))
	model.add(Dense(4))
	model.add(Activation('relu'))
	model.add(Dense(2))
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
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Error: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#calculate predictions
kreg.fit(X, Y, batch_size=16, epochs=NB_EPOCH)
to_predict = np.array([101,120])
predictions = kreg.predict(to_predict)
print("Prediction:", predictions)