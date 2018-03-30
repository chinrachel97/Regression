# part 3 of research

from keras import backend as K
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from model import model
import numpy as np
import random, math

import csv

_EPSILON = K.epsilon()

# 1. dummy loss function
def loss_func(y_true, y_pred):
	return K.log(y_pred - y_true)

# 2. dummy loss function
def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

# 3. dummy loss function
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
	
np.random.seed(0)

#read the file for training
filename = 'func_vals.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)

#load dataset
dataset = np.loadtxt(raw_data, delimiter=",")

#split into input and output variables
X = dataset[:, 0:2]
Y = dataset[:, 2]

X = np.array(X)
Y = np.array(Y)
X = X.reshape(-2,2)
Y = Y.reshape(-1,1)

#read the file for testing
filename = 'func_vals_test.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)

#load dataset
dataset = np.loadtxt(raw_data, delimiter=",")

#split into input and output variables
X_test = dataset[:, 0:2]
Y_test = dataset[:, 2]

X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = X_test.reshape(-2,2)
Y_test = Y_test.reshape(-1,1)

#use ReLU
model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
model.fit(X, Y, shuffle = True, batch_size = 20,validation_split=0.4, epochs=20)

predictions = model.predict(X_test)
score = r2_score(Y_test, predictions)
print(predictions)
print(Y_test)
print("Score: ", score)