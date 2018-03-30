import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.layers import Dense, Input, merge

def baseline_model():
	model = Sequential()
	model.add(Dense(2, input_dim=2))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(1))
	
	return model
	
model = baseline_model()
print(model.summary())