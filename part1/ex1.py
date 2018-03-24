from keras.models import Sequential
from keras.layers import Dense

#defines a single layer with 12 artificial neurons
#expects 8 input variables (AKA features)
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))