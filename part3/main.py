# part 3 of research

from keras.optimizers import Adam
from custom_model import my_model as my_model
import numpy as np
import random, math

# normalize the data
def norm(min, max, input):
	z = (input - min) / (max - min)
	return z

# function whose values are approximated
def _func(x,y):
    return (x**2 + x*y + ((math.sqrt(x + 2*math.sqrt(x*y)))/(3*math.exp(-x)+1)))

# generate data set
def get_data(size):
    x1, x2, y = [], [], []
    for i in range(size):
        t1 = norm(0, 100, random.randint(0, 100))
        t2 = norm(0, 100, random.randint(0, 100))
        x1.append([t1])
        x2.append([t2])
        max = _func(100, 100)
        min = _func(0, 0)
        res_norm = norm(min, max, _func(t1,t2))
        y.append([res_norm])
    return [np.array(x1), np.array(x2)], np.array(y)

# dummy loss function
def loss_func(y_true, y_pred):
	return y_pred - y_true

np.random.seed(0)
X,y = get_data(10000)

my_model.compile(optimizer=Adam(lr=0.05), loss='mean_squared_error')
my_model.fit(X,y, shuffle = True, batch_size = 10,validation_split=0.4, epochs=20)

X,y = get_data(10)
print(my_model.predict(X))
print(y)