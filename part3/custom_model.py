import keras
from keras import backend as K
from keras.layers import Dense, Input, merge
from keras.models import Model

USE_LOG = 1

def g_1(x):
    return K.square(x)

def g_3(x):
    return x

def g_4(x):
    return K.sqrt(x)

def g_5(x):
    return K.pow(x,-1)

def g_6(x):
    return K.exp(x)

def g_7(x):
    return K.log(x+100)

x = Input(shape=(1,))
y = Input(shape=(1,))

l1 = Dense(1, activation=g_6)(x)
l2 = Dense(1, activation=g_5)(l1)
l3 = Dense(1, activation=g_3)(l2)
l4 = Dense(1, activation=g_5)(l3)

# implementing g_2(x,y) = x*y
if USE_LOG:
    l5 = Dense(1, activation=g_7)(x)
    l6 = Dense(1, activation=g_7)(y)
    l7 = keras.layers.Add()([l5,l6])
    l8 = Dense(1, activation=g_3)(l7)
    l9 = Dense(1, activation=g_6)(l8)
else:
    l5 = Dense(1, activation=g_7)(x)
    l6 = Dense(1, activation=g_7)(y)
    l9 = K.dot(l5,l6)

l10 = Dense(1, activation=g_4)(l9)

if USE_LOG:
    l11 = Dense(1, activation=g_7)(x)
    l12 = Dense(1, activation=g_7)(l10)
    l13 = keras.layers.Add()([l11,l12]) 
    l14 = Dense(1, activation=g_3)(l13)
    l15 = Dense(1, activation=g_6)(l14)
else:
    l11 = Dense(1, activation=g_7)(x)
    l12 = Dense(1, activation=g_7)(l10)
    l15 = K.batch_dot(l11,l12)


l16 = Dense(1, activation=g_4)(l15)

if USE_LOG:
    l17 = Dense(1, activation=g_7)(l4)
    l18 = Dense(1, activation=g_7)(l16)
    l19 = keras.layers.Add()([l17,l18])
    l20 = Dense(1, activation=g_3)(l19)
    l21 = Dense(1, activation=g_6)(l20)
else:
    l17 = Dense(1, activation=g_7)(l4)
    l18 = Dense(1, activation=g_7)(l16)
    l21 = K.batch_dot(l17,l18)

if USE_LOG:
    l22 = Dense(1, activation=g_7)(x)
    l23 = Dense(1, activation=g_7)(y)
    l24 = keras.layers.Add()([l22,l23])
    l25 = Dense(1, activation=g_3)(l24)
    l26 = Dense(1, activation=g_6)(l25)
else:
    l22 = Dense(1, activation=g_7)(x)
    l23 = Dense(1, activation=g_7)(y)
    l26 = K.batch_dot(l22,l23)

l27 = Dense(1, activation=g_1)(x)
l28 = keras.layers.Add()([l26,l27])
l29 = Dense(1, activation=g_3)(l28)

l30 = keras.layers.Add()([l21,l29])
l31 = Dense(1, activation=g_3)(l30)

my_model = Model(input=[x,y], output=l31)
print(my_model.summary())