import csv
import random
import math 
import numpy as np

#normalize data
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

#write to file	
with open('func_vals.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		t1 = norm(0, 100, random.randint(0, 100))
		t2 = norm(0, 100, random.randint(0, 100))
		max = _func(100, 100)
		min = _func(0, 0)
		res_norm = norm(min, max, _func(t1,t2))
		y = res_norm
		w.writerow([t1, t2, y])
		
#write to file	
with open('func_vals_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10):
		t1 = norm(0, 100, random.randint(0, 100))
		t2 = norm(0, 100, random.randint(0, 100))
		max = _func(100, 100)
		min = _func(0, 0)
		res_norm = norm(min, max, _func(t1,t2))
		y = res_norm
		w.writerow([t1, t2, y])