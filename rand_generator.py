import csv
import random
import math 

#normalize data
def norm(min, max, input):
	z = (input - min) / (max - min)
	return z
	
#f(x)=x+50
with open('lin_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(0, 1000)
		y = x+50
		w.writerow([norm(0, 1000, x), norm(50, 1050, y)])
		
with open('lin_values_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(0, 2000)
		y = x+50
		w.writerow([norm(0, 2000, x), norm(50, 2050, y)])

#f(x)=x^2
with open('quad_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(0, 1000)
		y = x*x
		w.writerow([norm(0, 1000, x), norm(0, 1000000, y)])
		

with open('quad_values_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(0, 2000)
		y = x*x
		w.writerow([norm(0, 2000, x), norm(0, 4000000, y)])
		
#f(x)=x^3 + 2*x
with open('cube_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(0, 1000)
		y = pow(x,3) + (2*x)
		max = pow(1000,3) + (2*1000)
		w.writerow([norm(0, 1000, x), norm(0, max, y)])
		

with open('cube_values_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(0, 2000)
		y = pow(x,3) + (2*x)
		max = pow(2000,3) + (2*2000)
		w.writerow([norm(0, 2000, x), norm(0, max, y)])
	
#f(x)=3x^2 + 7sinx + cos(x^4)
with open('train_1.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(0, 1000)
		y = (3*pow(x,2)) + (7*math.sin(x)) + (math.cos(pow(x,4)))
		min = (3*pow(0,2)) + (7*math.sin(0)) + (math.cos(pow(0,4)))
		max = (3*pow(1000,2)) + (7*math.sin(1000)) + (math.cos(pow(1000,4)))
		max = (3*pow(1000,2)) + (7*math.sin(1000)) + (math.cos(pow(1000,4)))
		w.writerow([norm(0, 1000, x), norm(min, max, y)])
		

with open('test_1.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(0, 2000)
		y = (3*pow(x,2)) + (7*math.sin(x)) + (math.cos(pow(x,4)))
		min = (3*pow(0,2)) + (7*math.sin(0)) + (math.cos(pow(0,4)))
		max = (3*pow(2000,2)) + (7*math.sin(2000)) + (math.cos(pow(2000,4)))
		w.writerow([norm(0, 2000, x), norm(min, max, y)])