import csv
import random

#f(x)=x+50
with open('lin_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(1, 1001)
		y = x+50
		w.writerow([x/1000, y/1000])
		
with open('lin_values_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(1, 1001)
		y = x+50
		w.writerow([x/1000, y/1000])

#f(x)=x^2
with open('quad_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(1, 1001)
		y = x*x
		w.writerow([x/1000, y/1000000])
		

with open('quad_values_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(1, 1001)
		y = x*x
		w.writerow([x/1000, y/1000000])
		
#f(x)=x^3 + 2*x
with open('cube_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(20000):
		x = random.randint(1, 1001)
		y = pow(x,3) + (2*x)
		w.writerow([x/1000, y/1000000000])
		

with open('cube_values_test.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(1, 1001)
		y = pow(x,3) + (2*x)
		w.writerow([x/1000, y/1000000000])
		
