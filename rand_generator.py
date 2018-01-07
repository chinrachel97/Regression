import csv
import random

with open('quad_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(1, 1001)
		y = x*x
		w.writerow([x, y])
		
with open('lin_values.csv', 'w', newline='') as csvfile:
	w = csv.writer(csvfile, delimiter=',')
	for i in range(10000):
		x = random.randint(1, 1001)
		y = x+1
		w.writerow([x, y])