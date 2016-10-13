import csv
import numpy as np
import matplotlib
import math
from matplotlib import pyplot as plt
from scipy import linalg as LA

def plotter(transformed , answer , high , low):
	x = np.array(transformed[0])
	y = np.array(transformed[1])
	for i in xrange(len(x)):
		if answer[i] == 'Iris-setosa':
			plt.plot(x[i],y[i],'bo')
		elif answer[i] == 'Iris-versicolor':
			plt.plot(x[i],y[i],'go')
		else:
			plt.plot(x[i],y[i],'ro')
	
	red_dot, = plt.plot(1, "ro", markersize=5)
	green_dot, = plt.plot(1, "go", markersize=5)
	blue_dot, = plt.plot(1, "bo", markersize=5)
	plt.legend([green_dot, blue_dot, red_dot], ["Iris-versicolor", "Iris-setosa" , "Iris-Verginica"])

	plt.title("PCA Projections")
	plt.xlabel("PCA" + low)
	plt.ylabel("PCA" + high)
	plt.show()

input_file = open('iris.data' , 'rt')
reader = csv.reader(input_file)
reader = list(reader)
A = list()
answer = list()
for i in reader:
	answer.append(i[-1])
	i = i[:-1]
	for j in xrange(len(i)):
		i[j] = float(i[j])
	A.append(i)
L = np.array(A)
A = np.matrix(A)
B = A.transpose()
new = B*A
eigenvalues, eigenvectors = LA.eig(new)
eigv = list()
for i in xrange(len(eigenvalues)):
	eigv.append((eigenvalues[i],eigenvectors[i]))
eigv.sort(key=lambda tup: tup[0] , reverse = True)
new = list()
print eigv
for i in eigv:
	new.append(i[1])
new = np.array(new)

# PCA2 VS PCA1 

transformed = L.dot(new[:2].T)
transformed =  transformed.T
plotter(transformed , answer , '2' , '1')

# PCA3 vs PCA2
transformed = L.dot(new[1:3].T)
transformed =  transformed.T
plotter(transformed , answer ,'3','2')

#PCA3 vs PCA1
arr = list()
arr.append(new[0])
arr.append(new[2])
arr = np.array(arr)
transformed = L.dot(arr.T)
transformed = transformed.T
plotter(transformed , answer , '3' , '1')