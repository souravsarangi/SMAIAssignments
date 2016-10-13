import csv
import sys
from collections import defaultdict
from random import shuffle
import numpy as np
import matplotlib
import math
from matplotlib import pyplot as plt
from scipy import linalg as LA
import matplotlib.patches as mpatches

def plotter(transformed , answer):
	x = np.array(transformed[0])
	for i in xrange(len(x)):
		if answer[i] == 'Iris-setosa':
			plt.plot(x[i],0,'bo',markersize=15)
		elif answer[i] == 'Iris-versicolor':
			plt.plot(x[i],0,'go',markersize=15)
		else:
			plt.plot(x[i],0,'ro',markersize=15)
	red_dot, = plt.plot(1, "ro", markersize=5)
	green_dot, = plt.plot(1, "go", markersize=5)
	blue_dot, = plt.plot(1, "bo", markersize=5)

	plt.legend([green_dot, blue_dot, red_dot], ["Iris-versicolor", "Iris-setosa" , "Iris-Verginica"])
	plt.title("LDA Projections")
	plt.xlabel("LDA1")
	plt.ylabel("-")
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
A = np.array(A)
labels = ['Iris-setosa' , 'Iris-versicolor' ,'Iris-virginica']
means = list()

for i in labels:
	cond = [A[j] for j in xrange(len(A)) if answer[j]==i]
	means.append(np.mean(cond , axis=0))

mean_on_data = np.mean(A, axis=0)
scatter_between = np.zeros((4,4))
scatter_within = np.zeros((4,4))

for i,mean_vec in zip(labels,means):
	cond = np.array([A[j] for j in xrange(len(A)) if answer[j]==i])  
	n = cond.shape[0]
	mean_vec = mean_vec.reshape(4,1) 
	overall_mean = mean_on_data.reshape(4,1) 
	scatter_between += n * (mean_vec - mean_on_data).dot((mean_vec - mean_on_data).T)

for i,mv in zip(labels, means):
	class_sc_mat = np.zeros((4,4)) 
	cond = [A[j] for j in xrange(len(A)) if answer[j]==i]
	for row in cond:
		row, mv = row.reshape(4,1), mv.reshape(4,1) 
		class_sc_mat += (row-mv).dot((row-mv).T)
	scatter_within += class_sc_mat                             

eigenvalues, eigenvectors = LA.eig(LA.inv(scatter_within).dot(scatter_between))
eigv = list()
for i in xrange(len(eigenvalues)):
	eigv.append((eigenvalues[i],eigenvectors[i]))
eigv.sort(key=lambda tup: tup[0] , reverse = True)
print eigv

new = list()
for i in eigv:
	new.append(i[1])
new = np.array(new)
transformed = A.dot(new[:1].T)
transformed =  transformed.T
plotter(transformed,answer)