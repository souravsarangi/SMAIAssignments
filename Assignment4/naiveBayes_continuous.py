import csv
from collections import defaultdict
from random import shuffle
import numpy as np
import math as Math
input_file = open('wdbc.data' , 'rt')
data = list()
success = list()
yes_rec = 0
no_rec =0
reader = csv.reader(input_file)
j=0
reader = list(reader)
accuracy = list()
iterations = 10
e = Math.pow(2.718,-0.5)
denom_factor  = 1/Math.sqrt(2*3.1416)
reader_ = reader
while(iterations):
	shuffle(reader_)
	testdata = reader_[0:len(reader_)/2]
	reader = reader_[len(reader_)/2 : -1]
	total_features = len(reader[0])
	feature_list_M = list()
	feature_list_B = list()
	mean_list_M = list()
	mean_list_B = list()
	variance_list_M = list()
	variance_list_B = list()

	for i in xrange(total_features):
		feature_list_M.append(list())
		feature_list_B.append(list())
	#reader = reader[len(reader)/2:-1]
	for row in reader:
		row_data = row
		row_data[1] = row_data[1].replace('"','')
		if row_data[1] == "M":
			yes_rec+=1
		else:
			no_rec+=1
		for i in xrange(2,len(row_data)):
			row_data[i] = row_data[i].replace('"','')
			if row_data[1] == "M":
				feature_list_M[i].append(float(row_data[i]))
			else:
				feature_list_B[i].append(float(row_data[i]))
	
	for i in xrange(2,total_features):
		mean_list_M.append(np.mean(feature_list_M[i]))
		mean_list_B.append(np.mean(feature_list_B[i]))
		variance_list_M.append(np.std(feature_list_M[i]))
		variance_list_B.append(np.std(feature_list_B[i]))


	correct = 0
	yes_yes =0
	yes_no = 0
	no_no =0
	no_yes =0
	counter =0
	for row in testdata:
		row_data = row
		prob = 1
		prob_y = 1
		for i in xrange(2,len(row_data)):
			row_data[i] = row_data[i].replace('"','')
			val = float(row_data[i])
			val = denom_factor/variance_list_M[i-2] * Math.pow(e,Math.pow(((val-mean_list_M[i-2])/variance_list_M[i-2]),2))
			if val ==0:
				continue
			prob*=val
		prob*= yes_rec
		prob_y = prob
		prob = 1
		for i in xrange(2,len(row_data)):
			row_data[i] = row_data[i].replace('"','')
			val = float(row_data[i])
			val = denom_factor/variance_list_B[i-2] * Math.pow(e,Math.pow(((val-mean_list_B[i-2])/variance_list_B[i-2]),2))
			if val ==0:
				continue
			prob*=val
			
		prob*= no_rec
		row_data[1] = row_data[1].replace('"','')
		if prob > prob_y:
			if row_data[1] == 'B':
				correct+=1
				no_no+=1
			else:
				yes_no+=1
				if counter < 3:
					print row_data
					counter+=1
		else:
			if row_data[1] == 'M':
				correct+=1
				yes_yes+=1
			else:
				no_yes+=1
				if counter <3:
					print row_data
					counter+=1
	answer = float(correct)/len(testdata)
	accuracy.append(answer)
	iterations-=1
	print yes_yes , yes_no
	print no_yes , no_no
print accuracy
print np.mean(accuracy)
print np.std(accuracy)




