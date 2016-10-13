import csv
from collections import defaultdict
from random import shuffle
import numpy as np
input_file = open('bank/bank.csv' , 'rt')
data = list()
success = list()
yes_rec = 0
no_rec =0
reader = csv.reader(input_file)
j=0
field_entries = next(reader)
field_entries = field_entries[0].split(';');
for i in xrange(len(field_entries)):
	data.append(defaultdict(int))
	success.append(defaultdict(int))

reader = list(reader)
reader_ = reader
accuracy = list()
iterations = 1
while(iterations):
	shuffle(reader_)
	testdata = reader_[0:len(reader_)/2]
	reader = reader_[len(reader_)/2:-1]
	for row in reader:
		row_data = row[0].split(';');
		row_data[-1] = row_data[-1].replace('"','')
		if row_data[-1] == "yes":
			yes_rec+=1
		else:
			no_rec+=1
		for i in xrange(len(row_data)-1):
			row_data[i] = row_data[i].replace('"','')
			'''
			if(i == 0):
				data[i][int(int(row_data[i])/5)] += 1
				if row_data[-1] == 'yes':
					success[i][int(int(row_data[i])/5)]+=1
			elif i==5:
				data[i][int(int(row_data[i])/500)] += 1
				if row_data[-1] == 'yes':
					success[i][int(int(row_data[i])/500)]+=1
			elif i==11:
				data[i][int(int(row_data[i])/50)] += 1
				if row_data[-1] == 'yes':
					success[i][int(int(row_data[i])/50)]+=1
			'''
			if i==0 or i==5 or i==11 or i==12 or i==14 or i==13:
				continue
			else:
				data[i][row_data[i]] +=1
				if row_data[-1] == 'yes':
					success[i][row_data[i]]+=1
	correct = 0
	yes_yes =0
	yes_no = 0
	no_no =0
	no_yes=0
	for i in xrange(17):
		if i==0 or i==5 or i==11 or i==12 or i==14 or i==16 or i==9 or i==10 or i==13:
			continue
		for key in success[i]:
			print "P('yes'/" + str(key) + ")=" + str(success[i][key]) + '/' + str(yes_rec)
	counter = 0
	for row in testdata:
		row_data = row[0].split(';')
		prob = 1
		prob_y = 1
		for i in xrange(len(row_data)-1):
			row_data[i] = row_data[i].replace('"','')
			if i==0 or i==5 or i==11 or i==12 or i==14 or i==16 or i==9 or i==10 or i==13:
				continue
			'''
			if i==0:
				val = int(int(row_data[i])/5)
				val = success[i][val]
				if val ==0:
					val = 0.01
					continue
				prob*= float(val)/yes_rec
			elif i==5:
				val = int(int(row_data[i])/500)
				val = success[i][val]
				if val ==0 :
					val =0.01
					continue
				prob*= float(val)/yes_rec
			elif i==11:
				val = int(int(row_data[i])/50)
				val = success[i][val]
				if val == 0:
					val =0.01
					continue
				prob*=float(val)/yes_rec
			'''
			val = success[i][row_data[i]]
			if val ==0:
				val =0.01
				continue
			prob*= float(val)/yes_rec
		prob*= yes_rec
		prob_y = prob
		prob = 1
		
		for i in xrange(len(row_data)-1):
			if i==0 or i==5 or i==11 or i==12 or i==14 or i==16 or i==9 or i==10 or i==13:
				continue
			'''if i==0:
				val = int(int(row_data[i])/5)
				val = data[i][val] - success[i][val]
				if val ==0:
					val = 0.01
					continue
				prob*= float(val)/no_rec
			elif i==5:
				val = int(int(row_data[i])/500)
				val = data[i][val] - success[i][val]
				if val ==0 :
					continue
				prob*= float(val)/no_rec
			elif i==11:
				val = int(int(row_data[i])/50)
				val = data[i][val] - success[i][val]
				if val == 0:
					val =0.01
					continue
				prob*=float(val)/no_rec
			'''
			val = data[i][row_data[i]] - success[i][row_data[i]]
			if val ==0:
				continue
			prob*= float(val)/no_rec
		prob*= no_rec
		row_data[-1] = row_data[-1].replace('"','')
		if prob > prob_y:
			if row_data[-1] == 'no':
				correct+=1
				no_no+=1
			else:
				yes_no+=1
				if counter < 3:
					print row_data
					counter+=1
		else:
			if row_data[-1] == 'yes':
				correct+=1
				yes_yes+=1
			else:
				no_yes+=1
				if counter < 3:
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




