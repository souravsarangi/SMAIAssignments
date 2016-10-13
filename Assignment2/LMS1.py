import numpy as np
import matplotlib.pyplot as plt
import math

def getValue(a,i): # a equation of line , b is point
    return a[0]*i[0]+a[1]*i[1]+a[2]

w1 = [(1,6),(7,2),(8,9),(9,9),(4,8),(8,5)]
w2 = [(2,1),(3,3),(2,4),(7,1),(1,3),(5,2)]

data = [[],[],[]]
for i in w1:
	data[0].append(1)
	data[1].append(i[0])
	data[2].append(i[1])
for i in w2:
	data[0].append(-1)
	data[1].append(-i[0])
	data[2].append(-i[1])

a = [3.0,10.0,1.0] #initial
b = 1
eta = 0.028
theta = 0.01
i,k = 0,0

while 1:
	i = i+1
	flag = 0
	compute = [0,0,0]
	for k in xrange(12):
		point = [data[0][k], data[1][k], data[2][k]]

		compute[0] = (eta/(i))* ( (b  - (a[0]*point[0] + a[1]*point[1] + a[2]*point[2]) )*point[0])
		compute[1] = (eta/(i)) * ( (b  - (a[0]*point[0] + a[1]*point[1] + a[2]*point[2]) )*point[1])
		compute[2] = (eta/(i)) * ( (b  - (a[0]*point[0] + a[1]*point[1] + a[2]*point[2]) )*point[2])

		a[0] = a[0] + compute[0]
		a[1] = a[1] + compute[1]
		a[2] = a[2] + compute[2]
		if math.sqrt(compute[0]**2+compute[1]**2+compute[2]**2)>theta:
			flag=1
	
	if flag==0:
		break

print i
predict = [(8,8),(6,6),(2,2),(0,0),(4,0),(4,4)]
for i in predict:
	print getValue(a,i)


x1 = 0 
y1 = -((a[1]*x1) + a[0])/a[2]
x2 = 10
y2 = -((a[1]*x2) + a[0])/a[2]
#a[0]x+a[1]y+a[2]=0

#x1 = 0.0
#y1 = (-a[2]-a[0]*x1)/a[1]

#y2 = 10.0
#x2 = (-a[2]-a[1]*y2)/a[0] 


############################# plot curve ######################################
print x1, y1, x2,y2
fig = plt.figure()
fig.suptitle('Widrow-Hoff (LMS) Procedure', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.plot([x1,x2],[y1,y2],color='black')
type1 = ax.scatter(*zip(*w1),color='red')
type2 = ax.scatter(*zip(*w2))
type3 = ax.scatter(*zip(*predict),color="yellow")
ax.legend([type1, type2,type3], ["Class w1","Class w2","to predict"], loc=2)
ax.text(0.95, 0.01, 'Weight Vector is :%s '%a,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=10)
plt.savefig('LMS.png')
