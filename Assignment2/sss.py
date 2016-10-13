import numpy as np
import matplotlib.pyplot as plt

w1= [(1, 6), (7, 2), (8, 9), (9, 9), (4, 8), (8, 5)]
w2=[(2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]

a = np.random.rand(3)
iters = 0


while 1:
	iters += 1
	count = 0

	for i in w1:
		if (a[0]*i[0] + a[1]*i[1] + a[2]) < 0:
			count += 1
			a[0] += i[0]
			a[1] += i[1]
			a[2] += 1			

	for i in w2:
		if (a[0]*i[0] + a[1]*i[1] + a[2]) > 0:
			count += 1
			a[0] -= i[0]
			a[1] -= i[1]
			a[2] -= 1			

	print iters
	if count == 0:
		break

print iters
#line equation a[0]x + a[1]y + a[2] = 0
x1 = 0
y1 = -(a[2]*1.0/a[1])
x2 = 15
y2 = -(a[2] + 15*a[0])*1.0/a[1]
#x2 = -(a[2]*1.0/a[1])
#y2 = 0
#print x1, y1, x2,y2
fig = plt.figure()
fig.suptitle('Single Sample Perceptron', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.plot([x1,x2],[y1,y2],color='green')
type1 = ax.scatter(*zip(*w1),color='red')
type2 = ax.scatter(*zip(*w2))
ax.legend([type1, type2], ["Class w1","Class w2"], loc=2)
ax.text(0.95, 0.01, 'Weight Vector is :%s '%a,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=10)
plt.savefig('ssp.png')