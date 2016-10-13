import os, struct
from pylab import *
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
warnings.simplefilter('error')



def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    
    image=[]
    for i in xrange(len(images)):
        k=[]
        for j in xrange(28):
            for l in xrange(28):
                k.append(1.0*images[j][l]/255)
        k.append(1)
        image.append(k)
    return np.array(image),np.array(labels)  

import warnings
warnings.filterwarnings('error')

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    for i in xrange(len(x)):
        for j in xrange(len(x[i])):
            try:
                x[i][j]=1/(1+np.exp(-x[i][j]))
            except RuntimeWarning:
                x[i][j]=0.0
    return x


trainX, trainy = load_mnist('training')

itrain2=[]
ltrain2=[]
k=0
for i in xrange(60000):
    if ltrain2.count(trainy[i])<200:
        itrain2.append(trainX[i])
        ltrain2.append(trainy[i])
        k+=1

trainX, trainy= np.array(itrain2),np.array(ltrain2)

testX, testy = load_mnist('testing')


syn0=[]
syn1=[]
eta=[0.008,0.008,0.008,0.08]
for i in xrange(4):
    np.random.seed(1)
    #no of hidden layer nodes
    hidden=600
    # randomly initialize our weights with mean 0(between -1 and 1)
    syn0.append(2*np.random.random((785,hidden))-1)
    syn1.append(2*np.random.random((hidden,1))-1)
    for j in xrange(1000):

        # Feed forward through layers 0, 1, and 2
        l0 = trainX
        l1 = nonlin(np.dot(l0,syn0[i]))
        l2 = nonlin(np.dot(l1,syn1[i]))

        # how much did we miss the target value?
        l2_error = ((trainy>>i)&1) - l2
        
        if (j% 1) == 0:
            print "Bit= "+str(i)+" iteration= "+str(j)+" Error:" + str(np.mean(np.abs(l2_error)))
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1[i].T)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1[i] += eta[i]*l1.T.dot(l2_delta)
        syn0[i] += eta[i]*l0.T.dot(l1_delta)
        



def change(x):
    for i in xrange(len(x)):
        if(x[i]>0.5):
            x[i]=1
        else:
            x[i]=0
    return x

l2=""
for i in xrange(4):
    l0 = testX
    l1 = nonlin(np.dot(l0,syn0[i]))
    if l2=="":
        l2 = change(nonlin(np.dot(l1,syn1[i])))
    else:
        l2=(change(nonlin(np.dot(l1,syn1[i])))*2)+l2
      
confmat = []
for i in range (0, 10):
    m=[]
    for j in range (0, 10):
        m.append(0)
    confmat.append(m)
correct=0
incorrect=0
warnings.filterwarnings("ignore")

for i in xrange(len(l2)):
    if int(l2[i][0])==testy[i][0]:
        correct+=1
    else:
        incorrect+=1
    if l2[i]>9:
        l2[i]=9
    confmat[testy[i][0]][int(l2[i][0])]+=1
print "acc="+str(1.0*correct/(correct+incorrect))

for i in confmat:
    for j in i:
        print j,
    print
    

