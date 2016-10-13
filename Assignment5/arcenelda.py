import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv(
    filepath_or_buffer='arcene/arcene_train.data', 
    header=None, 
    sep='\s+')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
df.tail()
X = df.as_matrix()

y=[]
with open('arcene/arcene_train.labels',"r") as f: 
    k=f.readline()
    while k!='':
        y.append(int(k))
        k=f.readline()
y = np.array(y)


label_dict = {1: 'Cancer', -1: 'No Cancer'}

np.set_printoptions(precision=10000)

mean_vectors = []
for cl in range(1,10000):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))


S_W = np.zeros((10000,10000))
for cl,mv in zip(range(1,10000), mean_vectors):
    class_sc_mat = np.zeros((10000,10000))                  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(10000,1), mv.reshape(10000,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)

overall_mean = np.mean(X, axis=0)

S_B = np.zeros((10000,10000))
for i,mean_vec in enumerate(mean_vectors):  
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(10000,1) # make column vector
    overall_mean = overall_mean.reshape(10000,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(10000,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(10000,1), eig_pairs[1][1].reshape(10000,1)))
print('Matrix W:\n', W.real)

X_lda = X.dot(W)

from matplotlib import pyplot as plt

def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,10000),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()

