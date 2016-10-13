import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv(
    filepath_or_buffer='iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


X = df.ix[:,0:4].values
y = df.ix[:,4].values

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("variance",var_exp)

matrix_wpc12 = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix Wpc12:\n', matrix_wpc12)
Ypc12 = X_std.dot(matrix_wpc12)
x1=Ypc12[:,0].tolist()
y1=Ypc12[:,1].tolist()
use_colours = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
plt.scatter(x1,y1,c=[use_colours[x] for x in y.tolist()])
plt.title("PC12")
plt.show()

matrix_wpc13 = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[2][1].reshape(4,1)))
print('Matrix Wpc13:\n', matrix_wpc13)
Ypc13 = X_std.dot(matrix_wpc13)
x1=Ypc13[:,0].tolist()
y1=Ypc13[:,1].tolist()
use_colours = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
plt.scatter(x1,y1,c=[use_colours[x] for x in y.tolist()])
plt.title("PC13")
plt.show()

matrix_wpc23 = np.hstack((eig_pairs[1][1].reshape(4,1), 
                      eig_pairs[2][1].reshape(4,1)))
print('Matrix Wpc23:\n', matrix_wpc23)
Ypc23 = X_std.dot(matrix_wpc23)
x1=Ypc23[:,0].tolist()
y1=Ypc23[:,1].tolist()
use_colours = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
plt.scatter(x1,y1,c=[use_colours[x] for x in y.tolist()])
plt.title("PC23")
plt.show()








