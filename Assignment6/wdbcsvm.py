from __future__ import print_function

from time import time
import logging
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC



train = pd.read_csv(
    filepath_or_buffer='wdbc.csv', 
    #header=None, 
    sep=',',
    low_memory=False)


print(train.columns)

y = train['Diagnosis']
X = train.drop(['ID','Diagnosis'],axis=1)

n_features = X.shape[1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

kf = KFold(train.shape[0], n_folds=5, random_state=1)

for trainkf, test in kf:
    X_train, X_test, y_train, y_test=X.iloc[trainkf,:],X.iloc[test,:],y.iloc[trainkf],y.iloc[test]
    
    n_components = 5

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    #eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))


    ###############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)


    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting Malignant or Not")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

