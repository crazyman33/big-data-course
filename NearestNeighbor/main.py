# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:37:50 2016

@author: HDU
"""
import matplotlib.pyplot as plt
import numpy as np
import cifar

datatr, labeltr, datate, labelte = cifar.load_CIFAR10("../cifar-10-batches-py/")

# print "Xte:%d" %(len(dataTest))
# print "Yte:%d" %(len(labelTest))
Xtr = np.asarray(datatr)
Xte = np.asarray(datate)
Ytr = np.asarray(labeltr)
Yte = np.asarray(labelte)
print Xtr.shape
print Xte.shape
print Ytr.shape
print Yte.shape
print type(Xtr)

num_training = 500
mask = range(num_training)
Xtr = Xtr[mask]
Ytr = Ytr[mask]

num_test = num_training
mask = range(num_test)
Xte = Xte[mask]
Yte = Yte[mask]

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072
Xtr_rows = np.reshape(Xtr, (Xtr.shape[0], -1))
Xte_rows = np.reshape(Xte, (Xte.shape[0], -1))
Xtr_rows=Xtr_rows.astype(np.float32)
Xte_rows=Xte_rows.astype(np.float32)
Ytr=Ytr.astype(np.float32)
Yte=Yte.astype(np.float32)
if 1:
    import NearestNeighbor;
    nn = NearestNeighbor.NearestNeighbor()  # create a Nearest Neighbor classifier class
    nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
    Yte_predict = nn.predict(Xte_rows)  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print 'accuracy: %f' % (np.mean(Yte_predict == Yte))

if 0:
    import KNearestNeighbor
    knn = KNearestNeighbor.KNearestNeighbor()

    # assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
    # recall Xtr_rows is 50,000 x 3072 matrix
    num_val=num_training/5
    Xval_rows = Xtr_rows[:num_val, :]  # take first 1000 for validation
    Yval = Ytr[:num_val]
    Xtr_rows = Xtr_rows[num_val:, :]  # keep last 49,000 for train
    Ytr = Ytr[num_val:]

    # find hyperparameters that work best on the validation set
    validation_accuracies = []
    for k in [1, 3, 5, 10, 20, 50, 100]:
        # use a particular value of k and evaluation on validation data
        knn.train(Xtr_rows, Ytr)
        # here we assume a modified NearestNeighbor class that can take a k as input
        Yval_predict = knn.predict(Xval_rows, k=k)
        acc = np.mean(Yval_predict == Yval)
        print 'accuracy : %f' % (acc,)

        # keep track of what works on the validation set
        validation_accuracies.append((k, acc))

    knn.train(Xtr_rows, Ytr)
    Yte_predict = knn.predict(Xte_rows,k=10,num_loops=2)  # predict labels on the test images
    print 'accuracy: %f' % (np.mean(Yte_predict == Yte))
