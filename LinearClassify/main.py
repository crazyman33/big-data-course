# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:37:50 2016

@author: HDU
"""
import matplotlib.pyplot as plt
import numpy as np
import cifar
import linear_classifier

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

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(Ytr == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        # plt.imshow(Xtr[idx].reshape(3,32,32)[0,:,:])
        plt.imshow(np.transpose(Xtr[idx].reshape(3, 32, 32), (1, 2, 0)))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

num_training = 10000
mask = range(num_training)
Xtr = Xtr[mask]
Ytr = Ytr[mask]

num_test = num_training
mask = range(num_test)
Xte = Xte[mask]
Yte = Yte[mask]

Xtr=Xtr.astype(np.float32)
Xte=Xte.astype(np.float32)

#PCA
D_new=15
if 1:
    mean_val=np.mean(Xtr, axis=0)
    std_val = np.std(Xtr, axis=0)
    Xtr -= mean_val
    Xtr /= std_val
    Xte -= mean_val
    Xte /= std_val

    cov = np.dot(Xtr.T, Xtr) / Xtr.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xtr = np.dot(Xtr, U[:,:D_new])
    Xtr = Xtr / np.sqrt(S[:D_new] + 1e-5)

    Xte = np.dot(Xte, U[:,:D_new])
    Xte = Xte / np.sqrt(S[:D_new] + 1e-5)

Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])

svm = linear_classifier.LinearSVM()
loss_hist = svm.train(np.transpose(Xtr, (1,0)), Ytr, learning_rate=0.001,
                      batch_size=500, num_iters=1000, verbose=True)
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

y_train_pred = svm.predict(np.transpose(Xte, (1,0)))
print 'test accuracy: %f' % (np.mean(Yte == y_train_pred),)

