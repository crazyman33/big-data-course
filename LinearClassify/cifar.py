import cPickle
import numpy as np

def unpickle(file):
    fo = open(file,'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_CIFAR10(file):
    #get the training data
    dataTrain = []
    labelTrain = []
    for i in range(1,6):
        dic = unpickle(file+"\\data_batch_"+str(i))
    for item in dic["data"]:
        dataTrain.append(item)
    for item in dic["labels"]:
        labelTrain.append(item)

    #get test data
    dataTest = []
    labelTest = []
    dic = unpickle(file+"\\test_batch")
    for item in dic["data"]:
        dataTest.append(item)
    for item in dic["labels"]:
        labelTest.append(item)
    return (dataTrain,labelTrain,dataTest,labelTest)