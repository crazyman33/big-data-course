##北方工业大学人工智能系张远，考试参考程序，请同学们不要上传网络进行传播
import csv
import numpy as np
print('开始执行')

#读入训练数据
with open('training_data.csv', newline='') as f:
    reader = csv.reader(f)
    train_data = []
    for item in reader:
        if (reader.line_num == 1):
            continue
        train_data.append(item)
train_y = np.arange(len(train_data))
for idx in range(len(train_data)):
    if train_data[idx][4] == 'setosa':
        train_y[idx] = 1
    if train_data[idx][4] == 'versicolor':
        train_y[idx] = 0
train_x=np.zeros((len(train_data),4))
for idx in range(len(train_data)):
    train_x[idx]=np.array(train_data[idx][0:4],dtype=float)

#读入测试数据
with open('test_data.csv', newline='') as f:
    reader = csv.reader(f)
    test_data = []
    for item in reader:
        if (reader.line_num == 1):
            continue
        test_data.append(item)
test_y = np.arange(len(test_data))
for idx in range(len(test_data)):
    if test_data[idx][4] == 'setosa':
        test_y[idx] = 1
    if test_data[idx][4] == 'versicolor':
        test_y[idx] = 0
test_x=np.zeros((len(test_data),4))
for idx in range(len(test_data)):
    test_x[idx]=np.array(test_data[idx][0:4],dtype=float)

#################################################################
#############同学们使用自己的分类器替换以下代码###################
#感知机分类器
class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.loss = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.output(xi))*self.d_sigmoid(self.z(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += (target - self.output(xi))*(target - self.output(xi))/2
            self.loss.append(errors)
        return self

    def output(self,X):
        return self.sigmoid(self.z(X))

    def z(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def d_sigmoid(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def predict(self, X):
        return np.where(self.output(X) >= 0.5, 1, 0)

    def test(self, X):
        return self.predict(X)

#使用分类器训练
pp = Perceptron(epochs=100, eta=0.01)
pp.train(train_x, train_y)
predict_y=pp.test(test_x)
#使用分类器测试
print('分类精度：', np.mean( predict_y == test_y )*100, '%')
print('程序结束')

##############迭代过程展示，这部分不要求##########################
#Loss
import matplotlib.pyplot as plt
plt.plot(range(1, len(pp.loss)+1), pp.loss, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
##北方工业大学人工智能系张远，考试参考程序，请同学们不要上传网络进行传播