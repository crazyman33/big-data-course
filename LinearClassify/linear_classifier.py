# -*- coding: utf-8 -*-
import numpy as np

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    dim, num_train = X.shape

    f = np.dot(W,X)     # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=0), (num_train, 1))   # N by 1
    prob = np.exp(f - np.transpose(f_max, (1,0))) / np.sum(np.exp(f - np.transpose(f_max, (1,0))), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[y, range(num_train)] = 1.0    # N by C
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train+0.5 * reg * np.sum(W * W)#向量化直接操作即可
    dW += -np.dot(y_trueClass - prob, X.T) / num_train + reg * W

    return loss, dW
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.Inputs and outputs
    are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    scores = np.dot(W,X)        # N by C
    num_train = X.shape[1]
    num_classes = W.shape[0]
    scores_correct = scores[y, np.arange(num_train)]   # 1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
    margins = scores.transpose((1, 0)) - scores_correct + 1.0     # N by C
    margins[np.arange(num_train), y] = 0.0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    # compute the gradient
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum
    dW += np.transpose(np.dot(X, margins)/num_train, (1,0)) + reg * W     # D by C

    return loss, dW

class LinearClassifier:

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-1, num_iters=100,
              batch_size=100, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        dim, num_train = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            rand_idx = np.random.choice(num_train, batch_size)
            X_batch = X[:, rand_idx]
            y_batch = y[rand_idx]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.W += -1 * learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 10 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = np.dot(self.W, X)
        y_pred = scores.argmax(axis=0)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.
        Inputs:
        - X_batch: D x N array of data; each column is a data point.
        - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - reg: (float) regularization strength.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


