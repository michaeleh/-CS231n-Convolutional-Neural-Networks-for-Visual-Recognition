import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    # scores function
    scores = X.dot(W)

    # read correct scores into a column array of height N
    correct_score = scores[list(range(num_train)), y]
    correct_score = correct_score.reshape(num_train, -1)

    scores += 1 - correct_score

    # make sure correct scores themselves don't contribute to loss function
    scores[list(range(num_train)), y] = 0

    # construct loss function
    loss = np.sum(np.fmax(scores, 0)) / num_train
    loss += reg * np.sum(W * W)

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    X_mask = np.zeros(scores.shape)
    # 1 for every score bigger than zero
    X_mask[scores > 0] = 1
    # subtracting from y places
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
    # increasing X by margin
    dW = X.T.dot(X_mask)

    dW /= num_train
    dW += 2 * reg * W

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
