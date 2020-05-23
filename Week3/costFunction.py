import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(np.dot(X, theta))
    J = (1 / m) * (-np.dot(np.log(h).T, y) - np.dot(np.log(1 - h).T, (1 - y)))
    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


def gradient(theta, X, y):
    ''' calculate gradient descent for logistic regression'''
    m = y.size
    theta = theta.reshape(-1, 1)
    h = sigmoid((np.dot(X, theta)))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return (grad.flatten())  # returns copy of array in one dimension
