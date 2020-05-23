import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(np.dot(X, theta))
    theta_J = theta[1:]
    regparameter = (reg/(2*m))*np.sum(np.square(theta_J)) # the value added to the cost function
    J = -1*(1/m)*(np.dot((np.log(h)).T, y) + np.dot((np.log(1-h)).T,(1-y))) + regparameter
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def gradientReg(theta, reg, X, y):
    '''returns the gradient for input values of theta, reg, X and y'''
    m = y.size
    theta = theta.reshape(-1,1)
    h = sigmoid(np.dot(X, theta))
    grad = (1/m)*np.dot(X.T, h-y) + (reg/m)*np.r_[[[0]], theta[1:]]
    return(grad.flatten())