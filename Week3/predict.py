import numpy as np
from sigmoid import sigmoid


def predict(theta, X, threshold=0.5):
    p = sigmoid(np.dot(X, theta.T)) >= threshold
    return p.astype('int')