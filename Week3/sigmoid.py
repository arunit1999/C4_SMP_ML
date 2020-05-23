import numpy as np


def sigmoid(z):
    sgm = (1 / (1 + np.exp(-z)))
    return sgm