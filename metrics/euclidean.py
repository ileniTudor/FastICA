import numpy as np


def compute_distance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))
