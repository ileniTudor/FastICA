import numpy as np


def compute_distance(x,y):
    return np.abs(np.abs((x * y).sum()) - 1)