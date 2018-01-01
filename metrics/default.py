import numpy as np


def compute_distance(x,y):
    # print(x, y,"dist",np.abs(np.abs((x * y).sum()) - 1))
    return np.abs(np.abs((x * y).sum()) - 1)