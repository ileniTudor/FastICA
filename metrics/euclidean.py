import numpy as np
from scipy.spatial import distance

def compute_distance(x, y):
    # print(x,y)
    x1=np.abs(x)
    y1=np.abs(y)
    return np.sqrt(np.sum(np.power(x1 - y1, 2)))
    # return distance.euclidean(x,y)