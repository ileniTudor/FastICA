import numpy as np


def sqrt_root(x):
    return np.sqrt(np.sum(x * x))


def compute_distance(x, y):
    x1=np.abs(x)
    y1=np.abs(y)
    return 1- (np.sum(x1 * y1) / (sqrt_root(x1) * sqrt_root(y1)))


if __name__ == "__main__":
    assert np.round(compute_distance(np.array([3, 45, 7, 2]), np.array([2, 54, 13, 15])), 3) == 0.972
