import numpy as np


def sqrt_root(x):
    return np.sqrt(np.sum(x * x))


def compute_distance(x, y):
    return 1- (np.sum(x * y) / (sqrt_root(x) * sqrt_root(y)))


if __name__ == "__main__":
    assert np.round(compute_distance(np.array([3, 45, 7, 2]), np.array([2, 54, 13, 15])), 3) == 0.972
