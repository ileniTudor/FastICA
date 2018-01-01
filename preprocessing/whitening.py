import numpy as np


def whitening(x):
    # subtract the mean of t to have a centered input matrix
    X_mean = x.mean(axis=-1)
    x -= X_mean[:, np.newaxis]

    cov = np.cov(x)

    # Calculate eigenvalues and eigenvectors of the covariance matrix.
    d, E = np.linalg.eigh(cov)

    # Generate a diagonal matrix with the eigenvalues as diagonal elements.
    D = np.diag(d)
    # Inverse of sqrt of D
    Di = np.sqrt(np.linalg.inv(D))

    # Perform whitening. x_whiten is the whitened matrix.
    x_whiten = np.dot(E, np.dot(Di, np.dot(E.T, x)))

    return x_whiten