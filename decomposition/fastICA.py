"""
    The implementation of FastICA based in the paper:
    [1] 'Hyvarinen, Aapo. Fast and robust fixed-point algorithms for independent component analysis, IEEE transactions
    on Neural Networks 10.3 (1999): 626-634", where it was first introduced.
"""
import numpy as np
from contrast_function import  contrast_function_factory
from preprocessing.whitening import whitening
from metrics.distance_metrics_factory import get_similarity_metric_fn

def calculate_new_w(w, X):
    """
    :param w: the old value of w
    :return: a new value of v computed based on algorithm described in [1] pp.7
    :remark: may be used multiple contrast function ([1] pp. 6):
            -- 'logcosh' is used for general purpose
            -- 'exp' (exponential) may be more robust (suitable for super-Gaussian signals)
            -- 'kurtosis' (estimation of sub-Gaussian independent component)
    """
    fn_and_der = contrast_function_factory.get_contrast_function("exp")

    g, g_der = fn_and_der(np.dot(w.T, X), None)

    # gwtx, g_wtx = _exp(np.dot(w.T, X))
    w_new = (X * g).mean(axis=1) - g_der.mean() * w

    w_new /= np.sqrt((w_new ** 2).sum())

    return w_new


def ica(X, max_iter: int, tolerance: float = 1e-5, do_whitening: bool = True,verbose:bool=True):
    """
    :param X: input sources matrix. It is a NxM matrix. Where N is the number of rows corresponding to
              the number of components and M is represents the number of samples for each component.
    :param max_iter: if not converges faster, it will run max_iter steps for each component
    :param tolerance: used to check if two vectors are equal
    :param do_whitening: when True perform the whitening preprocessing
    :param verbose: write to console the progress of the decomposition

    :return: S - sources matrix, each row corresponding to a independent component
    """

    if do_whitening:
        X = whitening(X)
    components_nr = X.shape[0]
    if verbose:
        print('look for ', components_nr, 'components')
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    for i in range(components_nr):  # estimate the i'th component of W
        w = np.random.rand(components_nr)
        for j in range(max_iter):
            w_new = calculate_new_w(w, X)
            if i >= 1:
                # decorrelation of outputs w by Gram-Schmidth decorrelation algorithm ([1] pp. 9)
                # in order to prevent the new output w to converge to an already found maxima
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            distance_fn = get_similarity_metric_fn("euclidean")
            dist = distance_fn(w,w_new)
            w = w_new

            if dist < tolerance:
                break
        if verbose:
            print("w", i, "converges in", j, "steps")
        W[i, :] = w

    S = np.dot(W, X)
    return S