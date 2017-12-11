import numpy as np


def mix_sources(mixtures:list):
    # normalize data
    for i,mix in enumerate(mixtures):
        if np.max(mix) > 1 or np.min(mix) < 1:
            mixtures[i] = mix / (np.max(mix)/2) - 1.0
    X = np.c_[[mix for mix in mixtures]].T
    # add random noise
    X += 0.2 * np.random.normal(size=X.shape)
    return X