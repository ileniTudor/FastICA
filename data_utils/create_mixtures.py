import numpy as np


def mix_sources(mixtures: list, apply_noise:bool = False):
    # normalize data
    for i in range(len(mixtures)):
        max_val = np.max(mixtures[i])
        if max_val > 1 or np.min(mixtures[i]) < 1:
            # continue
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5
    X = np.c_[[mix for mix in mixtures]]
    if apply_noise:
        # add random noise
        X += 0.02 * np.random.normal(size=X.shape)
    return X
