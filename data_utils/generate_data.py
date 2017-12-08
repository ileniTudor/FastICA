import numpy as np
from scipy import signal

def generate_toy_signals():
    np.random.seed(0)
    n_samples = 2000  # x axis
    time = np.linspace(0, 8, n_samples)  # y axis
    # signals
    s1 = np.sin(2 * time)  # sinusoidal
    s2 = np.sign(np.sin(3 * time))  # square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # saw tooth signal
    return s1, s2, s3