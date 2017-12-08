import numpy as np


def fn_and_der(u,args):
    exp = np.exp(- np.power(u, 2) / 2)
    g = u * exp
    g_der = (1 - np.power(u, 2)) * exp
    return g, g_der
