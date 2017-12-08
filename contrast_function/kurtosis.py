import numpy as np


def fn_and_der(u,args):
    return 1/4 * np.power(u,4),np.power(u,3)