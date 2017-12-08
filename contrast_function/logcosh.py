import numpy as np


def fn_and_der(u,args:dict):
    try:
        alph = args['alph']
    except:
        alph = 1.0
    return 1/alph * np.log(np.cosh(alph * u)),np.tanh(alph * u)