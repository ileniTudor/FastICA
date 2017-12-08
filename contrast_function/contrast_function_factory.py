from contrast_function import kurtosis
from contrast_function import exponential
from contrast_function import logcosh


def get_contrast_function(name: str):
    functions_map = {
        'logcosh': logcosh,
        'exp': exponential,
        'kurtosis': kurtosis
    }

    def contrast_fn(u,args):
        return functions_map[name].fn_and_der(u,args)

    return contrast_fn
