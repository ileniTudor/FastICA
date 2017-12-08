from metrics import euclidean, manhattan, chebyshev, cosine


def get_similarity_metric_fn(name: str):
    functions_map = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
        'cosine':cosine
    }
    # values = [v for v in functions_map.values()]
    assert name in [v for v in functions_map.keys()]

    def metric_fn(x,y):
        return functions_map[name].compute_distance(x,y)

    return metric_fn