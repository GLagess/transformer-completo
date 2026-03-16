import numpy as np

EPSILON = 1e-6


def layer_norm(x, epsilon=EPSILON):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True) + epsilon
    return (x - mean) / np.sqrt(var)


def add_and_norm(x, sublayer_output):
    return layer_norm(x + sublayer_output)
