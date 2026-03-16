import numpy as np


def ffn(x, W1, b1, W2, b2):
    hidden = x @ W1 + b1
    relu_out = np.maximum(0, hidden)
    return relu_out @ W2 + b2
