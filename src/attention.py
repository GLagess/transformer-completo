import numpy as np

EPS = 1e-9


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x - x_max, -50, 50))
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPS)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn_weights = softmax(scores, axis=-1)
    return np.matmul(attn_weights, V)
