import numpy as np

from .attention import scaled_dot_product_attention
from .layernorm import add_and_norm
from .ffn import ffn


def encoder_block(x, Wq, Wk, Wv, W1, b1, W2, b2):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    attn_out = scaled_dot_product_attention(Q, K, V, mask=None)
    x = add_and_norm(x, attn_out)

    ffn_out = ffn(x, W1, b1, W2, b2)
    x = add_and_norm(x, ffn_out)
    return x


def encoder_stack(x, layer_params):
    for params in layer_params:
        x = encoder_block(x, *params)
    return x
