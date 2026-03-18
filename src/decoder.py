import numpy as np

from .attention import scaled_dot_product_attention, softmax
from .layernorm import add_and_norm
from .ffn import ffn
from .causal_mask import create_causal_mask


def decoder_block(y, Z, Wq_self, Wk_self, Wv_self, Wq_cross, Wk_cross, Wv_cross, W1, b1, W2, b2):
    seq_len = y.shape[1]
    causal_mask = create_causal_mask(seq_len)

    Qs = y @ Wq_self
    Ks = y @ Wk_self
    Vs = y @ Wv_self
    self_attn_out = scaled_dot_product_attention(Qs, Ks, Vs, mask=causal_mask)
    y = add_and_norm(y, self_attn_out)

    Qc = y @ Wq_cross
    Kc = Z @ Wk_cross
    Vc = Z @ Wv_cross
    cross_out = scaled_dot_product_attention(Qc, Kc, Vc, mask=None)
    y = add_and_norm(y, cross_out)

    ffn_out = ffn(y, W1, b1, W2, b2)
    y = add_and_norm(y, ffn_out)
    return y


def decoder_stack(y, Z, layer_params):
    for params in layer_params:
        y = decoder_block(y, Z, *params)
    return y


def project_to_vocab(decoder_out, W_proj):
    return decoder_out @ W_proj


def decoder_logits_to_probs(logits):
    return softmax(logits, axis=-1)
