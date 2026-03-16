import numpy as np


def create_causal_mask(seq_len):
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask
