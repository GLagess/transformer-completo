import numpy as np


def obter_embeddings(ids, emb_matrix):
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[np.newaxis, :]
    return emb_matrix[ids]


def positional_encoding(seq_len, d_model):
    pe = np.zeros((1, seq_len, d_model), dtype=np.float64)
    position = np.arange(seq_len)[:, np.newaxis].astype(np.float64)
    div_term = np.exp(np.arange(0, d_model, 2).astype(np.float64) * (-np.log(10000.0) / d_model))
    pe[0, :, 0::2] = np.sin(position * div_term)
    pe[0, :, 1::2] = np.cos(position * div_term)
    return pe
