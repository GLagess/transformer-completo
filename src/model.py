import numpy as np

from .embeddings import obter_embeddings, positional_encoding
from .encoder import encoder_stack
from .decoder import decoder_stack, project_to_vocab, decoder_logits_to_probs

D_MODEL = 512
D_K = 64
D_FF = 2048
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
SEED = 42


def _init_attention_weights(d_model, d_k, rng):
    Wq = rng.standard_normal((d_model, d_k)).astype(np.float64) * 0.1
    Wk = rng.standard_normal((d_model, d_k)).astype(np.float64) * 0.1
    Wv = rng.standard_normal((d_model, d_model)).astype(np.float64) * 0.1
    return Wq, Wk, Wv


def _init_ffn_weights(d_model, d_ff, rng):
    W1 = rng.standard_normal((d_model, d_ff)).astype(np.float64) * 0.1
    b1 = np.zeros(d_ff)
    W2 = rng.standard_normal((d_ff, d_model)).astype(np.float64) * 0.1
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


def build_encoder_params(n_layers, d_model, d_k, d_ff, rng):
    layer_params = []
    for _ in range(n_layers):
        Wq, Wk, Wv = _init_attention_weights(d_model, d_k, rng)
        W1, b1, W2, b2 = _init_ffn_weights(d_model, d_ff, rng)
        layer_params.append((Wq, Wk, Wv, W1, b1, W2, b2))
    return layer_params


def build_decoder_params(n_layers, d_model, d_k, d_ff, rng):
    layer_params = []
    for _ in range(n_layers):
        Wq_self, Wk_self, Wv_self = _init_attention_weights(d_model, d_k, rng)
        Wq_cross, Wk_cross, Wv_cross = _init_attention_weights(d_model, d_k, rng)
        W1, b1, W2, b2 = _init_ffn_weights(d_model, d_ff, rng)
        layer_params.append(
            (Wq_self, Wk_self, Wv_self, Wq_cross, Wk_cross, Wv_cross, W1, b1, W2, b2)
        )
    return layer_params


def run_encoder(encoder_input_ids, enc_emb_matrix, encoder_params):
    X = obter_embeddings(encoder_input_ids, enc_emb_matrix)
    seq_len = X.shape[1]
    X = X + positional_encoding(seq_len, D_MODEL)
    Z = encoder_stack(X, encoder_params)
    return Z


def run_decoder(decoder_input_ids, Z, dec_emb_matrix, decoder_params, W_proj):
    Y = obter_embeddings(decoder_input_ids, dec_emb_matrix)
    seq_len = Y.shape[1]
    Y = Y + positional_encoding(seq_len, D_MODEL)
    out = decoder_stack(Y, Z, decoder_params)
    logits = project_to_vocab(out, W_proj)
    probs = decoder_logits_to_probs(logits)
    return logits, probs


def build_full_model(vocab_size, rng):
    enc_emb = rng.standard_normal((vocab_size, D_MODEL)).astype(np.float64) * 0.1
    dec_emb = rng.standard_normal((vocab_size, D_MODEL)).astype(np.float64) * 0.1

    encoder_params = build_encoder_params(
        N_ENCODER_LAYERS, D_MODEL, D_K, D_FF, rng
    )
    decoder_params = build_decoder_params(
        N_DECODER_LAYERS, D_MODEL, D_K, D_FF, rng
    )
    W_proj = rng.standard_normal((D_MODEL, vocab_size)).astype(np.float64) * 0.1

    return {
        "enc_emb": enc_emb,
        "dec_emb": dec_emb,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "W_proj": W_proj,
    }
