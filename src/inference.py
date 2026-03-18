import numpy as np

from .model import run_encoder, run_decoder
from .vocab import get_toy_vocab, TOKEN_START, TOKEN_EOS, FRASE_ENCODER


def run_autoregressive_inference(model, encoder_phrase=FRASE_ENCODER, max_steps=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    token_to_id, id_to_token, _ = get_toy_vocab()
    enc_ids = [token_to_id[t] for t in encoder_phrase.strip().split() if t in token_to_id]
    if not enc_ids:
        enc_ids = [token_to_id.get("Thinking", 3), token_to_id.get("Machines", 4)]

    encoder_input = np.array([enc_ids], dtype=np.int64)
    Z = run_encoder(encoder_input, model["enc_emb"], model["encoder_params"])

    id_start = token_to_id[TOKEN_START]
    id_eos = token_to_id[TOKEN_EOS]
    context = [id_start]

    for _ in range(max_steps):
        decoder_input = np.array([context], dtype=np.int64)
        _, probs = run_decoder(
            decoder_input, Z,
            model["dec_emb"], model["decoder_params"], model["W_proj"],
        )
        next_probs = probs[0, -1, :].copy()
        if len(context) == 1:
            next_probs[id_eos] = 0.0
            next_probs = next_probs / next_probs.sum()
        next_id = int(np.argmax(next_probs))
        context.append(next_id)
        if next_id == id_eos:
            break

    tokens = [id_to_token.get(i, "w{0}".format(i)) for i in context[1:-1] if i != id_eos]
    return " ".join(tokens)
