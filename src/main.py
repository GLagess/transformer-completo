import numpy as np

from .attention import scaled_dot_product_attention
from .ffn import ffn
from .layernorm import add_and_norm
from .causal_mask import create_causal_mask
from .vocab import get_toy_vocab, FRASE_ENCODER
from .model import build_full_model, run_encoder, run_decoder
from .inference import run_autoregressive_inference

SEED = 42


def tarefa1():
    print("Tarefa 1 - Refatoracao")
    rng = np.random.default_rng(SEED)
    batch, seq, d_model, d_k, d_ff = 1, 3, 4, 2, 8
    x = rng.standard_normal((batch, seq, d_model)).astype(np.float64) * 0.1
    Q = rng.standard_normal((batch, seq, d_k)).astype(np.float64) * 0.1
    K = rng.standard_normal((batch, seq, d_k)).astype(np.float64) * 0.1
    V = rng.standard_normal((batch, seq, d_model)).astype(np.float64) * 0.1
    out_attn = scaled_dot_product_attention(Q, K, V, mask=None)
    x = add_and_norm(x, out_attn)
    mask = create_causal_mask(seq)
    _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    W1 = rng.standard_normal((d_model, d_ff)).astype(np.float64) * 0.1
    b1 = np.zeros(d_ff)
    W2 = rng.standard_normal((d_ff, d_model)).astype(np.float64) * 0.1
    b2 = np.zeros(d_model)
    x = add_and_norm(x, ffn(x, W1, b1, W2, b2))
    print("  ok - attention, ffn, add_and_norm integrados\n")


def tarefa2():
    print("Tarefa 2 - Encoder")
    token_to_id, _, vocab_size = get_toy_vocab()
    rng = np.random.default_rng(SEED)
    model = build_full_model(vocab_size, rng)
    enc_ids = [token_to_id["Thinking"], token_to_id["Machines"]]
    Z = run_encoder(np.array([enc_ids], dtype=np.int64), model["enc_emb"], model["encoder_params"])
    print("  entrada:", FRASE_ENCODER)
    print("  Z shape:", Z.shape, "\n")


def tarefa3():
    print("Tarefa 3 - Decoder")
    token_to_id, _, vocab_size = get_toy_vocab()
    rng = np.random.default_rng(SEED)
    model = build_full_model(vocab_size, rng)
    enc_ids = [token_to_id["Thinking"], token_to_id["Machines"]]
    Z = run_encoder(np.array([enc_ids], dtype=np.int64), model["enc_emb"], model["encoder_params"])
    dec_ids = [token_to_id["<START>"], token_to_id["w5"]]
    logits, probs = run_decoder(
        np.array([dec_ids], dtype=np.int64), Z,
        model["dec_emb"], model["decoder_params"], model["W_proj"],
    )
    print("  logits shape:", logits.shape, "probs shape:", probs.shape, "\n")


def tarefa4():
    print("Tarefa 4 - Inferencia")
    rng = np.random.default_rng(SEED)
    _, _, vocab_size = get_toy_vocab()
    model = build_full_model(vocab_size, rng)
    frase = run_autoregressive_inference(model, FRASE_ENCODER, max_steps=20, rng=rng)
    print("  encoder input:", FRASE_ENCODER)
    print("  frase gerada:", frase, "\n")


def main():
    np.random.seed(SEED)
    tarefa1()
    tarefa2()
    tarefa3()
    tarefa4()


if __name__ == "__main__":
    main()
