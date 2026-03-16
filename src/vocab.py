TOKEN_START = "<START>"
TOKEN_EOS = "<EOS>"
TOKEN_PAD = "<pad>"
FRASE_ENCODER = "Thinking Machines"


def get_toy_vocab():
    tokens = [
        TOKEN_START,
        TOKEN_EOS,
        TOKEN_PAD,
        "Thinking",
        "Machines",
        "Maquinas",
        "Pensantes",
        "w5",
        "w6",
        "w7",
    ]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for t, i in token_to_id.items()}
    return token_to_id, id_to_token, len(tokens)
