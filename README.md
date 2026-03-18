# Lab 4 - Transformer Encoder-Decoder

Implementacao do transformer completo (encoder + decoder) com frase toy "Thinking Machines" e inferencia ate <EOS>.

## Como rodar

Instalar dependencias:
```
py -m pip install -r requirements.txt
```
Rodar o lab:
```
py -m src.main
```
(ou `python -m src.main` se tiver no PATH.)

## Arquivos

- `src/attention.py` - scaled_dot_product_attention com mask opcional
- `src/ffn.py` - feed-forward position-wise (512 -> 2048 -> 512, ReLU)
- `src/layernorm.py` - layer norm e add_and_norm
- `src/causal_mask.py` - mascara causal pro decoder
- `src/encoder.py` - encoder_block e encoder_stack
- `src/decoder.py` - decoder_block, decoder_stack, linear + softmax
- `src/embeddings.py` - embeddings e positional encoding
- `src/vocab.py` - vocabulario toy
- `src/model.py` - montagem do modelo, run_encoder, run_decoder
- `src/inference.py` - loop auto-regressivo
- `src/main.py` - chama as 4 tarefas do lab

Partes geradas/complementadas com IA, revisadas por Gabriel Lages.
