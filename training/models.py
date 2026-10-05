"""
The three temporal architectures under comparison.

All three share the interface ``build_model(name, input_dim, num_classes) -> keras.Model``
and are compiled with the identical optimiser, loss (with label smoothing) and
metrics, so any performance difference is attributable to the architecture.

    LSTM      : Input -> LSTM(128) -> LSTM(64) -> Dense -> Softmax
    GRU       : Input -> GRU(128)  -> GRU(64)  -> Dense -> Softmax
    GRU+MHA   : Input -> GRU(128) -> GRU(64) -> Multi-Head Self-Attention
                        -> temporal pooling -> Dense -> Softmax

The attention block is a real, trainable ``keras.layers.MultiHeadAttention``
layer with a residual connection and layer normalisation - not a placeholder.
"""

from __future__ import annotations

from typing import Tuple

import config


def _keras():
    import tensorflow as tf

    return tf, tf.keras, tf.keras.layers


def build_lstm(input_dim: int, num_classes: int, seq_len: int | None = None):
    """Model A - LSTM baseline."""
    tf, keras, L = _keras()
    units = tuple(config.LSTM_UNITS)
    inp = keras.Input(shape=(None, input_dim), name="landmarks")
    x = L.LSTM(units[0], return_sequences=True, name="lstm_1")(inp)
    x = L.LSTM(units[1], name="lstm_2")(x)
    x = L.Dropout(config.DROPOUT, name="dropout")(x)
    x = L.Dense(config.DENSE_UNITS, activation="relu", name="dense")(x)
    out = L.Dense(num_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="lstm_baseline")


def build_gru(input_dim: int, num_classes: int, seq_len: int | None = None):
    """Model B - GRU baseline."""
    tf, keras, L = _keras()
    units = tuple(config.GRU_UNITS)
    inp = keras.Input(shape=(None, input_dim), name="landmarks")
    x = L.GRU(units[0], return_sequences=True, name="gru_1")(inp)
    x = L.GRU(units[1], name="gru_2")(x)
    x = L.Dropout(config.DROPOUT, name="dropout")(x)
    x = L.Dense(config.DENSE_UNITS, activation="relu", name="dense")(x)
    out = L.Dense(num_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="gru_baseline")


def build_gru_mha(input_dim: int, num_classes: int, seq_len: int | None = None):
    """Model C - proposed GRU + Multi-Head Self-Attention + temporal pooling."""
    tf, keras, L = _keras()
    units = tuple(config.GRU_UNITS)
    inp = keras.Input(shape=(None, input_dim), name="landmarks")
    x = L.GRU(units[0], return_sequences=True, name="gru_1")(inp)
    x = L.GRU(units[1], return_sequences=True, name="gru_2")(x)

    # ---- trainable multi-head self-attention (queries = keys = values = GRU states)
    attn = L.MultiHeadAttention(num_heads=config.ATTENTION_HEADS,
                                key_dim=config.ATTENTION_KEY_DIM,
                                name="multi_head_attention")(x, x)
    x = L.Add(name="attention_residual")([x, attn])
    x = L.LayerNormalization(name="attention_norm")(x)
    x = L.Dropout(config.DROPOUT, name="attention_dropout")(x)

    # ---- temporal pooling over the attended sequence
    avg = L.GlobalAveragePooling1D(name="temporal_avg_pool")(x)
    mx = L.GlobalMaxPooling1D(name="temporal_max_pool")(x)
    pooled = L.Concatenate(name="temporal_pool")([avg, mx])

    h = L.Dense(config.DENSE_UNITS, activation="relu", name="dense")(pooled)
    h = L.Dropout(config.DROPOUT, name="dropout")(h)
    out = L.Dense(num_classes, activation="softmax", name="softmax")(h)
    return keras.Model(inp, out, name="gru_multihead_attention")


BUILDERS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "gru_mha": build_gru_mha,
}


def build_model(name: str, input_dim: int | None = None, num_classes: int | None = None):
    """Build and compile one of the three models."""
    if name not in BUILDERS:
        raise KeyError(f"unknown model '{name}' (expected one of {sorted(BUILDERS)})")
    tf, keras, _ = _keras()
    input_dim = int(input_dim or config.MODEL_INPUT_DIM if config.MASK_ENABLED else input_dim or config.FEATURE_DIM)
    num_classes = int(num_classes or config.NUM_CLASSES)
    model = BUILDERS[name](input_dim, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model


def parameter_count(model) -> int:
    return int(model.count_params())


def architecture_table() -> str:  # pragma: no cover - reporting helper
    rows = [
        ("lstm", f"LSTM({config.LSTM_UNITS[0]}) -> LSTM({config.LSTM_UNITS[1]}) -> Dense({config.DENSE_UNITS}) -> softmax"),
        ("gru", f"GRU({config.GRU_UNITS[0]}) -> GRU({config.GRU_UNITS[1]}) -> Dense({config.DENSE_UNITS}) -> softmax"),
        ("gru_mha",
         f"GRU({config.GRU_UNITS[0]}) -> GRU({config.GRU_UNITS[1]}) -> MultiHeadAttention(heads={config.ATTENTION_HEADS}, "
         f"key_dim={config.ATTENTION_KEY_DIM}) + residual + LayerNorm -> [avg;max] temporal pooling -> "
         f"Dense({config.DENSE_UNITS}) -> softmax"),
    ]
    return "\n".join(f"  {n:8s} {d}" for n, d in rows)
