"""Version-tolerant shims for the two HF CLIP APIs that changed in transformers 5.

Every function here works on transformers 4.x *and* 5.x -- the sd15/sdxl venv and
the flow-model ``.venv-flux`` can be pinned differently without this drifting.
Nothing imports torch at module scope, so the tier split in CLAUDE.md holds.

* :func:`image_features` -- ``CLIPModel.get_image_features`` returned a plain
  ``(B, D)`` tensor through 4.x. In 5.x it returns a ``BaseModelOutputWithPooling``
  whose ``pooler_output`` holds the projected features (the vision tower's own
  pooled state is overwritten in place). Call sites just want the tensor.
* :func:`encoder_hidden_states` -- the text encoder is driven from
  ``inputs_embeds`` here (PEZ optimizes embeddings, not ids), which means going
  through ``text_model.encoder`` directly. That entry point lost
  ``causal_attention_mask`` and ``output_hidden_states`` in 5.x: it now takes the
  causal mask as ``attention_mask`` and returns only the last hidden state. So we
  drive the layer stack ourselves and collect every hidden state, dispatching on
  the layer signature -- 4.x wants ``(h, attention_mask, causal_attention_mask)``
  and returns a tuple, 5.x wants ``(h, attention_mask)`` and returns the tensor.

Both are additive ``-inf`` masks that land in the same ``softmax`` addend either
way, so :func:`causal_mask` needs no versioning.
"""

from __future__ import annotations

import inspect


def image_features(clip, inputs):
    """CLIP image embeddings as a ``(B, D)`` tensor (not L2-normalized)."""
    out = clip.get_image_features(**inputs)
    pooled = getattr(out, "pooler_output", None)     # 5.x: ModelOutput wrapper
    return out if pooled is None else pooled         # 4.x: already the tensor


def causal_mask(length, device, dtype=None):
    """Additive ``(1, 1, L, L)`` upper-triangular ``-inf`` mask."""
    import torch
    if dtype is None:
        dtype = torch.float32
    m = torch.full((length, length), float("-inf"), device=device, dtype=dtype)
    return m.triu(1)[None, None]


def _layer_takes_causal(layer) -> bool:
    return "causal_attention_mask" in inspect.signature(layer.forward).parameters


def encoder_hidden_states(text_model, seq, mask):
    """Run a CLIP text encoder over ``seq`` (already embeddings + positions).

    Returns the list of hidden states, embeddings-output first, exactly like
    ``output_hidden_states=True`` used to: ``[-1]`` is the final pre-layer-norm
    state, ``[-2]`` the penultimate layer diffusers conditions on.
    """
    layers = text_model.encoder.layers
    if not layers:
        return [seq]
    legacy = _layer_takes_causal(layers[0])          # checked once, not per layer
    hidden, h = [seq], seq
    for layer in layers:
        h = layer(h, None, mask) if legacy else layer(h, mask)
        if isinstance(h, tuple):                     # 4.x returned a 1-tuple
            h = h[0]
        hidden.append(h)
    return hidden
