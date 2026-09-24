"""
What the loaded model is made of, and where it looks when it reads a prompt.

model/transformer.py is frozen by the Stage 6.2 preregistration, so nothing here
changes it. CausalSelfAttention.forward keeps its attention weights in a local
variable (weights = F.softmax(scores)), which no hook can reach. Its q_proj,
k_proj and v_proj projections and the out_proj input are modules, though, so
forward hooks capture Q, K, V and the attention output of one ordinary forward
pass, and the weights are recomputed from Q and K with the same operations
forward() uses.

Every capture is checked before it is returned: weights @ V, merged across heads,
must equal bit for bit the tensor the model itself passed to out_proj. If any
block fails, the request fails; unverified weights are never shown.
"""

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from eclipse.inference import EclipseError, GenerationRequestError

MAX_ATTENTION_TOKENS = 30
MAX_ATTENTION_CHARS = 400


class AttentionCaptureError(EclipseError):
    """The recomputed weights did not reproduce the model's attention output."""


def _component(name):
    """Map a parameter name to (block index or None, component)."""
    parts = name.split(".")
    if parts[0] == "blocks":
        sub = parts[2]
        if sub == "attn":
            return int(parts[1]), "attention"
        if sub in ("ff_in", "ff_out"):
            return int(parts[1]), "feed_forward"
        return int(parts[1]), sub                     # ln_1, ln_2
    return None, parts[0]                             # token_embedding, position_embedding, ln_final


def describe_architecture(engine):
    """The loaded checkpoint's structure, with parameter counts taken from the model itself."""
    info = engine.info
    arch = info.architecture
    top = defaultdict(int)
    blocks = [defaultdict(int) for _ in range(arch["num_blocks"])]
    for name, param in engine.model.named_parameters():
        block, component = _component(name)
        (blocks[block] if block is not None else top)[component] += param.numel()

    counted = sum(top.values()) + sum(sum(b.values()) for b in blocks)
    if counted != info.parameters:
        raise EclipseError(f"parameter breakdown sums to {counted:,}, "
                           f"the model has {info.parameters:,}")
    return {
        "name": info.name,
        "checkpoint": info.checkpoint,
        "step": info.step,
        "parameters": info.parameters,
        "architecture": {**arch, "head_dim": arch["d_model"] // arch["num_heads"]},
        "architecture_sources": info.architecture_sources,
        "components": {
            "token_embedding": top["token_embedding"],
            "position_embedding": top["position_embedding"],
            "blocks": [{"ln_1": b["ln_1"], "attention": b["attention"],
                        "ln_2": b["ln_2"], "feed_forward": b["feed_forward"],
                        "total": sum(b.values())} for b in blocks],
            "ln_final": top["ln_final"],
            # The logits are x @ token_embedding.weight.T: no parameters of its own.
            "output_head": 0,
        },
    }


def _recompute(attn, q, k, v, T):
    """CausalSelfAttention.forward's weights and merged output, from its own Q, K, V."""
    H, Dh = attn.num_heads, attn.head_dim
    q = q.view(1, T, H, Dh).transpose(1, 2)
    k = k.view(1, T, H, Dh).transpose(1, 2)
    v = v.view(1, T, H, Dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    scores = scores + attn.causal_mask[:T, :T]
    weights = F.softmax(scores, dim=-1)
    merged = (weights @ v).transpose(1, 2).contiguous().view(1, T, H * Dh)
    return weights, merged


def capture_weights(engine, ids):
    """Verified attention weights, one [heads, T, T] tensor per block, for token ids."""
    T = len(ids)
    model = engine.model
    captured, handles = {}, []

    def keep_output(key):
        return lambda module, inputs, output: captured.__setitem__(key, output.detach())

    def keep_input(key):
        return lambda module, inputs: captured.__setitem__(key, inputs[0].detach())

    # The engine's lock keeps a concurrent generate() from running through the hooks.
    with engine._lock:
        try:
            for i, block in enumerate(model.blocks):
                for name in ("q_proj", "k_proj", "v_proj"):
                    handles.append(getattr(block.attn, name).register_forward_hook(
                        keep_output((i, name))))
                handles.append(block.attn.out_proj.register_forward_pre_hook(
                    keep_input((i, "merged"))))
            with torch.inference_mode():
                model(torch.tensor([ids], device=engine.device))
        finally:
            for handle in handles:
                handle.remove()

    blocks = []
    with torch.inference_mode():
        for i, block in enumerate(model.blocks):
            weights, merged = _recompute(block.attn, captured[(i, "q_proj")],
                                         captured[(i, "k_proj")], captured[(i, "v_proj")], T)
            if not torch.equal(merged, captured[(i, "merged")]):
                raise AttentionCaptureError(
                    f"block {i + 1}: recomputed attention does not reproduce the model's own "
                    "attention output; refusing to show unverified weights")
            blocks.append(weights[0])
    return blocks


def capture_attention(engine, prompt):
    """One forward pass over the prompt; every block's and head's attention weights."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationRequestError("the prompt is empty")
    if len(prompt) > MAX_ATTENTION_CHARS:
        raise GenerationRequestError(f"the prompt is {len(prompt):,} characters; the limit "
                                     f"for the attention view is {MAX_ATTENTION_CHARS}")
    ids = engine.tokenizer.encode(prompt)
    if len(ids) > MAX_ATTENTION_TOKENS:
        raise GenerationRequestError(f"the prompt is {len(ids)} tokens; the attention view "
                                     f"takes at most {MAX_ATTENTION_TOKENS}")
    blocks = [[[[round(w, 6) for w in row] for row in head] for head in weights.cpu().tolist()]
              for weights in capture_weights(engine, ids)]

    return {
        "prompt": prompt,
        "tokens": [engine.tokenizer.decode([i]) for i in ids],
        "token_ids": ids,
        "num_blocks": len(blocks),
        "num_heads": len(blocks[0]),
        "attention": blocks,                          # [block][head][query][key]
        "verified": True,
        "verification": ("for every block, these weights times V reproduced bit for bit "
                         "the attention output the model computed in this forward pass"),
        "rounding": "weights are rounded to 6 decimal places for display",
    }
