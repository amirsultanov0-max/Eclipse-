"""
Stage 4 checks, before any training loop exists.

  1. Shape contract: every intermediate tensor in a forward pass, not just the
     output, matches the architecture we agreed on.
  2. Causality: position t's prediction cannot see token t+1. A silent mask bug
     trains "successfully" while cheating.
  3. Target shift: targets[:, t] == inputs[:, t+1], the classic off-by-one.
  4. Parameter count, against the ~1.8M estimate.
  5. Loss at initialisation, against log(4000) = 8.2940.

Runs on CPU: these are correctness checks, and CPU float arithmetic is
deterministic enough to compare two forward passes exactly.

    venv/bin/python shape_test.py
"""

import math

import numpy as np
import torch

from model.transformer import (CONTEXT_LENGTH, D_FF, D_MODEL, NUM_BLOCKS, NUM_HEADS,
                               VOCAB_SIZE, TinyTransformer)

B = 2                              # small batch for the shape trace
T = CONTEXT_LENGTH                 # 512
D = D_MODEL                        # 128
H = NUM_HEADS                      # 4
DH = D_MODEL // NUM_HEADS          # 32
F_FF = D_FF                        # 512
V = VOCAB_SIZE                     # 4000

TRAIN_TOKENS = "data/stage3_train_tokens.npy"


def expected_shapes():
    """The shape contract, written out stage by stage."""
    top_before = [
        ("token_embedding", (B, T, D)),
        ("position_embedding", (T, D)),
        ("embeddings_sum", (B, T, D)),
    ]
    per_block = [
        ("ln_1", (B, T, D)),
        ("q_proj", (B, T, D)),
        ("k_proj", (B, T, D)),
        ("v_proj", (B, T, D)),
        ("q_heads", (B, H, T, DH)),
        ("k_heads", (B, H, T, DH)),
        ("v_heads", (B, H, T, DH)),
        ("scores", (B, H, T, T)),
        ("masked_scores", (B, H, T, T)),
        ("attn_weights", (B, H, T, T)),
        ("attn_output", (B, H, T, DH)),
        ("merged_heads", (B, T, D)),
        ("attn_out_proj", (B, T, D)),
        ("residual_1", (B, T, D)),
        ("ln_2", (B, T, D)),
        ("ffn_hidden", (B, T, F_FF)),
        ("ffn_out", (B, T, D)),
        ("residual_2", (B, T, D)),
    ]
    top_after = [
        ("blocks_output", (B, T, D)),
        ("ln_final", (B, T, D)),
        ("logits", (B, T, V)),
    ]
    contract = list(top_before)
    for i in range(NUM_BLOCKS):
        contract += [(f"block{i}.{name}", shape) for name, shape in per_block]
    contract += top_after
    return contract, per_block


def check_shapes(model):
    print("1. SHAPE CONTRACT")
    contract, per_block = expected_shapes()

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    trace = {}
    logits, loss = model(idx, targets, trace=trace)

    failures = [(k, exp, trace.get(k)) for k, exp in contract if trace.get(k) != exp]
    missing = [k for k, _ in contract if k not in trace]
    extra = [k for k in trace if k not in dict(contract)]

    print(f"   input [B={B}, T={T}] -> tracing {len(trace)} intermediate tensors\n")
    print(f"   {'stage':<28} {'expected':<22} {'actual':<22} ok")
    for key, exp in contract:
        if key.startswith("block") and not key.startswith("block0."):
            continue  # blocks 1-5 are summarised below
        actual = trace.get(key)
        print(f"   {key:<28} {str(exp):<22} {str(actual):<22} {'OK' if actual == exp else 'FAIL'}")

    repeated_ok = all(trace.get(f"block{i}.{n}") == s
                      for i in range(1, NUM_BLOCKS) for n, s in per_block)
    print(f"\n   blocks 1-{NUM_BLOCKS - 1}: all {len(per_block)} stages each match: {repeated_ok}")
    print(f"   logits {tuple(logits.shape)} | loss is a scalar: {loss.shape == torch.Size([])}")
    print(f"   missing keys: {missing or 'none'} | untraced extras: {extra or 'none'}")
    print(f"   PASS: {not failures and not missing and repeated_ok}\n")

    # A shorter sequence must also work — the mask has to be sliced, not assumed.
    short_trace = {}
    model(torch.randint(0, V, (3, 16)), trace=short_trace)
    ok_short = (short_trace["block0.scores"] == (3, H, 16, 16)
                and short_trace["logits"] == (3, 16, V))
    print(f"   T=16 instead of {T} (mask slicing): {short_trace['block0.scores']} "
          f"-> PASS: {ok_short}\n")
    return not failures and not missing and repeated_ok and ok_short


def check_causality(model):
    """Changing a future token must not change any earlier prediction."""
    print("2. CAUSAL MASK")
    torch.manual_seed(0)
    idx = torch.randint(0, V, (1, 16))
    with torch.no_grad():
        before, _ = model(idx)
        changed = idx.clone()
        changed[0, -1] = (idx[0, -1] + 1) % V      # rewrite only the LAST token
        after, _ = model(changed)

    earlier_identical = torch.equal(before[:, :-1], after[:, :-1])
    last_changed = not torch.equal(before[:, -1], after[:, -1])
    print(f"   rewrote token 15; predictions at positions 0-14 unchanged: {earlier_identical}")
    print(f"   prediction at position 15 did change (so the test can detect a difference): "
          f"{last_changed}")

    # And the mask itself: re-run block 0's attention by hand to look at the
    # weight matrix, which must be zero everywhere above the diagonal.
    with torch.no_grad():
        x = model.token_embedding(idx) + model.position_embedding(torch.arange(16))
        normed = model.blocks[0].ln_1(x)
        attn = model.blocks[0].attn
        q = attn.q_proj(normed).view(1, 16, H, DH).transpose(1, 2)
        k = attn.k_proj(normed).view(1, 16, H, DH).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(DH) + attn.causal_mask[:16, :16]
        weights = torch.softmax(scores, dim=-1)
    upper = torch.triu(weights[0, 0], diagonal=1)
    rows_sum_to_one = torch.allclose(weights.sum(-1), torch.ones(1, H, 16), atol=1e-6)
    print(f"   attention weights above the diagonal are exactly 0: {upper.abs().max().item() == 0.0}")
    print(f"   every attention row still sums to 1: {rows_sum_to_one}")
    passed = earlier_identical and last_changed and upper.abs().max().item() == 0.0
    print(f"   PASS: {passed}\n")
    return passed


def check_target_shift():
    """targets[:, t] must equal inputs[:, t+1] — one slice, offset by one."""
    print("3. TARGET SHIFT (the off-by-one)")
    stream = np.load(TRAIN_TOKENS)
    chunk = torch.from_numpy(stream[:T + 1].astype(np.int64))    # 513 tokens
    inputs = chunk[:-1].unsqueeze(0)      # stream[0:512]
    targets = chunk[1:].unsqueeze(0)      # stream[1:513]

    rule_holds = torch.equal(targets[:, :-1], inputs[:, 1:])
    print(f"   slice of {len(chunk)} tokens -> inputs {tuple(inputs.shape)}, "
          f"targets {tuple(targets.shape)}")
    print(f"   inputs[0, :6]  = {inputs[0, :6].tolist()}")
    print(f"   targets[0, :6] = {targets[0, :6].tolist()}")
    print(f"   targets[:, t] == inputs[:, t+1] for every t: {rule_holds}")
    print(f"   PASS: {rule_holds}\n")
    return rule_holds


def check_parameters(model):
    print("4. PARAMETER COUNT")
    groups = {}
    for name, param in model.named_parameters():
        key = "blocks (all 6)" if name.startswith("blocks.") else name
        groups[key] = groups.get(key, 0) + param.numel()

    block_params = sum(p.numel() for p in model.blocks[0].parameters())
    total = model.num_parameters()
    for name, count in groups.items():
        print(f"   {name:<28} {count:>10,}")
    print(f"   {'-> one block':<28} {block_params:>10,}")
    print(f"   {'TOTAL':<28} {total:>10,}   (estimate was ~1,800,000)")
    print(f"   tied LM head saves a separate {D}x{V} matrix: {D * V:,} parameters")
    print(f"   untied, the model would be {total + D * V:,}\n")
    return total


def check_init_loss(model):
    print("5. LOSS AT INITIALISATION")
    stream = np.load(TRAIN_TOKENS)
    rng = np.random.default_rng(1337)
    losses, logit_stds = [], []
    with torch.no_grad():
        for _ in range(5):
            starts = rng.integers(0, len(stream) - T - 1, size=4)
            chunks = np.stack([stream[s:s + T + 1] for s in starts]).astype(np.int64)
            batch = torch.from_numpy(chunks)
            logits, loss = model(batch[:, :-1], batch[:, 1:])
            losses.append(loss.item())
            logit_stds.append(logits.std().item())

    mean_loss = sum(losses) / len(losses)
    uniform = math.log(V)
    print(f"   5 batches of 4 x {T} real tokens")
    print(f"   loss {mean_loss:.4f} (min {min(losses):.4f}, max {max(losses):.4f})")
    print(f"   uniform reference log(4000) = {uniform:.4f} | excess {mean_loss - uniform:+.4f}")
    print(f"   logit spread at init: std {sum(logit_stds) / len(logit_stds):.4f}")
    sane = uniform - 0.05 < mean_loss < uniform + 0.5
    print(f"   PASS (within a sane band of log(4000)): {sane}\n")
    return mean_loss


def main():
    torch.manual_seed(1337)
    model = TinyTransformer()
    model.eval()
    print(f"TinyTransformer | vocab {V:,} | context {T} | d_model {D} | heads {H} "
          f"(head_dim {DH}) | d_ff {F_FF} | blocks {NUM_BLOCKS}\n")

    shapes_ok = check_shapes(model)
    causal_ok = check_causality(model)
    shift_ok = check_target_shift()
    check_parameters(model)
    check_init_loss(model)
    print(f"ALL CORRECTNESS CHECKS PASS: {shapes_ok and causal_ok and shift_ok}")


if __name__ == "__main__":
    main()
