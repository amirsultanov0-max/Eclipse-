"""
Shared fixtures. No trained checkpoint is needed: stand-in checkpoints are
written with the real TinyTransformer, the real AdamW parameter groups and
train_transformer.save_checkpoint itself, with the config keys the target run's
training code saved, so the loader is tested against the format it will meet.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eclipse.inference import Eclipse, weights_sha256  # noqa: E402
from model.transformer import TinyTransformer  # noqa: E402
from tokenizer.bpe_tokenizer import SAVE_FILE, BPETokenizer  # noqa: E402
# Imported read-only, to write checkpoints exactly as training did.
from train_transformer import make_param_groups, save_checkpoint  # noqa: E402

TARGET_MANIFEST = PROJECT_ROOT / "results" / "seed_replication" / "sr_data200mb_s1337.provenance.json"


def pre_flag_config(manifest_ref):
    """The config keys train_transformer.py saved at commit 0d55208, which trained
    sr_data200mb_s1337: no d_model, n_heads, d_ff or parameters."""
    m = json.loads(TARGET_MANIFEST.read_text(encoding="utf-8"))
    return {"steps": 10_000, "new_steps": 10_000, "resumed_from": None,
            "train_tokens": m["args"]["train_tokens"], "batch_size": 16, "lr": 0.001,
            "weight_decay": 0.01, "grad_clip": 1.0, "seed": 1337, "eval_seed": 1337,
            "generation_seed": 11_337, "context_length": 512, "vocab_size": 4000,
            "args": m["args"],
            "provenance": {"git": m["git"], "data": m["data"],
                           "fingerprints": {k: m["fingerprints"][k] for k in
                                            ("initial_weights_sha256", "eval_batches_sha256")},
                           "manifest": str(manifest_ref)}}


def post_flag_config(manifest_ref, model, d_model, n_heads, d_ff):
    """The config after the Stage 6.2 architecture flags (commit 1893e6f onward)."""
    config = pre_flag_config(manifest_ref)
    config.update({"d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
                   "parameters": model.num_parameters()})
    return config


def write_training_checkpoint(path, model, config, step=10_000, best_val=2.086):
    groups, _ = make_param_groups(model)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    x = torch.randint(0, 4000, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    optimizer.step()                    # populates real AdamW state
    optimizer.zero_grad(set_to_none=True)
    save_checkpoint(path, model, optimizer, step, best_val, config)
    return path


def write_manifest(path, final_step, final_weights_sha256):
    path.write_text(json.dumps({"fingerprints": {
        "final_step": final_step, "final_weights_sha256": final_weights_sha256}}))
    return path


def build_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    return TinyTransformer(**kwargs)


@pytest.fixture(scope="session")
def tokenizer():
    return BPETokenizer.load(SAVE_FILE)


@pytest.fixture(scope="session")
def d128_checkpoint(tmp_path_factory):
    """Reference architecture, pre-flag config, final step, matching manifest."""
    root = tmp_path_factory.mktemp("d128")
    model = build_model()
    manifest = root / "run.provenance.json"
    path = write_training_checkpoint(root / "d128_best.pt", model, pre_flag_config(manifest))
    write_manifest(manifest, 10_000, weights_sha256(model.state_dict()))
    return path, model.state_dict()


@pytest.fixture(scope="session")
def small_checkpoint(tmp_path_factory):
    """A different architecture (d_model 64, 2 heads, d_ff 256), post-flag config,
    a mid-run step, so provenance is unverifiable rather than verified."""
    root = tmp_path_factory.mktemp("small")
    model = build_model(seed=1, d_model=64, num_heads=2, d_ff=256)
    manifest = write_manifest(root / "run.provenance.json", 10_000, "0" * 64)
    config = post_flag_config(manifest, model, 64, 2, 256)
    path = write_training_checkpoint(root / "small_step500.pt", model, config, step=500)
    return path, model.state_dict()


@pytest.fixture(scope="session")
def small_engine(small_checkpoint):
    return Eclipse.load(small_checkpoint[0])
