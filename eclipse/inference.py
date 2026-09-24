"""
Eclipse inference: load the trained TinyTransformer checkpoint and continue text.

Inference only. Nothing here trains, and nothing here imports the training or
evaluation scripts; the model and tokenizer are the research code itself
(model/transformer.py, tokenizer/bpe_tokenizer.py), used unchanged.

Loading a checkpoint:
  1. The file must exist. A missing checkpoint is an error, never a fallback.
  2. It is read with torch.load(weights_only=True), so a checkpoint cannot run
     arbitrary code when it is opened.
  3. The architecture is taken from the checkpoint itself: from its saved
     `config` where that records a value, and from the shapes of its weights
     otherwise, with the two cross-checked. Checkpoints written before the
     Stage 6.2 architecture flags (including sr_data200mb_s1337) record neither
     d_model, d_ff nor n_heads. The first two are read from the weight shapes;
     the head count leaves no trace in the weights, and those checkpoints were
     all built by TinyTransformer() with its default head count, so that
     default is used and reported as such.
  4. The tokenizer file must be the frozen one whose SHA256 the Stage 6.2
     preregistration records, and its vocabulary must match the model's.
  5. If the checkpoint's config names a provenance manifest and the checkpoint
     is that run's final step, the weights are hashed exactly as
     train_transformer.py hashed them and must match the manifest.
"""

import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from eclipse.common import PROJECT_ROOT, display_path, pick_device
from model.transformer import NUM_HEADS, TinyTransformer
from tokenizer.bpe_tokenizer import SAVE_FILE as TOKENIZER_FILE
from tokenizer.bpe_tokenizer import BPETokenizer

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sr_data200mb_s1337_best.pt"
CHECKPOINT_ENV_VAR = "ECLIPSE_CHECKPOINT"
# Where train_transformer.py wrote this checkpoint on the training machine.
TRAINING_OUTPUT_PATH = ("checkpoints/seed_replication/sr_data200mb_s1337/"
                        "sr_data200mb_s1337_best.pt")
# eval/stage6_2_preregistration.md, section 9.
TOKENIZER_SHA256 = "e3876f558896533d13b7818282c6b19b55cbf205390ad4ee9e982d105762d3df"

DEFAULT_MAX_NEW_TOKENS = 150      # the cap every Stage 5 generation evaluation used
MAX_NEW_TOKENS_LIMIT = 512
DEFAULT_TEMPERATURE = 1.0
MAX_TEMPERATURE = 2.0
DEFAULT_TOP_K = 40                # the decoding the Stage 5 error analysis adopted
MAX_PROMPT_CHARS = 4000
SEED_LIMIT = 2 ** 32


class EclipseError(Exception):
    """Base class for everything this module reports to a caller."""


class CheckpointNotFoundError(EclipseError):
    pass


class CheckpointError(EclipseError):
    """The file exists but cannot be served as Eclipse."""


class GenerationRequestError(EclipseError, ValueError):
    """The prompt or the generation settings are invalid."""


def resolve_checkpoint_path(explicit=None):
    """--checkpoint beats $ECLIPSE_CHECKPOINT beats the default. Relative = repo-relative."""
    chosen = explicit or os.environ.get(CHECKPOINT_ENV_VAR) or DEFAULT_CHECKPOINT
    path = Path(chosen).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_sha256(state_dict):
    """The same digest as train_transformer.sha256_tensors(sorted(state_dict.items()))."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def resolve_architecture(state_dict, config):
    """Returns (architecture, sources): TinyTransformer kwargs and where each came from."""
    try:
        vocab_size, d_model = state_dict["token_embedding.weight"].shape
        context_length = state_dict["position_embedding.weight"].shape[0]
        d_ff = state_dict["blocks.0.ff_in.weight"].shape[0]
    except (KeyError, AttributeError, ValueError) as e:
        raise CheckpointError(f"the weights are not a TinyTransformer state_dict ({e!r})") from None
    block_ids = {int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")}
    if block_ids != set(range(len(block_ids))):
        raise CheckpointError(f"non-contiguous transformer blocks in the weights: {sorted(block_ids)}")

    from_weights = {"vocab_size": vocab_size, "context_length": context_length,
                    "d_model": d_model, "d_ff": d_ff, "num_blocks": len(block_ids)}
    architecture, sources = {}, {}
    for key, value in from_weights.items():
        saved = config.get(key)
        if saved is not None and saved != value:
            raise CheckpointError(f"checkpoint config says {key}={saved} but its weights "
                                  f"have {key}={value}")
        architecture[key] = value
        sources[key] = "config, matches weights" if saved is not None else "weights"

    num_heads = config.get("n_heads")
    if num_heads is not None:
        sources["num_heads"] = "config"
    elif "d_model" in config:
        raise CheckpointError("checkpoint config records d_model but not n_heads; "
                              "the head count cannot be recovered from the weights")
    else:
        num_heads = NUM_HEADS
        sources["num_heads"] = ("model default: this checkpoint predates the architecture "
                                "flags, when training always used it")
    if d_model % num_heads:
        raise CheckpointError(f"d_model {d_model} does not divide into {num_heads} heads")
    architecture["num_heads"] = num_heads
    return architecture, sources


def check_provenance(checkpoint, state_dict):
    """Compare the weights with the run's provenance manifest, when that is possible."""
    config = checkpoint.get("config") or {}
    manifest_ref = (config.get("provenance") or {}).get("manifest")
    if not manifest_ref:
        return {"status": "unavailable", "detail": "the checkpoint names no provenance manifest"}
    manifest_path = PROJECT_ROOT / manifest_ref
    if not manifest_path.is_file():
        return {"status": "unavailable", "manifest": manifest_ref,
                "detail": f"the provenance manifest {manifest_ref} was not found"}
    fingerprints = json.loads(manifest_path.read_text(encoding="utf-8")).get("fingerprints", {})
    final_step, expected = fingerprints.get("final_step"), fingerprints.get("final_weights_sha256")
    step = checkpoint.get("step")
    if expected is None or step != final_step:
        return {"status": "unverifiable", "manifest": manifest_ref,
                "detail": f"the manifest fingerprints only the final step ({final_step}); "
                          f"this checkpoint is step {step}"}
    actual = weights_sha256(state_dict)
    if actual != expected:
        raise CheckpointError(
            f"the weights do not match the research record: {manifest_ref} records "
            f"final_weights_sha256 {expected} for step {final_step}, this file hashes to "
            f"{actual}. Use the checkpoint that run produced, or pass "
            f"--skip-provenance-check to serve an unverified file.")
    return {"status": "verified", "manifest": manifest_ref, "weights_sha256": actual,
            "detail": f"weights match the manifest's final_weights_sha256 (step {step})"}


def load_tokenizer(path=TOKENIZER_FILE):
    actual = sha256_file(path)
    if actual != TOKENIZER_SHA256:
        raise CheckpointError(f"tokenizer {display_path(path)} has SHA256 {actual}, not the "
                              f"frozen research tokenizer {TOKENIZER_SHA256}")
    return BPETokenizer.load(Path(path))


def _read_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise CheckpointError(f"could not read {display_path(path)} as a PyTorch checkpoint: "
                              f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
                              ) from None
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise CheckpointError(f"{display_path(path)} is not a training checkpoint "
                              "(expected a dict with a 'model' state_dict)")
    config = checkpoint.get("config")
    if config is not None and not isinstance(config, dict):
        raise CheckpointError("the checkpoint's 'config' is not a dict")
    return checkpoint


def top_k_filter(logits, k):
    """Keep the k highest logits (ties at the k-th value included). As eval_sampling.py."""
    kth_value = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def sample_next(logits, temperature, top_k, generator):
    """One token id from a [V] row of logits. Temperature 0 is greedy (argmax)."""
    if temperature == 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k is not None and top_k < logits.numel():
        logits = top_k_filter(logits, top_k)
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator))


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int | None = DEFAULT_TOP_K
    seed: int | None = None

    def validate(self, vocab_size):
        def is_int(v):
            return isinstance(v, int) and not isinstance(v, bool)

        if not is_int(self.max_new_tokens) or not 1 <= self.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise GenerationRequestError(
                f"max_new_tokens must be an integer from 1 to {MAX_NEW_TOKENS_LIMIT}")
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
                or not 0 <= self.temperature <= MAX_TEMPERATURE):
            raise GenerationRequestError(
                f"temperature must be a number from 0 to {MAX_TEMPERATURE} (0 = greedy)")
        if self.top_k is not None and (not is_int(self.top_k) or not 1 <= self.top_k <= vocab_size):
            raise GenerationRequestError(
                f"top_k must be an integer from 1 to {vocab_size}, or null for no top-k")
        if self.seed is not None and (not is_int(self.seed) or not 0 <= self.seed < SEED_LIMIT):
            raise GenerationRequestError(f"seed must be an integer from 0 to {SEED_LIMIT - 1}")


@dataclass
class Generation:
    prompt: str
    completion: str
    prompt_tokens: int
    prompt_unknown_tokens: int
    generated_tokens: int
    stop_reason: str              # "eos" or "max_new_tokens"
    seed: int
    settings: GenerationSettings
    elapsed_seconds: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelInfo:
    checkpoint: str
    step: int | None
    best_val_loss: float | None
    parameters: int
    architecture: dict
    architecture_sources: dict
    training: dict
    tokenizer: dict
    provenance: dict
    device: str
    name: str = "Eclipse"
    limits: dict = field(default_factory=lambda: {
        "max_new_tokens": MAX_NEW_TOKENS_LIMIT, "max_temperature": MAX_TEMPERATURE,
        "max_prompt_chars": MAX_PROMPT_CHARS, "seed_limit": SEED_LIMIT})
    defaults: dict = field(default_factory=lambda: asdict(GenerationSettings()))

    def to_dict(self):
        return asdict(self)


class Eclipse:
    """A loaded model and tokenizer, ready to continue prompts. Load once, reuse."""

    def __init__(self, model, tokenizer, context_length, device=torch.device("cpu"), info=None):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = torch.device(device)
        self.info = info
        self._lock = threading.Lock()

    @classmethod
    def load(cls, checkpoint_path=None, device="cpu", verify_provenance=True):
        path = resolve_checkpoint_path(checkpoint_path)
        if not path.is_file():
            raise CheckpointNotFoundError(
                f"Eclipse checkpoint not found: {path}\n"
                f"Eclipse serves only the trained research checkpoint and has no fallback model. "
                f"Copy sr_data200mb_s1337_best.pt (written by training to {TRAINING_OUTPUT_PATH}) "
                f"to {display_path(DEFAULT_CHECKPOINT)}, or point --checkpoint / "
                f"${CHECKPOINT_ENV_VAR} at it.")

        checkpoint = _read_checkpoint(path)
        state_dict, config = checkpoint["model"], checkpoint.get("config") or {}
        architecture, sources = resolve_architecture(state_dict, config)

        tokenizer = load_tokenizer()
        if len(tokenizer) != architecture["vocab_size"]:
            raise CheckpointError(f"the tokenizer has {len(tokenizer)} tokens but the model's "
                                  f"vocabulary is {architecture['vocab_size']}")

        model = TinyTransformer(**architecture)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise CheckpointError(f"weights do not fit the resolved architecture: {e}") from None
        if config.get("parameters") not in (None, model.num_parameters()):
            raise CheckpointError(f"checkpoint config records {config['parameters']:,} parameters, "
                                  f"the rebuilt model has {model.num_parameters():,}")

        provenance = (check_provenance(checkpoint, state_dict) if verify_provenance else
                      {"status": "skipped", "detail": "provenance check disabled by the operator"})

        device = pick_device(device)
        model.to(device).eval()
        info = ModelInfo(
            checkpoint=display_path(path),
            step=checkpoint.get("step"),
            best_val_loss=checkpoint.get("best_val_loss"),
            parameters=model.num_parameters(),
            architecture=architecture,
            architecture_sources=sources,
            training={k: config.get(k) for k in ("train_tokens", "seed", "steps", "batch_size", "lr")},
            tokenizer={"file": display_path(TOKENIZER_FILE), "sha256": TOKENIZER_SHA256,
                       "vocab_size": len(tokenizer)},
            provenance=provenance,
            device=device.type,
        )
        return cls(model, tokenizer, architecture["context_length"], device, info)

    def generate(self, prompt, settings=None):
        settings = settings or GenerationSettings()
        settings.validate(len(self.tokenizer))
        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerationRequestError("the prompt is empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise GenerationRequestError(f"the prompt is {len(prompt):,} characters; "
                                         f"the limit is {MAX_PROMPT_CHARS:,}")
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.context_length:
            raise GenerationRequestError(f"the prompt is {len(prompt_ids)} tokens; the model's "
                                         f"context is {self.context_length}")

        seed = settings.seed if settings.seed is not None else secrets.randbelow(SEED_LIMIT)
        generator = torch.Generator().manual_seed(seed)
        start = time.perf_counter()
        with self._lock:
            ids, stop_reason = self._generate_ids(prompt_ids, settings, generator)
        return Generation(
            prompt=prompt,
            completion=self.tokenizer.decode(ids),
            prompt_tokens=len(prompt_ids),
            prompt_unknown_tokens=prompt_ids.count(self.tokenizer.unk_id),
            generated_tokens=len(ids),
            stop_reason=stop_reason,
            seed=seed,
            settings=settings,
            elapsed_seconds=time.perf_counter() - start,
        )

    @torch.inference_mode()
    def _generate_ids(self, prompt_ids, settings, generator):
        # Once prompt + output outgrow the context, the model sees the most recent
        # context_length tokens, as in every generation loop in the research code.
        idx = torch.tensor([prompt_ids], device=self.device)
        generated = []
        for _ in range(settings.max_new_tokens):
            logits, _ = self.model(idx[:, -self.context_length:])
            # Sampling on the CPU keeps a seed's output independent of the device's RNG.
            next_id = sample_next(logits[0, -1].float().cpu(), settings.temperature,
                                  settings.top_k, generator)
            if next_id == self.tokenizer.eos_id:
                return generated, "eos"
            generated.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=self.device)], dim=1)
        return generated, "max_new_tokens"
