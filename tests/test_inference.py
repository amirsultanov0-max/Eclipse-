"""
Checkpoint loading, tokenizer/model compatibility, generation and its limits,
the command-line program, and the absence of any external-model fallback.

The tests at the bottom use the real trained checkpoint and are skipped, with
the reason printed, until checkpoints/sr_data200mb_s1337_best.pt exists.
"""

import math
import pickle
import re
import subprocess

import pytest
import torch

import generate as generate_cli
from conftest import (PROJECT_ROOT, build_model, post_flag_config, pre_flag_config,
                      write_manifest, write_training_checkpoint)
from eclipse import inference
from eclipse.inference import (DEFAULT_CHECKPOINT, MAX_NEW_TOKENS_LIMIT, TOKENIZER_SHA256,
                               CheckpointError, CheckpointNotFoundError, Eclipse,
                               GenerationRequestError, GenerationSettings, sha256_file,
                               top_k_filter, weights_sha256)
from model.transformer import TinyTransformer
from tokenizer.bpe_tokenizer import SAVE_FILE


class ScriptedModel(torch.nn.Module):
    """Emits a fixed sequence of token ids (then repeats the last), and records
    the length of every input it is shown."""

    def __init__(self, script, vocab_size=4000):
        super().__init__()
        self.script, self.vocab_size, self.seen = list(script), vocab_size, []

    def forward(self, idx):
        self.seen.append(idx.shape[1])
        token = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab_size)
        logits[:, -1, token] = 100.0
        return logits, None


def scripted_engine(tokenizer, script, context_length=512):
    return Eclipse(ScriptedModel(script), tokenizer, context_length)


# --- checkpoint location and missing checkpoint ------------------------------

def test_default_checkpoint_is_the_200mb_seed_1337_reference_run():
    assert DEFAULT_CHECKPOINT == PROJECT_ROOT / "checkpoints" / "sr_data200mb_s1337_best.pt"


def test_checkpoint_path_is_gitignored():
    result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q",
                             "checkpoints/sr_data200mb_s1337_best.pt"])
    assert result.returncode == 0, "checkpoints/ must stay out of git"


def test_missing_checkpoint_is_reported_not_replaced(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(CheckpointNotFoundError) as info:
        Eclipse.load(missing)
    message = str(info.value)
    assert str(missing) in message
    assert "no fallback model" in message
    assert "sr_data200mb_s1337_best.pt" in message


def test_checkpoint_path_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv(inference.CHECKPOINT_ENV_VAR, raising=False)
    assert inference.resolve_checkpoint_path() == DEFAULT_CHECKPOINT
    monkeypatch.setenv(inference.CHECKPOINT_ENV_VAR, str(tmp_path / "env.pt"))
    assert inference.resolve_checkpoint_path() == tmp_path / "env.pt"
    assert inference.resolve_checkpoint_path(tmp_path / "cli.pt") == tmp_path / "cli.pt"
    assert inference.resolve_checkpoint_path("checkpoints/x.pt") == PROJECT_ROOT / "checkpoints/x.pt"


# --- loading: architecture from the checkpoint --------------------------------

def test_pre_flag_reference_checkpoint_loads(d128_checkpoint):
    path, state = d128_checkpoint
    engine = Eclipse.load(path)
    info = engine.info
    assert info.architecture == {"vocab_size": 4000, "context_length": 512, "d_model": 128,
                                 "d_ff": 512, "num_blocks": 6, "num_heads": 4}
    assert info.architecture_sources["d_model"] == "weights"
    assert info.architecture_sources["vocab_size"] == "config, matches weights"
    assert info.architecture_sources["num_heads"].startswith("model default")
    assert info.parameters == 1_767_424
    assert info.step == 10_000
    assert info.training["train_tokens"] == "data/stage6_1_train_tokens.npy"
    assert info.training["seed"] == 1337
    assert not engine.model.training
    for name, tensor in engine.model.state_dict().items():
        assert torch.equal(tensor, state[name]), name


def test_other_architecture_is_built_from_the_checkpoint_not_hardcoded(small_checkpoint):
    path, state = small_checkpoint
    engine = Eclipse.load(path)
    assert engine.info.architecture["d_model"] == 64
    assert engine.info.architecture["num_heads"] == 2
    assert engine.info.architecture_sources["num_heads"] == "config"
    assert engine.info.architecture["d_ff"] == 256
    assert engine.model.blocks[0].attn.num_heads == 2
    for name, tensor in engine.model.state_dict().items():
        assert torch.equal(tensor, state[name]), name


def test_config_that_contradicts_the_weights_is_rejected(tmp_path):
    model = build_model(d_model=64, num_heads=2, d_ff=256)
    config = post_flag_config(tmp_path / "m.json", model, 128, 4, 512)
    path = write_training_checkpoint(tmp_path / "bad.pt", model, config, step=5)
    with pytest.raises(CheckpointError, match="d_model=128 but its weights have d_model=64"):
        Eclipse.load(path)


def test_config_parameter_count_must_match(tmp_path):
    model = build_model(d_model=64, num_heads=2, d_ff=256)
    config = post_flag_config(tmp_path / "m.json", model, 64, 2, 256)
    config["parameters"] += 1
    path = write_training_checkpoint(tmp_path / "bad.pt", model, config, step=5)
    with pytest.raises(CheckpointError, match="parameters"):
        Eclipse.load(path)


def test_post_flag_config_without_head_count_is_rejected(tmp_path):
    model = build_model(d_model=64, num_heads=2, d_ff=256)
    config = post_flag_config(tmp_path / "m.json", model, 64, 2, 256)
    del config["n_heads"]
    path = write_training_checkpoint(tmp_path / "bad.pt", model, config, step=5)
    with pytest.raises(CheckpointError, match="n_heads"):
        Eclipse.load(path)


def test_vocabulary_must_match_the_tokenizer(tmp_path):
    model = build_model(vocab_size=3999)
    config = pre_flag_config(tmp_path / "m.json")
    config["vocab_size"] = 3999
    path = write_training_checkpoint(tmp_path / "bad.pt", model, config, step=5)
    with pytest.raises(CheckpointError, match="tokenizer has 4000 tokens"):
        Eclipse.load(path)


def test_file_that_is_not_a_checkpoint_is_rejected(tmp_path):
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="could not read"):
        Eclipse.load(garbage)
    tensor_only = tmp_path / "tensor.pt"
    torch.save(torch.zeros(3), tensor_only)
    with pytest.raises(CheckpointError, match="not a training checkpoint"):
        Eclipse.load(tensor_only)


class Payload:
    def __reduce__(self):
        return (print, ("code ran while loading",))


def test_checkpoint_cannot_run_code_when_loaded(tmp_path, capsys):
    path = tmp_path / "evil.pt"
    with path.open("wb") as f:
        pickle.dump({"model": {}, "config": Payload()}, f, protocol=2)
    with pytest.raises(CheckpointError):
        Eclipse.load(path)
    assert "code ran" not in capsys.readouterr().out


# --- provenance ---------------------------------------------------------------

def test_final_step_weights_are_verified_against_the_manifest(d128_checkpoint):
    provenance = Eclipse.load(d128_checkpoint[0]).info.provenance
    assert provenance["status"] == "verified"


def test_weights_that_differ_from_the_manifest_are_refused(tmp_path):
    model = build_model()
    manifest = write_manifest(tmp_path / "m.json", 10_000, "f" * 64)
    path = write_training_checkpoint(tmp_path / "c.pt", model, pre_flag_config(manifest))
    with pytest.raises(CheckpointError, match="do not match the research record"):
        Eclipse.load(path)
    skipped = Eclipse.load(path, verify_provenance=False)
    assert skipped.info.provenance["status"] == "skipped"


def test_mid_run_checkpoint_is_reported_unverifiable(small_checkpoint):
    assert Eclipse.load(small_checkpoint[0]).info.provenance["status"] == "unverifiable"


def test_missing_manifest_is_reported_unavailable(tmp_path):
    model = build_model(d_model=64, num_heads=2, d_ff=256)
    config = post_flag_config(tmp_path / "absent.json", model, 64, 2, 256)
    path = write_training_checkpoint(tmp_path / "c.pt", model, config)
    assert Eclipse.load(path).info.provenance["status"] == "unavailable"


def test_weights_hash_is_the_training_scripts_hash():
    from train_transformer import sha256_tensors
    state = build_model(d_model=64, num_heads=2, d_ff=256).state_dict()
    assert weights_sha256(state) == sha256_tensors(sorted(state.items()))


# --- tokenizer ----------------------------------------------------------------

def test_tokenizer_is_the_frozen_research_tokenizer(tokenizer):
    assert sha256_file(SAVE_FILE) == TOKENIZER_SHA256
    assert len(tokenizer) == 4000
    text = 'Tom said, "Look at the big, red ball!"\nLily smiled.'
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_modified_tokenizer_is_rejected(tmp_path):
    copy = tmp_path / "bpe_4000.json"
    copy.write_bytes(SAVE_FILE.read_bytes() + b" ")
    with pytest.raises(CheckpointError, match="not the frozen research tokenizer"):
        inference.load_tokenizer(copy)


# --- sampling -----------------------------------------------------------------

def test_top_k_filter_matches_the_stage5_implementation():
    from eval_sampling import top_k_filter as research_top_k
    logits = torch.randn(4000, generator=torch.Generator().manual_seed(0))
    logits[10] = logits[11]            # a tie
    for k in (1, 2, 40, 3999, 4000):
        assert torch.equal(top_k_filter(logits, k), research_top_k(logits, k))


def test_greedy_is_deterministic_and_equals_top_k_1(small_engine):
    prompt = "Once upon a time"
    a = small_engine.generate(prompt, GenerationSettings(max_new_tokens=20, temperature=0))
    b = small_engine.generate(prompt, GenerationSettings(max_new_tokens=20, temperature=0,
                                                         seed=99))
    c = small_engine.generate(prompt, GenerationSettings(max_new_tokens=20, temperature=1.0,
                                                         top_k=1))
    assert a.completion == b.completion == c.completion


def test_greedy_matches_the_stage5_greedy_loop(small_engine, tokenizer):
    from eval_generation import greedy_generate
    prompt = "Lily found a shiny key under the bed. She"
    ids, _ = greedy_generate(small_engine.model, tokenizer, prompt, 25, torch.device("cpu"))
    result = small_engine.generate(prompt, GenerationSettings(max_new_tokens=25, temperature=0))
    assert result.completion == tokenizer.decode(ids)


def test_seed_reproduces_a_sample(small_engine):
    settings = GenerationSettings(max_new_tokens=30, temperature=1.0, top_k=40, seed=1337)
    first = small_engine.generate("Tom picked up the red ball and", settings)
    second = small_engine.generate("Tom picked up the red ball and", settings)
    assert first.completion == second.completion and first.seed == 1337
    outputs = {small_engine.generate("Tom picked up the red ball and",
                                     GenerationSettings(max_new_tokens=30, seed=s)).completion
               for s in range(5)}
    assert len(outputs) > 1


def test_seed_is_reported_when_not_given(small_engine):
    result = small_engine.generate("The dog", GenerationSettings(max_new_tokens=5))
    again = small_engine.generate("The dog", GenerationSettings(max_new_tokens=5,
                                                                seed=result.seed))
    assert again.completion == result.completion


# --- generation limits --------------------------------------------------------

def test_max_new_tokens_is_respected(small_engine):
    for n in (1, 7, 40):
        result = small_engine.generate("Once", GenerationSettings(max_new_tokens=n, seed=0))
        assert result.generated_tokens <= n
        if result.stop_reason == "max_new_tokens":
            assert result.generated_tokens == n


def test_generation_stops_at_eos(tokenizer):
    word = tokenizer.encode(" happy")[0]
    engine = scripted_engine(tokenizer, [word, word, tokenizer.eos_id])
    result = engine.generate("She was", GenerationSettings(max_new_tokens=50, temperature=0))
    assert result.stop_reason == "eos"
    assert result.completion == " happy happy"
    assert "<|endoftext|>" not in result.completion


def test_model_never_sees_more_than_its_context(tokenizer):
    word = tokenizer.encode(" the")[0]
    engine = scripted_engine(tokenizer, [word], context_length=512)
    prompt = " the" * 500
    result = engine.generate(prompt, GenerationSettings(max_new_tokens=60, temperature=0))
    assert result.generated_tokens == 60
    assert max(engine.model.seen) == 512
    assert engine.model.seen[0] == 500


def test_prompt_longer_than_the_context_is_rejected(small_engine):
    with pytest.raises(GenerationRequestError, match="context is 512"):
        small_engine.generate(" the" * 600)


def test_prompt_character_limit(small_engine):
    with pytest.raises(GenerationRequestError, match="characters"):
        small_engine.generate("a" * (inference.MAX_PROMPT_CHARS + 1))


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t", None, 42])
def test_empty_or_non_text_prompt_is_rejected(small_engine, prompt):
    with pytest.raises(GenerationRequestError):
        small_engine.generate(prompt)


@pytest.mark.parametrize("settings", [
    dict(max_new_tokens=0), dict(max_new_tokens=MAX_NEW_TOKENS_LIMIT + 1),
    dict(max_new_tokens=2.5), dict(max_new_tokens=True),
    dict(temperature=-0.1), dict(temperature=2.01), dict(temperature=math.nan),
    dict(temperature=math.inf), dict(temperature="1"),
    dict(top_k=0), dict(top_k=4001), dict(top_k=3.0),
    dict(seed=-1), dict(seed=2 ** 32), dict(seed="7"),
])
def test_invalid_settings_are_rejected(small_engine, settings):
    with pytest.raises(GenerationRequestError):
        small_engine.generate("Once upon a time", GenerationSettings(**settings))


def test_unknown_characters_are_counted(small_engine):
    result = small_engine.generate("The 🙂 and ж were", GenerationSettings(max_new_tokens=3))
    assert result.prompt_unknown_tokens == 2


# --- command-line program -----------------------------------------------------

def test_cli_missing_checkpoint_fails_clearly(tmp_path, capsys):
    code = generate_cli.main(["Once", "--checkpoint", str(tmp_path / "missing.pt")])
    err = capsys.readouterr().err
    assert code == 1
    assert "checkpoint not found" in err and "no fallback model" in err


def test_cli_generates(small_checkpoint, capsys):
    code = generate_cli.main(["Once upon a time", "--checkpoint", str(small_checkpoint[0]),
                              "--max-new-tokens", "10", "--seed", "4", "--top-k", "0"])
    out, err = capsys.readouterr()
    assert code == 0
    assert out.startswith("Once upon a time")
    assert "seed 4" in err


def test_cli_rejects_invalid_settings(small_checkpoint, capsys):
    code = generate_cli.main(["Once", "--checkpoint", str(small_checkpoint[0]),
                              "--temperature", "5"])
    assert code == 1
    assert "temperature" in capsys.readouterr().err


# --- no external model or API -------------------------------------------------

PHASE3_SOURCES = ["eclipse/__init__.py", "eclipse/common.py", "eclipse/inference.py",
                  "server.py", "generate.py", "web/index.html", "web/app.js", "web/style.css"]
FORBIDDEN = re.compile(r"openai|anthropic|claude|huggingface|hf_hub|transformers|"
                       r"from_pretrained|pipeline\(|ollama|cohere|mistral|gemini|"
                       r"requests\.|urllib|http\.client|aiohttp", re.IGNORECASE)


@pytest.mark.parametrize("relative", PHASE3_SOURCES)
def test_program_references_no_external_model_or_network_client(relative):
    text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text)


def test_web_interface_loads_nothing_from_other_hosts():
    for name in ("index.html", "app.js", "style.css"):
        text = (PROJECT_ROOT / "web" / name).read_text(encoding="utf-8")
        assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", text, re.IGNORECASE), name


def test_requirements_contain_no_model_hub_or_llm_client():
    lines = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
    packages = {re.split(r"[=<>]", line)[0].strip().lower()
                for line in lines if line.strip() and not line.startswith("#")}
    assert packages == {"torch", "numpy", "fastapi", "uvicorn", "pytest", "httpx2"}


def test_the_model_class_is_the_research_transformer(small_engine):
    assert type(small_engine.model) is TinyTransformer


# --- the real checkpoint (skipped until it is in place) ------------------------

needs_checkpoint = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.is_file(),
    reason=f"trained checkpoint not present at {DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)}")


@pytest.fixture(scope="module")
def real_engine():
    return Eclipse.load()


@needs_checkpoint
def test_real_checkpoint_is_the_recorded_research_run(real_engine):
    info = real_engine.info
    assert info.architecture == {"vocab_size": 4000, "context_length": 512, "d_model": 128,
                                 "d_ff": 512, "num_blocks": 6, "num_heads": 4}
    assert info.parameters == 1_767_424
    assert info.step == 10_000
    assert round(info.best_val_loss, 4) == 2.0860     # results/seed_replication/sr_data200mb_s1337.md
    assert info.training["train_tokens"] == "data/stage6_1_train_tokens.npy"
    assert info.training["seed"] == 1337
    assert info.provenance["status"] == "verified"
    assert info.provenance["weights_sha256"] == (
        "f2892b7a6bdef41b03c8e7adadd96c31ecc528fbd553ef4d0e33a7f973c2b9b5")


@needs_checkpoint
def test_real_checkpoint_generates(real_engine):
    greedy = real_engine.generate("Once upon a time, there was a tiny robot who wanted to",
                                  GenerationSettings(max_new_tokens=60, temperature=0))
    assert greedy.generated_tokens > 0 and greedy.completion.strip()
    settings = GenerationSettings(max_new_tokens=60, seed=1337)
    first = real_engine.generate("Lily found a shiny key under the bed. She", settings)
    second = real_engine.generate("Lily found a shiny key under the bed. She", settings)
    assert first.completion == second.completion
