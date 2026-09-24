"""
The architecture and attention views: that the frozen model file is untouched,
that hooked attention weights are the model's own (checked bit for bit against
the literal softmax output inside forward()), that hooks leave no trace, and
the API and page. The last test needs the real checkpoint and is skipped
until it is in place.
"""

import hashlib
import re
import threading

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from torch.overrides import TorchFunctionMode

import server
from conftest import PROJECT_ROOT, build_model
from eclipse import attention
from eclipse.attention import (MAX_ATTENTION_CHARS, MAX_ATTENTION_TOKENS, AttentionCaptureError,
                               capture_attention, capture_weights, describe_architecture)
from eclipse.inference import DEFAULT_CHECKPOINT, Eclipse, GenerationRequestError, GenerationSettings

# eval/stage6_2_preregistration.md, section 9: both arms' model/transformer.py.
FROZEN_MODEL_SHA256 = "63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0"
PROMPT = "Lily found a shiny key under the bed. She"


class SoftmaxSpy(TorchFunctionMode):
    """Records every softmax output computed inside forward(): the literal weights."""

    def __init__(self):
        super().__init__()
        self.outputs = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in (F.softmax, torch.softmax, torch.Tensor.softmax):
            self.outputs.append(out.detach().clone())
        return out


@pytest.fixture(scope="module")
def peaked_engine(tokenizer):
    """Weights pushed away from their near-uniform init so attention is uneven."""
    model = build_model(seed=3).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p, generator=torch.Generator().manual_seed(4)) * 0.2)
    return Eclipse(model, tokenizer, 512)


def attn_hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


# --- the frozen model file --------------------------------------------------------

def test_model_file_is_still_the_preregistered_file():
    digest = hashlib.sha256((PROJECT_ROOT / "model" / "transformer.py").read_bytes()).hexdigest()
    assert digest == FROZEN_MODEL_SHA256


# --- hooked weights are the model's own ------------------------------------------

def test_hooked_weights_are_bitwise_the_softmax_inside_forward(peaked_engine, tokenizer):
    ids = tokenizer.encode(PROMPT)
    hooked = capture_weights(peaked_engine, ids)
    spy = SoftmaxSpy()
    with torch.inference_mode(), spy:
        peaked_engine.model(torch.tensor([ids]))
    assert len(spy.outputs) == len(hooked) == 6
    for block, (mine, literal) in enumerate(zip(hooked, spy.outputs)):
        assert torch.equal(mine, literal[0]), f"block {block}"
    assert max(w.max().item() for w in hooked[1:]) > 0.5, "attention should not be uniform here"


def test_weights_are_causal_probabilities(peaked_engine, tokenizer):
    ids = tokenizer.encode(PROMPT)
    for weights in capture_weights(peaked_engine, ids):
        H, T, _ = weights.shape
        assert torch.allclose(weights.sum(-1), torch.ones(H, T))
        assert (weights.triu(1) == 0).all()
        assert (weights >= 0).all()
        assert torch.equal(weights[:, 0, 0], torch.ones(H))


def test_capture_does_not_change_the_model(peaked_engine, tokenizer):
    ids = torch.tensor([tokenizer.encode(PROMPT)])
    with torch.inference_mode():
        before, _ = peaked_engine.model(ids)
    settings = GenerationSettings(max_new_tokens=20, seed=5)
    generated_before = peaked_engine.generate("Once upon a time", settings).completion
    capture_attention(peaked_engine, PROMPT)
    with torch.inference_mode():
        after, _ = peaked_engine.model(ids)
    assert torch.equal(before, after)
    assert peaked_engine.generate("Once upon a time", settings).completion == generated_before
    assert attn_hook_count(peaked_engine.model) == 0


def test_hooks_are_removed_when_the_forward_pass_fails(peaked_engine):
    def fail(module, inputs):
        raise RuntimeError("boom")
    blocker = peaked_engine.model.blocks[2].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            capture_attention(peaked_engine, PROMPT)
    finally:
        blocker.remove()
    assert attn_hook_count(peaked_engine.model) == 0


def test_verification_failure_refuses_to_return_weights(peaked_engine, monkeypatch):
    real = attention._recompute

    def tampered(*args):
        weights, merged = real(*args)
        return weights, merged + 1e-6
    monkeypatch.setattr(attention, "_recompute", tampered)
    with pytest.raises(AttentionCaptureError, match="refusing to show unverified weights"):
        capture_attention(peaked_engine, PROMPT)
    assert attn_hook_count(peaked_engine.model) == 0


def test_capture_waits_for_a_running_generation(peaked_engine):
    done = threading.Event()
    with peaked_engine._lock:
        worker = threading.Thread(target=lambda: (capture_attention(peaked_engine, PROMPT),
                                                  done.set()))
        worker.start()
        assert not done.wait(0.3), "capture ran while the engine was busy"
    worker.join(5)
    assert done.is_set()


# --- the formatted result and its limits -----------------------------------------

def test_result_shape_and_tokens(small_engine, tokenizer):
    result = capture_attention(small_engine, PROMPT)
    T = len(tokenizer.encode(PROMPT))
    assert result["tokens"] == [tokenizer.decode([i]) for i in tokenizer.encode(PROMPT)]
    assert "".join(result["tokens"]) == PROMPT
    assert result["num_blocks"] == 6 and result["num_heads"] == 2
    assert len(result["attention"]) == 6
    for block in result["attention"]:
        assert len(block) == 2
        for head in block:
            assert len(head) == T and all(len(row) == T for row in head)
            for r, row in enumerate(head):
                assert abs(sum(row) - 1) < 1e-4
                assert all(w == 0 for w in row[r + 1:])
    assert result["verified"] is True


def test_token_limit(small_engine, tokenizer):
    ok = " the" * MAX_ATTENTION_TOKENS
    assert len(tokenizer.encode(ok)) == MAX_ATTENTION_TOKENS
    assert len(capture_attention(small_engine, ok)["tokens"]) == MAX_ATTENTION_TOKENS
    with pytest.raises(GenerationRequestError, match="at most 30"):
        capture_attention(small_engine, ok + " the")


@pytest.mark.parametrize("prompt", ["", "  \n", None, 7, "a" * (MAX_ATTENTION_CHARS + 1)])
def test_invalid_prompts(small_engine, prompt):
    with pytest.raises(GenerationRequestError):
        capture_attention(small_engine, prompt)


# --- architecture -----------------------------------------------------------------

def test_reference_architecture_breakdown(d128_checkpoint):
    a = describe_architecture(Eclipse.load(d128_checkpoint[0]))
    assert a["parameters"] == 1_767_424
    assert a["architecture"] == {"vocab_size": 4000, "context_length": 512, "d_model": 128,
                                 "d_ff": 512, "num_blocks": 6, "num_heads": 4, "head_dim": 32}
    c = a["components"]
    assert c["token_embedding"] == 512_000 and c["position_embedding"] == 65_536
    assert c["ln_final"] == 256 and c["output_head"] == 0
    assert len(c["blocks"]) == 6
    assert c["blocks"][0] == {"ln_1": 256, "attention": 66_048, "ln_2": 256,
                              "feed_forward": 131_712, "total": 198_272}


def test_architecture_follows_the_checkpoint(small_engine):
    a = describe_architecture(small_engine)
    assert a["architecture"]["d_model"] == 64 and a["architecture"]["num_heads"] == 2
    assert a["architecture"]["head_dim"] == 32
    c = a["components"]
    total = (c["token_embedding"] + c["position_embedding"] + c["ln_final"]
             + sum(b["total"] for b in c["blocks"]))
    assert total == a["parameters"] == small_engine.model.num_parameters()
    assert c["token_embedding"] == 4000 * 64


# --- API and page -----------------------------------------------------------------

@pytest.fixture(scope="module")
def client(small_engine):
    with TestClient(server.create_app(small_engine)) as c:
        yield c


def test_architecture_endpoint(client, small_engine):
    response = client.get("/api/architecture")
    assert response.status_code == 200
    assert response.json() == describe_architecture(small_engine)


def test_attention_endpoint(client, small_engine):
    response = client.post("/api/attention", json={"prompt": PROMPT})
    assert response.status_code == 200
    data = response.json()
    assert data == capture_attention(small_engine, PROMPT)
    assert data["verified"] is True


@pytest.mark.parametrize("body", [
    {}, {"prompt": ""}, {"prompt": 5}, {"prompt": "hi", "block": 1},
    {"prompt": "a" * (MAX_ATTENTION_CHARS + 1)}, {"prompt": " the" * 31}, {"prompt": " \t"},
])
def test_attention_endpoint_rejects_invalid_input(client, body):
    response = client.post("/api/attention", json=body)
    assert response.status_code == 422
    assert response.json()["error"]


def test_attention_endpoint_reports_a_failed_verification(client, monkeypatch):
    real = attention._recompute
    monkeypatch.setattr(attention, "_recompute",
                        lambda *a: (real(*a)[0], real(*a)[1] * 2))
    response = client.post("/api/attention", json={"prompt": PROMPT})
    assert response.status_code == 500
    assert "unverified" in response.json()["error"]


def test_attention_endpoint_method(client):
    assert client.get("/api/attention").status_code == 405


def test_model_page(client):
    response = client.get("/model")
    assert response.status_code == 200
    assert "<title>How Eclipse works</title>" in response.text
    assert "real attention weights" in response.text
    for name in ("model.js", "model.css"):
        assert client.get(f"/static/{name}").status_code == 200


def test_main_page_links_to_the_model_page(client):
    assert 'href="/model"' in client.get("/").text


def test_generate_endpoint_still_works(client):
    response = client.post("/api/generate", json={"prompt": "Once", "max_new_tokens": 5, "seed": 1})
    assert response.status_code == 200


NEW_SOURCES = ["eclipse/attention.py", "web/model.html", "web/model.js", "web/model.css"]
FORBIDDEN = re.compile(r"openai|anthropic|claude|huggingface|hf_hub|transformers|"
                       r"from_pretrained|pipeline\(|ollama|cohere|mistral|gemini|"
                       r"requests\.|urllib|http\.client|aiohttp", re.IGNORECASE)


@pytest.mark.parametrize("relative", NEW_SOURCES)
def test_new_files_reference_no_external_model_or_host(relative):
    source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    assert not FORBIDDEN.search(source), FORBIDDEN.search(source)
    # The SVG namespace is an identifier, not a request.
    source = source.replace("http://www.w3.org/2000/svg", "")
    assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", source, re.IGNORECASE)


# --- the real checkpoint (skipped until it is in place) ----------------------------

@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(),
                    reason="trained checkpoint not present at checkpoints/sr_data200mb_s1337_best.pt")
def test_real_checkpoint_attention():
    engine = Eclipse.load()
    assert describe_architecture(engine)["parameters"] == 1_767_424
    ids = engine.tokenizer.encode(PROMPT)
    hooked = capture_weights(engine, ids)
    spy = SoftmaxSpy()
    with torch.inference_mode(), spy:
        engine.model(torch.tensor([ids], device=engine.device))
    assert all(torch.equal(h, s[0]) for h, s in zip(hooked, spy.outputs))
    result = capture_attention(engine, PROMPT)
    assert result["num_blocks"] == 6 and result["num_heads"] == 4 and result["verified"]
