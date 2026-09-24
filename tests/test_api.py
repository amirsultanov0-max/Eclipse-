"""
The HTTP API and the pages it serves, through FastAPI's test client, with a
stand-in checkpoint loaded by the real loader. The last test uses the real
trained checkpoint and is skipped until it is in place.
"""

import pytest
from fastapi.testclient import TestClient

import server
from eclipse.inference import DEFAULT_CHECKPOINT, MAX_NEW_TOKENS_LIMIT, MAX_PROMPT_CHARS, Eclipse


@pytest.fixture(scope="module")
def client(small_engine):
    with TestClient(server.create_app(small_engine)) as c:
        yield c


def generate(client, **body):
    return client.post("/api/generate", json={"prompt": "Once upon a time", **body})


# --- pages --------------------------------------------------------------------

def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    page = response.text
    assert "<title>Eclipse</title>" in page
    assert "TinyStories" in page and "not an assistant" in page
    assert "/static/app.js" in page and "/static/style.css" in page


@pytest.mark.parametrize("name", ["app.js", "style.css", "index.html"])
def test_static_files(client, name):
    assert client.get(f"/static/{name}").status_code == 200


def test_no_path_traversal_through_static(client):
    assert client.get("/static/../server.py").status_code == 404
    assert client.get("/static/%2e%2e/server.py").status_code == 404


def test_api_docs_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


# --- /api/info ----------------------------------------------------------------

def test_info_describes_the_loaded_checkpoint(client, small_engine):
    info = client.get("/api/info").json()
    assert info["name"] == "Eclipse"
    assert info["checkpoint"] == small_engine.info.checkpoint
    assert info["architecture"]["d_model"] == 64
    assert info["architecture"]["num_heads"] == 2
    assert info["parameters"] == small_engine.model.num_parameters()
    assert info["provenance"]["status"] == "unverifiable"
    assert info["limits"]["max_new_tokens"] == MAX_NEW_TOKENS_LIMIT
    assert info["defaults"] == {"max_new_tokens": 150, "temperature": 1.0, "top_k": 40,
                                "seed": None}
    assert info["device"] == "cpu"


# --- /api/generate: valid requests --------------------------------------------

def test_generate_returns_a_continuation(client):
    response = generate(client, max_new_tokens=20, seed=5)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Once upon a time"
    assert isinstance(data["completion"], str)
    assert 0 <= data["generated_tokens"] <= 20
    assert data["stop_reason"] in {"eos", "max_new_tokens"}
    assert data["seed"] == 5
    assert data["prompt_tokens"] == 4
    assert data["settings"] == {"max_new_tokens": 20, "temperature": 1.0, "top_k": 40, "seed": 5}
    assert data["elapsed_ms"] >= 0


def test_defaults_apply_when_settings_are_omitted(client):
    data = client.post("/api/generate", json={"prompt": "The dog"}).json()
    assert data["settings"]["max_new_tokens"] == 150
    assert data["settings"]["top_k"] == 40
    assert data["generated_tokens"] <= 150


def test_seed_makes_the_api_reproducible(client):
    a = generate(client, seed=1337, max_new_tokens=30).json()
    b = generate(client, seed=1337, max_new_tokens=30).json()
    assert a["completion"] == b["completion"]


def test_greedy_and_no_top_k_are_accepted(client):
    greedy = [generate(client, temperature=0, max_new_tokens=15).json()["completion"]
              for _ in range(2)]
    assert greedy[0] == greedy[1]
    assert generate(client, top_k=None, temperature=1, max_new_tokens=5).status_code == 200


def test_api_output_equals_direct_inference(client, small_engine):
    from eclipse.inference import GenerationSettings
    api = generate(client, seed=11, max_new_tokens=25).json()
    direct = small_engine.generate("Once upon a time", GenerationSettings(max_new_tokens=25,
                                                                          seed=11))
    assert api["completion"] == direct.completion


# --- /api/generate: invalid requests ------------------------------------------

@pytest.mark.parametrize("body", [
    {},                                                   # no prompt
    {"prompt": ""},
    {"prompt": "a" * (MAX_PROMPT_CHARS + 1)},
    {"prompt": 123},
    {"prompt": ["Once"]},
    {"prompt": "Once", "max_new_tokens": 0},
    {"prompt": "Once", "max_new_tokens": MAX_NEW_TOKENS_LIMIT + 1},
    {"prompt": "Once", "max_new_tokens": "10"},
    {"prompt": "Once", "max_new_tokens": 10.5},
    {"prompt": "Once", "max_new_tokens": True},
    {"prompt": "Once", "temperature": -0.5},
    {"prompt": "Once", "temperature": 2.5},
    {"prompt": "Once", "temperature": "hot"},
    {"prompt": "Once", "top_k": 0},
    {"prompt": "Once", "seed": -1},
    {"prompt": "Once", "seed": 2 ** 32},
    {"prompt": "Once", "model": "gpt-4"},                 # unknown fields are refused
])
def test_invalid_request_is_422_with_a_message(client, body):
    response = client.post("/api/generate", json=body)
    assert response.status_code == 422
    assert response.json()["error"]


def test_whitespace_prompt_is_rejected(client):
    response = client.post("/api/generate", json={"prompt": "   \n "})
    assert response.status_code == 422
    assert "empty" in response.json()["error"]


def test_prompt_beyond_the_context_is_rejected(client):
    response = client.post("/api/generate", json={"prompt": " the" * 600})
    assert response.status_code == 422
    assert "context is 512" in response.json()["error"]


def test_top_k_beyond_the_vocabulary_is_rejected(client):
    response = generate(client, top_k=4001)
    assert response.status_code == 422
    assert "top_k" in response.json()["error"]


def test_nan_temperature_is_rejected(client):
    response = client.post("/api/generate", content=b'{"prompt": "Once", "temperature": NaN}',
                           headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_malformed_json_is_rejected(client):
    response = client.post("/api/generate", content=b'{"prompt": ',
                           headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_wrong_method(client):
    assert client.get("/api/generate").status_code == 405


# --- starting the server ------------------------------------------------------

def test_server_refuses_to_start_without_the_checkpoint(tmp_path, monkeypatch, capsys):
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("server started"))
    with pytest.raises(SystemExit) as exit_info:
        server.main(["--checkpoint", str(tmp_path / "missing.pt")])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "checkpoint not found" in err and "no fallback model" in err


def test_server_starts_with_a_loadable_checkpoint(small_checkpoint, monkeypatch, capsys):
    import uvicorn
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: started.update(
        app=app, host=host, port=port))
    server.main(["--checkpoint", str(small_checkpoint[0]), "--port", "8123"])
    assert started["host"] == "127.0.0.1" and started["port"] == 8123
    assert started["app"].state.engine.info.architecture["d_model"] == 64
    assert "provenance: unverifiable" in capsys.readouterr().out


# --- the real checkpoint (skipped until it is in place) ------------------------

@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(),
                    reason="trained checkpoint not present at checkpoints/sr_data200mb_s1337_best.pt")
def test_real_checkpoint_through_the_api():
    with TestClient(server.create_app(Eclipse.load())) as client:
        info = client.get("/api/info").json()
        assert info["parameters"] == 1_767_424
        assert info["provenance"]["status"] == "verified"
        data = generate(client, max_new_tokens=40, seed=1337).json()
        assert data["generated_tokens"] > 0
