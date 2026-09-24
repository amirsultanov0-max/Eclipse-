"""
The 3D page's embedding data: that PCA coordinates, explained variance and
cosine neighbours are computed correctly from the loaded embedding matrix
(checked against independent computations from the state_dict), the endpoint
limits, the page, and the pinned vendored Three.js files. The last test needs
the real checkpoint and is skipped until it is in place.
"""

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from conftest import PROJECT_ROOT
from eclipse import embedding
from eclipse.embedding import (MAX_COUNT, MAX_K, MAX_NEIGHBORS, MIN_COUNT, embedding_neighbors,
                               embedding_projection, principal_components, selected_token_ids)
from eclipse.inference import DEFAULT_CHECKPOINT, Eclipse, GenerationRequestError

VENDOR = PROJECT_ROOT / "web" / "vendor" / "three-0.185.0"
# SHA256 of each file as extracted from three-0.185.0.tgz on the npm registry, whose
# sha512 matched npm's published integrity value.
VENDOR_SHA256 = {
    "three.module.min.js": "86bcee248b64f44bcfc23c331ae74619061957d59cab040171dcb6fb5900beb6",
    "three.core.min.js": "0e9dd2793e01d0d9eb4f2ab00b4ffcdd4488275ebebee5c31fa8d347bc29f0bf",
    "OrbitControls.js": "faabb4e8dfd9235ee4a9fd7c9a3d75f90f1689dbd4944bd6fd32117dacec5f93",
    "LICENSE": "8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc",
}


def state_dict_embedding(engine):
    """Read independently of eclipse/embedding.py: straight from the state_dict."""
    return engine.model.state_dict()["token_embedding.weight"].double().numpy()


def cosine(E, a, b):
    return float(E[a] @ E[b] / (np.linalg.norm(E[a]) * np.linalg.norm(E[b])))


# --- PCA ----------------------------------------------------------------------------

def test_pca_matches_the_covariance_eigendecomposition(small_engine):
    E = state_dict_embedding(small_engine)
    centred = E - E.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(centred, rowvar=False))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    pca = principal_components(small_engine)
    for i in range(3):
        assert abs(pca["components"][i] @ evecs[:, i]) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(pca["coords"], centred @ pca["components"].T)
    assert np.allclose(pca["explained_variance_ratio"], evals[:3] / evals.sum())


def test_pca_signs_are_fixed_and_cached(small_engine):
    pca = principal_components(small_engine)
    for row in pca["components"]:
        assert row[np.argmax(np.abs(row))] > 0
    assert principal_components(small_engine) is pca


def test_returned_coordinates_are_the_pca_coordinates(small_engine):
    data = embedding_projection(small_engine)
    coords = principal_components(small_engine)["coords"]
    for token in data["tokens"]:
        assert token["xyz"] == pytest.approx(coords[token["id"]].tolist(), abs=5e-6)
    ratio = principal_components(small_engine)["explained_variance_ratio"]
    assert data["explained_variance_ratio"] == pytest.approx(ratio.tolist(), abs=1e-6)
    assert data["explained_variance_total"] == pytest.approx(float(ratio.sum()), abs=1e-6)
    assert 0 < data["explained_variance_total"] < 1


# --- token selection ----------------------------------------------------------------

def test_selection_is_the_first_word_tokens_in_learned_order(small_engine, tokenizer):
    data = embedding_projection(small_engine, count=300)
    ids = [t["id"] for t in data["tokens"]]
    expected = [i for i in range(len(tokenizer))
                if re.fullmatch(r" [A-Za-z]{3,}", tokenizer.id_to_token[i])][:300]
    assert ids == expected
    assert [t["text"] for t in data["tokens"][:3]] == [" the", " and", " was"]
    assert all(t["text"] == tokenizer.id_to_token[t["id"]] for t in data["tokens"])
    assert data["count"] == 300
    for probe in (" happy", " sad", " dog", " cat", " mom", " dad"):
        assert tokenizer.token_to_id[probe] in ids
    assert ids == selected_token_ids(small_engine)[:300]


# --- neighbours are real cosine similarities -----------------------------------------

@pytest.mark.parametrize("k", [1, 3, 5])
def test_drawn_neighbors_are_the_true_top_k_cosines(small_engine, k):
    E = state_dict_embedding(small_engine)
    data = embedding_projection(small_engine, count=120, k=k)
    ids = [t["id"] for t in data["tokens"]]
    for token in data["tokens"]:
        brute = sorted(((cosine(E, token["id"], j), j) for j in ids if j != token["id"]),
                       reverse=True)[:k]
        assert len(token["neighbors"]) == k
        for (nid, sim), (true_sim, true_id) in zip(token["neighbors"], brute):
            assert sim == pytest.approx(true_sim, abs=6e-5)
            assert sim == pytest.approx(cosine(E, token["id"], nid), abs=6e-5)
        sims = [s for _, s in token["neighbors"]]
        assert sims == sorted(sims, reverse=True)
        assert token["id"] not in [n for n, _ in token["neighbors"]]
        assert set(n for n, _ in token["neighbors"]) <= set(ids)


def test_full_vocabulary_neighbors_match_brute_force(small_engine, tokenizer):
    E = state_dict_embedding(small_engine)
    tid = tokenizer.token_to_id[" happy"]
    result = embedding_neighbors(small_engine, tid, k=10)
    brute = sorted(((cosine(E, tid, j), j) for j in range(len(E)) if j != tid), reverse=True)[:10]
    assert [n["id"] for n in result["neighbors"]] == [j for _, j in brute]
    for n, (true_sim, _) in zip(result["neighbors"], brute):
        assert n["similarity"] == pytest.approx(true_sim, abs=6e-5)
        assert n["text"] == tokenizer.id_to_token[n["id"]]
    assert result["text"] == " happy"
    assert "4,000" in result["scope"]


# --- limits and payload size --------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(count=MIN_COUNT - 1), dict(count=MAX_COUNT + 1), dict(count="300"), dict(count=True),
    dict(k=0), dict(k=MAX_K + 1), dict(k=2.0),
])
def test_projection_rejects_bad_arguments(small_engine, kwargs):
    with pytest.raises(GenerationRequestError):
        embedding_projection(small_engine, **kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(token_id=-1), dict(token_id=4000), dict(token_id=1.5), dict(token_id="7"),
    dict(token_id=5, k=0), dict(token_id=5, k=MAX_NEIGHBORS + 1),
])
def test_neighbors_reject_bad_arguments(small_engine, kwargs):
    with pytest.raises(GenerationRequestError):
        embedding_neighbors(small_engine, **kwargs)


def test_payload_sizes(small_engine):
    default = len(json.dumps(embedding_projection(small_engine)))
    largest = len(json.dumps(embedding_projection(small_engine, count=MAX_COUNT, k=MAX_K)))
    assert default < 64 * 1024
    assert largest < 256 * 1024


# --- API and page -------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(small_engine):
    with TestClient(server.create_app(small_engine)) as c:
        yield c


def test_projection_endpoint(client, small_engine):
    response = client.get("/api/embedding-projection?count=150&k=4")
    assert response.status_code == 200
    assert response.json() == embedding_projection(small_engine, count=150, k=4)
    assert client.get("/api/embedding-projection").json()["count"] == 300


def test_neighbors_endpoint(client, small_engine):
    response = client.get("/api/embedding-neighbors?token_id=242&k=6")
    assert response.status_code == 200
    assert response.json() == embedding_neighbors(small_engine, 242, 6)


@pytest.mark.parametrize("query", [
    "/api/embedding-projection?count=9", "/api/embedding-projection?count=801",
    "/api/embedding-projection?k=0", "/api/embedding-projection?k=9",
    "/api/embedding-projection?count=abc", "/api/embedding-neighbors",
    "/api/embedding-neighbors?token_id=4000", "/api/embedding-neighbors?token_id=-1",
    "/api/embedding-neighbors?token_id=x", "/api/embedding-neighbors?token_id=3&k=21",
])
def test_endpoints_return_422_with_a_message(client, query):
    response = client.get(query)
    assert response.status_code == 422
    assert response.json()["error"]


def test_model_3d_page(client):
    response = client.get("/model-3d")
    assert response.status_code == 200
    page = response.text
    assert "<title>Eclipse in 3D</title>" in page
    importmap = json.loads(re.search(r'<script type="importmap">(.*?)</script>', page, re.S).group(1))
    assert importmap == {"imports": {"three": "/static/vendor/three-0.185.0/three.module.min.js"}}
    assert '<script type="module" src="/static/model3d.js"></script>' in page
    assert "real attention weights" in page
    for name in ("model3d.js", "model3d.css"):
        assert client.get(f"/static/{name}").status_code == 200


@pytest.mark.parametrize("name", sorted(VENDOR_SHA256))
def test_vendored_three_is_the_pinned_release(client, name):
    data = (VENDOR / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == VENDOR_SHA256[name]
    response = client.get(f"/static/vendor/three-0.185.0/{name}")
    assert response.status_code == 200
    assert response.content == data
    if name.endswith(".js"):
        assert "javascript" in response.headers["content-type"]


def test_vendor_directory_holds_only_the_pinned_files():
    assert sorted(p.name for p in VENDOR.iterdir()) == sorted(VENDOR_SHA256)
    assert "MIT License" in (VENDOR / "LICENSE").read_text()


def test_existing_pages_do_not_load_the_3d_code(client):
    for path in ("/", "/model"):
        page = client.get(path).text
        assert "three" not in page.lower() and "model3d" not in page


NEW_SOURCES = ["eclipse/embedding.py", "web/model3d.html", "web/model3d.js", "web/model3d.css"]
FORBIDDEN = re.compile(r"openai|anthropic|claude|huggingface|hf_hub|transformers|"
                       r"from_pretrained|pipeline\(|ollama|cohere|mistral|gemini|"
                       r"requests\.|urllib|http\.client|aiohttp|fetch\(\s*['\"`]https?:", re.IGNORECASE)


@pytest.mark.parametrize("relative", NEW_SOURCES)
def test_new_files_reference_no_external_model_or_host(relative):
    source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    assert not FORBIDDEN.search(source), FORBIDDEN.search(source)
    assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", source, re.IGNORECASE)


# --- the real checkpoint (skipped until it is in place) ------------------------------

@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(),
                    reason="trained checkpoint not present at checkpoints/sr_data200mb_s1337_best.pt")
def test_real_checkpoint_embeddings(capsys):
    engine = Eclipse.load()
    data = embedding_projection(engine)
    assert (data["vocab_size"], data["d_model"], data["count"]) == (4000, 128, 300)
    assert 0 < data["explained_variance_total"] < 1
    for token in data["tokens"]:
        sims = [s for _, s in token["neighbors"]]
        assert all(-1 <= s <= 1 for s in sims) and sims == sorted(sims, reverse=True)
    # Printed for a person to read (pytest -s); no claim about what they should be.
    with capsys.disabled():
        print(f"\nexplained variance, first 3 PCs: {data['explained_variance_ratio']} "
              f"(total {data['explained_variance_total']:.4f})")
        for probe in (" happy", " dog", " mom", " big"):
            n = embedding_neighbors(engine, engine.tokenizer.token_to_id[probe], k=8)
            print(f"{probe!r:>9} -> " + ", ".join(f"{x['text']!r} {x['similarity']:.3f}"
                                                  for x in n["neighbors"]))
