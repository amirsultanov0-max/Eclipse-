"""
The loaded model's token embedding matrix, for the 3D view.

Everything here is read from engine.model.token_embedding.weight, the
vocab_size x d_model matrix the checkpoint was trained with. The matrix is tied,
so the same rows also produce the output logits. Nothing is modified.

- Positions: the first three principal components of that matrix, from an SVD
  of the mean-centred rows. Three components keep only part of the variance, so
  the explained-variance ratio is always returned alongside them.
- Similarities: cosine similarity between full d_model-dimensional rows, never
  between the 3D positions.
"""

import re
import weakref

import numpy as np

from eclipse.inference import GenerationRequestError

DEFAULT_COUNT, MIN_COUNT, MAX_COUNT = 300, 10, 800
DEFAULT_K, MAX_K = 3, 8
DEFAULT_NEIGHBORS, MAX_NEIGHBORS = 10, 20
WORD_TOKEN = re.compile(r"^ [A-Za-z]{3,}$")
SELECTION_RULE = ("the first {count} tokens, in the order the tokenizer learned them (roughly "
                  "most frequent first), that are a space followed by at least 3 letters")

_pca_cache = weakref.WeakKeyDictionary()


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _embedding(engine):
    return engine.model.token_embedding.weight.detach().cpu().double().numpy()


def _unit_rows(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def principal_components(engine):
    """Top-3 PCA of the embedding rows, computed once per loaded model."""
    model = engine.model
    if model in _pca_cache:
        return _pca_cache[model]
    E = _embedding(engine)
    centred = E - E.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:3].copy()
    # SVD leaves each component's sign arbitrary; fix it so the layout is stable.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    variance = singular ** 2
    result = {"coords": centred @ components.T, "components": components,
              "explained_variance_ratio": variance[:3] / variance.sum()}
    _pca_cache[model] = result
    return result


def selected_token_ids(engine):
    """Word-start tokens in learned (ID) order; see SELECTION_RULE."""
    tokenizer = engine.tokenizer
    return [i for i in range(len(tokenizer)) if WORD_TOKEN.match(tokenizer.id_to_token[i])]


def embedding_projection(engine, count=DEFAULT_COUNT, k=DEFAULT_K):
    if not _is_int(count) or not MIN_COUNT <= count <= MAX_COUNT:
        raise GenerationRequestError(f"count must be an integer from {MIN_COUNT} to {MAX_COUNT}")
    if not _is_int(k) or not 1 <= k <= MAX_K:
        raise GenerationRequestError(f"k must be an integer from 1 to {MAX_K}")

    pca = principal_components(engine)
    ids = selected_token_ids(engine)[:count]
    unit = _unit_rows(_embedding(engine)[ids])
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    text = engine.tokenizer.id_to_token
    ratio = pca["explained_variance_ratio"]
    E_shape = engine.model.token_embedding.weight.shape

    return {
        "vocab_size": E_shape[0],
        "d_model": E_shape[1],
        "count": len(ids),
        "k": k,
        "explained_variance_ratio": [round(float(r), 6) for r in ratio],
        "explained_variance_total": round(float(ratio.sum()), 6),
        "selection": SELECTION_RULE.format(count=len(ids)),
        "projection": ("first 3 principal components of the mean-centred "
                       f"{E_shape[0]:,} x {E_shape[1]} token embedding matrix"),
        "similarity": (f"cosine similarity of the full {E_shape[1]}-dimensional embedding rows; "
                       "neighbors are the most similar tokens among the ones returned"),
        # neighbors: [[token_id, cosine similarity], ...], most similar first
        "tokens": [{"id": tid, "text": text[tid],
                    "xyz": [round(float(v), 5) for v in pca["coords"][tid]],
                    "neighbors": [[ids[j], round(float(sims[r, j]), 4)] for j in order[r]]}
                   for r, tid in enumerate(ids)],
    }


def embedding_neighbors(engine, token_id, k=DEFAULT_NEIGHBORS):
    """The k most similar tokens to token_id across the whole vocabulary."""
    vocab_size = engine.model.token_embedding.weight.shape[0]
    if not _is_int(token_id) or not 0 <= token_id < vocab_size:
        raise GenerationRequestError(f"token_id must be an integer from 0 to {vocab_size - 1}")
    if not _is_int(k) or not 1 <= k <= MAX_NEIGHBORS:
        raise GenerationRequestError(f"k must be an integer from 1 to {MAX_NEIGHBORS}")
    unit = _unit_rows(_embedding(engine))
    sims = unit @ unit[token_id]
    sims[token_id] = -np.inf
    top = np.argsort(-sims, kind="stable")[:k]
    text = engine.tokenizer.id_to_token
    return {
        "id": token_id,
        "text": text[token_id],
        "scope": f"all {vocab_size:,} tokens in the vocabulary",
        "neighbors": [{"id": int(i), "text": text[int(i)], "similarity": round(float(sims[i]), 4)}
                      for i in top],
    }
