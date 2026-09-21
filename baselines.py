"""
Stage 3 reference losses, to give the trained model's number a meaning.

  1. uniform        log(4,000) = 8.294. No knowledge at all; where we start.
  2. unigram        Always predict the overall token frequencies, ignoring the
                    previous token. The gap between this and the trained model
                    is what one token of context is actually worth.
  3. bigram counts  The best any bigram model can do, worked out directly by
                    counting pairs instead of by gradient descent. Our trained
                    model cannot beat this, so it tells us whether training
                    converged or simply ran out of steps.

All losses are mean cross-entropy in nats, the same units train.py reports.
"""

import math

import numpy as np


def uniform_loss(vocab_size):
    return math.log(vocab_size)


def unigram_losses(train_stream, val_stream, vocab_size):
    """Loss from predicting overall token frequencies, ignoring context."""
    train = train_stream.astype(np.int64)
    val = val_stream.astype(np.int64)

    counts = np.bincount(train, minlength=vocab_size).astype(np.float64)
    # Add-1 smoothing, so a token that only ever appears in the validation
    # split cannot get probability zero (which would make the loss infinite).
    # With 4.7M tokens it changes the frequent tokens by a negligible amount.
    log_probs = np.log((counts + 1.0) / (counts.sum() + vocab_size))

    # The targets are every token except the first one in the stream.
    return float(-log_probs[train[1:]].mean()), float(-log_probs[val[1:]].mean())


SMOOTHING_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TUNE_FRACTION = 0.05  # share of the TRAINING stream held back to choose k


def _pair_counts(stream, vocab_size):
    """Count every (current, next) pair. Returns the flat table and row sums."""
    # Flattening each pair into a single index lets one bincount produce the
    # whole 4,000 x 4,000 table.
    pairs = stream[:-1] * vocab_size + stream[1:]
    counts = np.bincount(pairs, minlength=vocab_size * vocab_size)
    row_totals = counts.reshape(vocab_size, vocab_size).sum(axis=1)
    return counts, row_totals


def _add_k_loss(counts, row_totals, stream, vocab_size, k):
    """Mean cross-entropy of add-k smoothed counts on `stream`."""
    pairs = stream[:-1] * vocab_size + stream[1:]
    pair_counts = counts[pairs].astype(np.float64)
    totals = row_totals[stream[:-1]].astype(np.float64)
    return float(-np.log((pair_counts + k) / (totals + k * vocab_size)).mean())


def select_smoothing_k(train_stream, vocab_size, ks=SMOOTHING_GRID, tune_fraction=TUNE_FRACTION):
    """
    Choose the add-k smoothing constant using TRAINING data only.

    The last `tune_fraction` of the training stream is held back: counts come
    from the front part, and each candidate k is scored on the held-back tail.
    The validation stream is never consulted, so the baseline it later produces
    is not tuned on the data it is measured against.

    Returns (best_k, rows) with rows as (k, tuning_loss).
    """
    train = train_stream.astype(np.int64)
    split = int(len(train) * (1 - tune_fraction))
    counts, row_totals = _pair_counts(train[:split], vocab_size)
    rows = [(k, _add_k_loss(counts, row_totals, train[split:], vocab_size, k)) for k in ks]
    return min(rows, key=lambda row: row[1])[0], rows


def count_bigram_losses(train_stream, val_stream, vocab_size, k):
    """
    The count-based bigram: P(next | current) estimated by counting pairs,
    smoothed with a FIXED k (chosen beforehand by select_smoothing_k).

    Counts are fitted on the whole training stream. Returns
    (mle_train_loss, train_loss, val_loss).

    Unsmoothed counts are the true floor for TRAINING loss, but they score
    infinitely badly on validation: any pair that never occurred in training
    gets probability zero. Add-k smoothing shaves probability off the pairs we
    saw and spreads it over the ones we did not, which is what makes a finite
    validation number possible.
    """
    train = train_stream.astype(np.int64)
    val = val_stream.astype(np.int64)

    counts, row_totals = _pair_counts(train, vocab_size)

    train_pairs = train[:-1] * vocab_size + train[1:]
    mle_train = float(-np.log(
        counts[train_pairs].astype(np.float64) / row_totals[train[:-1]].astype(np.float64)
    ).mean())

    return (mle_train,
            _add_k_loss(counts, row_totals, train, vocab_size, k),
            _add_k_loss(counts, row_totals, val, vocab_size, k))


def top_next_tokens(train_stream, vocab_size, token_id, n=5):
    """The n most common tokens that follow `token_id` in the training stream."""
    train = train_stream.astype(np.int64)
    following = train[1:][train[:-1] == token_id]
    counts = np.bincount(following, minlength=vocab_size)
    order = np.argsort(counts)[::-1][:n]
    total = counts.sum()
    return [(int(i), int(counts[i]), float(counts[i] / total)) for i in order if counts[i] > 0]
