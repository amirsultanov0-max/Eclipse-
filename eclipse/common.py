"""
Helpers shared by the Phase 3 program.

Lifted from train_transformer.py rather than imported from it, so that the
inference program has no dependency on the frozen Phase 2 training script.
"""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def display_path(path):
    """A path relative to the repo when it is inside it, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
