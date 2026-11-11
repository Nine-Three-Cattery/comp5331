"""Clean-room Trajectron++ pedestrian core.

This package intentionally has no import path into the pinned official
Trajectron++ repository.  It consumes only the model-neutral files under
``data/neutral`` and exposes a small training/prediction interface for the
course experiment.
"""

from .data import CachedNeutralDataset, NeutralWindowDataset, collate_windows, load_split
from .model import IndependentTrajectronPP

__all__ = [
    "NeutralWindowDataset",
    "CachedNeutralDataset",
    "collate_windows",
    "load_split",
    "IndependentTrajectronPP",
]
