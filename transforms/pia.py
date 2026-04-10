"""Deprecated shim for legacy imports.

PIA augmentation now lives in augmentation/pia.
Riemannian alignment is provided in transforms/riemann.py.
"""

from .base import BaseTransform, NoOpTransform
from .riemann import RiemannianAlignTransform

# Backward compatibility alias (avoid breaking old imports).
PIARawTransform = RiemannianAlignTransform

__all__ = [
    "BaseTransform",
    "NoOpTransform",
    "RiemannianAlignTransform",
    "PIARawTransform",
]
