"""WheatScopeNet -- lightweight organ-level wheat canopy segmentation network.

Reference implementation of the segmentation model of the WheatScoper framework
("WheatScoper: A lightweight organ-based framework for multi-view wheat
phenotyping using time-series RGB images"). The network segments RGB canopy
images into three classes -- background, spike and leaf.

Typical usage::

    from wheatscopenet import WheatScopeNet

    model = WheatScopeNet(num_classes=3)
    logits = model(images)          # [B, 3, H, W]

The public model symbols are imported lazily so that ``import wheatscopenet``
stays cheap and does not require the optional CUDA selective-scan kernel to be
present at package-import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

__all__ = ["WheatScopeNet", "build_wheatscopenet", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - import for static type checkers only
    from .models import WheatScopeNet, build_wheatscopenet


def __getattr__(name: str) -> Any:
    """Resolve the re-exported model symbols on first access (PEP 562)."""
    if name in ("WheatScopeNet", "build_wheatscopenet"):
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
