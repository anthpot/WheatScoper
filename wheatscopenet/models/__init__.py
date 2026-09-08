"""Model package for WheatScopeNet (paper Figures 3-6).

Public entry points:

* :class:`WheatScopeNet` / :func:`build_wheatscopenet` -- the full network (Fig. 3).
* :class:`PHSBlock`, :class:`PH_SS2D`, :class:`SS2D_OP`, :class:`SS2D`,
  :class:`SimpleFusionGate` -- the parallel hybrid spatial block (Fig. 4, Eq. 1-5).
* :class:`SCAB`, :class:`LightCSF` -- the cross-scale bridge (Fig. 5, 6, Eq. 6-11).
* :class:`SEBlock` -- the squeeze-and-excitation unit shared by the bridge modules.
"""

from .bridge import SCAB, LightCSF
from .layers import SEBlock
from .phs_block import PH_SS2D, PHSBlock, SimpleFusionGate
from .ss2d import SS2D, SS2D_OP
from .wheatscopenet import WheatScopeNet, build_wheatscopenet

__all__ = [
    "WheatScopeNet",
    "build_wheatscopenet",
    "PHSBlock",
    "PH_SS2D",
    "SS2D_OP",
    "SS2D",
    "SimpleFusionGate",
    "SCAB",
    "LightCSF",
    "SEBlock",
]
