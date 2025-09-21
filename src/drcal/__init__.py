#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The main mrcal Python package

This package doesn't contain any code itself, but all the mrcal.mmm submodules
export their symbols here for convenience. So any function that can be called as
mrcal.mmm.fff() can be called as mrcal.fff() instead. The latter is preferred.

"""

# The C wrapper is written by us in mrcal-pywrap.c
from .bindings import optimize

# The C wrapper is generated from mrcal-genpywrap.py
from . import bindings_npsp as _drcal_npsp

from .projections import *
from .cameramodel import cameramodel
from .poseutils import rt_from_Rt

# The C wrapper is generated from poseutils-genpywrap.py
from . import bindings_poseutils_npsp as _poseutils_npsp
from .stereo import *
from .visualization import *
from .model_analysis import *
from .synthetic_data import *
from .calibration import compute_chessboard_corners
from .image_transforms import *
from .utils import *
from .triangulation import *


__all__ = [
    "_drcal_npsp",
    "_poseutils_npsp",
    "rt_from_Rt",
    "optimize",
    "cameramodel",
    "compute_chessboard_corners",
]
