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
# from .bindings import optimize

# The C wrapper is generated from mrcal-genpywrap.py
# from . import bindings_npsp as _drcal_npsp

# from .projections import *
# from .cameramodel import cameramodel
# from .poseutils import rt_from_Rt

# The C wrapper is generated from poseutils-genpywrap.py
# from . import bindings_poseutils_npsp as _poseutils_npsp
# from .stereo import *
# from .visualization import *
# from .model_analysis import *
# from .synthetic_data import *
# from .calibration import compute_chessboard_corners
# from .image_transforms import *
# from .utils import *
# from .triangulation import *

from .model_analysis import projection_diff
from . import cahvor
from .bindings_poseutils_npsp import identity_Rt
from .projections import unproject
from .visualization import (
    show_projection_diff,
    show_splined_model_correction,
    show_projection_uncertainty,
    show_valid_intrinsics_region,
    show_residuals_vectorfield,
    show_residuals_magnitudes,
    show_residuals_directions,
    show_residuals_regional,
    show_distortion_off_pinhole,
    show_distortion_off_pinhole_radial,
    show_geometry,
    show_residuals_board_observation,
    show_residuals_histogram,
)
from .utils import (
    align_procrustes_points_Rt01,
    hypothesis_board_corner_positions,
    measurements_board,
)
from .cameramodel import cameramodel
from .poseutils import invert_Rt, rt_from_Rt, Rt_from_rt, compose_Rt, compose_rt
from .bindings import (
    lensmodel_metadata_and_config,
    lensmodel_num_params,
    load_image,
    optimize,
)
from .calibration import (
    compute_chessboard_corners,
    estimate_joint_frame_poses,
    estimate_monocular_calobject_poses_Rt_tocam,
    seed_stereographic,
)


__all__ = [
    "projection_diff",
    "cahvor",
    "identity_Rt",
    "unproject",
    "show_projection_diff",
    "show_residuals_histogram",
    "show_residuals_board_observation",
    "show_geometry",
    "show_residuals_directions",
    "show_residuals_regional",
    "show_distortion_off_pinhole",
    "show_distortion_off_pinhole_radial",
    "show_valid_intrinsics_region",
    "show_projection_uncertainty",
    "show_splined_model_correction",
    "show_residuals_vectorfield",
    "show_residuals_magnitudes",
    "align_procrustes_points_Rt01",
    "hypothesis_board_corner_positions",
    "measurements_board",
    "cameramodel",
    "invert_Rt",
    "rt_from_Rt",
    "Rt_from_rt",
    "compose_Rt",
    "compose_rt",
    "lensmodel_metadata_and_config",
    "lensmodel_num_params",
    "load_image",
    "optimize",
    "compute_chessboard_corners",
    "estimate_joint_frame_poses",
    "estimate_monocular_calobject_poses_Rt_tocam",
    "seed_stereographic",
]
