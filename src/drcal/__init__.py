"""The main drcal Python package"""

from .image_transforms import (
    image_transformation_map,
    pinhole_model_for_reprojection,
    transform_image,
)
from .model_analysis import is_within_valid_intrinsics_region, projection_diff
from .bindings_poseutils_npsp import identity_Rt
from .projections import unproject
from .visualization import (
    annotate_image__valid_intrinsics_region,
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
from .poseutils import (
    compose_r,
    invert_Rt,
    rt_from_Rt,
    Rt_from_rt,
    compose_Rt,
    compose_rt,
    r_from_R,
    R_from_r,
    R_from_quat,
    Rt_from_qt,
    invert_R,
    rotate_point_r,
    rotate_point_R,
    invert_rt,
    qt_from_Rt,
    transform_point_Rt,
    transform_point_rt,
)
from .bindings import (
    lensmodel_metadata_and_config,
    lensmodel_num_params,
    optimize,
)
from .calibration import (
    compute_chessboard_corners,
    estimate_joint_frame_poses,
    estimate_monocular_calobject_poses_Rt_tocam,
    seed_stereographic,
)

from .image_utils import save_image, load_image


__all__ = [
    "image_transformation_map",
    "pinhole_model_for_reprojection",
    "transform_image",
    "is_within_valid_intrinsics_region",
    "projection_diff",
    "identity_Rt",
    "unproject",
    "annotate_image__valid_intrinsics_region",
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
    "compose_r",
    "r_from_R",
    "R_from_r",
    "R_from_quat",
    "Rt_from_qt",
    "invert_R",
    "rotate_point_r",
    "rotate_point_R",
    "invert_rt",
    "qt_from_Rt",
    "transform_point_Rt",
    "transform_point_rt",
    "lensmodel_metadata_and_config",
    "lensmodel_num_params",
    "load_image",
    "save_image",
    "optimize",
    "compute_chessboard_corners",
    "estimate_joint_frame_poses",
    "estimate_monocular_calobject_poses_Rt_tocam",
    "seed_stereographic",
]
