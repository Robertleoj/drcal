import numpy as np
from . import numpy_utils as npu

from .calibration_objects import ref_calibration_object

from .bindings import optimizer_callback
from .utils import hypothesis_board_corner_positions

from .poseutils import (
    Rt_from_rt,
    compose_Rt,
    invert_Rt,
    rotate_point_R,
    transform_point_Rt,
)
from .projections import project, unproject
from .model_analysis import _observed_pixel_uncertainty_from_inputs

from .bindings_poseutils_npsp import identity_Rt


def synthesize_board_observations(
    models,
    *,
    object_width_n,
    object_height_n,
    object_spacing,
    calobject_warp,
    rt_ref_boardcenter,
    rt_ref_boardcenter__noiseradius,
    Nframes,
    max_oblique_angle_deg=None,
    pcamera_nominal_ref=np.array((0, 0, 0), dtype=float),
    which="all-cameras-must-see-full-board",
):
    r"""Produce synthetic chessboard observations

SYNOPSIS

    models = [drcal.cameramodel("0.cameramodel"),
              drcal.cameramodel("1.cameramodel"),]

    # shapes (Nframes, Ncameras, object_height_n, object_width_n, 2) and
    #        (Nframes, 4, 3)
    q,Rt_ref_boardref = \
        drcal.synthesize_board_observations( \
          models,

          # board geometry
          object_width_n  = 10,
          object_height_n = 12,
          object_spacing  = 0.1,
          calobject_warp  = None,

          # mean board pose and the radius of the added uniform noise
          rt_ref_boardcenter              = rt_ref_boardcenter,
          rt_ref_boardcenter__noiseradius = rt_ref_boardcenter__noiseradius,

          # How many frames we want
          Nframes = 100,

          which = 'some-cameras-must-see-half-board')

    # q now contains the synthetic pixel observations, but some of them will be
    # out of view. I construct an (x,y,weight) observations array, as expected
    # by the optimizer, and I set the weight for the out-of-view points to -1 to
    # tell the optimizer to ignore those points


    # Set the weights to 1 initially
    # shape (Nframes, Ncameras, object_height_n, object_width_n, 3)
    observations = npu.glue(q,
                            np.ones( q.shape[:-1] + (1,) ),
                            axis = -1)

    # shape (Ncameras, 1, 1, 2)
    imagersizes = npu.mv( npu.cat(*[ m.imagersize() for m in models ]),
                          -2, -4 )

    observations[ np.any( q              < 0, axis=-1 ), 2 ] = -1.
    observations[ np.any( q-imagersizes >= 0, axis=-1 ), 2 ] = -1.

Given a description of a calibration object and of the cameras observing it,
produces perfect pixel observations of the objects by those cameras. We return a
dense observation array: every corner observation from every chessboard pose
will be reported for every camera. Some of these observations MAY be
out-of-view, depending on the value of the 'which' argument; see description
below. The example above demonstrates how to mark such out-of-bounds
observations as outliers to tell the optimization to ignore these.

The "models" provides the intrinsics and extrinsics.

The calibration objects are nominally have pose rt_ref_boardcenter in the
reference coordinate system, with each pose perturbed uniformly with radius
rt_ref_boardcenter__noiseradius. This is nonstandard since here I'm placing the
board origin at its center instead of the corner (as
drcal.ref_calibration_object() does). But this is more appropriate to the usage
of this function. The returned Rt_ref_boardref transformation DOES use the
normal corner-referenced board geometry

Returns the point observations and the chessboard poses that produced these
observations.

ARGUMENTS

- models: an array of drcal.cameramodel objects, one for each camera we're
  simulating. This is the intrinsics and the extrinsics. Ncameras = len(models)

- object_width_n:  the number of horizontal points in the calibration object grid

- object_height_n: the number of vertical points in the calibration object grid

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical.

- calobject_warp: a description of the calibration board warping. None means "no
  warping": the object is flat. Otherwise this is an array of shape (2,). See
  the docs for ref_calibration_object() for the meaning of the values in this
  array.

- rt_ref_boardcenter: the nominal pose of the calibration object, in the
  reference coordinate system. This is an rt transformation from a
  center-referenced calibration object to the reference coordinate system

- rt_ref_boardcenter__noiseradius: the deviation-from-nominal for the chessboard
  pose for each frame. I add uniform noise to rt_ref_boardcenter, with each
  element sampled independently, with the radius given here.

- Nframes: how many frames of observations to return

- which: a string, defaulting to 'all-cameras-must-see-full-board'. Controls the
  requirements on the visibility of the returned points. Valid values:

  - 'all-cameras-must-see-full-board': We return only those chessboard poses
    that produce observations that are FULLY visible by ALL the cameras.

  - 'some-cameras-must-see-full-board': We return only those chessboard poses
    that produce observations that are FULLY visible by AT LEAST ONE camera.

  - 'all-cameras-must-see-half-board': We return only those chessboard poses
    that produce observations that are AT LEAST HALF visible by ALL the cameras.

  - 'some-cameras-must-see-half-board': We return only those chessboard poses
    that produce observations that are AT LEAST HALF visible by AT LEAST ONE
    camera.

- max_oblique_angle_deg: optional value, defaulting to None. If non-None, we
  only return observations where the board normal is within this angle of the
  vector to the nominal camera (at pcamera_nominal_ref). This ensures that the
  boards all "face" the camera to a certain degree

- pcamera_nominal_ref: optional vector, defaulting to (0,0,0). Used in
  conjunction with max_oblique_angle_deg to make sure the observation angle
  isn't too oblique

RETURNED VALUES

We return a tuple:

- q: an array of shape (Nframes, Ncameras, object_height, object_width, 2)
  containing the pixel coordinates of the generated observations

- Rt_ref_boardref: an array of shape (Nframes, 4,3) containing the poses of the
  chessboards. This transforms the object returned by ref_calibration_object()
  to the pose that was projected, in the ref coord system

    """

    # Can visualize results with this script:
    r"""
    r = np.array((30, 0, 0,), dtype=float) * np.pi/180.

    model = drcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                             np.array((1000., 1000., 1000., 1000.,))),
                               imagersize = np.array((2000,2000)) )
    Rt_ref_boardref = \
        drcal.synthesize_board_observations([model],
                                            object_width_n                  = 5,
                                            object_height_n                 = 20,
                                            object_spacing                  = 0.1,
                                            calobject_warp                  = None,
                                            rt_ref_boardcenter              = npu.glue(r, np.array((0,0,3.)), axis=-1),
                                            rt_ref_boardcenter__noiseradius = np.array((0,0,0., 0,0,0)),
                                            Nframes                         = 1) [1]
    drcal.show_geometry( models_or_extrinsics_rt_fromref = np.zeros((1,1,6), dtype=float),
                         frames_rt_toref                 = drcal.rt_from_Rt(Rt_ref_boardref),
                         object_width_n                  = 20,
                         object_height_n                 = 5,
                         object_spacing                  = 0.1,
                         _set = 'xyplane 0',
                         wait = 1 )
    """

    which_valid = (
        "all-cameras-must-see-full-board",
        "some-cameras-must-see-full-board",
        "all-cameras-must-see-half-board",
        "some-cameras-must-see-half-board",
    )

    if which not in which_valid:
        raise Exception(f"'which' argument must be one of {which_valid}")

    Ncameras = len(models)

    # I move the board, and keep the cameras stationary.
    #
    # Camera coords: x,y with pixels, z forward
    # Board coords:  x,y in-plane. z forward (i.e. back towards the camera)

    # The center of the board is at the origin (ignoring warping)
    board_center = np.array(
        (
            (object_width_n - 1) * object_spacing / 2.0,
            (object_height_n - 1) * object_spacing / 2.0,
            0,
        )
    )

    # shape: (Nh,Nw,3)
    board_reference = (
        ref_calibration_object(
            object_width_n,
            object_height_n,
            object_spacing,
            calobject_warp=calobject_warp,
        )
        - board_center
    )

    # Transformation from the board returned by ref_calibration_object() to
    # the one I use here. It's a shift to move the origin to the center of the
    # board
    Rt_boardref_origboardref = identity_Rt()
    Rt_boardref_origboardref[3, :] = -board_center

    if max_oblique_angle_deg is not None:
        max_cos_oblique_angle = np.cos(max_oblique_angle_deg * np.pi / 180.0)
    else:
        max_cos_oblique_angle = None

    def get_observation_chunk():
        """Make Nframes observations, and return them all, even the out-of-view ones"""

        # I compute the full random block in one shot. This is useful for
        # simulations that want to see identical poses when asking for N-1
        # random poses and when asking for the first N-1 of a set of N random
        # poses

        # shape (Nframes,6)
        randomblock = np.random.uniform(low=-1.0, high=1.0, size=(Nframes, 6))

        # shape(Nframes,4,3)
        Rt_ref_boardref = Rt_from_rt(
            rt_ref_boardcenter + randomblock * rt_ref_boardcenter__noiseradius
        )

        # shape = (Nframes, Nh,Nw,3)
        boards_ref = transform_point_Rt(  # shape (Nframes, 1,1,4,3)
            npu.mv(Rt_ref_boardref, 0, -5),
            # shape ( Nh,Nw,3)
            board_reference,
        )

        # I project full_board. Shape: (Nframes,Ncameras,Nh,Nw,2)
        q = npu.mv(
            npu.cat(
                *[
                    project(
                        transform_point_Rt(
                            models[i].extrinsics_Rt_fromref(), boards_ref
                        ),
                        *models[i].intrinsics(),
                    )
                    for i in range(Ncameras)
                ]
            ),
            0,
            1,
        )

        return q, Rt_ref_boardref

    def cull(q, Rt_ref_boardref, which):
        # q               has shape (Nframes,Ncameras,Nh,Nw,2)
        # Rt_ref_boardref has shape (Nframes,4,3)

        ######## Throw out extreme oblique views
        if max_cos_oblique_angle is not None:
            nref_position = Rt_ref_boardref[..., 3, :] - pcamera_nominal_ref
            nref_position /= npu.dummy(npu.mag(nref_position), -1)
            nref_orientation = Rt_ref_boardref[..., :3, 2]
            costh = np.abs(npu.inner(nref_position, nref_orientation))
            i = costh > max_cos_oblique_angle

            q = q[i]
            Rt_ref_boardref = Rt_ref_boardref[i]

        ######## I pick only those frames where at least one cameras sees the
        ######## whole board

        # shape (Nframes,Ncameras,Nh,Nw)
        mask_visible = (q[..., 0] >= 0) * (q[..., 1] >= 0)
        for i in range(Ncameras):
            W, H = models[i].imagersize()
            mask_visible[:, i, ...] *= (q[:, i, :, :, 0] <= W - 1) * (
                q[:, i, :, :, 1] <= H - 1
            )

        # shape (Nframes, Ncameras)
        Nvisible = np.count_nonzero(mask_visible, axis=(-1, -2))

        Nh, Nw = q.shape[2:4]
        if which == "all-cameras-must-see-full-board":
            iframe = np.all(Nvisible == Nh * Nw, axis=-1)
        elif which == "some-cameras-must-see-full-board":
            iframe = np.any(Nvisible == Nh * Nw, axis=-1)
        elif which == "all-cameras-must-see-half-board":
            iframe = np.all(Nvisible > Nh * Nw // 2, axis=-1)
        elif which == "some-cameras-must-see-half-board":
            iframe = np.any(Nvisible > Nh * Nw // 2, axis=-1)
        else:
            raise Exception(
                "Unknown 'which' argument. This is a bug. I checked for the valid options at the top of this function"
            )

        # q               has shape (Nframes_inview,Ncameras,Nh*Nw,2)
        # Rt_ref_boardref has shape (Nframes_inview,4,3)
        return q[iframe, ...], Rt_ref_boardref[iframe, ...]

    # shape (Nframes_sofar,Ncameras,Nh,Nw,2)
    q = np.zeros((0, Ncameras, object_height_n, object_width_n, 2), dtype=float)
    # shape (Nframes_sofar,4,3)
    Rt_ref_boardref = np.zeros((0, 4, 3), dtype=float)

    # I keep creating data, until I get Nframes-worth of in-view observations
    while True:
        q_here, Rt_ref_boardref_here = get_observation_chunk()

        q_here, Rt_ref_boardref_here = cull(q_here, Rt_ref_boardref_here, which)

        q = npu.glue(q, q_here, axis=-5)
        Rt_ref_boardref = npu.glue(Rt_ref_boardref, Rt_ref_boardref_here, axis=-3)
        if q.shape[0] >= Nframes:
            q = q[:Nframes, ...]
            Rt_ref_boardref = Rt_ref_boardref[:Nframes, ...]
            break

    return q, compose_Rt(Rt_ref_boardref, Rt_boardref_origboardref)


def _noisy_observation_vectors_for_triangulation(
    p, Rt01, intrinsics0, intrinsics1, Nsamples, sigma
):
    # p has shape (...,3)

    # shape (..., 2)
    q0 = project(p, *intrinsics0)
    q1 = project(transform_point_Rt(invert_Rt(Rt01), p), *intrinsics1)

    # shape (..., 1,2). Each has x,y
    q0 = npu.dummy(q0, -2)
    q1 = npu.dummy(q1, -2)

    q_noise = np.random.randn(*p.shape[:-1], Nsamples, 2, 2) * sigma
    # shape (..., Nsamples,2). Each has x,y
    q0_noise = q_noise[..., :, 0, :]
    q1_noise = q_noise[..., :, 1, :]

    q0_noisy = q0 + q0_noise
    q1_noisy = q1 + q1_noise

    # shape (..., Nsamples, 3)
    v0local_noisy = unproject(q0_noisy, *intrinsics0)
    v1local_noisy = unproject(q1_noisy, *intrinsics1)
    v0_noisy = v0local_noisy
    v1_noisy = rotate_point_R(Rt01[:3, :], v1local_noisy)

    # All have shape (..., Nsamples,3)
    return v0local_noisy, v1local_noisy, v0_noisy, v1_noisy, q0, q1, q0_noisy, q1_noisy


def make_perfect_observations(optimization_inputs, *, observed_pixel_uncertainty=None):
    r"""Write perfect observations with perfect noise into the optimization_inputs

    SYNOPSIS

        model = drcal.cameramodel("0.cameramodel")
        optimization_inputs = model.optimization_inputs()

        optimization_inputs['calobject_warp'] = np.array((1e-3, -1e-3))
        drcal.make_perfect_observations(optimization_inputs)

        # We now have perfect data assuming a slightly WARPED chessboard. Let's use
        # this data to compute a calibration assuming a FLAT chessboard
        optimization_inputs['calobject_warp'] *= 0.
        optimization_inputs['do_optimize_calobject_warp'] = False

        drcal.optimize(**optimization_inputs)

        model = drcal.cameramodel(optimization_inputs = optimization_inputs,
                                  icam_intrinsics     = model.icam_intrinsics())
        model.write("reoptimized.cameramodel")

        # We can now look at the residuals and diffs to see how much a small
        # chessboard deformation affects our results

    Tracking down all the sources of error in real-world models computed by drcal is
    challenging: the models never fit perfectly, and the noise never follows the
    assumed distribution exactly. It is thus really useful to be able to run
    idealized experiments where both the models and the noise are perfect. We can
    then vary only one variable to judge its effects. Since everything else is
    perfect, we can be sure that any imperfections in the results are due only to
    the variable we tweaked. In the sample above we evaluated the effect of a small
    chessboard deformation.

    This function ingests optimization_inputs from a completed calibration. It then
    assumes that all the geometry and intrinsics are perfect, and sets the
    observations to projections of that perfect geometry. If requested, perfect
    gaussian noise is then added to the observations.

    THIS FUNCTION MODIFIES THE INPUT OPTIMIZATION_INPUTS

    ARGUMENTS

    - optimization_inputs: the input from a calibrated model. Usually the output of
      drcal.cameramodel.optimization_inputs() call. The output is written into
      optimization_inputs['observations_board'] and
      optimization_inputs['observations_point']

    - observed_pixel_uncertainty: optional standard deviation of the noise to apply.
      By default the noise applied has same variance as the noise in the input
      optimization_inputs. If we want to omit the noise, pass
      observed_pixel_uncertainty = 0

    RETURNED VALUES

    None

    """

    x = optimizer_callback(
        **optimization_inputs, no_jacobian=True, no_factorization=True
    )[1]

    if observed_pixel_uncertainty is None:
        observed_pixel_uncertainty = _observed_pixel_uncertainty_from_inputs(
            optimization_inputs, x=x
        )

    if (
        "indices_frame_camintrinsics_camextrinsics" in optimization_inputs
        and optimization_inputs["indices_frame_camintrinsics_camextrinsics"] is not None
        and optimization_inputs["indices_frame_camintrinsics_camextrinsics"].size
    ):
        # shape (Nobservations, Nheight, Nwidth, 3)
        pcam = hypothesis_board_corner_positions(**optimization_inputs)[0]
        i_intrinsics = optimization_inputs["indices_frame_camintrinsics_camextrinsics"][
            :, 1
        ]
        # shape (Nobservations,1,1,Nintrinsics)
        intrinsics = npu.mv(optimization_inputs["intrinsics"][i_intrinsics], -2, -4)
        optimization_inputs["observations_board"][..., :2] = project(
            pcam, optimization_inputs["lensmodel"], intrinsics
        )

    if (
        "indices_point_camintrinsics_camextrinsics" in optimization_inputs
        and optimization_inputs["indices_point_camintrinsics_camextrinsics"] is not None
        and optimization_inputs["indices_point_camintrinsics_camextrinsics"].size
    ):
        indices_point_camintrinsics_camextrinsics = optimization_inputs[
            "indices_point_camintrinsics_camextrinsics"
        ]

        # shape (Nobservations,3)
        pref = optimization_inputs["points"][
            indices_point_camintrinsics_camextrinsics[:, 0]
        ]

        # shape (Nobservations,4,3)
        Rt_cam_ref = npu.glue(
            identity_Rt(),
            Rt_from_rt(optimization_inputs["extrinsics_rt_fromref"]),
            axis=-3,
        )[indices_point_camintrinsics_camextrinsics[:, 2] + 1]

        # shape (Nobservations,3)
        pcam = transform_point_Rt(Rt_cam_ref, pref)

        # shape (Nobservations,Nintrinsics)
        intrinsics = optimization_inputs["intrinsics"][
            indices_point_camintrinsics_camextrinsics[:, 1]
        ]
        optimization_inputs["observations_point"][..., :2] = project(
            pcam, optimization_inputs["lensmodel"], intrinsics
        )

    ########### I have perfect data. Now add perfect noise
    if observed_pixel_uncertainty == 0:
        return

    for what in ("observations_board", "observations_point"):
        if (
            what in optimization_inputs
            and optimization_inputs[what] is not None
            and optimization_inputs[what].size
        ):
            noise_nominal = observed_pixel_uncertainty * np.random.randn(
                *optimization_inputs[what][..., :2].shape
            )

            weight = npu.dummy(optimization_inputs[what][..., 2], axis=-1)
            weight[weight <= 0] = 1.0  # to avoid dividing by 0

            optimization_inputs[what][..., :2] += noise_nominal / weight
