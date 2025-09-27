import numpy as np
from . import numpy_utils as npu


def ref_calibration_object(
    W=None,
    H=None,
    object_spacing=None,
    *,
    optimization_inputs=None,
    calobject_warp=None,
    x_corner0=0,
    x_corner1=None,
    Nx=None,
    y_corner0=0,
    y_corner1=None,
    Ny=None,
):
    r"""Return the geometry of the calibration object

    SYNOPSIS

        import drcal.gnuplotlib as gp
        import numpysane as nps

        obj = drcal.ref_calibration_object( 10,6, 0.1 )

        print(obj.shape)
        ===> (6, 10, 3)

        gp.plot( nps.clump( obj[...,:2], n=2),
                 tuplesize = -2,
                 _with     = 'points',
                 _xrange   = (-0.1,1.0),
                 _yrange   = (-0.1,0.6),
                 unset     = 'grid',
                 square    = True,
                 terminal  = 'dumb 74,45')

         0.6 +---------------------------------------------------------------+
             |     +          +           +           +          +           |
             |                                                               |
         0.5 |-+   A     A    A     A     A     A     A     A    A     A   +-|
             |                                                               |
             |                                                               |
         0.4 |-+   A     A    A     A     A     A     A     A    A     A   +-|
             |                                                               |
             |                                                               |
         0.3 |-+   A     A    A     A     A     A     A     A    A     A   +-|
             |                                                               |
             |                                                               |
         0.2 |-+   A     A    A     A     A     A     A     A    A     A   +-|
             |                                                               |
             |                                                               |
         0.1 |-+   A     A    A     A     A     A     A     A    A     A   +-|
             |                                                               |
             |                                                               |
           0 |-+   A     A    A     A     A     A     A     A    A     A   +-|
             |                                                               |
             |     +          +           +           +          +           |
        -0.1 +---------------------------------------------------------------+
                   0         0.2         0.4         0.6        0.8          1

    Returns the geometry of a calibration object in its own reference coordinate
    system in a (H,W,3) array. Only a grid-of-points calibration object is
    supported, possibly with some deformation (i.e. what the internal drcal solver
    supports). Each row of the output is an (x,y,z) point. The origin is at the
    corner of the grid, so ref_calibration_object(...)[0,0,:] is np.array((0,0,0)).
    The grid spans x and y, with z representing the depth: z=0 for a flat
    calibration object.

    A simple parabolic board warping model is supported by passing a (2,) array in
    calobject_warp. These 2 values describe additive flex along the x axis and along
    the y axis, in that order. In each direction the flex is a parabola, with the
    parameter k describing the max deflection at the center. If the edges were at
    +-1 we'd have

        z = k*(1 - x^2)

    The edges we DO have are at (0,N-1), so the equivalent expression is

        xr = x / (N-1)
        z = k*( 1 - 4*xr^2 + 4*xr - 1 ) =
            4*k*(xr - xr^2) =
            4*k*xr*(1 - xr)

    By default we return the coordinates of the chessboard CORNERS only, but this
    function can return the position of ANY point on the chessboard. This can be
    controlled by passing the x_corner0,x_corner1,Nx arguments (and/or their y-axis
    versions). This selects the grid of points we return, in chessboard-corner
    coordinates (0 is the first corner, 1 is the second corner, etc). We use
    np.linspace(x_corner0, x_corner1, Nx). By default we have

    - x_corner0 = 0
    - x_corner1 = W-1
    - Nx        = W

    So we only return the coordinates of the corners by default. The points returned
    along the y axis work similarly, using their variables.

    If optimization_inputs is given, we get H,W,object_spacing and calobject_warp
    from the inputs. In this case, (H,W,object_spacing) must all be None. Otherwise
    all of (H,W,object_spacing) must be given. Thus it's possible to call this
    function like this:

        model = drcal.cameramodel('calibration.cameramodel')
        obj = drcal.ref_calibration_object(optimization_inputs =
                                           optimization_inputs)

    ARGUMENTS

    - W: how many chessboard corners we have in the horizontal direction

    - H: how many chessboard corners we have in the vertical direction

    - object_spacing: the distance between adjacent points in the calibration
      object. If a scalar is given, a square object is assumed, and the vertical and
      horizontal distances are assumed to be identical. An array of shape (..., 2)
      can be given: the last dimension is (spacing_h, spacing_w), and the preceding
      dimensions are used for broadcasting

    - calobject_warp: optional array of shape (2,) defaults to None. Describes the
      warping of the calibration object. If None, the object is flat. If an array is
      given, the values describe the maximum additive deflection along the x and y
      axes. Extended array can be given for broadcasting

    - optimization_inputs: the input from a calibrated model. Usually the output of
      drcal.cameramodel.optimization_inputs() call. If given,
      (H,W,object_spacing,calobject_warp) are all read from these inputs, and must
      not be given separately.

    - x_corner0: optional value, defaulting to 0. Selects the first point in the
      linear horizontal grid we're returning. This indexes the chessboard corners,
      and we start with the first corner by default

    - x_corner1: optional value, defaulting to W-1. Selects the last point in the
      linear horizontal grid we're returning. This indexes the chessboard corners,
      and we end with the last corner by default

    - Nx: optional value, defaulting to W. Selects the number of points we return in
      the horizontal direction, between x_corner0 and x_corner1 inclusive.

    - y_corner0,y_corner1,Ny: same as x_corner0,x_corner1,Nx but acting in the
      vertical direction

    This function supports broadcasting across object_spacing and calobject_warp

    RETURNED VALUES

    The calibration object geometry in a (..., Ny,Nx,3) array, with the leading
    dimensions set by the broadcasting rules. Usually Ny = H and Nx = W

    """
    Noptions_base = 0
    options_base = ("W", "H", "object_spacing")
    for o in options_base:
        if locals()[o] is not None:
            Noptions_base += 1
    if not (Noptions_base == 0 or Noptions_base == len(options_base)):
        raise Exception(
            f"Options '{options_base}': ALL must be given, or NONE must be given"
        )
    if Noptions_base > 0 and optimization_inputs is not None:
        raise Exception(
            f"Options '{options_base}' and 'optimization_inputs' cannot both be given"
        )
    if Noptions_base == 0 and optimization_inputs is None:
        raise Exception(
            f"One of options '{options_base}' and 'optimization_inputs' MUST be given"
        )

    if optimization_inputs is not None:
        H, W = optimization_inputs["observations_board"].shape[-3:-1]

        object_spacing = optimization_inputs["calibration_object_spacing"]
        calobject_warp = optimization_inputs["calobject_warp"]

    if Nx is None:
        Nx = W
    if Ny is None:
        Ny = H
    if x_corner1 is None:
        x_corner1 = W - 1
    if y_corner1 is None:
        y_corner1 = H - 1

    # shape (Ny,Nx)
    xx, yy = np.meshgrid(
        np.linspace(x_corner0, x_corner1, Nx), np.linspace(y_corner0, y_corner1, Ny)
    )

    # shape (Ny,Nx,3)
    full_object = npu.glue(
        npu.mv(npu.cat(xx, yy), 0, -1), np.zeros(xx.shape + (1,)), axis=-1
    )

    # object_spacing has shape (..., 2)
    object_spacing = np.array(object_spacing)
    if object_spacing.ndim == 0:
        object_spacing = np.array((1, 1)) * object_spacing
    object_spacing = npu.dummy(object_spacing, -2, -2)
    # object_spacing now has shape (..., 1,1,2)

    if object_spacing.ndim > 3:
        # extend full_object to the output shape I want
        full_object = full_object * np.ones(object_spacing.shape[:-3] + (1, 1, 1))
    full_object[..., :2] *= object_spacing

    if calobject_warp is not None:
        xr = xx / (W - 1)
        yr = yy / (H - 1)
        dx = 4.0 * xr * (1.0 - xr)
        dy = 4.0 * yr * (1.0 - yr)

        # To allow broadcasting over calobject_warp
        if calobject_warp.ndim > 1:
            # shape (..., 1,1,2)
            calobject_warp = npu.dummy(calobject_warp, -2, -2)
            # extend full_object to the output shape I want
            full_object = full_object * np.ones(calobject_warp.shape[:-3] + (1, 1, 1))
        full_object[..., 2] += calobject_warp[..., 0] * dx
        full_object[..., 2] += calobject_warp[..., 1] * dy

    return full_object
