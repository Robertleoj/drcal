"""
Low-level routines for core mrcal operations

This is the written-in-C Python extension module that underlies the routines in
mrcal.h. Most of the functions in this module (those prefixed with "_") are
not meant to be called directly, but have Python wrappers that should be used
instead.

All functions are exported into the mrcal module. So you can call these via
mrcal._mrcal.fff() or mrcal.fff(). The latter is preferred.
"""
from __future__ import annotations
__all__: list[str] = ['CHOLMOD_factorization', 'corresponding_icam_extrinsics', 'decode_observation_indices_points_triangulated', 'drt_ref_refperturbed__dbpacked', 'knots_for_splined_models', 'lensmodel_metadata_and_config', 'lensmodel_num_params', 'load_image', 'measurement_index_boards', 'measurement_index_points', 'measurement_index_points_triangulated', 'measurement_index_regularization', 'num_intrinsics_optimization_params', 'num_measurements', 'num_measurements_boards', 'num_measurements_points', 'num_measurements_points_triangulated', 'num_measurements_regularization', 'num_states', 'num_states_calobject_warp', 'num_states_extrinsics', 'num_states_frames', 'num_states_intrinsics', 'num_states_points', 'optimize', 'optimizer_callback', 'pack_state', 'save_image', 'state_index_calobject_warp', 'state_index_extrinsics', 'state_index_frames', 'state_index_intrinsics', 'state_index_points', 'supported_lensmodels', 'traverse_sensor_links', 'unpack_state']
class CHOLMOD_factorization:
    """
    
    A basic Python interface to CHOLMOD
    
    SYNOPSIS
    
        from scipy.sparse import csr_matrix
    
        indptr  = np.array([0, 2, 3, 6, 8])
        indices = np.array([0, 2, 2, 0, 1, 2, 1, 2])
        data    = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    
        Jsparse = csr_matrix((data, indices, indptr))
        Jdense  = Jsparse.toarray()
        print(Jdense)
        ===> [[1. 0. 2.] 
              [0. 0. 3.] 
              [4. 5. 6.] 
              [0. 7. 8.]]
    
        bt = np.array(((1., 5., 3.), (2., -2., -8)))
        print(nps.transpose(bt))
        ===> [[ 1.  2.] 
              [ 5. -2.] 
              [ 3. -8.]]
    
        F  = mrcal.CHOLMOD_factorization(Jsparse)
        xt = F.solve_xt_JtJ_bt(bt)
        print(nps.transpose(xt))
        ===> [[ 0.02199662  0.33953751] 
              [ 0.31725888  0.46982516] 
              [-0.21996616 -0.50648618]]
    
        print(nps.matmult(nps.transpose(Jdense), Jdense, nps.transpose(xt)))
        ===> [[ 1.  2.] 
              [ 5. -2.] 
              [ 3. -8.]]
    
    The core of the mrcal optimizer is a sparse linear least squares solver using
    CHOLMOD to solve a large, sparse linear system. CHOLMOD is a C library, but it
    is sometimes useful to invoke it from Python.
    
    The CHOLMOD_factorization class factors a matrix JtJ, and this method uses that
    factorization to efficiently solve the linear equation JtJ x = b. The usual
    linear algebra conventions refer to column vectors, but numpy generally deals
    with row vectors, so I talk about solving the equivalent transposed problem: xt
    JtJ = bt. The difference is purely notational.
    
    The class takes a sparse array J as an argument in __init__(). J is optional,
    but there's no way in Python to pass it later, so from Python you should always
    pass J. This is optional for internal initialization from C code.
    
    J must be given as an instance of scipy.sparse.csr_matrix. csr is a row-major
    sparse representation. CHOLMOD wants column-major matrices, so it see this
    matrix J as a transpose: the CHOLMOD documentation refers to this as "At". And
    the CHOLMOD documentation talks about factoring AAt, while I talk about
    factoring JtJ. These are the same thing.
    
    The factorization of JtJ happens in __init__(), and we use this factorization
    later (as many times as we want) to solve JtJ x = b by calling
    solve_xt_JtJ_bt().
    
    This class carefully checks its input for validity, but makes no effort to be
    flexible: anything that doesn't look right will result in an exception.
    Specifically:
    
    - J.data, J.indices, J.indptr must all be numpy arrays
    
    - J.data, J.indices, J.indptr must all have exactly one dimension
    
    - J.data, J.indices, J.indptr must all be C-contiguous (the normal numpy order)
    
    - J.data must hold 64-bit floating-point values (dtype=float)
    
    - J.indices, J.indptr must hold 32-bit integers (dtype=np.int32)
    
    ARGUMENTS
    
    The __init__() function takes
    
    - J: a sparse array in a scipy.sparse.csr_matrix object
    
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def solve_xt_JtJ_bt(*args, **kwargs):
        """
        
        Solves the linear system JtJ x = b using CHOLMOD
        
        SYNOPSIS
        
            from scipy.sparse import csr_matrix
        
            indptr  = np.array([0, 2, 3, 6, 8])
            indices = np.array([0, 2, 2, 0, 1, 2, 1, 2])
            data    = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        
            Jsparse = csr_matrix((data, indices, indptr))
            Jdense  = Jsparse.toarray()
            print(Jdense)
            ===> [[1. 0. 2.] 
                  [0. 0. 3.] 
                  [4. 5. 6.] 
                  [0. 7. 8.]]
        
            bt = np.array(((1., 5., 3.), (2., -2., -8)))
            print(nps.transpose(bt))
            ===> [[ 1.  2.] 
                  [ 5. -2.] 
                  [ 3. -8.]]
        
            F  = mrcal.CHOLMOD_factorization(Jsparse)
            xt = F.solve_xt_JtJ_bt(bt)
            print(nps.transpose(xt))
            ===> [[ 0.02199662  0.33953751] 
                  [ 0.31725888  0.46982516] 
                  [-0.21996616 -0.50648618]]
        
            print(nps.matmult(nps.transpose(Jdense), Jdense, nps.transpose(xt)))
            ===> [[ 1.  2.] 
                  [ 5. -2.] 
                  [ 3. -8.]]
        
        The core of the mrcal optimizer is a sparse linear least squares solver using
        CHOLMOD to solve a large, sparse linear system. CHOLMOD is a C library, but it
        is sometimes useful to invoke it from Python.
        
        The CHOLMOD_factorization class factors a matrix JtJ, and this method uses that
        factorization to efficiently solve the linear equation JtJ x = b. The usual
        linear algebra conventions refer to column vectors, but numpy generally deals
        with row vectors, so I talk about solving the equivalent transposed problem: xt
        JtJ = bt. The difference is purely notational.
        
        As many vectors b as we'd like may be given at one time (in rows of bt). The
        dimensions of the returned array xt will match the dimensions of the given array
        bt.
        
        Broadcasting is supported: any leading dimensions will be processed correctly,
        as long as bt has shape (..., Nstate)
        
        This function carefully checks its input for validity, but makes no effort to be
        flexible: anything that doesn't look right will result in an exception.
        Specifically:
        
        - bt must be C-contiguous (the normal numpy order)
        
        - bt must contain 64-bit floating-point values (dtype=float)
        
        This function is now able to pass different values of "sys" to the internal
        cholmod_solve2() call. This is specified with the "mode" argument. By default,
        we use CHOLMOD_A, which is the default behavior: we solve JtJ x = b. All the
        other modes supported by CHOLMOD are supported. From cholmod.h:
        
          CHOLMOD_A:    solve Ax=b
          CHOLMOD_LDLt: solve LDL'x=b
          CHOLMOD_LD:   solve LDx=b
          CHOLMOD_DLt:  solve DL'x=b
          CHOLMOD_L:    solve Lx=b
          CHOLMOD_Lt:   solve L'x=b
          CHOLMOD_D:    solve Dx=b
          CHOLMOD_P:    permute x=Px
          CHOLMOD_Pt:   permute x=P'x
        
        See the CHOLMOD documentation and source for details.
        
        ARGUMENTS
        
        - bt: a numpy array of shape (..., Nstate). This array must be C-contiguous and
          it must have dtype=float
        
        - sys: optional string, defaulting to "A": solve JtJ x = b. Selects the specific
          problem being solved; see the description above. The value passed to "sys"
          should be the string with or without the "CHOLMOD_" prefix
        
        RETURNED VALUE
        
        The transpose of the solution array x, in a numpy array of the same shape as the
        input bt
        """
    def __str__(self):
        """
        Return str(self).
        """
    def rcond(self):
        """
        
        Compute rough estimate of reciprocal of condition number
        
        SYNOPSIS
        
            b, x, J, factorization = \\
                mrcal.optimizer_callback(**optimization_inputs)
        
            rcond = factorization.rcond()
        
        Calls cholmod_rcond(). Its documentation says:
        
          Returns a rough estimate of the reciprocal of the condition number: the
          minimum entry on the diagonal of L (or absolute entry of D for an LDLT
          factorization) divided by the maximum entry. L can be real, complex, or
          zomplex. Returns -1 on error, 0 if the matrix is singular or has a zero or NaN
          entry on the diagonal of L, 1 if the matrix is 0-by-0, or
          min(diag(L))/max(diag(L)) otherwise. Never returns NaN; if L has a NaN on the
          diagonal it returns zero instead.
        
        ARGUMENTS
        
        - None
        
        RETURNED VALUE
        
        A single floating point value: an estimate of the reciprocal of the condition
        number
        
        
        """
def _rectification_maps(*args, **kwargs):
    """
    
    Construct image transformation maps to make rectified images
    
    This is an internal function. You probably want mrcal.rectification_maps(). See
    the docs for that function for details.
    
    """
def _rectified_resolution(*args, **kwargs):
    """
    
    Compute the resolution to be used for the rectified system
    
    This is an internal function. You probably want mrcal.rectified_resolution(). See the
    docs for that function for details.
    
    """
def _rectified_system(*args, **kwargs):
    """
    
    Build rectified models for stereo rectification
    
    This is an internal function. You probably want mrcal.rectified_system(). See the
    docs for that function for details.
    
    """
def corresponding_icam_extrinsics(*args, **kwargs):
    """
    
    Return the icam_extrinsics corresponding to a given icam_intrinsics
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        icam_intrinsics = m.icam_intrinsics()
    
        icam_extrinsics = \\
            mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                                **optimization_inputs)
    
        if icam_extrinsics >= 0:
            extrinsics_rt_fromref_at_calibration_time = \\
                optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics]
        else:
            extrinsics_rt_fromref_at_calibration_time = \\
                mrcal.identity_rt()
    
    When calibrating cameras, each observation is associated with some camera
    intrinsics (lens parameters) and some camera extrinsics (geometry). Those two
    chunks of data live in different parts of the optimization vector, and are
    indexed independently. If we have STATIONARY cameras, then each set of camera
    intrinsics is associated with exactly one set of camera extrinsics, and we can
    use THIS function to query this correspondence. If we have moving cameras, then
    a single physical camera would have one set of intrinsics but many different
    extrinsics, and this function will throw an exception.
    
    Furthermore, it is possible that a camera's pose is used to define the reference
    coordinate system of the optimization. In this case this camera has no explicit
    extrinsics (they are an identity transfomration, by definition), and we return
    -1, successfully.
    
    In order to determine the camera mapping, we need quite a bit of context. If we
    have the full set of inputs to the optimization function, we can pass in those
    (as shown in the example above). Or we can pass the individual arguments that
    are needed (see ARGUMENTS section for the full list). If the optimization inputs
    and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - icam_intrinsics: an integer indicating which camera we're asking about
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
    - Nobservations_board
    - Nobservations_point
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    - indices_frame_camintrinsics_camextrinsics: array of dims (Nobservations_board,
      3). For each observation these are an
      (iframe,icam_intrinsics,icam_extrinsics) tuple. icam_extrinsics == -1
      means this observation came from a camera in the reference coordinate system.
      iframe indexes the "frames_rt_toref" array, icam_intrinsics indexes the
      "intrinsics_data" array, icam_extrinsics indexes the "extrinsics_rt_fromref"
      array
    
      All of the indices are guaranteed to be monotonic. This array contains 32-bit
      integers.
    
    
    RETURNED VALUE
    
    The integer reporting the index of the camera extrinsics in the optimization
    vector. If this camera is at the reference of the coordinate system, return -1
    
    """
def decode_observation_indices_points_triangulated(*args, **kwargs):
    """
    
    NOT DONE YET; fill this in
    """
def drt_ref_refperturbed__dbpacked(*args, **kwargs):
    """
    
    write this
    """
def knots_for_splined_models(*args, **kwargs):
    """
    
    Return a tuple of locations of x and y spline knots
    
    SYNOPSIS
    
        print(mrcal.knots_for_splined_models('LENSMODEL_SPLINED_STEREOGRAPHIC_order=2_Nx=4_Ny=3_fov_x_deg=200'))
    
        ( array([-3.57526078, -1.19175359,  1.19175359,  3.57526078]),
          array([-2.38350719,  0.        ,  2.38350719]))
    
    Splined models are defined by the locations of their control points. These are
    arranged in a grid, the size and density of which is set by the model
    configuration. This function returns a tuple:
    
    - the locations of the knots along the x axis
    - the locations of the knots along the y axis
    
    The values in these arrays correspond to whatever is used to index the splined
    surface. In the case of LENSMODEL_SPLINED_STEREOGRAPHIC, these are the
    normalized stereographic projection coordinates. These can be unprojected to
    observation vectors at the knots:
    
        ux,uy = mrcal.knots_for_splined_models('LENSMODEL_SPLINED_STEREOGRAPHIC_order=2_Nx=4_Ny=3_fov_x_deg=200')
        u  = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(ux,uy)), 0, -1))
        v  = mrcal.unproject_stereographic(u)
    
        # v[index_y, index_x] is now an observation vector that will project to this
        # knot
    
    ARGUMENTS
    
    - lensmodel: the "LENSMODEL_..." string we're querying. This function only makes
      sense for "LENSMODEL_SPLINED_..." models
    
    RETURNED VALUE
    
    A tuple:
    
    - An array of shape (Nx,) representing the knot locations along the x axis
    
    - An array of shape (Ny,) representing the knot locations along the y axis
    
    """
def lensmodel_metadata_and_config(*args, **kwargs):
    """
    
    Returns a model's meta-information and configuration
    
    SYNOPSIS
    
      import pprint
      pprint.pprint(mrcal.lensmodel_metadata_and_config('LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=14_fov_x_deg=200'))
    
        {'Nx': 16,
         'Ny': 14,
         'can_project_behind_camera': 1,
         'fov_x_deg': 200,
         'has_core': 1,
         'has_gradients': 1,
         'order': 3}
    
    Each lens model has some metadata (inherent properties of a model family) and
    may have some configuration (parameters that specify details about the model,
    but aren't subject to optimization). The configuration parameters are embedded
    in the model string. This function returns a dict containing the metadata and
    all the configuration values. See the documentation for details:
    
      https://mrcal.secretsauce.net/lensmodels.html#representation
    
    ARGUMENTS
    
    - lensmodel: the "LENSMODEL_..." string we're querying
    
    RETURNED VALUE
    
    A dict containing all the metadata and configuration properties for that model
    
    
    """
def lensmodel_num_params(*args, **kwargs):
    """
    
    Get the number of lens parameters for a particular model type
    
    SYNOPSIS
    
        print(mrcal.lensmodel_num_params('LENSMODEL_OPENCV4'))
    
        8
    
    I support a number of lens models, which have different numbers of parameters.
    Given a lens model, this returns how many parameters there are. Some models have
    no configuration, and there's a static mapping between the lensmodel string and
    the parameter count. Some other models DO have some configuration values inside
    the model string (LENSMODEL_SPLINED_STEREOGRAPHIC_... for instance), and the
    number of parameters is computed using the configuration values. The lens model
    is given as a string such as
    
      LENSMODEL_PINHOLE
      LENSMODEL_OPENCV4
      LENSMODEL_CAHVOR
      LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=100
    
    The full list can be obtained with mrcal.supported_lensmodels()
    
    Note that when optimizing a lens model, some lens parameters may be locked down,
    resulting in fewer parameters than this function returns. To retrieve the number
    of parameters used to represent the intrinsics of a camera in an optimization,
    call mrcal.num_intrinsics_optimization_params(). Or to get the number of
    parameters used to represent the intrinsics of ALL the cameras in an
    optimization, call mrcal.num_states_intrinsics()
    
    ARGUMENTS
    
    - lensmodel: the "LENSMODEL_..." string we're querying
    
    RETURNED VALUE
    
    An integer number of parameters needed to describe a lens of the given type
    
    """
def load_image(*args, **kwargs):
    """
    
    Load an image from disk into a numpy array
    
    SYNOPSIS
    
        image = \\
            mrcal.load_image("scene.jpg",
                             bits_per_pixel = 8,
                             channels       = 1)
    
        ## image is now a numpy array of shape (height,width) containing the
        ## pixel data
    
    This is a completely uninteresting image-loading routine. It's like any other
    image-loading routine out there; use any that you like. This exists because cv2
    is very slow.
    
    This wraps the mrcal_image_TYPE_load() functions. At this time I support only
    these 3 data formats:
    
    - bits_per_pixel = 8,  channels = 1: 8-bit grayscale data
    - bits_per_pixel = 16, channels = 1: 16-bit grayscale data
    - bits_per_pixel = 24, channels = 3: BGR color data
    
    With the exception of 16-bit grayscale data, the load function will convert the
    input image to the requested format. At this time, asking for 16-bit grayscale
    data requires that the input image matches that format.
    
    If we ask for an 8-bit image, but pass a 16-bit image file, we will apply
    stretch equalization to the input first.
    
    If the bits_per_pixel, channels arguments are omitted or set to <= 0, we will
    load the image in whatever format it appears on disk.
    
    ARGUMENTS
    
    - filename: the image on disk to load
    
    - bits_per_pixel: optional integer describing the requested bit depth. Must be 8
      or 16 or 24. If omitted or <= 0, we use the bit depth of the image on disk
    
    - channels: optional integer describing the number of channels in the image.
      Integer. Must be 1 or 3. If omitted or <= 0, we use the channel count of the
      image on disk
    
    RETURNED VALUE
    
    A numpy array containing the pixel data
    
    
    """
def measurement_index_boards(*args, **kwargs):
    """
    
    Return the measurement index of the start of a given board observation
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x = mrcal.optimizer_callback(**optimization_inputs)[1]
    
        Nmeas   = mrcal.num_measurements_boards (   **optimization_inputs)
        i_meas0 = mrcal.measurement_index_boards(0, **optimization_inputs)
    
        x_boards_all = x[i_meas0:i_meas0+Nmeas]
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.measurement_index_...() functions report where particular items end up in
    the vector of measurements.
    
    THIS function reports the index in the measurement vector where a particular
    board observation begins. When solving calibration problems, most if not all of
    the measurements will come from these observations. These are stored
    contiguously.
    
    In order to determine the layout, we need quite a bit of context. If we have
    the full set of inputs to the optimization function, we can pass in those (as
    shown in the example above). Or we can pass the individual arguments that are
    needed (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - i_observation_board: an integer indicating which board observation we're
      querying
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable index in the measurements vector where the
    measurements for this particular board observation start
    
    """
def measurement_index_points(*args, **kwargs):
    """
    
    Return the measurement index of the start of a given point observation
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x = mrcal.optimizer_callback(**optimization_inputs)[1]
    
        Nmeas   = mrcal.num_measurements_points(    **optimization_inputs)
        i_meas0 = mrcal.measurement_index_points(0, **optimization_inputs)
    
        x_points_all = x[i_meas0:i_meas0+Nmeas]
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.measurement_index_...() functions report where particular items end up in
    the vector of measurements.
    
    THIS function reports the index in the measurement vector where a particular
    point observation begins. When solving structure-from-motion problems, most if
    not all of the measurements will come from these observations. These are stored
    contiguously.
    
    In order to determine the layout, we need quite a bit of context. If we have the
    full set of inputs to the optimization function, we can pass in those (as shown
    in the example above). Or we can pass the individual arguments that are needed
    (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - i_observation_point: an integer indicating which point observation we're
      querying
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable index in the measurements vector where the
    measurements for this particular point observation start
    
    """
def measurement_index_points_triangulated(*args, **kwargs):
    """
    
    NOT DONE YET; fill this in
    """
def measurement_index_regularization(*args, **kwargs):
    """
    
    Return the index of the start of the regularization measurements
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x = mrcal.optimizer_callback(**optimization_inputs)[1]
    
        Nmeas   = mrcal.num_measurements_regularization( **optimization_inputs)
        i_meas0 = mrcal.measurement_index_regularization(**optimization_inputs)
    
        x_regularization = x[i_meas0:i_meas0+Nmeas]
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.measurement_index_...() functions report where particular items end up in
    the vector of measurements.
    
    THIS function reports the index in the measurement vector where the
    regularization terms begin. These don't model physical effects, but guide the
    solver away from obviously-incorrect solutions, and resolve ambiguities. This
    helps the solver converge to the right solution, quickly.
    
    In order to determine the layout, we need quite a bit of context. If we have the
    full set of inputs to the optimization function, we can pass in those (as shown
    in the example above). Or we can pass the individual arguments that are needed
    (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting where in the measurement vector the regularization terms
    start
    
    """
def num_intrinsics_optimization_params(*args, **kwargs):
    """
    
    Get the number of optimization parameters for a single camera's intrinsics
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        f( m.optimization_inputs() )
    
    
        ...
    
        def f(optimization_inputs):
            Nstates  = mrcal.num_intrinsics_optimization_params(**optimization_inputs)
            ...
    
    
    Return the number of parameters used in the optimization of the intrinsics of a
    camera.
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what.
    
    This function reports how many optimization parameters are used to represent the
    intrinsics of a single camera. This is very similar to
    mrcal.lensmodel_num_params(), except THIS function takes into account the
    do_optimize_intrinsics_... variables used to lock down some parts of the
    intrinsics vector. Similarly, we have mrcal.num_states_intrinsics(), which takes
    into account the optimization details also, but reports the number of variables
    needed to describe ALL the cameras instead of just one.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the number of optimization parameters used to describe the
    intrinsics of a single camera
    
    """
def num_measurements(*args, **kwargs):
    """
    
    Return how many measurements we have in the full optimization problem
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x,J = mrcal.optimizer_callback(**optimization_inputs)[1:3]
    
        Nmeas   = mrcal.num_measurements(**optimization_inputs)
    
        print(x.shape[0] - Nmeas)
        ===>
        0
    
        print(J.shape[0] - Nmeas)
        ===>
        0
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_measurements_...() functions report where particular items end up in
    the vector of measurements.
    
    THIS function reports the total number of measurements we have. This corresponds
    to the number of elements in the vector x and to the number of rows in the
    jacobian matrix J.
    
    In order to determine the mapping, we need quite a bit of context. If we have
    the full set of inputs to the optimization function, we can pass in those (as
    shown in the example above). Or we can pass the individual arguments that are
    needed (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the size of the measurement vector x
    
    """
def num_measurements_boards(*args, **kwargs):
    """
    
    Return how many measurements we have from calibration object observations
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x = mrcal.optimizer_callback(**optimization_inputs)[1]
    
        Nmeas   = mrcal.num_measurements_boards (   **optimization_inputs)
        i_meas0 = mrcal.measurement_index_boards(0, **optimization_inputs)
    
        x_boards_all = x[i_meas0:i_meas0+Nmeas]
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_measurements_...() functions report how many measurements are produced
    by particular items.
    
    THIS function reports how many measurements come from the observations of the
    calibration object. When solving calibration problems, most if not all of the
    measurements will come from these observations. These are stored contiguously.
    
    In order to determine the layout, we need quite a bit of context. If we have
    the full set of inputs to the optimization function, we can pass in those (as
    shown in the example above). Or we can pass the individual arguments that are
    needed (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting how many elements of the measurement vector x come from
    the calibration object observations
    
    """
def num_measurements_points(*args, **kwargs):
    """
    
    Return how many measurements we have from point observations
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x = mrcal.optimizer_callback(**optimization_inputs)[1]
    
        Nmeas   = mrcal.num_measurements_points(    **optimization_inputs)
        i_meas0 = mrcal.measurement_index_points(0, **optimization_inputs)
    
        x_points_all = x[i_meas0:i_meas0+Nmeas]
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_measurements_...() functions report how many measurements are produced
    by particular items.
    
    THIS function reports how many measurements come from the observations of
    discrete points. When solving structure-from-motion problems, most if not all of
    the measurements will come from these observations. These are stored
    contiguously.
    
    In order to determine the layout, we need quite a bit of context. If we have the
    full set of inputs to the optimization function, we can pass in those (as shown
    in the example above). Or we can pass the individual arguments that are needed
    (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting how many elements of the measurement vector x come from
    observations of discrete points
    
    """
def num_measurements_points_triangulated(*args, **kwargs):
    """
    
    NOT DONE YET; fill this in
    """
def num_measurements_regularization(*args, **kwargs):
    """
    
    Return how many measurements we have from regularization
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        x = mrcal.optimizer_callback(**optimization_inputs)[1]
    
        Nmeas   = mrcal.num_measurements_regularization( **optimization_inputs)
        i_meas0 = mrcal.measurement_index_regularization(**optimization_inputs)
    
        x_regularization = x[i_meas0:i_meas0+Nmeas]
    
    The optimization algorithm tries to minimize the norm of a "measurements" vector
    x. The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_measurements_...() functions report where particular items end up in
    the vector of measurements.
    
    THIS function reports how many measurements come from the regularization terms
    of the optimization problem. These don't model physical effects, but guide the
    solver away from obviously-incorrect solutions, and resolve ambiguities. This
    helps the solver converge to the right solution, quickly.
    
    In order to determine the layout, we need quite a bit of context. If we have the
    full set of inputs to the optimization function, we can pass in those (as shown
    in the example above). Or we can pass the individual arguments that are needed
    (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_frames
      do_optimize_calobject_warp
      do_apply_regularization
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
      Nobservations_point
      calibration_object_width_n
      calibration_object_height_n
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting how many elements of the measurement vector x come from
    regularization terms
    
    """
def num_states(*args, **kwargs):
    """
    
    Get the total number of parameters in the optimization vector
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        f( m.optimization_inputs() )
    
    
        ...
    
        def f(optimization_inputs):
            Nstates  = mrcal.num_states (**optimization_inputs)
            ...
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_states_...() functions report how many variables in the optimization
    vector are taken up by each particular kind of measurement.
    
    THIS function reports how many variables are used to represent the FULL state
    vector.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the total variable count in the state vector
    
    """
def num_states_calobject_warp(*args, **kwargs):
    """
    
    Get the number of parameters in the optimization vector for the board warp
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_state0 = mrcal.state_index_calobject_warp(**optimization_inputs)
        Nstates  = mrcal.num_states_calobject_warp (**optimization_inputs)
    
        calobject_warp = b[i_state0:i_state0+Nstates]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_states_...() functions report how many variables in the optimization
    vector are taken up by each particular kind of measurement.
    
    THIS function reports how many variables are used to represent the
    calibration-object warping. This is stored contiguously as in memory. These
    warping parameters describe how the observed calibration object differs from the
    expected calibration object. There will always be some difference due to
    manufacturing tolerances and temperature and humidity effects.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable count of the calibration object warping
    parameters
    
    """
def num_states_extrinsics(*args, **kwargs):
    """
    
    Get the number of extrinsics parameters in the optimization vector
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_state0 = mrcal.state_index_extrinsics(0, **optimization_inputs)
        Nstates  = mrcal.num_states_extrinsics (   **optimization_inputs)
    
        extrinsics_rt_fromref_all = b[i_state0:i_state0+Nstates]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_states_...() functions report how many variables in the optimization
    vector are taken up by each particular kind of measurement.
    
    THIS function reports how many variables are used to represent ALL the camera
    extrinsics. The extrinsics are stored contiguously as an "rt transformation": a
    3-element rotation represented as a Rodrigues vector followed by a 3-element
    translation. These transform points represented in the reference coordinate
    system to the coordinate system of the specific camera. Note that mrcal allows
    the reference coordinate system to be tied to a particular camera. In this case
    the extrinsics of that camera do not appear in the state vector at all, and
    icam_extrinsics == -1 in the indices_frame_camintrinsics_camextrinsics
    array.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable count of extrinsics in the state vector
    
    """
def num_states_frames(*args, **kwargs):
    """
    
    Get the number of calibration object pose parameters in the optimization vector
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_state0 = mrcal.state_index_frames(0, **optimization_inputs)
        Nstates  = mrcal.num_states_frames (   **optimization_inputs)
    
        frames_rt_toref_all = b[i_state0:i_state0+Nstates]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_states_...() functions report how many variables in the optimization
    vector are taken up by each particular kind of measurement.
    
    THIS function reports how many variables are used to represent ALL the frame
    poses. Here a "frame" is a pose of the observed calibration object at some
    instant in time. The frames are stored contiguously as an "rt transformation": a
    3-element rotation represented as a Rodrigues vector followed by a 3-element
    translation. These transform points represented in the internal calibration
    object coordinate system to the reference coordinate system.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable count of frames in the state vector
    
    """
def num_states_intrinsics(*args, **kwargs):
    """
    
    Get the number of intrinsics parameters in the optimization vector
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_state0 = mrcal.state_index_intrinsics(0, **optimization_inputs)
        Nstates  = mrcal.num_states_intrinsics (   **optimization_inputs)
    
        intrinsics_all = b[i_state0:i_state0+Nstates]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_states_...() functions report how many variables in the optimization
    vector are taken up by each particular kind of measurement.
    
    THIS function reports how many optimization variables are used to represent ALL
    the camera intrinsics. The intrinsics are stored contiguously. They consist of a
    4-element "intrinsics core" (focallength-x, focallength-y, centerpixel-x,
    centerpixel-y) followed by a lensmodel-specific vector of "distortions". A
    similar function mrcal.num_intrinsics_optimization_params() is available to
    report the number of optimization variables used for just ONE camera. If all the
    intrinsics are being optimized, then the mrcal.lensmodel_num_params() returns
    the same value: the number of values needed to describe the intrinsics of a
    single camera. It is possible to lock down some of the intrinsics during
    optimization (by setting the do_optimize_intrinsics_... variables
    appropriately). These variables control what
    mrcal.num_intrinsics_optimization_params() and mrcal.num_states_intrinsics()
    return, but not mrcal.lensmodel_num_params().
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable count of intrinsics in the state vector
    
    """
def num_states_points(*args, **kwargs):
    """
    
    Get the number of point-position parameters in the optimization vector
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_state0 = mrcal.state_index_points(0, **optimization_inputs)
        Nstates  = mrcal.num_states_points (   **optimization_inputs)
    
        points_all = b[i_state0:i_state0+Nstates]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.num_states_...() functions report how many variables in the optimization
    vector are taken up by each particular kind of measurement.
    
    THIS function reports how many variables are used to represent ALL the points.
    The points are stored contiguously as a 3-element coordinates in the reference
    frame.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the variable count of points in the state vector
    
    """
def optimize(*args, **kwargs):
    """
    
    Invoke the calibration routine
    
    SYNOPSIS
    
        stats = mrcal.optimize( intrinsics_data,
                                extrinsics_rt_fromref,
                                frames_rt_toref, points,
                                observations_board, indices_frame_camintrinsics_camextrinsics,
                                observations_point, indices_point_camintrinsics_camextrinsics,
    
                                lensmodel,
                                imagersizes                       = imagersizes,
                                do_optimize_intrinsics_core       = True,
                                do_optimize_intrinsics_distortions= True,
                                calibration_object_spacing        = object_spacing,
                                point_min_range                   = 0.1,
                                point_max_range                   = 100.0,
                                do_apply_outlier_rejection        = True,
                                do_apply_regularization           = True,
                                verbose                           = False)
    
    Please see the mrcal documentation at
    https://mrcal.secretsauce.net/formulation.html for details.
    
    This is a flexible implementation of a calibration system core that uses sparse
    Jacobians, performs outlier rejection and reports some metrics back to the user.
    Measurements from any number of cameras can beat used simultaneously, and this
    routine is flexible-enough to solve structure-from-motion problems.
    
    The input is a combination of observations of a calibration board and
    observations of discrete points. The point observations MAY have a known
    range.
    
    The cameras and what they're observing is given in the arrays
    
    - intrinsics_data
    - extrinsics_rt_fromref
    - frames_rt_toref
    - points
    - indices_frame_camintrinsics_camextrinsics
    - indices_point_camintrinsics_camextrinsics
    
    intrinsics_data contains the intrinsics for all the physical cameras present in
    the problem. len(intrinsics_data) = Ncameras_intrinsics
    
    extrinsics_rt_fromref contains all the camera poses present in the problem,
    omitting any cameras that sit at the reference coordinate system.
    len(extrinsics_rt_fromref) = Ncameras_extrinsics.
    
    frames_rt_toref is all the poses of the calibration board in the problem, and
    points is all the discrete points being observed in the problem.
    
    indices_frame_camintrinsics_camextrinsics describes which board observations
    were made by which camera, and where this camera was. Each board observation is
    described by a tuple (iframe,icam_intrinsics,icam_extrinsics). The board at
    frames_rt_toref[iframe] was observed by camera
    intrinsics_data[icam_intrinsics], which was at
    extrinsics_rt_fromref[icam_extrinsics]
    
    indices_point_camintrinsics_camextrinsics is the same thing for discrete points.
    
    If we're solving a vanilla calibration problem, we have stationary cameras
    observing a moving target. By convention, camera 0 is at the reference
    coordinate system. So
    
    - Ncameras_intrinsics = Ncameras_extrinsics+1
    - All entries in indices_frame_camintrinsics_camextrinsics have
      icam_intrinsics = icam_extrinsics+1
    - frames_rt_toref, points describes the motion of the moving target we're
      observing
    
    Conversely, in a structure-from-motion problem we have some small number of
    moving cameras (often just 1) observing stationary target(s). We would have
    
    - Ncameras_intrinsics is small: it's how many physical cameras we have
    - Ncameras_extrinsics is large: it describes the motion of the cameras
    - frames_rt_toref, points is small: it describes the non-moving world we're
      observing
    
    Any combination of these extreme cases is allowed.
    
    REQUIRED ARGUMENTS
    
    - intrinsics: array of dims (Ncameras_intrinsics, Nintrinsics). The intrinsics
      of each physical camera. Each intrinsic vector is given as
    
        (focal_x, focal_y, center_pixel_x, center_pixel_y, distortion0, distortion1,
        ...)
    
      The focal lengths are given in pixels.
    
      On input this is a seed. On output the optimal data is returned. THIS ARRAY IS
      MODIFIED BY THIS CALL.
    
    - extrinsics_rt_fromref: array of dims (Ncameras_extrinsics, 6). The pose of
      each camera observation. Each pose is given as 6 values: a Rodrigues rotation
      vector followed by a translation. This represents a transformation FROM the
      reference coord system TO the coord system of each camera.
    
      On input this is a seed. On output the optimal data is returned. THIS ARRAY IS
      MODIFIED BY THIS CALL.
    
      If we only have one camera, pass either None or np.zeros((0,6))
    
    - frames_rt_toref: array of dims (Nframes, 6). The poses of the calibration
      object over time. Each pose is given as 6 values: a rodrigues rotation vector
      followed by a translation. This represents a transformation FROM the coord
      system of the calibration object TO the reference coord system. THIS IS
      DIFFERENT FROM THE CAMERA EXTRINSICS.
    
      On input this is a seed. On output the optimal data is returned. THIS ARRAY IS
      MODIFIED BY THIS CALL.
    
      If we don't have any frames, pass either None or np.zeros((0,6))
    
    - points: array of dims (Npoints, 3). The estimated positions of discrete points
      we're observing. These positions are represented in the reference coord
      system. The initial Npoints-Npoints_fixed points are optimized by this
      routine. The final Npoints_fixed points are fixed. By default
      Npoints_fixed==0, and we optimize all the points.
    
      On input this is a seed. On output the optimal data is returned. THIS ARRAY IS
      MODIFIED BY THIS CALL.
    
    - observations_board: array of dims (Nobservations_board,
                                         calibration_object_height_n,
                                         calibration_object_width_n,
                                         3).
      Each slice is an (x,y,weight) tuple where (x,y) are the observed pixel
      coordinates of the corners in the calibration object, and "weight" is the
      relative weight of this point observation. Most of the weights are expected to
      be 1.0, which implies that the noise on that observation has the nominal
      standard deviation of observed_pixel_uncertainty (in addition to the overall
      assumption of gaussian noise, independent on x,y). weight<0 indicates that
      this is an outlier. This is respected on input (even if
      !do_apply_outlier_rejection). New outliers are marked with weight<0 on output.
      Subpixel interpolation is assumed, so these contain 64-bit floating point
      values, like all the other data. The frame and camera that produced these
      observations are given in the indices_frame_camintrinsics_camextrinsics
    
      THIS ARRAY IS MODIFIED BY THIS CALL (to mark outliers)
    
    - indices_frame_camintrinsics_camextrinsics: array of dims (Nobservations_board,
      3). For each observation these are an
      (iframe,icam_intrinsics,icam_extrinsics) tuple. icam_extrinsics == -1
      means this observation came from a camera in the reference coordinate system.
      iframe indexes the "frames_rt_toref" array, icam_intrinsics indexes the
      "intrinsics_data" array, icam_extrinsics indexes the "extrinsics_rt_fromref"
      array
    
      All of the indices are guaranteed to be monotonic. This array contains 32-bit
      integers.
    
    - observations_point: array of dims (Nobservations_point, 3). Each slice is an
      (x,y,weight) tuple where (x,y) are the pixel coordinates of the observed
      point, and "weight" is the relative weight of this point observation. Most of
      the weights are expected to be 1.0, which implies that the noise on the
      observation is gaussian, independent on x,y, and has the nominal standard
      deviation of observed_pixel_uncertainty. weight<0 indicates that this is an
      outlier. This is respected on input (even if !do_apply_outlier_rejection). At
      this time, no new outliers are detected for point observations. Subpixel
      interpolation is assumed, so these contain 64-bit floating point values, like
      all the other data. The point index and camera that produced these
      observations are given in the indices_point_camera_points array.
    
    - indices_point_camintrinsics_camextrinsics: array of dims (Nobservations_point,
      3). For each observation these are an
      (i_point,icam_intrinsics,icam_extrinsics) tuple. Analogous to
      indices_frame_camintrinsics_camextrinsics, but for observations of discrete
      points.
    
      The indices can appear in any order. No monotonicity is required. This array
      contains 32-bit integers.
    
    - lensmodel: a string such as
    
      LENSMODEL_PINHOLE
      LENSMODEL_OPENCV4
      LENSMODEL_CAHVOR
      LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=100
    
    - imagersizes: integer array of dims (Ncameras_intrinsics,2)
    
    OPTIONAL ARGUMENTS
    
    - calobject_warp
    
      A numpy array of shape (2,) describing the non-flatness of the calibration
      board. If omitted or None, the board is assumed to be perfectly flat. And if
      do_optimize_calobject_warp then we optimize these parameters to find the
      best-fitting board shape.
    
    - Npoints_fixed
    
      Specifies how many points at the end of the points array are fixed, and remain
      unaffected by the optimization. This is 0 by default, and we optimize all the
      points.
    
    - do_optimize_intrinsics_core
    - do_optimize_intrinsics_distortions
    - do_optimize_extrinsics
    - do_optimize_frames
    - do_optimize_calobject_warp
    
      Indicate whether to optimize a specific set of variables. The intrinsics core
      is fx,fy,cx,cy. These all default to True so if we specify none of these, we
      will optimize ALL the variables.
    
    - calibration_object_spacing: the width of each square in a calibration board.
      Can be omitted if we have no board observations, just points. The calibration
      object has shape (calibration_object_height_n,calibration_object_width_n),
      given by the dimensions of "observations_board"
    
    - verbose: if True, write out all sorts of diagnostic data to STDERR. Defaults
      to False
    
    - do_apply_outlier_rejection: if False, don't bother with detecting or rejecting
      outliers. The outliers we get on input (observations_board[...,2] < 0) are
      honered regardless. Defaults to True
    
    - do_apply_regularization: if False, don't include regularization terms in the
      solver. Defaults to True
    
    - point_min_range, point_max_range: Required ONLY if point observations are
      given. These are lower, upper bounds for the distance of a point observation
      to its observing camera. Each observation outside of this range is penalized.
      This helps the solver by guiding it away from unreasonable solutions.
    
    We return a dict with various metrics describing the computation we just
    performed
    
    """
def optimizer_callback(*args, **kwargs):
    """
    
    Call the optimization callback function
    
    SYNOPSIS
    
        model               = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = model.optimization_inputs()
    
        b_packed,x,J_packed,factorization = \\
          mrcal.optimizer_callback( **optimization_inputs )
    
    Please see the mrcal documentation at
    https://mrcal.secretsauce.net/formulation.html for details.
    
    The main optimization routine in mrcal.optimize() searches for optimal
    parameters by repeatedly calling a function to evaluate each hypothethical
    parameter set. This evaluation function is available by itself here, separated
    from the optimization loop. The arguments are largely the same as those to
    mrcal.optimize(), but the inputs are all read-only. Some arguments that have
    meaning in calls to optimize() have no meaning in calls to optimizer_callback().
    These are accepted, and effectively ignored. Currently these are:
    
    - do_apply_outlier_rejection
    
    ARGUMENTS
    
    This function accepts lots of arguments, but they're the same as the arguments
    to mrcal.optimize() so please see that documentation for details. Arguments
    accepted by optimizer_callback() on top of those in optimize():
    
    - no_jacobian: optional boolean defaulting to False. If True, we do not compute
      a jacobian, which would speed up this function. We then return None in its
      place. if no_jacobian and not no_factorization then we still compute and
      return a jacobian, since it's needed for the factorization
    
    - no_factorization: optional boolean defaulting to False. If True, we do not
      compute a cholesky factorization of JtJ, which would speed up this function.
      We then return None in its place. if no_jacobian and not no_factorization then
      we still compute and return a jacobian, since it's needed for the
      factorization
    
    RETURNED VALUES
    
    The output is returned in a tuple:
    
    - b_packed: a numpy array of shape (Nstate,). This is the packed (unitless)
      state vector that represents the inputs, as seen by the optimizer. If the
      optimization routine was running, it would use this as a starting point in the
      search for different parameters, trying to find those that minimize norm2(x).
      This packed state can be converted to the expanded representation like this:
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0
        mrcal.unpack_state(b, **optimization_inputs)
    
    - x: a numpy array of shape (Nmeasurements,). This is the error vector. If the
      optimization routine was running, it would be testing different parameters,
      trying to find those that minimize norm2(x)
    
    - J: a sparse matrix of shape (Nmeasurements,Nstate). These are the gradients of
      the measurements in respect to the packed parameters. This is a SPARSE array
      of type scipy.sparse.csr_matrix. This object can be converted to a numpy array
      like this:
    
        b,x,J_sparse = mrcal.optimizer_callback(...)[:3]
        J_numpy      = J_sparse.toarray()
    
      Note that the numpy array is dense, so it is very inefficient for sparse data,
      and working with it could be very memory-intensive and slow.
    
      This jacobian matrix comes directly from the optimization callback function,
      which uses packed, unitless state. To convert a densified packed jacobian to
      full units, one can do this:
    
        J_sparse = mrcal.optimizer_callback(**optimization_inputs)[2]
        J_numpy      = J_sparse.toarray()
        mrcal.pack_state(J_numpy, **optimization_inputs)
    
      Note that we're calling pack_state() instead of unpack_state() because the
      packed variables are in the denominator
    
    - factorization: a Cholesky factorization of JtJ in a
      mrcal.CHOLMOD_factorization object. The core of the optimization algorithm is
      solving a linear system JtJ x = b. J is a large, sparse matrix, so we do this
      with a Cholesky factorization of J using the CHOLMOD library. This
      factorization is also useful in other contexts, such as uncertainty
      quantification, so we make it available here. If the factorization could not
      be computed (because JtJ isn't full-rank for instance), this is set to None
    
    """
def pack_state(*args, **kwargs):
    """
    
    Scales a state vector to the packed, unitless form used by the optimizer
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        Jpacked = mrcal.optimizer_callback(**optimization_inputs)[2].toarray()
    
        J = Jpacked.copy()
        mrcal.pack_state(J, **optimization_inputs)
    
    In order to make the optimization well-behaved, we scale all the variables in
    the state and the gradients before passing them to the optimizer. The internal
    optimization library thus works only with unitless (or "packed") data.
    
    This function takes a full numpy array of shape (...., Nstate), and scales it to
    produce packed data. This function applies the scaling directly to the input
    array; the input is modified, and nothing is returned.
    
    To unpack a state vector, you naturally call unpack_state(). To unpack a
    jacobian matrix, you would call pack_state() because in a jacobian, the state is
    in the denominator. This is shown in the example above.
    
    Broadcasting is supported: any leading dimensions will be processed correctly,
    as long as the given array has shape (..., Nstate).
    
    In order to know what the scale factors should be, and how they should map to
    each variable in the state vector, we need quite a bit of context. If we have
    the full set of inputs to the optimization function, we can pass in those (as
    shown in the example above). Or we can pass the individual arguments that are
    needed (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - b: a numpy array of shape (..., Nstate). This is the full state on input, and
      the packed state on output. The input array is modified.
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    None. The scaling is applied to the input array
    
    """
def save_image(*args, **kwargs):
    """
    
    Save a numpy array to an image on disk
    
    SYNOPSIS
    
        print(image.shape)
        ---> (768, 1024, 3)
    
        print(image.dtype)
        ---> dtype('uint8')
    
        mrcal.save_image("result.png", image)
    
        # wrote BGR color image to disk
    
    This is a completely uninteresting image-saving routine. It's like any other
    image-saving routine out there; use any that you like. This exists because cv2
    is very slow.
    
    This wraps the mrcal_image_TYPE_save() functions. At this time I support only
    these 3 data formats:
    
    - bpp = 8,  channels = 1: 8-bit grayscale data
    - bpp = 16, channels = 1: 16-bit grayscale data
    - bpp = 24, channels = 3: BGR color data
    
    ARGUMENTS
    
    - filename: the image on disk to save to
    
    - array: numpy array containing the input data. Must have shape (height,width)
      for grayscale data or (height,width,3) for color data. Each row must be stored
      densely, but a non-dense stride is supported when moving from column to
      column. The dtype must be either np.uint8 or np.uint16.
    
    RETURNED VALUE
    
    None on success. Exception thrown on error
    
    
    
    """
def state_index_calobject_warp(*args, **kwargs):
    """
    
    Return the index in the optimization vector of the calibration object warp
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_state = mrcal.state_index_calobject_warp(**optimization_inputs)
    
        calobject_warp = b[i_state:i_state+2]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.state_index_...() functions report where particular items end up in the
    state vector.
    
    THIS function reports the beginning of the calibration-object warping parameters
    in the state vector. This is stored contiguously as a 2-element vector. These
    warping parameters describe how the observed calibration object differs from the
    expected calibration object. There will always be some difference due to
    manufacturing tolerances and temperature and humidity effects.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the location in the state vector where the contiguous
    block of variables for the calibration object warping begins. If we're not
    optimizing the calibration object shape, returns None
    
    """
def state_index_extrinsics(*args, **kwargs):
    """
    
    Return the index in the optimization vector of the extrinsics of camera i
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        icam_extrinsics = 1
        i_state = mrcal.state_index_extrinsics(icam_extrinsics,
                                               **optimization_inputs)
    
        extrinsics_rt_fromref = b[i_state:i_state+6]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.state_index_...() functions report where particular items end up in the
    state vector.
    
    THIS function reports the beginning of the i-th camera extrinsics in the state
    vector. The extrinsics are stored contiguously as an "rt transformation": a
    3-element rotation represented as a Rodrigues vector followed by a 3-element
    translation. These transform points represented in the reference coordinate
    system to the coordinate system of the specific camera. Note that mrcal allows
    the reference coordinate system to be tied to a particular camera. In this case
    the extrinsics of that camera do not appear in the state vector at all, and
    icam_extrinsics == -1 in the indices_frame_camintrinsics_camextrinsics
    array.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - icam_extrinsics: an integer indicating which camera we're asking about
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the location in the state vector where the contiguous
    block of extrinsics for camera icam_extrinsics begins. If we're not optimizing
    the extrinsics, or we're asking for an out-of-bounds camera, returns None
    
    """
def state_index_frames(*args, **kwargs):
    """
    
    Return the index in the optimization vector of the pose of frame i
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        iframe = 1
        i_state = mrcal.state_index_frames(iframe,
                                           **optimization_inputs)
    
        frames_rt_toref = b[i_state:i_state+6]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.state_index_...() functions report where particular items end up in the
    state vector.
    
    THIS function reports the beginning of the i-th frame pose in the state vector.
    Here a "frame" is a pose of the observed calibration object at some instant in
    time. The frames are stored contiguously as an "rt transformation": a 3-element
    rotation represented as a Rodrigues vector followed by a 3-element translation.
    These transform points represented in the internal calibration object coordinate
    system to the reference coordinate system.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - iframe: an integer indicating which frame we're asking about
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the location in the state vector where the contiguous
    block of variables for frame iframe begins. If we're not optimizing the frames,
    or we're asking for an out-of-bounds frame, returns None
    
    """
def state_index_intrinsics(*args, **kwargs):
    """
    
    Return the index in the optimization vector of the intrinsics of camera i
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        icam_intrinsics = 1
        i_state = mrcal.state_index_intrinsics(icam_intrinsics,
                                               **optimization_inputs)
    
        Nintrinsics = mrcal.lensmodel_num_params(optimization_inputs['lensmodel'])
        intrinsics_data = b[i_state:i_state+Nintrinsics]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.state_index_...() functions report where particular items end up in the
    state vector.
    
    THIS function reports the beginning of the i-th camera intrinsics in the state
    vector. The intrinsics are stored contiguously. They consist of a 4-element
    "intrinsics core" (focallength-x, focallength-y, centerpixel-x, centerpixel-y)
    followed by a lensmodel-specific vector of "distortions". The number of
    intrinsics elements (including the core) for a particular lens model can be
    queried with mrcal.lensmodel_num_params(lensmodel). Note that
    do_optimize_intrinsics_core and do_optimize_intrinsics_distortions can be used
    to lock down one or both of those quantities, which would omit them from the
    optimization vector.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - icam_intrinsics: an integer indicating which camera we're asking about
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the location in the state vector where the contiguous
    block of intrinsics for camera icam_intrinsics begins. If we're not optimizing
    the intrinsics, or we're asking for an out-of-bounds camera, returns None
    """
def state_index_points(*args, **kwargs):
    """
    
    Return the index in the optimization vector of the position of point i
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b = mrcal.optimizer_callback(**optimization_inputs)[0]
        mrcal.unpack_state(b, **optimization_inputs)
    
        i_point = 1
        i_state = mrcal.state_index_points(i_point,
                                           **optimization_inputs)
    
        point = b[i_state:i_state+3]
    
    The optimization algorithm sees its world described in one, big vector of state.
    The optimizer doesn't know or care about the meaning of each element of this
    vector, but for later analysis, it is useful to know what's what. The
    mrcal.state_index_...() functions report where particular items end up in the
    state vector.
    
    THIS function reports the beginning of the i-th point in the state vector. The
    points are stored contiguously as a 3-element coordinates in the reference
    frame.
    
    In order to determine the variable mapping, we need quite a bit of context. If
    we have the full set of inputs to the optimization function, we can pass in
    those (as shown in the example above). Or we can pass the individual arguments
    that are needed (see ARGUMENTS section for the full list). If the optimization
    inputs and explicitly-given arguments conflict about the size of some array, the
    explicit arguments take precedence. If any array size is not specified, it is
    assumed to be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - i_point: an integer indicating which point we're asking about
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
      Nobservations_board
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    The integer reporting the location in the state vector where the contiguous
    block of variables for point i_point begins If we're not optimizing the points,
    or we're asking for an out-of-bounds point, returns None
    
    """
def supported_lensmodels():
    """
    
    Returns a tuple of strings for the various lens models we support
    
    SYNOPSIS
    
        print(mrcal.supported_lensmodels())
    
        ('LENSMODEL_PINHOLE',
         'LENSMODEL_STEREOGRAPHIC',
         'LENSMODEL_SPLINED_STEREOGRAPHIC_...',
         'LENSMODEL_OPENCV4',
         'LENSMODEL_OPENCV5',
         'LENSMODEL_OPENCV8',
         'LENSMODEL_OPENCV12',
         'LENSMODEL_CAHVOR',
         'LENSMODEL_CAHVORE_linearity=...')
    
    mrcal knows about some set of lens models, which can be queried here. The above
    list is correct as of this writing, but more models could be added with time.
    
    The returned lens models are all supported, with possible gaps in capabilities.
    The capabilities of each model are returned by lensmodel_metadata_and_config().
    
    Models ending in '...' have configuration parameters given in the model string,
    replacing the '...'.
    
    RETURNED VALUE
    
    A tuple of strings listing out all the currently-supported lens models
    
    """
def traverse_sensor_links(*args, **kwargs):
    """
    
    Finds optimal paths in a connectivity graph of sensors 
    
    SYNOPSIS
    
        # Sensor 4 only has shared observations with sensor 2
        # Otherwise, sensor 2 only has shared observations with sensor 1
        # Sensor 1 does share observations with sensor 0
        #
        # So we expect the best path to sensor 4 to be 0-1-2-4
        connectivity_matrix = np.array((( 0, 5, 0, 3, 0),
                                        ( 5, 0, 2, 5, 0),
                                        ( 0, 2, 0, 0, 5),
                                        ( 3, 5, 0, 0, 0),
                                        ( 0, 0, 5, 0, 0),),
                                       dtype=np.uint16)
    
        mrcal.traverse_sensor_links( \\
            connectivity_matrix  = connectivity_matrix,
            callback_sensor_link = lambda idx_to, idx_from: \\
                                          print(f"{idx_from}-{idx_to}") )
    
        ------>
        0-1
        0-3
        1-2
        2-4
    
    Traverses a connectivity graph of sensors to find the best connection from
    the root sensor (idx==0) to every other sensor. This is useful to seed a
    problem with sparse connections, where every sensor doesn't have overlapping
    observations with every other sensor.
    
    This uses a simple implmentation of Dijkstra's algorithm to optimize the number
    of links needed to reach each sensor, using the total number of shared
    observations as a tie-break.
    
    The main input to this function is a conectivity matrix: an (N,N) array where
    each element (i,j) contains the shared number of observations between sensors i
    and j. Some sensors may not share any observations, which would be indicated by
    a 0 in the connectivity matrix. This matrix is assumed to be symmetric and to
    have a 0 diagonal. The results are indicated by a callback for each optimal link
    in the chain.
    
    It is possible to have a disjoint graph, where there aren't any links from the
    root sensor to every other camera. This would result in the callback never being
    called for these disjoint sensors. It is the caller's job to catch and to think
    about this case.
    
    ARGUMENTS
    
    All arguments are required and must be specified with a keyword.
    
    - connectivity_matrix: a numpy array of shape (Nsensors,Nsensors) and
      dtype=np.uint16. This must be symmetric and have a 0 diagonal
    
    - callback_sensor_link: a callable invoked for each optimal link we report.
      Takes two arguments: idx_to,idx_from. Returns False if an error occured and we
      should exit
    
    RETURNED VALUE
    
    A true value on success
    
    """
def unpack_state(*args, **kwargs):
    """
    
    Scales a state vector from the packed, unitless form used by the optimizer
    
    SYNOPSIS
    
        m = mrcal.cameramodel('xxx.cameramodel')
    
        optimization_inputs = m.optimization_inputs()
    
        b_packed = mrcal.optimizer_callback(**optimization_inputs)[0]
    
        b = b_packed.copy()
        mrcal.unpack_state(b, **optimization_inputs)
    
    In order to make the optimization well-behaved, we scale all the variables in
    the state and the gradients before passing them to the optimizer. The internal
    optimization library thus works only with unitless (or "packed") data.
    
    This function takes a packed numpy array of shape (...., Nstate), and scales it
    to produce full data with real units. This function applies the scaling directly
    to the input array; the input is modified, and nothing is returned.
    
    To unpack a state vector, you naturally call unpack_state(). To unpack a
    jacobian matrix, you would call pack_state() because in a jacobian, the state is
    in the denominator.
    
    Broadcasting is supported: any leading dimensions will be processed correctly,
    as long as the given array has shape (..., Nstate).
    
    In order to know what the scale factors should be, and how they should map to
    each variable in the state vector, we need quite a bit of context. If we have
    the full set of inputs to the optimization function, we can pass in those (as
    shown in the example above). Or we can pass the individual arguments that are
    needed (see ARGUMENTS section for the full list). If the optimization inputs and
    explicitly-given arguments conflict about the size of some array, the explicit
    arguments take precedence. If any array size is not specified, it is assumed to
    be 0. Thus most arguments are optional.
    
    ARGUMENTS
    
    - b: a numpy array of shape (..., Nstate). This is the packed state on input,
      and the full state on output. The input array is modified.
    
    - **kwargs: if the optimization inputs are available, they can be passed-in as
      kwargs. These inputs contain everything this function needs to operate. If we
      don't have these, then the rest of the variables will need to be given
    
    - lensmodel: string specifying the lensmodel we're using (this is always
      'LENSMODEL_...'). The full list of valid models is returned by
      mrcal.supported_lensmodels(). This is required if we're not passing in the
      optimization inputs
    
    - do_optimize_intrinsics_core
      do_optimize_intrinsics_distortions
      do_optimize_extrinsics
      do_optimize_calobject_warp
      do_optimize_frames
    
      optional booleans; default to True. These specify what we're optimizing. See
      the documentation for mrcal.optimize() for details
    
    - Ncameras_intrinsics
      Ncameras_extrinsics
      Nframes
      Npoints
      Npoints_fixed
    
      optional integers; default to 0. These specify the sizes of various arrays in
      the optimization. See the documentation for mrcal.optimize() for details
    
    RETURNED VALUE
    
    None. The scaling is applied to the input array
    
    """
