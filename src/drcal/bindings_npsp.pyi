"""
Low-level routines for core mrcal operations

This is the written-in-C Python extension module that underlies the core
(un)project routines, and several low-level operations. Most of the functions in
this module (those prefixed with "_") are not meant to be called directly, but
have Python wrappers that should be used instead.

All functions are exported into the mrcal module. So you can call these via
mrcal._mrcal_npsp.fff() or mrcal.fff(). The latter is preferred.

"""

from __future__ import annotations

__all__: list[str] = ["apply_color_map", "apply_homography"]

def _A_Jt_J_At(*args, **kwargs):
    """
    Computes matmult(A,Jt,J,At) for a sparse J

    This is used in the internals of projection_uncertainty().

    A has shape (2,Nstate)

    J has shape (Nmeasurements,Nstate). J is large and sparse

    We use the Nleading_rows_J leading rows of J. This integer is passed-in as an
    argument.

    matmult(A, Jt, J, At) has shape (2,2)

    The input matrices are large, but the result is very small. I can't see a way to
    do this efficiently in pure Python, so I'm writing this.

    J is sparse, stored by row. This is the scipy.sparse.csr_matrix representation,
    and is also how CHOLMOD stores Jt (CHOLMOD stores by column, so the same data
    looks like Jt to CHOLMOD). The sparse J is given here as the p,i,x arrays from
    CHOLMOD, equivalent to the indptr,indices,data members of
    scipy.sparse.csr_matrix respectively.

    """

def _A_Jt_J_At__2(*args, **kwargs):
    """
    Computes matmult(A,Jt,J,At) for a sparse J where A.shape=(2,N)

    Exactly the same as _A_Jt_J_At(), but assumes that A.shape=(2,N) for efficiency.
    See the docs of _A_Jt_J_At() for details.

    """

def _Jt_x(*args, **kwargs):
    """
    Computes matrix-vector multiplication Jt*xt

    SYNOPSIS

        Jt_x = np.zeros( (J.shape[-1],), dtype=float)
        mrcal._mrcal_npsp._Jt_x(J.indptr,
                                J.indices,
                                J.data,
                                x,
                                out = Jt_x)

    Jt is the transpose of a (possibly very large) sparse array and x is a dense
    column vector. We pass in

    - J: the sparse array
    - xt: the row vector transpose of x

    The output is a dense row vector, the transpose of the multiplication

    J is sparse, stored by row. This is the scipy.sparse.csr_matrix representation,
    and is also how CHOLMOD stores Jt (CHOLMOD stores by column, so the same data
    looks like Jt to CHOLMOD). The sparse J is given here as the p,i,x arrays from
    CHOLMOD, equivalent to the indptr,indices,data members of
    scipy.sparse.csr_matrix respectively.

    Note: The output array MUST be passed-in because there's no way to know its
    shape beforehand. For the same reason, we cannot verify that its shape is
    correct, and the caller MUST do that, or else the program can crash.

    """

def _project(*args, **kwargs):
    """
    Internal point-projection routine

    This is the internals for mrcal.project(). As a user, please call THAT function,
    and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_withgrad

    - To make the broadcasting work, the argument order in this function is
      different. numpysane_pywrap broadcasts the leading arguments, so this function
      takes the lensmodel (the one argument that does not broadcast) last

    - To speed things up, this function doesn't call the C mrcal_project(), but uses
      the _mrcal_project_internal...() functions instead. That allows as much as
      possible of the outer init stuff to be moved outside of the slice computation
      loop

    This function is wrapped with numpysane_pywrap, so the points and the intrinsics
    broadcast as expected

    The outer logic (outside the loop-over-N-points) is duplicated in
    mrcal_project() and in the python wrapper definition in _project() and
    _project_withgrad() in mrcal-genpywrap.py. Please keep them in sync

    """

def _project_latlon(*args, **kwargs):
    """
    Internal projection routine

    This is the internals for mrcal.project_latlon(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_latlon_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_latlon_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients

    This is the internals for mrcal.project_latlon(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_latlon()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_lonlat(*args, **kwargs):
    """
    Internal projection routine

    This is the internals for mrcal.project_lonlat(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_lonlat_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_lonlat_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients

    This is the internals for mrcal.project_lonlat(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_lonlat()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_pinhole(*args, **kwargs):
    """
    Internal projection routine

    This is the internals for mrcal.project_pinhole(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_pinhole_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_pinhole_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients

    This is the internals for mrcal.project_pinhole(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_pinhole()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_stereographic(*args, **kwargs):
    """
    Internal projection routine

    This is the internals for mrcal.project_stereographic(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_stereographic_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_stereographic_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients

    This is the internals for mrcal.project_stereographic(). As a user, please call
    THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_stereographic()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _project_withgrad(*args, **kwargs):
    """
    Internal point-projection routine

    This is the internals for mrcal.project(). As a user, please call THAT function,
    and see the docs for that function. The differences:

    - This is just the gradients-returning function. The internal function that
      skips those is _project

    - To make the broadcasting work, the argument order in this function is
      different. numpysane_pywrap broadcasts the leading arguments, so this function
      takes the lensmodel (the one argument that does not broadcast) last

    - To speed things up, this function doesn't call the C mrcal_project(), but uses
      the _mrcal_project_internal...() functions instead. That allows as much as
      possible of the outer init stuff to be moved outside of the slice computation
      loop

    This function is wrapped with numpysane_pywrap, so the points and the intrinsics
    broadcast as expected

    The outer logic (outside the loop-over-N-points) is duplicated in
    mrcal_project() and in the python wrapper definition in _project() and
    _project_withgrad() in mrcal-genpywrap.py. Please keep them in sync

    """

def _stereo_range_dense(*args, **kwargs):
    """
    Internal wrapper of mrcal_stereo_range_dense()
    """

def _stereo_range_sparse(*args, **kwargs):
    """
    Internal wrapper of mrcal_stereo_range_sparse()
    """

def _unproject(*args, **kwargs):
    """
    Internal point-unprojection routine

    This is the internals for mrcal.unproject(). As a user, please call THAT
    function, and see the docs for that function. The differences:

    - To make the broadcasting work, the argument order in this function is
      different. numpysane_pywrap broadcasts the leading arguments, so this function
      takes the lensmodel (the one argument that does not broadcast) last

    - This function requires gradients, so it does not support some lens models;
      CAHVORE for instance

    - To speed things up, this function doesn't call the C mrcal_unproject(), but
      uses the _mrcal_unproject_internal...() functions instead. That allows as much
      as possible of the outer init stuff to be moved outside of the slice
      computation loop

    This function is wrapped with numpysane_pywrap, so the points and the intrinsics
    broadcast as expected

    The outer logic (outside the loop-over-N-points) is duplicated in
    mrcal_unproject() and in the python wrapper definition in _unproject()
    mrcal-genpywrap.py. Please keep them in sync
    """

def _unproject_latlon(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_latlon(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_latlon_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_latlon_withgrad(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_latlon(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_latlon()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_lonlat(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_lonlat(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_lonlat_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_lonlat_withgrad(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_lonlat(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_lonlat()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_pinhole(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_pinhole(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_pinhole_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_pinhole_withgrad(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_pinhole(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_pinhole()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_stereographic(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_stereographic(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_stereographic_withgrad()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def _unproject_stereographic_withgrad(*args, **kwargs):
    """
    Internal unprojection routine

    This is the internals for mrcal.unproject_stereographic(). As a user, please
    call THAT function, and see the docs for that function. The differences:

    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_stereographic()

    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected

    """

def apply_color_map(*args, **kwargs):
    """
    Color-code an array

    SYNOPSIS

        image = produce_data()

        print( image.shape )
        ===>
        (480, 640)

        image_colorcoded = mrcal.apply_color_map(image)

        print( image_colorcoded.shape )
        ===>
        (480, 640, 3)

        print( image_colorcoded.dtype )
        ===>
        dtype('uint8')

        mrcal.save_image('data.png', image_colorcoded)

    This is very similar to cv2.applyColorMap() but more flexible in several
    important ways. Differences:

    - Supports arrays of any shape. Most of the time the input is 2-dimensional
      images, but this isn't required

    - Supports any input data type, NOT limited to 8-bit images like
      cv2.applyColorMap()

    - Supports gnuplot color maps instead of MATLAB ones

    The color map is applied to each value in the input, each one producing an BGR
    row of shape (3,). So output.shape is input.shape + (3,).

    The output has dtype=numpy.uint8, so these arrays can be output as images, and
    visualized using any image-viewing tools.

    This function uses gnuplot's color maps, specified as rgbformulae:

      http://gnuplot.info/docs_6.0/loc14176.html
      http://gnuplot.info/docs_6.0/loc14246.html

    This is selected by passing (function_red,function_blue,function_green)
    integers, selecting different functions for each color channel. The default is
    the default gnuplot colormap: 7,5,15. This is a nice
    black-violet-blue-purple-red-orange-yellow map, appropriate for most usages. A
    colormap may be visualized with gnuplot. For instance to see the "AFM hot"
    colormap, run this gnuplot script:

      set palette rgbformulae 34,35,36
      test palette

    The definition of each colormap function is given by "show palette rgbformulae"
    in gnuplot:

        > show palette rgbformulae
         * there are 37 available rgb color mapping formulae:
            0: 0               1: 0.5             2: 1
            3: x               4: x^2             5: x^3
            6: x^4             7: sqrt(x)         8: sqrt(sqrt(x))
            9: sin(90x)       10: cos(90x)       11: |x-0.5|
           12: (2x-1)^2       13: sin(180x)      14: |cos(180x)|
           15: sin(360x)      16: cos(360x)      17: |sin(360x)|
           18: |cos(360x)|    19: |sin(720x)|    20: |cos(720x)|
           21: 3x             22: 3x-1           23: 3x-2
           24: |3x-1|         25: |3x-2|         26: (3x-1)/2
           27: (3x-2)/2       28: |(3x-1)/2|     29: |(3x-2)/2|
           30: x/0.32-0.78125 31: 2*x-0.84       32: 4x;1;-2x+1.84;x/0.08-11.5
           33: |2*x - 0.5|    34: 2*x            35: 2*x - 0.5
           36: 2*x - 1
         * negative numbers mean inverted=negative colour component
         * thus the ranges in `set pm3d rgbformulae' are -36..36

    ARGUMENTS

    - array: input numpy array

    - a_min: optional value indicating the lower bound of the values we color map.
      All input values outside of the range [a_min,a_max] are clipped. If omitted,
      we use array.min()

    - a_max: optional value indicating the upper bound of the values we color map.
      All input values outside of the range [a_min,a_max] are clipped. If omitted,
      we use array.max()

    - function_red
      function_green
      function_blue: optional integers selecting the color maps for each channel.
      See the full docstring for this function for detail

    RETURNED VALUE

    The color-mapped output array of shape array.shape + (3,) and containing 8-bit
    unsigned integers. The last row is the BGR color-mapped values.
    """

def apply_homography(*args, **kwargs):
    """
    Apply a homogeneous-coordinate homography to a set of 2D points

    SYNOPSIS

        print( H.shape )
        ===> (3,3)

        print( q0.shape )
        ===> (100, 2)

        q1 = mrcal.apply_homography(H10, q0)

        print( q1.shape )
        ===> (100, 2)

    A homography maps from pixel coordinates observed in one camera to pixel
    coordinates in another. For points represented in homogeneous coordinates ((k*x,
    k*y, k) to represent a pixel (x,y) for any k) a homography is a linear map H.
    Since homogeneous coordinates are unique only up-to-scale, the homography matrix
    H is also unique up to scale.

    If two pinhole cameras are observing a planar surface, there exists a homography
    that relates observations of the plane in the two cameras.

    This function supports broadcasting fully.

    ARGUMENTS

    - H: an array of shape (..., 3,3). This is the homography matrix. This is unique
      up-to-scale, so a homography H is functionally equivalent to k*H for any
      non-zero scalar k

    - q: an array of shape (..., 2). The pixel coordinates we are mapping

    RETURNED VALUE

    An array of shape (..., 2) containing the pixels q after the homography was
    applied


    """
