"""
Low-level routines for core drcal operations

This is the written-in-C Python extension module that underlies the core
(un)project routines, and several low-level operations. Most of the functions in
this module (those prefixed with "_") are not meant to be called directly, but
have Python wrappers that should be used instead.

All functions are exported into the drcal module. So you can call these via
drcal._drcal_npsp.fff() or drcal.fff(). The latter is preferred.

"""
from __future__ import annotations
__all__: list[str] = ['apply_homography']
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
        drcal._drcal_npsp._Jt_x(J.indptr,
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
    
    This is the internals for drcal.project(). As a user, please call THAT function,
    and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_withgrad
    
    - To make the broadcasting work, the argument order in this function is
      different. numpysane_pywrap broadcasts the leading arguments, so this function
      takes the lensmodel (the one argument that does not broadcast) last
    
    - To speed things up, this function doesn't call the C drcal_project(), but uses
      the _drcal_project_internal...() functions instead. That allows as much as
      possible of the outer init stuff to be moved outside of the slice computation
      loop
    
    This function is wrapped with numpysane_pywrap, so the points and the intrinsics
    broadcast as expected
    
    The outer logic (outside the loop-over-N-points) is duplicated in
    drcal_project() and in the python wrapper definition in _project() and
    _project_withgrad() in drcal-genpywrap.py. Please keep them in sync
    
    """
def _project_latlon(*args, **kwargs):
    """
    Internal projection routine
    
    This is the internals for drcal.project_latlon(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_latlon_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_latlon_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients
    
    This is the internals for drcal.project_latlon(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_latlon()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_lonlat(*args, **kwargs):
    """
    Internal projection routine
    
    This is the internals for drcal.project_lonlat(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_lonlat_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_lonlat_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients
    
    This is the internals for drcal.project_lonlat(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_lonlat()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_pinhole(*args, **kwargs):
    """
    Internal projection routine
    
    This is the internals for drcal.project_pinhole(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_pinhole_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_pinhole_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients
    
    This is the internals for drcal.project_pinhole(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_pinhole()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_stereographic(*args, **kwargs):
    """
    Internal projection routine
    
    This is the internals for drcal.project_stereographic(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _project_stereographic_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_stereographic_withgrad(*args, **kwargs):
    """
    Internal projection routine with gradients
    
    This is the internals for drcal.project_stereographic(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that
      does not report the gradients is _project_stereographic()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _project_withgrad(*args, **kwargs):
    """
    Internal point-projection routine
    
    This is the internals for drcal.project(). As a user, please call THAT function,
    and see the docs for that function. The differences:
    
    - This is just the gradients-returning function. The internal function that
      skips those is _project
    
    - To make the broadcasting work, the argument order in this function is
      different. numpysane_pywrap broadcasts the leading arguments, so this function
      takes the lensmodel (the one argument that does not broadcast) last
    
    - To speed things up, this function doesn't call the C drcal_project(), but uses
      the _drcal_project_internal...() functions instead. That allows as much as
      possible of the outer init stuff to be moved outside of the slice computation
      loop
    
    This function is wrapped with numpysane_pywrap, so the points and the intrinsics
    broadcast as expected
    
    The outer logic (outside the loop-over-N-points) is duplicated in
    drcal_project() and in the python wrapper definition in _project() and
    _project_withgrad() in drcal-genpywrap.py. Please keep them in sync
    
    """
def _stereo_range_sparse(*args, **kwargs):
    """
    Internal wrapper of drcal_stereo_range_sparse()
    """
def _unproject(*args, **kwargs):
    """
    Internal point-unprojection routine
    
    This is the internals for drcal.unproject(). As a user, please call THAT
    function, and see the docs for that function. The differences:
    
    - To make the broadcasting work, the argument order in this function is
      different. numpysane_pywrap broadcasts the leading arguments, so this function
      takes the lensmodel (the one argument that does not broadcast) last
    
    - This function requires gradients, so it does not support some lens models.
    
    - To speed things up, this function doesn't call the C drcal_unproject(), but
      uses the _drcal_unproject_internal...() functions instead. That allows as much
      as possible of the outer init stuff to be moved outside of the slice
      computation loop
    
    This function is wrapped with numpysane_pywrap, so the points and the intrinsics
    broadcast as expected
    
    The outer logic (outside the loop-over-N-points) is duplicated in
    drcal_unproject() and in the python wrapper definition in _unproject()
    drcal-genpywrap.py. Please keep them in sync 
    """
def _unproject_latlon(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_latlon(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_latlon_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_latlon_withgrad(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_latlon(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_latlon()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_lonlat(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_lonlat(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_lonlat_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_lonlat_withgrad(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_lonlat(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_lonlat()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_pinhole(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_pinhole(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_pinhole_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_pinhole_withgrad(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_pinhole(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_pinhole()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_stereographic(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_stereographic(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that reports the
      gradients also is _unproject_stereographic_withgrad()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def _unproject_stereographic_withgrad(*args, **kwargs):
    """
    Internal unprojection routine
    
    This is the internals for drcal.unproject_stereographic(). As a user, please
    call THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-reporting function. The internal function that does
      not report the gradients is _unproject_stereographic()
    
    This function is wrapped with numpysane_pywrap, so the points broadcast as
    expected
    
    """
def apply_homography(*args, **kwargs):
    """
    Apply a homogeneous-coordinate homography to a set of 2D points
    
    SYNOPSIS
    
        print( H.shape )
        ===> (3,3)
    
        print( q0.shape )
        ===> (100, 2)
    
        q1 = drcal.apply_homography(H10, q0)
    
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
