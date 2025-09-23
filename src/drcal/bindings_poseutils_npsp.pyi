"""
Low-level routines to manipulate poses, transformations and points

This is the written-in-C Python extension module. Most of the time you want to
use the drcal.poseutils wrapper module instead of this module directly. Any
functions not prefixed with "_" are meant to be called directly, without the
wrapper.

All functions are exported into the drcal module. So you can call these via
drcal._poseutils.fff() or drcal.fff(). The latter is preferred.

"""
from __future__ import annotations
__all__: list[str] = ['R_aligned_to_vector', 'R_from_quat', 'compose_r_tinyr0_gradientr0', 'compose_r_tinyr1_gradientr1', 'identity_R', 'identity_Rt', 'identity_r', 'identity_rt', 'skew_symmetric']
def R_aligned_to_vector(*args, **kwargs):
    """
    Compute a rotation to map a given vector to [0,0,1]
    
    SYNOPSIS
    
        # I have a plane that passes through a point p, and has a normal n. I
        # compute a transformation from the world to a coord system aligned to the
        # plane, with p at the origin. R_plane_world p + t_plane_world = 0:
    
        Rt_plane_world = np.zeros((4,3), dtype=float)
        Rt_plane_world[:3,:] = drcal.R_aligned_to_vector(n)
        Rt_plane_world[ 3,:] = -drcal.rotate_point_R(Rt_plane_world[:3,:],p)
    
    This rotation is not unique: adding any rotation around v still maps v to
    [0,0,1]. An arbitrary acceptable rotation is returned.
    
    ARGUMENTS
    
    - v: a numpy array of shape (3,). The vector that the computed rotation maps to
      [0,0,1]. Does not need to be normalized. Must be non-0
    
    RETURNED VALUES
    
    The rotation in a (3,3) array
    
        
    """
def R_from_quat(*args, **kwargs):
    """
    Convert a rotation defined as a unit quaternion rotation to a rotation matrix
    
    SYNOPSIS
    
        s    = np.sin(rotation_magnitude/2.)
        c    = np.cos(rotation_magnitude/2.)
        quat = nps.glue( c, s*rotation_axis, axis = -1)
    
        print(quat.shape)
        ===>
        (4,)
    
        R = drcal.R_from_quat(quat)
    
        print(R.shape)
        ===>
        (3,3)
    
    This is mostly for compatibility with some old stuff. drcal doesn't use
    quaternions anywhere. Test this thoroughly before using.
    
    This function supports broadcasting fully.
    
    ARGUMENTS
    
    - quat: array of shape (4,). The unit quaternion that defines the rotation. The
      values in the array are (u,i,j,k)
    
    - out: optional argument specifying the destination. By default, new numpy
      array(s) are created and returned. To write the results into existing (and
      possibly non-contiguous) arrays, specify them with the 'out' kwarg. If 'out'
      is given, we return the 'out' that was passed in. This is the standard
      behavior provided by numpysane_pywrap.
    
    RETURNED VALUE
    
    We return an array of rotation matrices. Each broadcasted slice has shape (3,3)
    
        
    """
def _R_from_r(*args, **kwargs):
    """
    Compute a rotation matrix from a Rodrigues vector
    
    This is an internal function. You probably want drcal.R_from_r(). See the docs
    for that function for details.
    """
def _R_from_r_withgrad(*args, **kwargs):
    """
    Compute a rotation matrix from a Rodrigues vector
    
    This is an internal function. You probably want drcal.R_from_r(). See the docs
    for that function for details.
    """
def _Rt_from_rt(*args, **kwargs):
    """
    Compute an Rt transformation from a rt transformation
    
    This is an internal function. You probably want drcal.Rt_from_rt(). See the docs
    for that function for details.
    """
def _Rt_from_rt_withgrad(*args, **kwargs):
    """
    Compute an Rt transformation from a rt transformation
    
    This is an internal function. You probably want drcal.Rt_from_rt(). See the docs
    for that function for details.
    """
def _align_procrustes_points_Rt01_noweights(*args, **kwargs):
    """
    Compute a rotation to align two sets of direction vectors or points
    
            This is the written-in-C Python extension module. Most of the time you want to
            use the drcal.poseutils wrapper module instead of this module directly. Any
            functions not prefixed with "_" are meant to be called directly, without the
            wrapper.
    
            All functions are exported into the drcal module. So you can call these via
            drcal._poseutils.fff() or drcal.fff(). The latter is preferred.
    
                
    """
def _align_procrustes_points_Rt01_weights(*args, **kwargs):
    """
    Compute a rotation to align two sets of direction vectors or points
    
            This is the written-in-C Python extension module. Most of the time you want to
            use the drcal.poseutils wrapper module instead of this module directly. Any
            functions not prefixed with "_" are meant to be called directly, without the
            wrapper.
    
            All functions are exported into the drcal module. So you can call these via
            drcal._poseutils.fff() or drcal.fff(). The latter is preferred.
    
                
    """
def _align_procrustes_vectors_R01_noweights(*args, **kwargs):
    """
    Compute a rotation to align two sets of direction vectors or points
    
            This is the written-in-C Python extension module. Most of the time you want to
            use the drcal.poseutils wrapper module instead of this module directly. Any
            functions not prefixed with "_" are meant to be called directly, without the
            wrapper.
    
            All functions are exported into the drcal module. So you can call these via
            drcal._poseutils.fff() or drcal.fff(). The latter is preferred.
    
                
    """
def _align_procrustes_vectors_R01_weights(*args, **kwargs):
    """
    Compute a rotation to align two sets of direction vectors or points
    
            This is the written-in-C Python extension module. Most of the time you want to
            use the drcal.poseutils wrapper module instead of this module directly. Any
            functions not prefixed with "_" are meant to be called directly, without the
            wrapper.
    
            All functions are exported into the drcal module. So you can call these via
            drcal._poseutils.fff() or drcal.fff(). The latter is preferred.
    
                
    """
def _compose_Rt(*args, **kwargs):
    """
    Composes two Rt transformations
    
    This is an internal function. You probably want drcal.compose_Rt(). See the docs
    for that function for details. This internal function differs from compose_Rt():
    
    - It supports exactly two arguments, while compose_Rt() can compose N
      transformations
    """
def _compose_r(*args, **kwargs):
    """
    Compose two angle-axis rotations
    
    This is an internal function. You probably want drcal.compose_r(). See the docs
    for that function for details. This internal function differs from compose_r():
    
    - It supports exactly two arguments, while compose_r() can compose N rotations
    
    - It never reports gradients
    """
def _compose_r_withgrad(*args, **kwargs):
    """
    Compose two angle-axis rotations; return (r,dr/dr0,dr/dr1)
    
    This is an internal function. You probably want drcal.compose_r(). See the docs
    for that function for details. This internal function differs from compose_r():
    
    - It supports exactly two arguments, while compose_r() can compose N rotations
    
    - It always reports gradients
    
    """
def _compose_rt(*args, **kwargs):
    """
    Compose two rt transformations
    
    This is an internal function. You probably want drcal.compose_rt(). See the docs
    for that function for details. This internal function differs from compose_rt():
    
    - It supports exactly two arguments, while compose_rt() can compose N
      transformations
    
    - It never reports gradients
    """
def _compose_rt_withgrad(*args, **kwargs):
    """
    Compose two rt transformations; return (rt,drt/drt0,drt/drt1)
    
    This is an internal function. You probably want drcal.compose_rt(). See the docs
    for that function for details. This internal function differs from compose_rt():
    
    - It supports exactly two arguments, while compose_rt() can compose N
      transformations
    
    - It always reports gradients
    
    Note that the C library returns limited gradients:
    
    - dr/dt0 is not returned: it is always 0
    - dr/dt1 is not returned: it is always 0
    
    THIS function combines these into the full drtout_drt0,drtout_drt1 arrays
    
    """
def _invert_R(*args, **kwargs):
    """
    Invert a rotation matrix
    
    This is an internal function. You probably want drcal.invert_R(). See the docs
    for that function for details.
    """
def _invert_Rt(*args, **kwargs):
    """
    Invert an Rt transformation
    
    This is an internal function. You probably want drcal.invert_Rt(). See the docs
    for that function for details.
    """
def _invert_rt(*args, **kwargs):
    """
    Invert an rt transformation
    
    This is an internal function. You probably want drcal.invert_rt(). See the docs
    for that function for details.
    """
def _invert_rt_withgrad(*args, **kwargs):
    """
    Invert an rt transformation
    
    This is an internal function. You probably want drcal.invert_rt(). See the docs
    for that function for details.
    
    Note that the C library returns limited gradients:
    
    - It returns dtout_drin,dtout_dtin only because
    
    - drout_drin always -I
    - drout_dtin always 0
    
    THIS function combines these into a full drtout_drtin array
    
    """
def _r_from_R(*args, **kwargs):
    """
    Compute a Rodrigues vector from a rotation matrix
    
    This is an internal function. You probably want drcal.r_from_R(). See the docs
    for that function for details.
    """
def _r_from_R_withgrad(*args, **kwargs):
    """
    Compute a Rodrigues vector from a rotation matrix
    
    This is an internal function. You probably want drcal.r_from_R(). See the docs
    for that function for details.
    """
def _rotate_point_R(*args, **kwargs):
    """
    Rotate a point using a rotation matrix
    
    This is an internal function. You probably want drcal.rotate_point_R(). See the
    docs for that function for details.
    
    """
def _rotate_point_R_withgrad(*args, **kwargs):
    """
    Rotate a point using a rotation matrix; report the result and gradients
    
    This is an internal function. You probably want drcal.rotate_point_R(). See the
    docs for that function for details.
    """
def _rotate_point_r(*args, **kwargs):
    """
    Rotate a point using a Rodrigues vector
    
    This is an internal function. You probably want drcal.rotate_point_r(). See the
    docs for that function for details.
    """
def _rotate_point_r_withgrad(*args, **kwargs):
    """
    Rotate a point using a Rodrigues vector; report the result and gradients
    
    This is an internal function. You probably want drcal.rotate_point_r(). See the
    docs for that function for details.
    """
def _rt_from_Rt(*args, **kwargs):
    """
    Compute an rt transformation from a Rt transformation
    
    This is an internal function. You probably want drcal.rt_from_Rt(). See the docs
    for that function for details.
    """
def _rt_from_Rt_withgrad(*args, **kwargs):
    """
    Compute an rt transformation from a Rt transformation
    
    This is an internal function. You probably want drcal.rt_from_Rt(). See the docs
    for that function for details.
    """
def _transform_point_Rt(*args, **kwargs):
    """
    Transform a point using an Rt transformation
    
    This is an internal function. You probably want drcal.transform_point_Rt(). See
    the docs for that function for details.
    """
def _transform_point_Rt_withgrad(*args, **kwargs):
    """
    Transform a point using an Rt transformation; report the result and gradients
    
    This is an internal function. You probably want drcal.transform_point_Rt(). See
    the docs for that function for details.
    """
def _transform_point_rt(*args, **kwargs):
    """
    Transform a point using an rt transformation
    
    This is an internal function. You probably want drcal.transform_point_rt(). See
    the docs for that function for details.
    """
def _transform_point_rt_withgrad(*args, **kwargs):
    """
    Transform a point using an rt transformation; report the result and gradients
    
    This is an internal function. You probably want drcal.transform_point_rt(). See
    the docs for that function for details.
    """
def compose_r_tinyr0_gradientr0(*args, **kwargs):
    """
    Special-case rotation composition for the uncertainty computation
    
    SYNOPSIS
    
        r1 = rotation_axis1 * rotation_magnitude1
    
        dr01_dr0 = compose_r_tinyr0_gradientr0(r1)
    
        ### Another way to get the same thing (but possibly less efficiently)
         _,dr01_dr0,_ = compose_r(np.zeros((3,),),
                                  r1,
                                  get_gradients=True)
    
    This is a special-case subset of compose_r(). It is the same, except:
    
    - r0 is assumed to be 0, so we don't ingest it, and we don't report the
      composition result
    - we ONLY report the dr01/dr0 gradient
    
    This special-case function is a part of the projection uncertainty computation,
    so it exists separate from compose_r(). See the documentation for compose_r()
    for all the details.
    
    This function supports broadcasting fully.
    
    ARGUMENTS
    
    - r1: the second of the two rotations being composed. The first rotation is an
      identity, so it's not given
    
    - out: optional argument specifying the destination. By default, a new numpy
      array is created and returned. To write the results into an existing (and
      possibly non-contiguous) array, specify it with the 'out' kwarg
    
    RETURNED VALUE
    
    We return a single array of shape (...,3,3): dr01/dr0
    
    """
def compose_r_tinyr1_gradientr1(*args, **kwargs):
    """
    Special-case rotation composition for the uncertainty computation
    
    SYNOPSIS
    
        r0 = rotation_axis0 * rotation_magnitude0
    
        dr01_dr1 = compose_r_tinyr1_gradientr1(r0)
    
        ### Another way to get the same thing (but possibly less efficiently)
         _,_,dr01_dr1 = compose_r(r0,
                                  np.zeros((3,),),
                                  get_gradients=True)
    
    This is a special-case subset of compose_r(). It is the same, except:
    
    - r1 is assumed to be 0, so we don't ingest it, and we don't report the
      composition result
    - we ONLY report the dr01/dr1 gradient
    
    This special-case function is a part of the projection uncertainty computation,
    so it exists separate from compose_r(). See the documentation for compose_r()
    for all the details.
    
    This function supports broadcasting fully.
    
    ARGUMENTS
    
    - r0: the first of the two rotations being composed. The second rotation is an
      identity, so it's not given
    
    - out: optional argument specifying the destination. By default, a new numpy
      array is created and returned. To write the results into an existing (and
      possibly non-contiguous) array, specify it with the 'out' kwarg
    
    RETURNED VALUE
    
    We return a single array of shape (...,3,3): dr01/dr1
    
    """
def identity_R(*args, **kwargs):
    """
    Return an identity rotation matrix
    
    SYNOPSIS
    
        print( drcal.identity_R() )
        ===>
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    
    As with all the poseutils functions, the output can be written directly into a
    (possibly-non-contiguous) array, by specifying the destination in the 'out'
    kwarg 
    """
def identity_Rt(*args, **kwargs):
    """
    Return an identity Rt transformation
    
    SYNOPSIS
    
        print( drcal.identity_Rt() )
        ===>
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]
         [0. 0. 0.]]
    
    As with all the poseutils functions, the output can be written directly into a
    (possibly-non-contiguous) array, by specifying the destination in the 'out'
    kwarg
    """
def identity_r(*args, **kwargs):
    """
    Return an identity Rodrigues rotation
    
    SYNOPSIS
    
        print( drcal.identity_r() )
        ===>
        [0. 0. 0.]
    
    As with all the poseutils functions, the output can be written directly into a
    (possibly-non-contiguous) array, by specifying the destination in the 'out'
    kwarg
    """
def identity_rt(*args, **kwargs):
    """
    Return an identity rt transformation
    
    SYNOPSIS
    
        print( drcal.identity_rt() )
        ===>
        [0. 0. 0. 0. 0. 0.]
    
    As with all the poseutils functions, the output can be written directly into a
    (possibly-non-contiguous) array, by specifying the destination in the 'out'
    kwarg
    """
def skew_symmetric(*args, **kwargs):
    """
    Return the skew-symmetric matrix used in a cross product
    
    SYNOPSIS
    
        a = np.array(( 1.,  5.,  7.))
        b = np.array(( 3., -.1, -10.))
    
        A = drcal.skew_symmetric(a)
    
        print( nps.inner(A,b) )
        ===>
        [-49.3  31.  -15.1]
    
        print( np.cross(a,b) )
        ===>
        [-49.3  31.  -15.1]
    
    A vector cross-product a x b can be represented as a matrix multiplication A*b
    where A is a skew-symmetric matrix based on the vector a. This function computes
    this matrix A from the vector a.
    
    This function supports broadcasting fully.
    
    ARGUMENTS
    
    - a: array of shape (3,)
    
    - out: optional argument specifying the destination. By default, new numpy
      array(s) are created and returned. To write the results into existing (and
      possibly non-contiguous) arrays, specify them with the 'out' kwarg. If 'out'
      is given, we return the 'out' that was passed in. This is the standard
      behavior provided by numpysane_pywrap.
    
    RETURNED VALUE
    
    We return the matrix A in a (3,3) numpy array
    
        
    """
