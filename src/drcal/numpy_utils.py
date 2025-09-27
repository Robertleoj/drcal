import numpy as np
from functools import reduce
import itertools
import inspect
import types


def atleast_dims(x, *dims):
    r"""Returns an array with extra length-1 dimensions to contain all given axes.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> nps.atleast_dims(a, -1).shape
        (2, 3)

        >>> nps.atleast_dims(a, -2).shape
        (2, 3)

        >>> nps.atleast_dims(a, -3).shape
        (1, 2, 3)

        >>> nps.atleast_dims(a, 0).shape
        (2, 3)

        >>> nps.atleast_dims(a, 1).shape
        (2, 3)

        >>> nps.atleast_dims(a, 2).shape
        [exception]

        >>> l = [-3,-2,-1,0,1]
        >>> nps.atleast_dims(a, l).shape
        (1, 2, 3)

        >>> l
        [-3, -2, -1, 1, 2]

    If the given axes already exist in the given array, the given array itself
    is returned. Otherwise length-1 dimensions are added to the front until all
    the requested dimensions exist. The given axis>=0 dimensions MUST all be
    in-bounds from the start, otherwise the most-significant axis becomes
    unaligned; an exception is thrown if this is violated. The given axis<0
    dimensions that are out-of-bounds result in new dimensions added at the
    front.

    If new dimensions need to be added at the front, then any axis>=0 indices
    become offset. For instance:

        >>> x.shape
        (2, 3, 4)

        >>> [x.shape[i] for i in (0,-1)]
        [2, 4]

        >>> x = nps.atleast_dims(x, 0, -1, -5)
        >>> x.shape
        (1, 1, 2, 3, 4)

        >>> [x.shape[i] for i in (0,-1)]
        [1, 4]

    Before the call, axis=0 refers to the length-2 dimension and axis=-1 refers
    to the length=4 dimension. After the call, axis=-1 refers to the same
    dimension as before, but axis=0 now refers to a new length=1 dimension. If
    it is desired to compensate for this offset, then instead of passing the
    axes as separate arguments, pass in a single list of the axes indices. This
    list will be modified to offset the axis>=0 appropriately. Ideally, you only
    pass in axes<0, and this does not apply. Doing this in the above example:

        >>> l
        [0, -1, -5]

        >>> x.shape
        (2, 3, 4)

        >>> [x.shape[i] for i in (l[0],l[1])]
        [2, 4]

        >>> x=nps.atleast_dims(x, l)
        >>> x.shape
        (1, 1, 2, 3, 4)

        >>> l
        [2, -1, -5]

        >>> [x.shape[i] for i in (l[0],l[1])]
        [2, 4]

    We passed the axis indices in a list, and this list was modified to reflect
    the new indices: The original axis=0 becomes known as axis=2. Again, if you
    pass in only axis<0, then you don't need to care about this.

    """

    if any(not isinstance(d, int) for d in dims):
        if len(dims) == 1 and isinstance(dims[0], list):
            dims = dims[0]
        else:
            raise ValueError(
                "atleast_dims() takes in axes as integers in separate arguments or\n"
                "as a single modifiable list"
            )

    if max(dims) >= x.ndim:
        raise ValueError(
            "Axis {} out of bounds because x.ndim={}.\n"
            "To keep the last dimension anchored, "
            "only <0 out-of-bounds axes are allowed".format(max(dims), x.ndim)
        )

    need_ndim = -min(d if d < 0 else -1 for d in dims)
    if x.ndim >= need_ndim:
        return x
    num_new_axes = need_ndim - x.ndim

    # apply an offset to any axes that need it
    if isinstance(dims, list):
        dims[:] = [d + num_new_axes if d >= 0 else d for d in dims]

    return x[(np.newaxis,) * (num_new_axes)]


def mv(x, axis_from, axis_to):
    r"""Moves a given axis to a new position. Similar to numpy.moveaxis().

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.mv( a, -1, 0).shape
        (4, 2, 3)

        >>> nps.mv( a, -1, -5).shape
        (4, 1, 1, 2, 3)

        >>> nps.mv( a, 0, -5).shape
        (2, 1, 1, 3, 4)

    New length-1 dimensions are added at the front, as required, and any axis>=0
    that are passed in refer to the array BEFORE these new dimensions are added.

    """
    axes = [axis_from, axis_to]
    x = atleast_dims(x, axes)

    # The below is equivalent to
    #   return np.moveaxis( x, *axes )
    # but some older installs have numpy 1.8, where this isn't available

    axis_from = axes[0] if axes[0] >= 0 else x.ndim + axes[0]
    axis_to = axes[1] if axes[1] >= 0 else x.ndim + axes[1]

    # python3 needs the list() cast
    order = list(range(0, axis_from)) + list(range((axis_from + 1), x.ndim))
    order.insert(axis_to, axis_from)
    return np.transpose(x, order)


def inner(a, b, out=None, dtype=None):
    """Inner product with broadcasting. Works like np.einsum('...i,...i->...', a, b)."""
    if dtype is None:
        dtype = float if out is None else out.dtype

    if out is None:
        return np.einsum("...i,...i->...", a, b, dtype=dtype)
    else:
        np.einsum("...i,...i->...", a, b, dtype=dtype, out=out)
        return out


def mag(a, out=None, dtype=None):
    r"""Magnitude of a vector. mag(x) is functionally identical to sqrt(inner(x,x))

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(3)
        >>> a
        array([0, 1, 2])

        >>> nps.mag(a)
        2.23606797749979

    This is a convenience function to compute a magnitude of a vector, with full
    broadcasting support.

    In-place operation is available with the "out" kwarg. The output dtype can
    be selected with the "dtype" kwarg. If omitted, dtype=float is selected.

    """

    if out is None:
        if dtype is None:
            dtype = float
        out = inner(a, a, dtype=dtype)

        if not isinstance(out, np.ndarray):
            # given two vectors, and without and 'out' array, inner() produces a
            # scalar, not an array. So I can't updated it inplace, and just
            # return a copy
            return np.sqrt(out)
    else:
        if dtype is None:
            dtype = out.dtype
        inner(a, a, out=out, dtype=dtype)

    # in-place sqrt
    np.sqrt.at(out, ())
    return out


def xchg(x, axis_a, axis_b):
    r"""Exchanges the positions of the two given axes. Similar to numpy.swapaxes()

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.xchg( a, -1, 0).shape
        (4, 3, 2)

        >>> nps.xchg( a, -1, -5).shape
        (4, 1, 2, 3, 1)

        >>> nps.xchg( a, 0, -5).shape
        (2, 1, 1, 3, 4)

    New length-1 dimensions are added at the front, as required, and any axis>=0
    that are passed in refer to the array BEFORE these new dimensions are added.

    """
    axes = [axis_a, axis_b]
    x = atleast_dims(x, axes)
    return np.swapaxes(x, *axes)


def transpose(x):
    r"""Reverses the order of the last two dimensions.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.transpose(a).shape
        (2, 4, 3)

        >>> nps.transpose( np.arange(3) ).shape
        (3, 1)

    A "matrix" is generally seen as a 2D array that we can transpose by looking
    at the 2 dimensions in the opposite order. Here we treat an n-dimensional
    array as an n-2 dimensional object containing 2D matrices. As usual, the
    last two dimensions contain the matrix.

    New length-1 dimensions are added at the front, as required, meaning that 1D
    input of shape (n,) is interpreted as a 2D input of shape (1,n), and the
    transpose is 2 of shape (n,1).

    """
    return xchg(atleast_dims(x, -2), -1, -2)


def matmult2(a, b, out=None):
    r"""Multiplication of two matrices

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6) .reshape(2,3)
        >>> b = np.arange(12).reshape(3,4)

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])

        >>> nps.matmult2(a,b)
        array([[20, 23, 26, 29],
               [56, 68, 80, 92]])

    This function is exposed publically mostly for legacy compatibility. Use
    numpysane.matmult() instead

    """

    if not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
        # two non-arrays (assuming two scalars)
        if out is not None:
            o = a * b
            out.setfield(o, out.dtype)
            out.resize([])
            return out
        return a * b

    if not isinstance(a, np.ndarray) or len(a.shape) == 0:
        # one non-array (assuming one scalar)
        if out is not None:
            out.setfield(a * b, out.dtype)
            out.resize(b.shape)
            return out
        return a * b

    if not isinstance(b, np.ndarray) or len(b.shape) == 0:
        # one non-array (assuming one scalar)
        if out is not None:
            out.setfield(a * b, out.dtype)
            out.resize(a.shape)
            return out
        return a * b

    if len(b.shape) == 1:
        b = b[np.newaxis, :]

    o = np.matmul(a, b, out)
    return o


def matmult(a, *rest, **kwargs):
    r"""Multiplication of N matrices

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6) .reshape(2,3)
        >>> b = np.arange(12).reshape(3,4)
        >>> c = np.arange(4) .reshape(4,1)

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])

        >>> c
        array([[0],
               [1],
               [2],
               [3]])

        >>> nps.matmult(a,b,c)
        array([[162],
               [504]])

        >>> abc = np.zeros((2,1), dtype=float)
        >>> nps.matmult(a,b,c, out=abc)
        >>> abc
        array([[162],
               [504]])

    This multiplies N matrices together by repeatedly calling matmult2() for
    each adjacent pair. In-place output is supported with the 'out' keyword
    argument

    This function supports broadcasting fully, in C internally

    """

    if len(rest) == 0:
        raise Exception("Need at least two terms to multiply")

    out = None
    if len(kwargs.keys()) > 1:
        raise Exception("Only ONE kwarg is supported: 'out'")
    if kwargs:
        # have exactly one kwarg
        if "out" not in kwargs:
            raise Exception("The only supported kwarg is 'out'")
        out = kwargs["out"]

    return matmult2(a, reduce(matmult2, rest), out=out)


def norm2(a, **kwargs):
    r"""Broadcast-aware 2-norm. norm2(x) is identical to inner(x,x)

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(3)
        >>> a
        array([0, 1, 2])

        >>> nps.norm2(a)
        5

    This is a convenience function to compute a 2-norm

    """
    return inner(a, a, **kwargs)


def glue(*args, **kwargs):
    r"""Concatenates a given list of arrays along the given 'axis' keyword argument.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> row = a[0,:] + 1000

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> row
        array([1000, 1001, 1002])

        >>> nps.glue(a,b, axis=-1)
        array([[  0,   1,   2, 100, 101, 102],
               [  3,   4,   5, 103, 104, 105]])

        # empty arrays ignored when glueing. Useful for initializing an accumulation
        >>> nps.glue(a,b, np.array(()), axis=-1)
        array([[  0,   1,   2, 100, 101, 102],
               [  3,   4,   5, 103, 104, 105]])

        >>> nps.glue(a,b,row, axis=-2)
        array([[   0,    1,    2],
               [   3,    4,    5],
               [ 100,  101,  102],
               [ 103,  104,  105],
               [1000, 1001, 1002]])

        >>> nps.glue(a,b, axis=-3)
        array([[[  0,   1,   2],
                [  3,   4,   5]],

               [[100, 101, 102],
                [103, 104, 105]]])

    The 'axis' must be given in a keyword argument.

    In order to count dimensions from the inner-most outwards, this function accepts
    only negative axis arguments. This is because numpy broadcasts from the last
    dimension, and the last dimension is the inner-most in the (usual) internal
    storage scheme. Allowing glue() to look at dimensions at the start would allow
    it to unalign the broadcasting dimensions, which is never what you want.

    To glue along the last dimension, pass axis=-1; to glue along the second-to-last
    dimension, pass axis=-2, and so on.

    Unlike in PDL, this function refuses to create duplicated data to make the
    shapes fit. In my experience, this isn't what you want, and can create bugs.
    For instance, PDL does this:

        pdl> p sequence(3,2)
        [
         [0 1 2]
         [3 4 5]
        ]

        pdl> p sequence(3)
        [0 1 2]

        pdl> p PDL::glue( 0, sequence(3,2), sequence(3) )
        [
         [0 1 2 0 1 2]   <--- Note the duplicated "0,1,2"
         [3 4 5 0 1 2]
        ]

    while numpysane.glue() does this:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a[0:1,:]


        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[0, 1, 2]])

        >>> nps.glue(a,b,axis=-1)
        [exception]

    Finally, this function adds as many length-1 dimensions at the front as
    required. Note that this does not create new data, just new degenerate
    dimensions. Example:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> res = nps.glue(a,b, axis=-5)
        >>> res
        array([[[[[  0,   1,   2],
                  [  3,   4,   5]]]],



               [[[[100, 101, 102],
                  [103, 104, 105]]]]])

        >>> res.shape
        (2, 1, 1, 2, 3)

    In numpysane older than 0.10 the semantics were slightly different: the axis
    kwarg was optional, and glue(*args) would glue along a new leading
    dimension, and thus would be equivalent to cat(*args). This resulted in very
    confusing error messages if the user accidentally omitted the kwarg. To
    request the legacy behavior, do

        nps.glue.legacy_version = '0.9'

    """
    axis = kwargs.get("axis")

    if axis is None:
        raise ValueError("glue() requires the axis to be given in the 'axis' kwarg")
    if axis >= 0:
        raise ValueError(
            "axis >= 0 can make broadcasting dimensions inconsistent, and is thus not allowed"
        )

    # deal with scalar (non-ndarray) args
    args = [np.asarray(x) for x in args]

    # Special case to support this common idiom:
    #
    #   accum = np.array(())
    #   while ...:
    #      x     = ...
    #      accum = nps.glue(accum, x,  axis = -2)
    #
    # Without special logic, this would throw an error since accum.shape starts
    # at (0,), which is almost certainly not compatible with x.shape. I support
    # both glue(empty,x) and glue(x,empty)
    if len(args) == 2:
        if args[0].shape == (0,) and args[1].size != 0:
            return atleast_dims(args[1], axis)
        if args[1].shape == (0,) and args[0].size != 0:
            return atleast_dims(args[0], axis)

    # Legacy behavior: if no axis is given, add a new axis at the front, and
    # glue along it
    max_ndim = max(x.ndim for x in args)
    if axis is None:
        axis = -1 - max_ndim

    # if we're glueing along a dimension beyond what we already have, expand the
    # target dimension count
    if max_ndim < -axis:
        max_ndim = -axis

    # Now I add dummy dimensions at the front of each array, to bring the source
    # arrays to the same dimensionality. After this is done, ndims for all the
    # matrices will be the same, and np.concatenate() should know what to do.
    args = [x[(np.newaxis,) * (max_ndim - x.ndim) + (Ellipsis,)] for x in args]
    return atleast_dims(np.concatenate(args, axis=axis), axis)


def cat(*args):
    r"""Concatenates a given list of arrays along a new first (outer) dimension.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> c = a - 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> c
        array([[-100,  -99,  -98],
               [ -97,  -96,  -95]])

        >>> res = nps.cat(a,b,c)
        >>> res
        array([[[   0,    1,    2],
                [   3,    4,    5]],

               [[ 100,  101,  102],
                [ 103,  104,  105]],

               [[-100,  -99,  -98],
                [ -97,  -96,  -95]]])

        >>> res.shape
        (3, 2, 3)

        >>> [x for x in res]
        [array([[0, 1, 2],
                [3, 4, 5]]),
         array([[100, 101, 102],
                [103, 104, 105]]),
         array([[-100,  -99,  -98],
                [ -97,  -96,  -95]])]
        ### Note that this is the same as [a,b,c]: cat is the reverse of
        ### iterating on an array

    This function concatenates the input arrays into an array shaped like the
    highest-dimensioned input, but with a new outer (at the start) dimension.
    The concatenation axis is this new dimension.

    As usual, the dimensions are aligned along the last one, so broadcasting
    will continue to work as expected. Note that this is the opposite operation
    from iterating a numpy array: iteration splits an array over its leading
    dimension, while cat joins a number of arrays via a new leading dimension.

    """
    if len(args) == 0:
        return np.array(())
    max_ndim = max(x.ndim for x in args)
    return glue(*args, axis=-1 - max_ndim)


def clump(x, **kwargs):
    r"""Groups the given n dimensions together.

    SYNOPSIS

        >>> import numpysane as nps
        >>> nps.clump( arr(2,3,4), n = -2).shape
        (2, 12)

    Reshapes the array by grouping together 'n' dimensions, where 'n' is given
    in a kwarg. If 'n' > 0, then n leading dimensions are clumped; if 'n' < 0,
    then -n trailing dimensions are clumped

    So for instance, if x.shape is (2,3,4) then nps.clump(x, n = -2).shape is
    (2,12) and nps.clump(x, n = 2).shape is (6, 4)

    In numpysane older than 0.10 the semantics were different: n > 0 was
    required, and we ALWAYS clumped the trailing dimensions. Thus the new
    clump(-n) is equivalent to the old clump(n). To request the legacy behavior,
    do

        nps.clump.legacy_version = '0.9'

    """
    n = kwargs.get("n")
    if n is None:
        raise ValueError("clump() requires a dimension count in the 'n' kwarg")

    if -1 <= n and n <= 1:
        return x

    if x.ndim < n:
        n = x.ndim
    if -x.ndim > n:
        n = -x.ndim

    if n < 0:
        s = list(x.shape[:n]) + [_product(x.shape[n:])]
    else:
        s = [_product(x.shape[:n])] + list(x.shape[n:])
    return x.reshape(s)


def _product(l):
    r"""Returns product of all values in the given list"""
    return reduce(lambda a, b: a * b, l)


def dummy(x, axis, *axes_rest):
    r"""Adds length-1 dimensions at the given positions.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(24).reshape(2,3,4)
        >>> a.shape
        (2, 3, 4)

        >>> nps.dummy(a, 0).shape
        (1, 2, 3, 4)

        >>> nps.dummy(a, 1).shape
        (2, 1, 3, 4)

        >>> nps.dummy(a, -1).shape
        (2, 3, 4, 1)

        >>> nps.dummy(a, -2).shape
        (2, 3, 1, 4)

        >>> nps.dummy(a, -2, -2).shape
        (2, 3, 1, 1, 4)

        >>> nps.dummy(a, -5).shape
        (1, 1, 2, 3, 4)

    This is similar to numpy.expand_dims(), but handles out-of-bounds dimensions
    better. New length-1 dimensions are added at the front, as required, and any
    axis>=0 that are passed in refer to the array BEFORE these new dimensions
    are added.

    """

    def dummy_inner(x, axis):
        need_ndim = axis + 1 if axis >= 0 else -axis
        if x.ndim >= need_ndim:
            # referring to an axis that already exists. expand_dims() thus works
            return np.expand_dims(x, axis)

        # referring to a non-existing axis. I simply add sufficient new axes, and
        # I'm done
        return atleast_dims(x, axis)

    axes = (axis,) + axes_rest
    for axis in axes:
        x = dummy_inner(x, axis)
    return x


def broadcast_define(prototype, prototype_output=None, out_kwarg=None):
    r"""Vectorizes an arbitrary function, expecting input as in the given prototype.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> @nps.broadcast_define( (('n',), ('n',)) )
        ... def inner_product(a, b):
        ...     return a.dot(b)

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> inner_product(a,b)
        array([ 305, 1250])

    The prototype defines the dimensionality of the inputs. In the inner product
    example above, the input is two 1D n-dimensional vectors. In particular, the
    'n' is the same for the two inputs. This function is intended to be used as
    a decorator, applied to a function defining the operation to be vectorized.
    Each element in the prototype list refers to each input, in order. In turn,
    each such element is a list that describes the shape of that input. Each of
    these shape descriptors can be any of

    - a positive integer, indicating an input dimension of exactly that length
    - a string, indicating an arbitrary, but internally consistent dimension

    The normal numpy broadcasting rules (as described elsewhere) apply. In
    summary:

    - Dimensions are aligned at the end of the shape list, and must match the
      prototype

    - Extra dimensions left over at the front must be consistent for all the
      input arguments, meaning:

      - All dimensions of length != 1 must match
      - Dimensions of length 1 match corresponding dimensions of any length in
        other arrays
      - Missing leading dimensions are implicitly set to length 1

    - The output(s) have a shape where
      - The trailing dimensions are whatever the function being broadcasted
        returns
      - The leading dimensions come from the extra dimensions in the inputs

    Calling a function wrapped with broadcast_define() with extra arguments
    (either positional or keyword), passes these verbatim to the inner function.
    Only the arguments declared in the prototype are broadcast.

    Scalars are represented as 0-dimensional numpy arrays: arrays with shape (),
    and these broadcast as one would expect:

        >>> @nps.broadcast_define( (('n',), ('n',), ()))
        ... def scaled_inner_product(a, b, scale):
        ...     return a.dot(b)*scale

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> scale = np.array((10,100))

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> scale
        array([ 10, 100])

        >>> scaled_inner_product(a,b,scale)
        array([[  3050],
               [125000]])

    Let's look at a more involved example. Let's say we have a function that
    takes a set of points in R^2 and a single center point in R^2, and finds a
    best-fit least-squares line that passes through the given center point. Let
    it return a 3D vector containing the slope, y-intercept and the RMS residual
    of the fit. This broadcasting-enabled function can be defined like this:

        import numpy as np
        import numpysane as nps

        @nps.broadcast_define( (('n',2), (2,)) )
        def fit(xy, c):
            # line-through-origin-model: y = m*x
            # E = sum( (m*x - y)**2 )
            # dE/dm = 2*sum( (m*x-y)*x ) = 0
            # ----> m = sum(x*y)/sum(x*x)
            x,y = (xy - c).transpose()
            m = np.sum(x*y) / np.sum(x*x)
            err = m*x - y
            err **= 2
            rms = np.sqrt(err.mean())
            # I return m,b because I need to translate the line back
            b = c[1] - m*c[0]

            return np.array((m,b,rms))

    And I can use broadcasting to compute a number of these fits at once. Let's
    say I want to compute 4 different fits of 5 points each. I can do this:

        n = 5
        m = 4
        c = np.array((20,300))
        xy = np.arange(m*n*2, dtype=np.float64).reshape(m,n,2) + c
        xy += np.random.rand(*xy.shape)*5

        res = fit( xy, c )
        mb  = res[..., 0:2]
        rms = res[..., 2]
        print "RMS residuals: {}".format(rms)

    Here I had 4 different sets of points, but a single center point c. If I
    wanted 4 different center points, I could pass c as an array of shape (4,2).
    I can use broadcasting to plot all the results (the points and the fitted
    lines):

        import gnuplotlib as gp

        gp.plot( *nps.mv(xy,-1,0), _with='linespoints',
                 equation=['{}*x + {}'.format(mb_single[0],
                                              mb_single[1]) for mb_single in mb],
                 unset='grid', square=1)

    The examples above all create a separate output array for each broadcasted
    slice, and copy the contents from each such slice into the larger output
    array that contains all the results. This is inefficient, and it is possible
    to pre-allocate an array to forgo these extra allocation and copy
    operations. There are several settings to control this. If the function
    being broadcasted can write its output to a given array instead of creating
    a new one, most of the inefficiency goes away. broadcast_define() supports
    the case where this function takes this array in a kwarg: the name of this
    kwarg can be given to broadcast_define() like so:

        @nps.broadcast_define( ....., out_kwarg = "out" )
        def func( ....., *, out):
            .....
            out[:] = result

    When used this way, the return value of the broadcasted function is ignored.
    In order for broadcast_define() to pass such an output array to the inner
    function, this output array must be available, which means that it must be
    given to us somehow, or we must create it.

    The most efficient way to make a broadcasted call is to create the full
    output array beforehand, and to pass that to the broadcasted function. In
    this case, nothing extra will be allocated, and no unnecessary copies will
    be made. This can be done like this:

        @nps.broadcast_define( (('n',), ('n',)), ....., out_kwarg = "out" )
        def inner_product(a, b, *, out):
            .....
            out.setfield(a.dot(b), out.dtype)

        out = np.empty((2,4), np.float)
        inner_product( np.arange(3), np.arange(2*4*3).reshape(2,4,3), out=out)

    In this example, the caller knows that it's calling an inner_product
    function, and that the shape of each output slice would be (). The caller
    also knows the input dimensions and that we have an extra broadcasting
    dimension (2,4), so the output array will have shape (2,4) + () = (2,4).
    With this knowledge, the caller preallocates the array, and passes it to the
    broadcasted function call. Furthermore, in this case the inner function will
    be called with an output array EVERY time, and this is the only mode the
    inner function needs to support.

    If the caller doesn't know (or doesn't want to pre-compute) the shape of the
    output, it can let the broadcasting machinery create this array for them. In
    order for this to be possible, the shape of the output should be
    pre-declared, and the dtype of the output should be known:

        @nps.broadcast_define( (('n',), ('n',)),
                               (),
                               out_kwarg = "out" )
        def inner_product(a, b, *, out, dtype):
            .....
            out.setfield(a.dot(b), out.dtype)

        out = inner_product( np.arange(3), np.arange(2*4*3).reshape(2,4,3), dtype=int)

    Note that the caller didn't need to specify the prototype of the output or
    the extra broadcasting dimensions (output prototype is in the
    broadcast_define() call, but not the inner_product() call). Specifying the
    dtype here is optional: it defaults to float if omitted. If the dtype IS
    given, the inner function must take a "dtype" argument; to use in cases
    where out_kwarg isn't given, and the output array must be created by the
    inner function.

    If we want the
    output array to be pre-allocated, the output prototype (it is () in this
    example) is required: we must know the shape of the output array in order to
    create it.

    Without a declared output prototype, we can still make mostly- efficient
    calls: the broadcasting mechanism can call the inner function for the first
    slice as we showed earlier, by creating a new array for the slice. This new
    array required an extra allocation and copy, but it contains the required
    shape information. This infomation will be used to allocate the output, and
    the subsequent calls to the inner function will be efficient:

        @nps.broadcast_define( (('n',), ('n',)),
                               out_kwarg = "out" )
        def inner_product(a, b, *, out=None):
            .....
            if out is None:
                return a.dot(b)
            out.setfield(a.dot(b), out.dtype)
            return out

        out = inner_product( np.arange(3), np.arange(2*4*3).reshape(2,4,3))

    Here we were slighly inefficient, but the ONLY required extra specification
    was out_kwarg: that's all you need. Also it is important to note that in
    this case the inner function is called both with passing it an output array
    to fill in, and with asking it to create a new one (by passing out=None to
    the inner function). This inner function then must support both modes of
    operation. If the inner function does not support filling in an output
    array, none of these efficiency improvements are possible.

    It is possible for a function to return more than one output, and this is
    supported by broadcast_define(). This case works exactly like the one-output
    case, except the output prototype is REQUIRED, and this output prototype
    contains multiple tuples, one for each output. The inner function must
    return the outputs in a tuple, and each individual output will be
    broadcasted as expected.

    broadcast_define() is analogous to thread_define() in PDL.

    """

    def inner_decorator_for_some_reason(func):
        # args broadcast, kwargs do not. All auxillary data should go into the
        # kwargs
        def broadcast_loop(*args, **kwargs):
            if len(args) < len(prototype):
                raise ValueError(
                    "Mismatched number of input arguments. Wanted at least {} but got {}".format(
                        len(prototype), len(args)
                    )
                )

            args_passthru = args[len(prototype) :]
            args = args[0 : len(prototype)]

            # make sure all the arguments are numpy arrays
            args = tuple(np.asarray(arg) for arg in args)

            # dims_extra: extra dimensions to broadcast through
            # dims_named: values of the named dimensions
            dims_extra, dims_named = _eval_broadcast_dims(args, prototype)

            # If None, the single output is either returned, or stored into
            # out_kwarg. If an integer, then a tuple is returned (or stored into
            # out_kwarg). If Noutputs==1 then we return a TUPLE of length 1
            Noutputs = None

            # substitute named variable values into the output prototype
            prototype_output_expanded = None
            if prototype_output is not None:
                # If a single prototype_output is given, wrap it in a tuple to indicate
                # that we only have one output
                if all(type(o) is int or type(o) is str for o in prototype_output):
                    prototype_output_expanded = [
                        d if type(d) is int else dims_named[d] for d in prototype_output
                    ]
                else:
                    Noutputs = len(prototype_output)
                    prototype_output_expanded = [
                        [
                            d if type(d) is int else dims_named[d]
                            for d in _prototype_output
                        ]
                        for _prototype_output in prototype_output
                    ]

            # I checked all the dimensions and aligned everything. I have my
            # to-broadcast dimension counts. Iterate through all the broadcasting
            # output, and gather the results
            output = None
            i_slice = 0

            if Noutputs is None:
                # We expect a SINGLE output

                # if the output was supposed to go to a particular place, set that
                if out_kwarg is not None and out_kwarg in kwargs:
                    output = kwargs[out_kwarg]
                    if prototype_output_expanded is not None:
                        expected_shape = dims_extra + prototype_output_expanded
                        if output.shape != tuple(expected_shape):
                            raise ValueError(
                                "Inconsistent output shape: expected {}, but got {}".format(
                                    expected_shape, output.shape
                                )
                            )
                # if we know enough to allocate the output, do that
                elif prototype_output_expanded is not None:
                    kwargs_dtype = {}
                    if "dtype" in kwargs:
                        kwargs_dtype["dtype"] = kwargs["dtype"]
                    output = np.empty(
                        dims_extra + prototype_output_expanded, **kwargs_dtype
                    )  # type: ignore

                # if no broadcasting involved, just call the function
                if not dims_extra:
                    # if the function knows how to write directly to an array,
                    # request that
                    if output is not None and out_kwarg is not None:
                        kwargs[out_kwarg] = output

                    sliced_args = args + args_passthru
                    result = func(*sliced_args, **kwargs)

                    if out_kwarg is not None and kwargs.get(out_kwarg) is not None:
                        # We wrote the output in-place. Return the output array
                        return kwargs.get(out_kwarg)

                    # Using the returned output. Run some checks, and return the
                    # returned value
                    if isinstance(result, tuple):
                        raise ValueError(
                            "Only a single output expected, but a tuple was returned!"
                        )
                    if prototype_output_expanded is not None and np.array(
                        result
                    ).shape != tuple(prototype_output_expanded):
                        raise ValueError(
                            "Inconsistent slice output shape: expected {}, but got {}".format(
                                prototype_output_expanded, np.array(result).shape
                            )
                        )
                    return result

                # reshaped output. I write to this array
                if output is not None:
                    output_flattened = clump(output, n=len(dims_extra))

                for x in _broadcast_iter_dim(args, prototype, dims_extra):
                    # if the function knows how to write directly to an array,
                    # request that
                    if output is not None and out_kwarg is not None:
                        kwargs[out_kwarg] = output_flattened[i_slice, ...]

                    sliced_args = x + args_passthru
                    result = func(*sliced_args, **kwargs)

                    if output is None or out_kwarg is None:
                        # We weren't writing directly into the output, so check
                        # the output for validity
                        if isinstance(result, tuple):
                            raise ValueError(
                                "Only a single output expected, but a tuple was returned!"
                            )
                        if not isinstance(result, np.ndarray):
                            result = np.array(result)

                        if prototype_output_expanded is None:
                            prototype_output_expanded = result.shape
                        else:
                            if result.shape != tuple(prototype_output_expanded):
                                raise ValueError(
                                    "Inconsistent slice output shape: expected {}, but got {}".format(
                                        prototype_output_expanded, result.shape
                                    )
                                )

                    if output is None:
                        # I didn't already have an output array because I didn't
                        # know how large it should be. But I now have the first
                        # slice, and I know how big the whole output should be.
                        # I create it
                        output = np.empty(
                            dims_extra + list(result.shape), dtype=result.dtype
                        )
                        output_flattened = output.reshape(
                            (_product(dims_extra),) + result.shape
                        )
                        output_flattened[i_slice, ...] = result

                    elif out_kwarg is None:
                        output_flattened[i_slice, ...] = result
                    # else:
                    #   I was writing directly to the output, so I don't need to
                    #   manually populate the slice

                    i_slice = i_slice + 1

            else:
                # We expect MULTIPLE outputs: a tuple of length Noutputs

                # if the output was supposed to go to a particular place, set that
                if out_kwarg is not None and out_kwarg in kwargs:
                    output = kwargs[out_kwarg]
                    if prototype_output_expanded is not None:
                        for i in range(Noutputs):
                            expected_shape = dims_extra + prototype_output_expanded[i]
                            if output[i].shape != tuple(expected_shape):
                                raise ValueError(
                                    "Inconsistent output shape for output {}: expected {}, but got {}".format(
                                        i, expected_shape, output[i].shape
                                    )
                                )
                # if we know enough to allocate the output, do that
                elif prototype_output_expanded is not None:
                    kwargs_dtype = {}
                    if "dtype" in kwargs:
                        kwargs_dtype["dtype"] = kwargs["dtype"]
                    output = [
                        np.empty(
                            dims_extra + prototype_output_expanded[i], **kwargs_dtype
                        )
                        for i in range(Noutputs)
                    ]
                # else:
                #   We don't have an output and we don't know its dimensions, so
                #   we can't allocate an array for it. Leave output as None. I
                #   will allocate it as soon I get the first slice; this will let
                #   me know how large the whole thing should be

                # if no broadcasting involved, just call the function
                if not dims_extra:
                    # if the function knows how to write directly to an array,
                    # request that
                    if output is not None and out_kwarg is not None:
                        kwargs[out_kwarg] = tuple(output)

                    sliced_args = args + args_passthru
                    result = func(*sliced_args, **kwargs)
                    if out_kwarg is not None and kwargs.get(out_kwarg) is not None:
                        # We wrote the output in-place. Return the output array
                        return kwargs.get(out_kwarg)

                    if not isinstance(result, tuple):
                        raise ValueError(
                            "A tuple of {} outputs is expected, but an object of type {} was returned".format(
                                Noutputs, type(result)
                            )
                        )
                    if len(result) != Noutputs:
                        raise ValueError(
                            "A tuple of {} outputs is expected, but a length-{} tuple was returned".format(
                                Noutputs, len(result)
                            )
                        )
                    if prototype_output_expanded is not None:
                        for i in range(Noutputs):
                            if np.array(result[i]).shape != tuple(
                                prototype_output_expanded[i]
                            ):
                                raise ValueError(
                                    "Inconsistent output {} shape: expected {}, but got {}".format(
                                        i,
                                        prototype_output_expanded[i],
                                        np.array(result[i]).shape,
                                    )
                                )
                    return result

                # reshaped output. I write to this array
                if output is not None:
                    output_flattened = [
                        clump(output[i], n=len(dims_extra)) for i in range(Noutputs)
                    ]

                for x in _broadcast_iter_dim(args, prototype, dims_extra):
                    # if the function knows how to write directly to an array,
                    # request that
                    if output is not None and out_kwarg is not None:
                        kwargs[out_kwarg] = tuple(
                            o[i_slice, ...] for o in output_flattened
                        )

                    sliced_args = x + args_passthru
                    result = func(*sliced_args, **kwargs)

                    if output is None or out_kwarg is None:
                        # We weren't writing directly into the output, so check
                        # the output for validity
                        if not isinstance(result, tuple):
                            raise ValueError(
                                "A tuple of {} outputs is expected, but an object of type {} was returned".format(
                                    Noutputs, type(result)
                                )
                            )
                        if len(result) != Noutputs:
                            raise ValueError(
                                "A tuple of {} outputs is expected, but a length-{} tuple was returned".format(
                                    Noutputs, len(result)
                                )
                            )

                        result = [
                            x if isinstance(x, np.ndarray) else np.array(x)
                            for x in result
                        ]

                        if prototype_output_expanded is None:
                            prototype_output_expanded = [
                                result[i].shape for i in range(Noutputs)
                            ]
                        else:
                            for i in range(Noutputs):
                                if result[i].shape != tuple(
                                    prototype_output_expanded[i]
                                ):
                                    raise ValueError(
                                        "Inconsistent slice output {} shape: expected {}, but got {}".format(
                                            i,
                                            prototype_output_expanded[i],
                                            result[i].shape,
                                        )
                                    )

                    if output is None:
                        # I didn't already have an output array because I didn't
                        # know how large it should be. But I now have the first
                        # slice, and I know how big the whole output should be.
                        # I create it
                        output = [
                            np.empty(
                                dims_extra + list(result[i].shape),
                                dtype=result[i].dtype,
                            )
                            for i in range(Noutputs)
                        ]
                        output_flattened = [
                            output[i].reshape((_product(dims_extra),) + result[i].shape)
                            for i in range(Noutputs)
                        ]
                        for i in range(Noutputs):
                            output_flattened[i][i_slice, ...] = result[i]

                    elif out_kwarg is None:
                        for i in range(Noutputs):
                            output_flattened[i][i_slice, ...] = result[i]
                    # else:
                    #   I was writing directly to the output, so I don't need to
                    #   manually populate the slice

                    i_slice = i_slice + 1

            return output

        if out_kwarg is not None and not isinstance(out_kwarg, str):
            raise ValueError("out_kwarg must be a string")

        # Make sure all dimensions are >=0 and that named output dimensions are
        # known from the input
        known_named_dims = set()
        if not isinstance(prototype, tuple):
            raise ValueError("Input prototype must be given as a tuple")
        for dims_input in prototype:
            if not isinstance(dims_input, tuple):
                raise ValueError("Input prototype dims must be given as a tuple")
            for dim in dims_input:
                if type(dim) is not int:
                    if type(dim) is not str:
                        raise ValueError(
                            "Prototype dimensions must be integers > 0 or strings. Got '{}' of type '{}'".format(
                                dim, type(dim)
                            )
                        )
                    known_named_dims.add(dim)
                else:
                    if dim < 0:
                        raise ValueError(
                            "Prototype dimensions must be > 0. Got '{}'".format(dim)
                        )

        if prototype_output is not None:
            if not isinstance(prototype_output, tuple):
                raise ValueError("Output prototype dims must be given as a tuple")

            # If a single prototype_output is given, wrap it in a tuple to indicate
            # that we only have one output
            if all(type(o) is int or type(o) is str for o in prototype_output):
                prototype_outputs = (prototype_output,)
            else:
                prototype_outputs = prototype_output
                if not all(isinstance(p, tuple) for p in prototype_outputs):
                    raise ValueError(
                        "Output dimensions must be integers > 0 or strings. Each output must be a tuple. Some given output aren't tuples: {}".format(
                            prototype_outputs
                        )
                    )

            for dims_output in prototype_outputs:
                for dim in dims_output:
                    if type(dim) is not int:
                        if type(dim) is not str:
                            raise ValueError(
                                "Output dimensions must be integers > 0 or strings. Got '{}' of type '{}'".format(
                                    dim, type(dim)
                                )
                            )
                        if dim not in known_named_dims:
                            raise ValueError(
                                "Output prototype has named dimension '{}' not seen in the input prototypes".format(
                                    dim
                                )
                            )
                    else:
                        if dim < 0:
                            raise ValueError(
                                "Prototype dimensions must be > 0. Got '{}'".format(dim)
                            )

        func_out = _clone_function(broadcast_loop, func.__name__)
        func_out.__doc__ = inspect.getdoc(func)
        if func_out.__doc__ is None:
            func_out.__doc__ = ""
        func_out.__doc__ += """\n\nThis function is broadcast-aware through numpysane.broadcast_define().
The expected inputs have input prototype:

    {prototype}

{output_prototype_text}

The first {nargs} positional arguments will broadcast. The trailing shape of
those arguments must match the input prototype; the leading shape must follow
the standard broadcasting rules. Positional arguments past the first {nargs} and
all the keyword arguments are passed through untouched.""".format(
            prototype=prototype,
            output_prototype_text="No output prototype is defined."
            if prototype_output is None
            else "and output prototype\n\n    {}".format(prototype_output),
            nargs=len(prototype),
        )
        return func_out

    return inner_decorator_for_some_reason


def _eval_broadcast_dims(args, prototype):
    r"""Helper function to evaluate a given list of arguments in respect to a given
    broadcasting prototype. This function will flag any errors in the
    dimensionality of the inputs. If no errors are detected, it returns

      dims_extra,dims_named

    where

      dims_extra is the outer shape of the broadcast
        This is a list: the union of all the leading shapes of all the
        arguments, after the trailing shapes of the prototype have been stripped

      dims_named is the sizes of the named dimensions
        This is a dict mapping dimension names to their sizes

    """

    # First I initialize dims_extra: the array containing the broadcasted
    # slices. Each argument calls for some number of extra dimensions, and the
    # overall array is as large as the biggest one of those
    Ndims_extra = 0
    for i_arg in range(len(args)):
        Ndims_extra_here = len(args[i_arg].shape) - len(prototype[i_arg])
        if Ndims_extra_here > Ndims_extra:
            Ndims_extra = Ndims_extra_here
    dims_extra = [1] * Ndims_extra

    def parse_dim(name_arg, shape_prototype, shape_arg, dims_named):
        def range_rev(n):
            r"""Returns a range from -1 to -n.

            Useful to index variable-sized lists while aligning their ends."""
            return range(-1, -n - 1, -1)

        # first, I make sure the input is at least as dimension-ful as the
        # prototype. I do this by prepending dummy dimensions of length-1 as
        # necessary
        if len(shape_prototype) > len(shape_arg):
            ndims_missing_here = len(shape_prototype) - len(shape_arg)
            shape_arg = (1,) * ndims_missing_here + shape_arg

        # MAKE SURE THE PROTOTYPE DIMENSIONS MATCH (the trailing dimensions)
        #
        # Loop through the dimensions. Set the dimensionality of any new named
        # argument to whatever the current argument has. Any already-known
        # argument must match
        for i_dim in range_rev(len(shape_prototype)):
            dim_prototype = shape_prototype[i_dim]

            if not isinstance(dim_prototype, int):
                # This is a named dimension. These can have any value, but ALL
                # dimensions of the same name must thave the SAME value
                # EVERYWHERE
                if dim_prototype not in dims_named:
                    dims_named[dim_prototype] = shape_arg[i_dim]
                dim_prototype = dims_named[dim_prototype]

            # The prototype dimension (named or otherwise) now has a numeric
            # value. Make sure it matches what I have
            if dim_prototype != shape_arg[i_dim]:
                raise ValueError(
                    "Argument {} dimension '{}': expected {} but got {}".format(
                        name_arg,
                        shape_prototype[i_dim],
                        dim_prototype,
                        shape_arg[i_dim],
                    )
                )

        # I now know that this argument matches the prototype. I look at the
        # extra dimensions to broadcast, and make sure they match with the
        # dimensions I saw previously
        Ndims_extra_here = len(shape_arg) - len(shape_prototype)

        # MAKE SURE THE BROADCASTED DIMENSIONS MATCH (the leading dimensions)
        #
        # This argument has Ndims_extra_here dimensions to broadcast. The
        # current shape to broadcast must be at least as large, and must match
        for i_dim in range_rev(Ndims_extra_here):
            dim_arg = shape_arg[i_dim - len(shape_prototype)]
            if dim_arg != 1:
                if dims_extra[i_dim] == 1:
                    dims_extra[i_dim] = dim_arg
                elif dims_extra[i_dim] != dim_arg:
                    raise ValueError(
                        "Argument {} prototype {} extra broadcast dim {} mismatch: previous arg set this to {}, but this arg wants {}".format(
                            name_arg, shape_prototype, i_dim, dims_extra[i_dim], dim_arg
                        )
                    )

    dims_named = {}  # parse_dim() adds to this
    for i_arg in range(len(args)):
        parse_dim(i_arg, prototype[i_arg], args[i_arg].shape, dims_named)

    return dims_extra, dims_named


def _broadcast_iter_dim(args, prototype, dims_extra):
    r"""Generator to iterate through all the broadcasting slices."""

    # pad the dimension of each arg with ones. This lets me use the full
    # dims_extra index on each argument, without worrying about overflow
    args = [
        atleast_dims(args[i], -(len(prototype[i]) + len(dims_extra)))
        for i in range(len(args))
    ]

    # per-arg dims_extra indexing varies: len-1 dimensions always index at 0. I
    # make a mask that I apply each time
    idx_slice_mask = np.ones((len(args), len(dims_extra)), dtype=int)
    for i in range(len(args)):
        idx_slice_mask[
            i, np.array(args[i].shape, dtype=int)[: len(dims_extra)] == 1
        ] = 0

    for idx_slice in itertools.product(*(range(x) for x in dims_extra)):
        # tuple(idx) because of wonky behavior differences:
        #     >>> a
        #     array([[0, 1, 2],
        #            [3, 4, 5]])
        #
        #     >>> a[tuple((1,1))]
        #     4
        #
        #     >>> a[list((1,1))]
        #     array([[3, 4, 5],
        #            [3, 4, 5]])
        yield tuple(
            args[i][tuple(idx_slice * idx_slice_mask[i])] for i in range(len(args))
        )


def _clone_function(f, name):
    r"""Returns a clone of a given function.

    This is useful to copy a function, updating its metadata, such as the
    documentation, name, etc. There are also differences here between python 2
    and python 3 that this function handles.

    """

    def get(f, what):
        what2 = "func_{}".format(what)
        what3 = "__{}__".format(what)
        try:
            return getattr(f, what2)
        except:
            try:
                return getattr(f, what3)
            except:
                pass
        return None

    return types.FunctionType(
        get(f, "code"), get(f, "globals"), name, get(f, "defaults"), get(f, "closure")
    )


def broadcast_generate(prototype, args):
    r"""A generator that produces broadcasted slices

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> for s in nps.broadcast_generate( (('n',), ('n',)), (a,b)):
        ...     print "slice: {}".format(s)
        slice: (array([0, 1, 2]), array([100, 101, 102]))
        slice: (array([3, 4, 5]), array([103, 104, 105]))

    The broadcasting operation of numpysane is described in detail in the
    numpysane.broadcast_define() docstring and in the main README of numpysane.
    This function can be used as a Python generator to produce each broadcasted
    slice one by one

    Since Python generators are inherently 1-dimensional, this function
    effectively flattens the broadcasted results. If the correct output shape
    needs to be reconstituted, the leading shape is available by calling
    numpysane.broadcast_extra_dims() with the same arguments as this function.

    """

    if len(args) != len(prototype):
        raise ValueError(
            "Mismatched number of input arguments. Wanted {} but got {}".format(
                len(prototype), len(args)
            )
        )

    # make sure all the arguments are numpy arrays
    args = tuple(np.asarray(arg) for arg in args)

    # dims_extra: extra dimensions to broadcast through
    # dims_named: values of the named dimensions
    dims_extra, dims_named = _eval_broadcast_dims(args, prototype)

    # I checked all the dimensions and aligned everything. I have my
    # to-broadcast dimension counts. Iterate through all the broadcasting
    # output, and gather the results
    for x in _broadcast_iter_dim(args, prototype, dims_extra):
        yield x


def broadcast_extra_dims(prototype, args):
    r"""Report the extra leading dimensions a broadcasted call would produce

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6). reshape(  2,3)
        >>> b = np.arange(15).reshape(5,1,3)

        >>> print(nps.broadcast_extra_dims((('n',), ('n',)),
                                           (a,b)))
        [5,2]

    The broadcasting operation of numpysane is described in detail in the
    numpysane.broadcast_define() docstring and in the main README of numpysane.
    This function applies the broadcasting rules to report the leading
    dimensions of a broadcasted result if a broadcasted function was called with
    the given arguments. This is most useful to reconstitute the desired shape
    from flattened output produced by numpysane.broadcast_generate()

    """

    if len(args) != len(prototype):
        raise ValueError(
            "Mismatched number of input arguments. Wanted {} but got {}".format(
                len(prototype), len(args)
            )
        )

    # make sure all the arguments are numpy arrays
    args = tuple(np.asarray(arg) for arg in args)

    # dims_extra: extra dimensions to broadcast through
    # dims_named: values of the named dimensions
    dims_extra, dims_named = _eval_broadcast_dims(args, prototype)

    return dims_extra


@broadcast_define((("n",), ("m",)), prototype_output=("n", "m"), out_kwarg="out")
def outer(a, b, out=None):
    r"""Outer product of two 1-dimensional n-long vectors.

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(3)
        >>> b = a+5
        >>> a
        array([0, 1, 2])

        >>> b
        array([5, 6, 7])

        >>> nps.outer(a,b)
        array([[ 0,  0,  0],
               [ 5,  6,  7],
               [10, 12, 14]])
    """
    if out is None:
        return np.outer(a, b)

    out.setfield(np.outer(a, b), out.dtype)
    return out


@broadcast_define(
    (
        (
            "n",
            "n",
        ),
    ),
    prototype_output=(),
)
def trace(a):
    r"""Broadcast-aware trace

    SYNOPSIS

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(3*4*4).reshape(3,4,4)
        >>> a
        array([[[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11],
                [12, 13, 14, 15]],

               [[16, 17, 18, 19],
                [20, 21, 22, 23],
                [24, 25, 26, 27],
                [28, 29, 30, 31]],

               [[32, 33, 34, 35],
                [36, 37, 38, 39],
                [40, 41, 42, 43],
                [44, 45, 46, 47]]])

        >>> nps.trace(a)
        array([ 30,  94, 158])
    """
    return np.trace(a)
