// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>
#include <structmember.h>
// Required for numpy 2. They now #include complex.h, so I is #defined to be the
// complex I, which conflicts with my usage here
#undef I

#include <signal.h>
#include "dogleg.h"

#if (CHOLMOD_VERSION > (CHOLMOD_VER_CODE(2, 2))) && \
    (CHOLMOD_VERSION < (CHOLMOD_VER_CODE(4, 0)))
#include <cholmod_function.h>
#endif

#include "mrcal-internal.h"
#include "mrcal.h"
#include "stereo.h"

#include "python-wrapping-utilities.h"

// adds a reference to P,I,X, unless an error is reported
static PyObject* csr_from_cholmod_sparse(
    PyObject* P,
    PyObject* I,
    PyObject* X
) {
    // I do the Python equivalent of this;
    // scipy.sparse.csr_matrix((data, indices, indptr))

    PyObject* result = NULL;

    PyObject* module = NULL;
    PyObject* method = NULL;
    PyObject* args = NULL;
    if (NULL == (module = PyImport_ImportModule("scipy.sparse"))) {
        BARF("Couldn't import scipy.sparse. I need that to represent J");
        goto done;
    }
    if (NULL == (method = PyObject_GetAttrString(module, "csr_matrix"))) {
        BARF("Couldn't find 'csr_matrix' in scipy.sparse");
        goto done;
    }

    // Here I'm assuming specific types in my cholmod arrays. I tried to
    // _Static_assert it, but internally cholmod uses void*, so I can't do that
    PyObject* MatrixDef = PyTuple_Pack(3, X, I, P);
    args = PyTuple_Pack(1, MatrixDef);
    Py_DECREF(MatrixDef);

    if (NULL == (result = PyObject_CallObject(method, args))) {
        goto done;  // reuse already-set error
    }

    // Testing code to dump out a dense representation of this matrix to a file.
    // Can compare that file to what this function returns like this:
    //   Jf = np.fromfile("/tmp/J_17014_444.dat").reshape(17014,444)
    //   np.linalg.norm( Jf - J.toarray() )
    // {
    // #define P(A, index) ((unsigned int*)((A)->p))[index]
    // #define I(A, index) ((unsigned int*)((A)->i))[index]
    // #define X(A, index) ((double*      )((A)->x))[index]
    //         char logfilename[128];
    //         sprintf(logfilename,
    //         "/tmp/J_%d_%d.dat",(int)Jt->ncol,(int)Jt->nrow); FILE* fp =
    //         fopen(logfilename, "w"); double* Jrow; Jrow =
    //         malloc(Jt->nrow*sizeof(double)); for(unsigned int icol=0;
    //         icol<Jt->ncol; icol++)
    //         {
    //             memset(Jrow, 0, Jt->nrow*sizeof(double));
    //             for(unsigned int i=P(Jt, icol); i<P(Jt, icol+1); i++)
    //             {
    //                 int irow = I(Jt,i);
    //                 double x = X(Jt,i);
    //                 Jrow[irow] = x;
    //             }
    //             fwrite(Jrow,sizeof(double),Jt->nrow,fp);
    //         }
    //         fclose(fp);
    //         free(Jrow);
    // #undef P
    // #undef I
    // #undef X
    // }

done:
    Py_XDECREF(module);
    Py_XDECREF(method);
    Py_XDECREF(args);

    return result;
}

// A container for a CHOLMOD factorization
typedef struct {
    PyObject_HEAD

        // if(inited_common), the "common" has been initialized
        // if(factorization), the factorization has been initialized
        //
        // So to use the object we need inited_common && factorization
        bool inited_common;
    cholmod_common common;
    cholmod_factor* factorization;

    // initialized the first time cholmod_solve2() is called
    cholmod_dense* Y;
    cholmod_dense* E;

    // optimizer_callback should return it
    // and I should have two solve methods:
} CHOLMOD_factorization;

// stolen from libdogleg
static int cholmod_error_callback(
    const char* s,
    ...
) {
    va_list ap;
    va_start(ap, s);
    int ret = vfprintf(stderr, s, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    return ret;
}

// for my internal C usage
static void _CHOLMOD_factorization_release_internal(
    CHOLMOD_factorization* self
) {
    if (self->E != NULL) {
        cholmod_free_dense(&self->E, &self->common);
        self->E = NULL;
    }
    if (self->Y != NULL) {
        cholmod_free_dense(&self->Y, &self->common);
        self->Y = NULL;
    }

    if (self->factorization) {
        cholmod_free_factor(&self->factorization, &self->common);
        self->factorization = NULL;
    }
    if (self->inited_common) {
        cholmod_finish(&self->common);
    }
    self->inited_common = false;
}

// for my internal C usage
static bool _CHOLMOD_factorization_init_from_cholmod_sparse(
    CHOLMOD_factorization* self,
    cholmod_sparse* Jt
) {
    if (!self->inited_common) {
        if (!cholmod_start(&self->common)) {
            BARF("Error trying to cholmod_start");
            return false;
        }
        self->inited_common = true;

        // stolen from libdogleg

        // I want to use LGPL parts of CHOLMOD only, so I turn off the
        // supernodal routines. This gave me a 25% performance hit in the solver
        // for a particular set of optical calibration data.
        self->common.supernodal = 0;

        // I want all output to go to STDERR, not STDOUT
#if (CHOLMOD_VERSION <= (CHOLMOD_VER_CODE(2, 2)))
        self->common.print_function = cholmod_error_callback;
#elif (CHOLMOD_VERSION < (CHOLMOD_VER_CODE(4, 0)))
        CHOLMOD_FUNCTION_DEFAULTS;
        CHOLMOD_FUNCTION_PRINTF(&self->common) = cholmod_error_callback;
#else
        SuiteSparse_config_printf_func_set(cholmod_error_callback);
#endif
    }

    self->factorization = cholmod_analyze(Jt, &self->common);

    if (self->factorization == NULL) {
        BARF("cholmod_analyze() failed");
        return false;
    }
    if (!cholmod_factorize(Jt, self->factorization, &self->common)) {
        BARF("cholmod_factorize() failed");
        return false;
    }
    if (self->factorization->minor != self->factorization->n) {
        BARF("Got singular JtJ!");
        return false;
    }
    return true;
}

static int CHOLMOD_factorization_init(
    CHOLMOD_factorization* self,
    PyObject* args,
    PyObject* kwargs
) {
    // Any existing factorization goes away. If this function fails, we lose the
    // existing factorization, which is fine. I'm placing this on top so that
    // __init__() will get rid of the old state
    _CHOLMOD_factorization_release_internal(self);

    // error by default
    int result = -1;

    char* keywords[] = {"J", NULL};
    PyObject* Py_J = NULL;
    PyObject* module = NULL;
    PyObject* csr_matrix_type = NULL;

    PyObject* Py_shape = NULL;
    PyObject* Py_nnz = NULL;
    PyObject* Py_data = NULL;
    PyObject* Py_indices = NULL;
    PyObject* Py_indptr = NULL;
    PyObject* Py_has_sorted_indices = NULL;

    PyObject* Py_h = NULL;
    PyObject* Py_w = NULL;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|O:CHOLMOD_factorization.__init__",
            keywords,
            &Py_J
        )) {
        goto done;
    }

    if (Py_J == NULL) {
        // Success. Nothing to do
        result = 0;
        goto done;
    }

    if (NULL == (module = PyImport_ImportModule("scipy.sparse"))) {
        BARF("Couldn't import scipy.sparse. I need that to represent J");
        goto done;
    }
    if (NULL ==
        (csr_matrix_type = PyObject_GetAttrString(module, "csr_matrix"))) {
        BARF("Couldn't find 'csr_matrix' in scipy.sparse");
        goto done;
    }
    if (!PyObject_IsInstance(Py_J, csr_matrix_type)) {
        BARF("Argument J is must have type scipy.sparse.csr_matrix");
        goto done;
    }

#define GETATTR(x)                                             \
    if (NULL == (Py_##x = PyObject_GetAttrString(Py_J, #x))) { \
        BARF("Couldn't get J." #x);                            \
        goto done;                                             \
    }

    GETATTR(shape);
    GETATTR(nnz);
    GETATTR(data);
    GETATTR(indices);
    GETATTR(indptr);
    GETATTR(has_sorted_indices);

    if (!PySequence_Check(Py_shape)) {
        BARF("J.shape should be an iterable");
        goto done;
    }
    int lenshape = PySequence_Length(Py_shape);
    if (lenshape != 2) {
        if (lenshape < 0) {
            BARF("Failed to get len(J.shape)");
        } else {
            BARF(
                "len(J.shape) should be exactly 2, but instead got %d",
                lenshape
            );
        }
        goto done;
    }

    Py_h = PySequence_GetItem(Py_shape, 0);
    if (Py_h == NULL) {
        BARF("Error getting J.shape[0]");
        goto done;
    }
    Py_w = PySequence_GetItem(Py_shape, 1);
    if (Py_w == NULL) {
        BARF("Error getting J.shape[1]");
        goto done;
    }

    long nnz;
    if (PyLong_Check(Py_nnz)) {
        nnz = PyLong_AsLong(Py_nnz);
    } else {
        BARF("Error interpreting nnz as an integer");
        goto done;
    }

    long h;
    if (PyLong_Check(Py_h)) {
        h = PyLong_AsLong(Py_h);
    } else {
        BARF("Error interpreting J.shape[0] as an integer");
        goto done;
    }
    long w;
    if (PyLong_Check(Py_w)) {
        w = PyLong_AsLong(Py_w);
    } else {
        BARF("Error interpreting J.shape[1] as an integer");
        goto done;
    }

#define CHECK_NUMPY_ARRAY(x, dtype)                                            \
    if (!PyArray_Check((PyArrayObject*)Py_##x)) {                              \
        BARF("J." #x " must be a numpy array");                                \
        goto done;                                                             \
    }                                                                          \
    if (1 != PyArray_NDIM((PyArrayObject*)Py_##x)) {                           \
        BARF(                                                                  \
            "J." #x                                                            \
            " must be a 1-dimensional numpy array. Instead got %d dimensions", \
            PyArray_NDIM((PyArrayObject*)Py_##x)                               \
        );                                                                     \
        goto done;                                                             \
    }                                                                          \
    if (PyArray_TYPE((PyArrayObject*)Py_##x) != dtype) {                       \
        BARF("J." #x " must have dtype: " #dtype);                             \
        goto done;                                                             \
    }                                                                          \
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)Py_##x)) {                    \
        BARF("J." #x " must live in contiguous memory");                       \
        goto done;                                                             \
    }

    CHECK_NUMPY_ARRAY(data, NPY_FLOAT64);
    CHECK_NUMPY_ARRAY(indices, NPY_INT32);
    CHECK_NUMPY_ARRAY(indptr, NPY_INT32);

    // OK, the input looks good. I guess I can tell CHOLMOD about it

    // My convention is to store row-major matrices, but CHOLMOD stores
    // col-major matrices. So I keep the same data, but tell CHOLMOD that I'm
    // storing Jt and not J
    cholmod_sparse Jt = {
        .nrow = w,
        .ncol = h,
        .nzmax = nnz,
        .p = PyArray_DATA((PyArrayObject*)Py_indptr),
        .i = PyArray_DATA((PyArrayObject*)Py_indices),
        .x = PyArray_DATA((PyArrayObject*)Py_data),
        .stype = 0,  // not symmetric
        .itype = CHOLMOD_INT,
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE,
        .sorted = PyObject_IsTrue(Py_has_sorted_indices),
        .packed = 1
    };

    if (!_CHOLMOD_factorization_init_from_cholmod_sparse(self, &Jt)) {
        goto done;
    }

    result = 0;

done:
    if (result != 0) {
        _CHOLMOD_factorization_release_internal(self);
    }

    Py_XDECREF(module);
    Py_XDECREF(csr_matrix_type);
    Py_XDECREF(Py_shape);
    Py_XDECREF(Py_nnz);
    Py_XDECREF(Py_data);
    Py_XDECREF(Py_indices);
    Py_XDECREF(Py_indptr);
    Py_XDECREF(Py_has_sorted_indices);
    Py_XDECREF(Py_h);
    Py_XDECREF(Py_w);

    return result;

#undef GETATTR
#undef CHECK_NUMPY_ARRAY
}

static void CHOLMOD_factorization_dealloc(
    CHOLMOD_factorization* self
) {
    _CHOLMOD_factorization_release_internal(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CHOLMOD_factorization_str(
    CHOLMOD_factorization* self
) {
    if (!(self->inited_common && self->factorization)) {
        return PyUnicode_FromString("No factorization given");
    }

    return PyUnicode_FromFormat(
        "Initialized with a valid factorization. N=%d",
        self->factorization->n
    );
}

static PyObject* CHOLMOD_factorization_solve_xt_JtJ_bt(
    CHOLMOD_factorization* self,
    PyObject* args,
    PyObject* kwargs
) {
    cholmod_dense* M = NULL;

    // error by default
    PyObject* result = NULL;
    PyObject* Py_out = NULL;

    char* keywords[] = {"bt", "sys", NULL};
    PyObject* Py_bt = NULL;
    char* sys = "A";

    if (!(self->inited_common && self->factorization)) {
        BARF("No factorization has been computed");
        goto done;
    }

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|s:CHOLMOD_factorization.solve_xt_JtJ_bt",
            keywords,
            &Py_bt,
            &sys
        )) {
        goto done;
    }

    if (Py_bt == NULL || !PyArray_Check((PyArrayObject*)Py_bt)) {
        BARF("bt must be a numpy array");
        goto done;
    }

    int ndim = PyArray_NDIM((PyArrayObject*)Py_bt);
    if (ndim < 1) {
        BARF(
            "bt must be at least a 1-dimensional numpy array. Instead got %d "
            "dimensions",
            ndim
        );
        goto done;
    }

#define LIST_SYS(_) \
    _(A)            \
    _(LDLt)         \
    _(LD)           \
    _(DLt)          \
    _(L)            \
    _(Lt)           \
    _(D)            \
    _(P)            \
    _(Pt)

#define SYS_CHECK(s)                                                  \
    else if (0 == strcmp(sys, #s) || 0 == strcmp(sys, "CHOLMOD_" #s)) \
        CHOLMOD_system = CHOLMOD_##s;

#define SYS_NAME(s) #s ","

    int CHOLMOD_system;
    if (0)
        ;
    LIST_SYS(SYS_CHECK)
    else {
        BARF(
            "Unknown sys '%s' given. Known values of sys: (" LIST_SYS(SYS_NAME
            ) ")",
            sys
        );
        goto done;
    }
#undef LIST_SYS
#undef SYS_CHECK
#undef SYS_NAME

    int Nstate = (int)PyArray_DIMS((PyArrayObject*)Py_bt)[ndim - 1];
    int Nrhs = (int)PyArray_SIZE((PyArrayObject*)Py_bt) / Nstate;

    if (self->factorization->n != (unsigned)Nstate) {
        BARF(
            "bt must be a 2-dimensional numpy array with %d cols (that's what "
            "the factorization has). Instead got %d cols",
            self->factorization->n,
            Nstate
        );
        goto done;
    }
    if (PyArray_TYPE((PyArrayObject*)Py_bt) != NPY_FLOAT64) {
        BARF("bt must have dtype=float");
        goto done;
    }
    if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)Py_bt)) {
        BARF("bt must live in contiguous memory");
        goto done;
    }

    // Alright. b looks good-enough to use
    if (0 == Nrhs) {
        // Degenerate input (0 columns). Just return it, and I'm done
        result = Py_bt;
        Py_INCREF(result);
        goto done;
    }

    cholmod_dense b = {
        .nrow = Nstate,
        .ncol = Nrhs,
        .nzmax = Nrhs * Nstate,
        .d = Nstate,
        .x = PyArray_DATA((PyArrayObject*)Py_bt),
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE
    };

    Py_out = PyArray_SimpleNew(
        ndim,
        PyArray_DIMS((PyArrayObject*)Py_bt),
        NPY_DOUBLE
    );
    if (Py_out == NULL) {
        BARF("Couldn't allocate Py_out");
        goto done;
    }

    cholmod_dense out = {
        .nrow = Nstate,
        .ncol = Nrhs,
        .nzmax = Nrhs * Nstate,
        .d = Nstate,
        .x = PyArray_DATA((PyArrayObject*)Py_out),
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE
    };

    M = &out;

    if (!cholmod_solve2(
            CHOLMOD_system,
            self->factorization,
            &b,
            NULL,
            &M,
            NULL,
            &self->Y,
            &self->E,
            &self->common
        )) {
        BARF("cholmod_solve2() failed");
        goto done;
    }
    if (M != &out) {
        BARF("cholmod_solve2() reallocated out! We leaked memory");
        goto done;
    }

    Py_INCREF(Py_out);
    result = Py_out;

done:
    Py_XDECREF(Py_out);

    return result;
}

static PyObject* CHOLMOD_factorization_rcond(
    CHOLMOD_factorization* self,
    PyObject* NPY_UNUSED(args)
) {
    if (!(self->inited_common && self->factorization)) {
        BARF("No factorization has been computed");
        return NULL;
    }

    return PyFloat_FromDouble(cholmod_rcond(self->factorization, &self->common)
    );
}

static const char CHOLMOD_factorization_docstring[] = R"(
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

)";

;
static const char CHOLMOD_factorization_solve_xt_JtJ_bt_docstring[] = R"(
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
)";

;
static const char CHOLMOD_factorization_rcond_docstring[] = R"(
Compute rough estimate of reciprocal of condition number

SYNOPSIS

    b, x, J, factorization = \
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


)";

static PyMethodDef CHOLMOD_factorization_methods[] = {
    PYMETHODDEF_ENTRY(
        CHOLMOD_factorization_,
        solve_xt_JtJ_bt,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(CHOLMOD_factorization_, rcond, METH_NOARGS),
    {}
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
// PyObject_HEAD_INIT throws
//   warning: missing braces around initializer []
// This isn't mine to fix, so I'm ignoring it
static PyTypeObject CHOLMOD_factorization_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "bindings.CHOLMOD_factorization",
    .tp_basicsize = sizeof(CHOLMOD_factorization),
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CHOLMOD_factorization_init,
    .tp_dealloc = (destructor)CHOLMOD_factorization_dealloc,
    .tp_methods = CHOLMOD_factorization_methods,
    .tp_str = (reprfunc)CHOLMOD_factorization_str,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = CHOLMOD_factorization_docstring,
};
#pragma GCC diagnostic pop

// For the C code. Create a new Python CHOLMOD_factorization object from a C
// cholmod_sparse structure
static PyObject* CHOLMOD_factorization_from_cholmod_sparse(
    cholmod_sparse* Jt
) {
    PyObject* self =
        PyObject_CallObject((PyObject*)&CHOLMOD_factorization_type, NULL);
    if (NULL == self) {
        return NULL;
    }

    if (!_CHOLMOD_factorization_init_from_cholmod_sparse(
            (CHOLMOD_factorization*)self,
            Jt
        )) {
        Py_DECREF(self);
        return NULL;
    }

    return self;
}

static bool parse_lensmodel_from_arg(
    // output
    mrcal_lensmodel_t* lensmodel,
    // input
    const char* lensmodel_cstring
) {
    mrcal_lensmodel_from_name(lensmodel, lensmodel_cstring);
    if (!mrcal_lensmodel_type_is_valid(lensmodel->type)) {
        switch (lensmodel->type) {
            case MRCAL_LENSMODEL_INVALID:
                // this should never (rarely?) happen
                BARF("Lens model '%s': error parsing", lensmodel_cstring);
                return false;
            case MRCAL_LENSMODEL_INVALID_BADCONFIG:
                BARF(
                    "Lens model '%s': error parsing the required configuration",
                    lensmodel_cstring
                );
                return false;
            case MRCAL_LENSMODEL_INVALID_MISSINGCONFIG:
                BARF(
                    "Lens model '%s': missing the required configuration",
                    lensmodel_cstring
                );
                return false;
            case MRCAL_LENSMODEL_INVALID_TYPE:
                BARF(
                    "Invalid lens model type was passed in: '%s'. Must be one "
                    "of " VALID_LENSMODELS_FORMAT,
                    lensmodel_cstring VALID_LENSMODELS_ARGLIST
                );
                return false;
            default:
                BARF(
                    "Lens model '%s' produced an unexpected error: "
                    "lensmodel->type=%d. This should never happen",
                    lensmodel_cstring,
                    (int)lensmodel->type
                );
                return false;
        }
        return false;
    }
    return true;
}

static PyObject* lensmodel_metadata_and_config(
    PyObject* NPY_UNUSED(self),
    PyObject* args
) {
    PyObject* result = NULL;

    char* lensmodel_string = NULL;
    if (!PyArg_ParseTuple(args, "s", &lensmodel_string)) {
        goto done;
    }
    mrcal_lensmodel_t lensmodel;
    if (!parse_lensmodel_from_arg(&lensmodel, lensmodel_string)) {
        goto done;
    }

    mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(&lensmodel);

#define MRCAL_ITEM_BUILDVALUE_DEF( \
    name,                          \
    type,                          \
    pybuildvaluecode,              \
    PRIcode,                       \
    SCNcode,                       \
    bitfield,                      \
    cookie                         \
)                                  \
    " s " pybuildvaluecode
#define MRCAL_ITEM_BUILDVALUE_VALUE( \
    name,                            \
    type,                            \
    pybuildvaluecode,                \
    PRIcode,                         \
    SCNcode,                         \
    bitfield,                        \
    cookie                           \
)                                    \
    , #name, cookie name

    if (lensmodel.type == MRCAL_LENSMODEL_CAHVORE) {
        result = Py_BuildValue(
            "{" MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                MRCAL_LENSMODEL_CAHVORE_CONFIG_LIST(
                    MRCAL_ITEM_BUILDVALUE_DEF,
                ) "}" MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, meta.)
                    MRCAL_LENSMODEL_CAHVORE_CONFIG_LIST(
                        MRCAL_ITEM_BUILDVALUE_VALUE,
                        lensmodel.LENSMODEL_CAHVORE__config.
                    )
        );
    } else if (lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC) {
        result = Py_BuildValue(
            "{" MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(
                    MRCAL_ITEM_BUILDVALUE_DEF,
                ) "}" MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, meta.)
                    MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(
                        MRCAL_ITEM_BUILDVALUE_VALUE,
                        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.
                    )
        );
    } else {
        result = Py_BuildValue("{" MRCAL_LENSMODEL_META_LIST(
            MRCAL_ITEM_BUILDVALUE_DEF,
        ) "}" MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, meta.));
    }

    Py_INCREF(result);

done:
    return result;
}

static PyObject* knots_for_splined_models(
    PyObject* NPY_UNUSED(self),
    PyObject* args
) {
    PyObject* result = NULL;
    PyArrayObject* py_ux = NULL;
    PyArrayObject* py_uy = NULL;

    char* lensmodel_string = NULL;
    if (!PyArg_ParseTuple(args, "s", &lensmodel_string)) {
        goto done;
    }
    mrcal_lensmodel_t lensmodel;
    if (!parse_lensmodel_from_arg(&lensmodel, lensmodel_string)) {
        goto done;
    }

    if (lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC) {
        BARF(
            "This function works only with the "
            "MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC model. %s passed in",
            lensmodel_string
        );
        goto done;
    }

    {
        double ux[lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx];
        double uy[lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny];
        if (!mrcal_knots_for_splined_models(ux, uy, &lensmodel)) {
            BARF("mrcal_knots_for_splined_models() failed");
            goto done;
        }

        npy_intp dims[1];

        dims[0] = lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx;
        py_ux = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (py_ux == NULL) {
            BARF("Couldn't allocate ux");
            goto done;
        }

        dims[0] = lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny;
        py_uy = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (py_uy == NULL) {
            BARF("Couldn't allocate uy");
            goto done;
        }

        memcpy(
            PyArray_DATA(py_ux),
            ux,
            lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx *
                sizeof(double)
        );
        memcpy(
            PyArray_DATA(py_uy),
            uy,
            lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny *
                sizeof(double)
        );
    }

    result = Py_BuildValue("OO", py_ux, py_uy);

done:
    Py_XDECREF(py_ux);
    Py_XDECREF(py_uy);
    return result;
}

static PyObject* lensmodel_num_params(
    PyObject* NPY_UNUSED(self),
    PyObject* args
) {
    PyObject* result = NULL;

    char* lensmodel_string = NULL;
    if (!PyArg_ParseTuple(args, "s", &lensmodel_string)) {
        goto done;
    }
    mrcal_lensmodel_t lensmodel;
    if (!parse_lensmodel_from_arg(&lensmodel, lensmodel_string)) {
        goto done;
    }

    int Nparams = mrcal_lensmodel_num_params(&lensmodel);

    result = Py_BuildValue("i", Nparams);

done:
    return result;
}

static PyObject* supported_lensmodels(
    PyObject* NPY_UNUSED(self),
    PyObject* NPY_UNUSED(args)
) {
    PyObject* result = NULL;
    const char* const* names = mrcal_supported_lensmodel_names();

    // I now have a NULL-terminated list of NULL-terminated strings. Get N
    int N = 0;
    while (names[N] != NULL) {
        N++;
    }

    result = PyTuple_New(N);
    if (result == NULL) {
        BARF("Failed PyTuple_New(%d)", N);
        goto done;
    }

    for (int i = 0; i < N; i++) {
        PyObject* name = Py_BuildValue("s", names[i]);
        if (name == NULL) {
            BARF("Failed Py_BuildValue...");
            Py_DECREF(result);
            result = NULL;
            goto done;
        }
        PyTuple_SET_ITEM(result, i, name);
    }

done:
    return result;
}

// just like PyArray_Converter(), but leave None as None
static int PyArray_Converter_leaveNone(
    PyObject* obj,
    PyObject** address
) {
    if (obj == Py_None) {
        *address = Py_None;
        Py_INCREF(Py_None);
        return 1;
    }
    return PyArray_Converter(obj, address);
}

// For various utility functions. Accepts ONE lens model, not N of them like the
// optimization function
#define LENSMODEL_ONE_ARGUMENTS(_, suffix)                 \
    _(lensmodel##suffix, char*, NULL, "s", , NULL, -1, {}) \
    _(intrinsics##suffix,                                  \
      PyArrayObject*,                                      \
      NULL,                                                \
      "O&",                                                \
      PyArray_Converter_leaveNone COMMA,                   \
      intrinsics##suffix,                                  \
      NPY_DOUBLE,                                          \
      {-1})

#define ARGDEF_observations_point_triangulated(_) \
    _(observations_point_triangulated,            \
      PyArrayObject*,                             \
      NULL,                                       \
      "O&",                                       \
      PyArray_Converter_leaveNone COMMA,          \
      observations_point_triangulated,            \
      NPY_DOUBLE,                                 \
      {-1 COMMA 3})
#define ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(_) \
    _(indices_point_triangulated_camintrinsics_camextrinsics,            \
      PyArrayObject*,                                                    \
      NULL,                                                              \
      "O&",                                                              \
      PyArray_Converter_leaveNone COMMA,                                 \
      indices_point_triangulated_camintrinsics_camextrinsics,            \
      NPY_INT32,                                                         \
      {-1 COMMA 3})

#define OPTIMIZE_ARGUMENTS_REQUIRED(_)             \
    _(intrinsics,                                  \
      PyArrayObject*,                              \
      NULL,                                        \
      "O&",                                        \
      PyArray_Converter_leaveNone COMMA,           \
      intrinsics,                                  \
      NPY_DOUBLE,                                  \
      {-1 COMMA - 1})                              \
    _(lensmodel, char*, NULL, "s", , NULL, -1, {}) \
    _(imagersizes,                                 \
      PyArrayObject*,                              \
      NULL,                                        \
      "O&",                                        \
      PyArray_Converter_leaveNone COMMA,           \
      imagersizes,                                 \
      NPY_INT32,                                   \
      {-1 COMMA 2})

// Defaults for do_optimize... MUST match those in ingest_packed_state()
//
// Accepting observed_pixel_uncertainty for backwards compatibility. It doesn't
// do anything anymore
#define OPTIMIZE_ARGUMENTS_OPTIONAL(_)                                        \
    _(extrinsics_rt_fromref,                                                  \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      extrinsics_rt_fromref,                                                  \
      NPY_DOUBLE,                                                             \
      {-1 COMMA 6})                                                           \
    _(frames_rt_toref,                                                        \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      frames_rt_toref,                                                        \
      NPY_DOUBLE,                                                             \
      {-1 COMMA 6})                                                           \
    _(points,                                                                 \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      points,                                                                 \
      NPY_DOUBLE,                                                             \
      {-1 COMMA 3})                                                           \
    _(observations_board,                                                     \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      observations_board,                                                     \
      NPY_DOUBLE,                                                             \
      {-1 COMMA - 1 COMMA - 1 COMMA 3})                                       \
    _(indices_frame_camintrinsics_camextrinsics,                              \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      indices_frame_camintrinsics_camextrinsics,                              \
      NPY_INT32,                                                              \
      {-1 COMMA 3})                                                           \
    _(observations_point,                                                     \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      observations_point,                                                     \
      NPY_DOUBLE,                                                             \
      {-1 COMMA 3})                                                           \
    _(indices_point_camintrinsics_camextrinsics,                              \
      PyArrayObject*,                                                         \
      NULL,                                                                   \
      "O&",                                                                   \
      PyArray_Converter_leaveNone COMMA,                                      \
      indices_point_camintrinsics_camextrinsics,                              \
      NPY_INT32,                                                              \
      {-1 COMMA 3})                                                           \
    ARGDEF_observations_point_triangulated(_                                  \
    ) ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(_         \
    ) _(observed_pixel_uncertainty, double, -1.0, "d", , NULL, -1, {}         \
    ) _(calobject_warp,                                                       \
        PyArrayObject*,                                                       \
        NULL,                                                                 \
        "O&",                                                                 \
        PyArray_Converter_leaveNone COMMA,                                    \
        calobject_warp,                                                       \
        NPY_DOUBLE,                                                           \
        {2}) _(Npoints_fixed, int, 0, "i", , NULL, -1, {})                    \
        _(do_optimize_intrinsics_core, int, -1, "p", , NULL, -1, {}           \
        ) _(do_optimize_intrinsics_distortions, int, -1, "p", , NULL, -1, {}  \
        ) _(do_optimize_extrinsics, int, -1, "p", , NULL, -1, {}              \
        ) _(do_optimize_frames, int, -1, "p", , NULL, -1, {}                  \
        ) _(do_optimize_calobject_warp, int, -1, "p", , NULL, -1, {}          \
        ) _(calibration_object_spacing, double, -1.0, "d", , NULL, -1, {}     \
        ) _(point_min_range, double, -1.0, "d", , NULL, -1, {}                \
        ) _(point_max_range, double, -1.0, "d", , NULL, -1, {}                \
        ) _(verbose, int, 0, "p", , NULL, -1, {}                              \
        ) _(do_apply_regularization, int, 1, "p", , NULL, -1, {}              \
        ) _(do_apply_regularization_unity_cam01, int, 0, "p", , NULL, -1, {}) \
            _(do_apply_outlier_rejection, int, 1, "p", , NULL, -1, {})        \
                _(imagepaths, PyObject*, NULL, "O", , NULL, -1, {})
/* imagepaths is in the argument list purely to make the
   mrcal-show-residuals-board-observation tool work. The python code doesn't
   actually touch it */

#define OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(_) \
    _(no_jacobian, int, 0, "p", , NULL, -1, {})        \
    _(no_factorization, int, 0, "p", , NULL, -1, {})

typedef enum {
    OPTIMIZEMODE_OPTIMIZE,
    OPTIMIZEMODE_CALLBACK,
    OPTIMIZEMODE_DRTRRP_DB
} optimizemode_t;

static bool lensmodel_one_validate_args(
    // out
    mrcal_lensmodel_t* mrcal_lensmodel,

    // in
    LENSMODEL_ONE_ARGUMENTS(ARG_LIST_DEFINE, ) bool do_check_layout
) {
    if (do_check_layout) {
        LENSMODEL_ONE_ARGUMENTS(CHECK_LAYOUT, );
    }

    if (!parse_lensmodel_from_arg(mrcal_lensmodel, lensmodel)) {
        return false;
    }
    int NlensParams = mrcal_lensmodel_num_params(mrcal_lensmodel);
    int NlensParams_have =
        PyArray_DIMS(intrinsics)[PyArray_NDIM(intrinsics) - 1];
    if (NlensParams != NlensParams_have) {
        BARF(
            "intrinsics.shape[-1] MUST be %d. Instead got %ld",
            NlensParams,
            NlensParams_have
        );
        return false;
    }

    return true;
done:
    return false;
}

// Using this for both optimize() and optimizer_callback()
static bool optimize_validate_args(
    // out
    mrcal_lensmodel_t* mrcal_lensmodel,

    // in
    optimizemode_t optimizemode,
    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
        OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)
            OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_LIST_DEFINE)

                void* dummy __attribute__((unused))
) {
    _Static_assert(
        sizeof(mrcal_pose_t) / sizeof(double) == 6,
        "mrcal_pose_t is assumed to contain 6 elements"
    );

    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(CHECK_LAYOUT);

    int Ncameras_intrinsics = PyArray_DIMS(intrinsics)[0];
    int Ncameras_extrinsics = PyArray_DIMS(extrinsics_rt_fromref)[0];
    if (PyArray_DIMS(imagersizes)[0] != Ncameras_intrinsics) {
        BARF(
            "Inconsistent Ncameras: 'intrinsics' says %ld, 'imagersizes' says "
            "%ld",
            Ncameras_intrinsics,
            PyArray_DIMS(imagersizes)[0]
        );
        return false;
    }

    long int Nobservations_board = PyArray_DIMS(observations_board)[0];
    if (PyArray_DIMS(indices_frame_camintrinsics_camextrinsics)[0] !=
        Nobservations_board) {
        BARF(
            "Inconsistent Nobservations_board: 'observations_board' says %ld, "
            "'indices_frame_camintrinsics_camextrinsics' says %ld",
            Nobservations_board,
            PyArray_DIMS(indices_frame_camintrinsics_camextrinsics)[0]
        );
        return false;
    }

    if (Nobservations_board > 0) {
        if (calibration_object_spacing <= 0.0) {
            BARF(
                "We have board observations, so calibration_object_spacing "
                "MUST be a valid float > 0"
            );
            return false;
        }

        if (do_optimize_calobject_warp && IS_NULL(calobject_warp)) {
            BARF(
                "do_optimize_calobject_warp is True, so calobject_warp MUST be "
                "given as an array to seed the optimization and to receive the "
                "results"
            );
            return false;
        }
    }

    int Nobservations_point = PyArray_DIMS(observations_point)[0];
    if (PyArray_DIMS(indices_point_camintrinsics_camextrinsics)[0] !=
        Nobservations_point) {
        BARF(
            "Inconsistent Nobservations_point: 'observations_point...' says "
            "%ld, 'indices_point_camintrinsics_camextrinsics' says %ld",
            Nobservations_point,
            PyArray_DIMS(indices_point_camintrinsics_camextrinsics)[0]
        );
        return false;
    }

    int Nobservations_point_triangulated =
        PyArray_DIMS(observations_point_triangulated)[0];
    if (PyArray_DIMS(indices_point_triangulated_camintrinsics_camextrinsics
        )[0] != Nobservations_point_triangulated) {
        BARF(
            "Inconsistent Nobservations_point_triangulated: "
            "'observations_point_triangulated...' says %ld, "
            "'indices_triangulated_point_camintrinsics_camextrinsics' says %ld",
            Nobservations_point_triangulated,
            PyArray_DIMS(indices_point_triangulated_camintrinsics_camextrinsics
            )[0]
        );
        return false;
    }

    // I reuse the single-lensmodel validation function. That function expects
    // ONE set of intrinsics instead of N intrinsics, like this function does.
    // But I already did the CHECK_LAYOUT() at the start of this function, and
    // I'm not going to do that again here: passing do_check_layout=false. So
    // that difference doesn't matter
    if (!lensmodel_one_validate_args(
            mrcal_lensmodel,
            lensmodel,
            intrinsics,
            false
        )) {
        return false;
    }

    // make sure the indices arrays are valid: the data is monotonic and
    // in-range
    int Nframes = PyArray_DIMS(frames_rt_toref)[0];
    int iframe_last = -1;
    int icam_intrinsics_last = -1;
    int icam_extrinsics_last = -1;
    for (int i_observation = 0; i_observation < Nobservations_board;
         i_observation++) {
        // check for monotonicity and in-rangeness
        int32_t iframe =
            ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics)
            )[i_observation * 3 + 0];
        int32_t icam_intrinsics =
            ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics)
            )[i_observation * 3 + 1];
        int32_t icam_extrinsics =
            ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics)
            )[i_observation * 3 + 2];

        // First I make sure everything is in-range
        if (iframe < 0 || iframe >= Nframes) {
            BARF(
                "iframe MUST be in [0,%d], instead got %d in row %d of "
                "indices_frame_camintrinsics_camextrinsics",
                Nframes - 1,
                iframe,
                i_observation
            );
            return false;
        }
        if (icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics) {
            BARF(
                "icam_intrinsics MUST be in [0,%d], instead got %d in row %d "
                "of indices_frame_camintrinsics_camextrinsics",
                Ncameras_intrinsics - 1,
                icam_intrinsics,
                i_observation
            );
            return false;
        }
        if (icam_extrinsics < -1 || icam_extrinsics >= Ncameras_extrinsics) {
            BARF(
                "icam_extrinsics MUST be in [-1,%d], instead got %d in row %d "
                "of indices_frame_camintrinsics_camextrinsics",
                Ncameras_extrinsics - 1,
                icam_extrinsics,
                i_observation
            );
            return false;
        }
        // And then I check monotonicity
        if (iframe == iframe_last) {
            if (icam_intrinsics < icam_intrinsics_last) {
                BARF(
                    "icam_intrinsics MUST be monotonically increasing in "
                    "indices_frame_camintrinsics_camextrinsics. Instead row %d "
                    "(frame %d) of indices_frame_camintrinsics_camextrinsics "
                    "has icam_intrinsics=%d after previously seeing "
                    "icam_intrinsics=%d",
                    i_observation,
                    iframe,
                    icam_intrinsics,
                    icam_intrinsics_last
                );
                return false;
            }
            if (icam_extrinsics < icam_extrinsics_last) {
                BARF(
                    "icam_extrinsics MUST be monotonically increasing in "
                    "indices_frame_camintrinsics_camextrinsics. Instead row %d "
                    "(frame %d) of indices_frame_camintrinsics_camextrinsics "
                    "has icam_extrinsics=%d after previously seeing "
                    "icam_extrinsics=%d",
                    i_observation,
                    iframe,
                    icam_extrinsics,
                    icam_extrinsics_last
                );
                return false;
            }
        } else if (iframe < iframe_last) {
            BARF(
                "iframe MUST be monotonically increasing in "
                "indices_frame_camintrinsics_camextrinsics. Instead row %d of "
                "indices_frame_camintrinsics_camextrinsics has iframe=%d after "
                "previously seeing iframe=%d",
                i_observation,
                iframe,
                iframe_last
            );
            return false;
        } else if (iframe - iframe_last != 1) {
            BARF(
                "iframe MUST be increasing sequentially in "
                "indices_frame_camintrinsics_camextrinsics. Instead row %d of "
                "indices_frame_camintrinsics_camextrinsics has iframe=%d after "
                "previously seeing iframe=%d",
                i_observation,
                iframe,
                iframe_last
            );
            return false;
        }

        iframe_last = iframe;
        icam_intrinsics_last = icam_intrinsics;
        icam_extrinsics_last = icam_extrinsics;
    }
    if (Nobservations_board > 0) {
        int i_observation_lastrow = Nobservations_board - 1;
        int32_t iframe_lastrow =
            ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics)
            )[i_observation_lastrow * 3 + 0];
        if (iframe_lastrow != Nframes - 1) {
            BARF(
                "iframe in indices_frame_camintrinsics_camextrinsics must "
                "cover ALL frames. Instead the last row of "
                "indices_frame_camintrinsics_camextrinsics has iframe=%d, but "
                "Nframes=%d",
                iframe_lastrow,
                Nframes
            );
            return false;
        }
    }

    int Npoints = PyArray_DIMS(points)[0];
    if (Npoints > 0) {
        if (Npoints_fixed > Npoints) {
            BARF(
                "I have Npoints=len(points)=%d, but Npoints_fixed=%d. "
                "Npoints_fixed > Npoints makes no sense",
                Npoints,
                Npoints_fixed
            );
            return false;
        }
        if (point_min_range <= 0.0 || point_max_range <= 0.0 ||
            point_min_range >= point_max_range) {
            BARF(
                "Point observations were given, so point_min_range and "
                "point_max_range MUST have been given usable values > 0 and "
                "max>min"
            );
            return false;
        }
    } else {
        if (Npoints_fixed) {
            BARF(
                "No 'points' were given, so it's 'Npoints_fixed' doesn't do "
                "anything, and shouldn't be given"
            );
            return false;
        }
    }

    // I allow i_point to be non-monotonic, but I do make sure that it covers
    // all Npoints of my array.
    int32_t i_point_biggest = -1;
    for (int i_observation = 0; i_observation < Nobservations_point;
         i_observation++) {
        int32_t i_point =
            ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics)
            )[i_observation * 3 + 0];
        int32_t icam_intrinsics =
            ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics)
            )[i_observation * 3 + 1];
        int32_t icam_extrinsics =
            ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics)
            )[i_observation * 3 + 2];

        // First I make sure everything is in-range
        if (i_point < 0 || i_point >= Npoints) {
            BARF(
                "i_point MUST be in [0,%d], instead got %d in row %d of "
                "indices_point_camintrinsics_camextrinsics",
                Npoints - 1,
                i_point,
                i_observation
            );
            return false;
        }
        if (icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics) {
            BARF(
                "icam_intrinsics MUST be in [0,%d], instead got %d in row %d "
                "of indices_point_camintrinsics_camextrinsics",
                Ncameras_intrinsics - 1,
                icam_intrinsics,
                i_observation
            );
            return false;
        }
        if (icam_extrinsics < -1 || icam_extrinsics >= Ncameras_extrinsics) {
            BARF(
                "icam_extrinsics MUST be in [-1,%d], instead got %d in row %d "
                "of indices_point_camintrinsics_camextrinsics",
                Ncameras_extrinsics - 1,
                icam_extrinsics,
                i_observation
            );
            return false;
        }

        if (i_point > i_point_biggest) {
            if (i_point > i_point_biggest + 1) {
                BARF(
                    "indices_point_camintrinsics_camextrinsics should contain "
                    "i_point that extend the existing set by one point at a "
                    "time at most. However row %d has i_point=%d while the "
                    "biggest-seen-so-far i_point=%d",
                    i_observation,
                    i_point,
                    i_point_biggest
                );
                return false;
            }
            i_point_biggest = i_point;
        }
    }
    if (i_point_biggest != Npoints - 1) {
        BARF(
            "indices_point_camintrinsics_camextrinsics should cover all point "
            "indices in [0,%d], but there are gaps. The biggest i_point=%d",
            Npoints - 1,
            i_point_biggest
        );
        return false;
    }

    i_point_biggest = -1;
    for (int i_observation = 0;
         i_observation < Nobservations_point_triangulated;
         i_observation++) {
        int32_t i_point = ((int32_t*)PyArray_DATA(
            indices_point_triangulated_camintrinsics_camextrinsics
        ))[i_observation * 3 + 0];
        int32_t icam_intrinsics = ((int32_t*)PyArray_DATA(
            indices_point_triangulated_camintrinsics_camextrinsics
        ))[i_observation * 3 + 1];
        int32_t icam_extrinsics = ((int32_t*)PyArray_DATA(
            indices_point_triangulated_camintrinsics_camextrinsics
        ))[i_observation * 3 + 2];

        if (icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics) {
            BARF(
                "icam_intrinsics MUST be in [0,%d], instead got %d in row %d "
                "of indices_point_triangulated_camintrinsics_camextrinsics",
                Ncameras_intrinsics - 1,
                icam_intrinsics,
                i_observation
            );
            return false;
        }
        if (icam_extrinsics < -1 || icam_extrinsics >= Ncameras_extrinsics) {
            BARF(
                "icam_extrinsics MUST be in [-1,%d], instead got %d in row %d "
                "of indices_point_triangulated_camintrinsics_camextrinsics",
                Ncameras_extrinsics - 1,
                icam_extrinsics,
                i_observation
            );
            return false;
        }

        if (i_point > i_point_biggest) {
            if (i_point > i_point_biggest + 1) {
                BARF(
                    "indices_point_triangulated_camintrinsics_camextrinsics "
                    "should contain i_point that extend the existing set by "
                    "one point at a time at most. However row %d has "
                    "i_point=%d while the biggest-seen-so-far i_point=%d",
                    i_observation,
                    i_point,
                    i_point_biggest
                );
                return false;
            }
            i_point_biggest = i_point;
        }
    }

    // There are more checks for triangulated points, but I run them later, in
    // fill_c_observations_point_triangulated()

    return true;
done:
    return false;
}

static void fill_c_observations_board(
    // out
    mrcal_observation_board_t* c_observations_board,

    // in
    int Nobservations_board,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics
) {
    for (int i_observation = 0; i_observation < Nobservations_board;
         i_observation++) {
        int32_t iframe = ((const int32_t*)PyArray_DATA(
            (PyArrayObject*)indices_frame_camintrinsics_camextrinsics
        ))[i_observation * 3 + 0];
        int32_t icam_intrinsics = ((const int32_t*)PyArray_DATA(
            (PyArrayObject*)indices_frame_camintrinsics_camextrinsics
        ))[i_observation * 3 + 1];
        int32_t icam_extrinsics = ((const int32_t*)PyArray_DATA(
            (PyArrayObject*)indices_frame_camintrinsics_camextrinsics
        ))[i_observation * 3 + 2];

        c_observations_board[i_observation].icam.intrinsics = icam_intrinsics;
        c_observations_board[i_observation].icam.extrinsics = icam_extrinsics;
        c_observations_board[i_observation].iframe = iframe;
    }
}

static void fill_c_observations_point(
    // out
    mrcal_observation_point_t* c_observations_point,

    // in
    int Nobservations_point,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics
) {
    for (int i_observation = 0; i_observation < Nobservations_point;
         i_observation++) {
        int32_t i_point = ((const int32_t*)PyArray_DATA(
            (PyArrayObject*)indices_point_camintrinsics_camextrinsics
        ))[i_observation * 3 + 0];
        int32_t icam_intrinsics = ((const int32_t*)PyArray_DATA(
            (PyArrayObject*)indices_point_camintrinsics_camextrinsics
        ))[i_observation * 3 + 1];
        int32_t icam_extrinsics = ((const int32_t*)PyArray_DATA(
            (PyArrayObject*)indices_point_camintrinsics_camextrinsics
        ))[i_observation * 3 + 2];

        c_observations_point[i_observation].icam.intrinsics = icam_intrinsics;
        c_observations_point[i_observation].icam.extrinsics = icam_extrinsics;
        c_observations_point[i_observation].i_point = i_point;
    }
}

static bool fill_c_observations_point_triangulated_validate_arguments(
    const PyArrayObject* observations_point_triangulated,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics
) {
    if (observations_point_triangulated != NULL) {
        ARGDEF_observations_point_triangulated(CHECK_LAYOUT);
    }
    ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(CHECK_LAYOUT);

    return true;
done:
    return false;
}

static bool fill_c_observations_point_triangulated_finish_set(
    mrcal_observation_point_triangulated_t* c_observations_point_triangulated,
    int ipoint_last_in_set,
    int Npoints_in_this_set
) {
    if (ipoint_last_in_set < 0) {
        return true;
    }

    c_observations_point_triangulated[ipoint_last_in_set].last_in_set = true;
    if (Npoints_in_this_set < 2) {
        BARF(
            "Error in indices...[%d]. Each point must be observed at least 2 "
            "times",
            ipoint_last_in_set
        );
        return false;
    }
    return true;
}

// return the number of points, or <0 on error
static int fill_c_observations_point_triangulated(
    // output. I fill in the given arrays
    mrcal_observation_point_triangulated_t* c_observations_point_triangulated,

    // input
    const PyArrayObject* observations_point_triangulated,  // may be NULL
    // used only if observations_point_triangulated != NULL
    const mrcal_lensmodel_t* lensmodel,
    const double* intrinsics,

    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics
) {
    if (indices_point_triangulated_camintrinsics_camextrinsics == NULL) {
        return 0;
    }

    if (!fill_c_observations_point_triangulated_validate_arguments(
            observations_point_triangulated,
            indices_point_triangulated_camintrinsics_camextrinsics
        )) {
        return -1;
    }

    int N = (int
    )PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    if (observations_point_triangulated != NULL) {
        if (N != (int)PyArray_DIM(observations_point_triangulated, 0)) {
            BARF(
                "Inconsistent point counts. "
                "observations_point_triangulated.shape[0] = %d, but "
                "indices_point_triangulated_camintrinsics_camextrinsics.shape["
                "0] = %d",
                (int)PyArray_DIM(observations_point_triangulated, 0),
                N
            );
            return -1;
        }
    }

    const double* observations_point_triangulated__data =
        (observations_point_triangulated != NULL)
            ? (const double*)PyArray_DATA(
                  (PyArrayObject*)observations_point_triangulated
              )
            : NULL;
    const int32_t*
        indices_point_triangulated_camintrinsics_camextrinsics__data =
            (const int32_t*)PyArray_DATA(
                (PyArrayObject*)
                    indices_point_triangulated_camintrinsics_camextrinsics
            );

    int ipoint_current = -1;
    int Npoints_in_this_set = 0;

    // Needed for the unproject() below
    int Nintrinsics_state = 0;
    mrcal_projection_precomputed_t precomputed;
    if (lensmodel != NULL) {
        Nintrinsics_state = mrcal_lensmodel_num_params(lensmodel);
        mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(lensmodel);
        if (!meta.has_gradients) {
            BARF(
                "mrcal_unproject(lensmodel='%s') is not yet implemented: we "
                "need gradients",
                mrcal_lensmodel_name_unconfigured(lensmodel)
            );
            return -1;
        }
        _mrcal_precompute_lensmodel_data(&precomputed, lensmodel);
    }

    for (int i = 0; i < N; i++) {
        const int32_t* row =
            &indices_point_triangulated_camintrinsics_camextrinsics__data
                [3 * i];

        const int32_t ipoint = row[0];
        const int32_t icam_intrinsics = row[1];
        const int32_t icam_extrinsics = row[2];

        c_observations_point_triangulated[i].last_in_set = false;
        c_observations_point_triangulated[i].outlier = false;
        c_observations_point_triangulated[i].icam = (mrcal_camera_index_t
        ){.intrinsics = icam_intrinsics, .extrinsics = icam_extrinsics};
        if (observations_point_triangulated__data != NULL) {
            const mrcal_point3_t* px_weight =
                (const mrcal_point3_t*)(&observations_point_triangulated__data
                                            [3 * i]);

            // For now the triangulated observations are local observation
            // vectors
            if (!_mrcal_unproject_internal(  // out
                    &c_observations_point_triangulated[i].px,

                    // in
                    (const mrcal_point2_t*)(px_weight->xyz),
                    1,
                    lensmodel,
                    &intrinsics[icam_intrinsics * Nintrinsics_state],
                    &precomputed
                )) {
                BARF("mrcal_unproject() failed");
                return -1;
            }

            c_observations_point_triangulated[i].outlier =
                (px_weight->z <= 0.0);
        } else {
            c_observations_point_triangulated[i].px = (mrcal_point3_t){};
        }

        if (ipoint < 0) {
            BARF(
                "Error in "
                "indices_point_triangulated_camintrinsics_camextrinsics[%d]. "
                "Saw ipoint=%d. Each one must be >=0",
                i,
                ipoint
            );
            return -1;
        } else if (ipoint == ipoint_current) {
            Npoints_in_this_set++;
        } else if (ipoint == ipoint_current + 1) {
            // The previous point was the last in the set
            if (!fill_c_observations_point_triangulated_finish_set(
                    c_observations_point_triangulated,
                    i - 1,
                    Npoints_in_this_set
                )) {
                return -1;
            }

            ipoint_current = ipoint;
            Npoints_in_this_set = 1;
        } else {
            BARF(
                "Error in "
                "indices_point_triangulated_camintrinsics_camextrinsics[%d]. "
                "All ipoint must be consecutive and monotonic",
                i
            );
            return -1;
        }
    }
    if (!fill_c_observations_point_triangulated_finish_set(
            c_observations_point_triangulated,
            N - 1,
            Npoints_in_this_set
        )) {
        return -1;
    }

    return N;
}

#define PROBLEM_SELECTIONS_SET_BIT(x) .x = x,
#define CONSTRUCT_PROBLEM_SELECTIONS()                                        \
    ({                                                                        \
        /* By default we optimize everything we can; these are default at <0  \
         */                                                                   \
        if (do_optimize_intrinsics_core < 0)                                  \
            do_optimize_intrinsics_core = Ncameras_intrinsics > 0;            \
        if (do_optimize_intrinsics_distortions < 0)                           \
            do_optimize_intrinsics_core = Ncameras_intrinsics > 0;            \
        if (do_optimize_extrinsics < 0)                                       \
            do_optimize_extrinsics = Ncameras_extrinsics > 0;                 \
        if (do_optimize_frames < 0)                                           \
            do_optimize_frames = Nframes > 0;                                 \
        if (do_optimize_calobject_warp < 0)                                   \
            do_optimize_calobject_warp = Nobservations_board > 0;             \
        /* stuff not in the above if doesn't have a <0 default; those are all \
         * 0 or 1 already */                                                  \
        (mrcal_problem_selections_t                                           \
        ){MRCAL_PROBLEM_SELECTIONS_LIST(PROBLEM_SELECTIONS_SET_BIT)};         \
    })

static PyObject* _optimize(
    optimizemode_t optimizemode,
    PyObject* args,
    PyObject* kwargs
) {
    PyObject* result = NULL;

    PyArrayObject* b_packed_final = NULL;
    PyArrayObject* x_final = NULL;
    PyObject* pystats = NULL;

    PyArrayObject* P = NULL;
    PyArrayObject* I = NULL;
    PyArrayObject* X = NULL;
    PyObject* factorization = NULL;
    PyObject* jacobian = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_DEFINE);

    int calibration_object_height_n = -1;
    int calibration_object_width_n = -1;

    SET_SIGINT();

    if (optimizemode == OPTIMIZEMODE_OPTIMIZE ||
        optimizemode == OPTIMIZEMODE_DRTRRP_DB) {
        char* keywords[] = {OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                                OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST) NULL};
        if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE
                ) "|$" OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE) ":mrcal.optimize",

                keywords,

                OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                    OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL
            )) {
            goto done;
        }
    } else if (optimizemode == OPTIMIZEMODE_CALLBACK) {
        char* keywords[] = {OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST
        ) OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST
        ) OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(NAMELIST) NULL};
        if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE
                ) "|$" OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
                    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(PARSECODE
                    ) ":mrcal.optimizer_callback",

                keywords,

                OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG
                ) OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG)
                    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(PARSEARG) NULL
            )) {
            goto done;
        }
    } else {
        BARF("ERROR: Unknown optimizemode=%d. Giving up", (int)optimizemode);
        goto done;
    }

    // Some of my input arguments can be empty (None). The code all assumes that
    // everything is a properly-dimensioned numpy array, with "empty" meaning
    // some dimension is 0. Here I make this conversion. The user can pass None,
    // and we still do the right thing.
    //
    // There's a silly implementation detail here: if you have a preprocessor
    // macro M(x), and you pass it M({1,2,3}), the preprocessor see 3 separate
    // args, not 1. That's why I have a __VA_ARGS__ here and why I instantiate a
    // separate dims[] (PyArray_SimpleNew is a macro too)
#define SET_SIZE0_IF_NONE(x, type, ...)                                        \
    ({                                                                         \
        if (IS_NULL(x)) {                                                      \
            if (x != NULL)                                                     \
                Py_DECREF(x);                                                  \
            npy_intp dims[] = {__VA_ARGS__};                                   \
            x = (PyArrayObject*)                                               \
                PyArray_SimpleNew(sizeof(dims) / sizeof(dims[0]), dims, type); \
        }                                                                      \
    })

    SET_SIZE0_IF_NONE(extrinsics_rt_fromref, NPY_DOUBLE, 0, 6);

    SET_SIZE0_IF_NONE(frames_rt_toref, NPY_DOUBLE, 0, 6);
    SET_SIZE0_IF_NONE(
        observations_board,
        NPY_DOUBLE,
        0,
        179,
        171,
        3
    );  // arbitrary numbers; shouldn't matter
    SET_SIZE0_IF_NONE(
        indices_frame_camintrinsics_camextrinsics,
        NPY_INT32,
        0,
        3
    );

    SET_SIZE0_IF_NONE(points, NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(observations_point, NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(
        indices_point_camintrinsics_camextrinsics,
        NPY_INT32,
        0,
        3
    );
    SET_SIZE0_IF_NONE(observations_point_triangulated, NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(
        indices_point_triangulated_camintrinsics_camextrinsics,
        NPY_INT32,
        0,
        3
    );
    SET_SIZE0_IF_NONE(imagersizes, NPY_INT32, 0, 2);
#undef SET_NULL_IF_NONE

    mrcal_lensmodel_t mrcal_lensmodel;
    // Check the arguments for optimize(). If optimizer_callback, then the other
    // stuff is defined, but it all has valid, default values
    if (!optimize_validate_args(
            &mrcal_lensmodel,
            optimizemode,
            OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_CALL
            ) OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_CALL)
                OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_LIST_CALL) NULL
        )) {
        goto done;
    }

    // Can't compute a factorization without a jacobian. That's what we're
    // factoring
    if (!no_factorization) {
        no_jacobian = false;
    }

    {
        int Ncameras_intrinsics = PyArray_DIMS(intrinsics)[0];
        int Ncameras_extrinsics = PyArray_DIMS(extrinsics_rt_fromref)[0];
        int Nframes = PyArray_DIMS(frames_rt_toref)[0];
        int Npoints = PyArray_DIMS(points)[0];
        int Nobservations_board = PyArray_DIMS(observations_board)[0];
        int Nobservations_point = PyArray_DIMS(observations_point)[0];

        if (Nobservations_board > 0) {
            calibration_object_height_n = PyArray_DIMS(observations_board)[1];
            calibration_object_width_n = PyArray_DIMS(observations_board)[2];
        }

        // The checks in optimize_validate_args() make sure these casts are
        // kosher
        double* c_intrinsics = (double*)PyArray_DATA(intrinsics);
        mrcal_pose_t* c_extrinsics =
            (mrcal_pose_t*)PyArray_DATA(extrinsics_rt_fromref);
        mrcal_pose_t* c_frames = (mrcal_pose_t*)PyArray_DATA(frames_rt_toref);
        mrcal_point3_t* c_points = (mrcal_point3_t*)PyArray_DATA(points);
        mrcal_calobject_warp_t* c_calobject_warp =
            IS_NULL(calobject_warp)
                ? NULL
                : (mrcal_calobject_warp_t*)PyArray_DATA(calobject_warp);

        // Is contiguous; I made sure above
        mrcal_point3_t* c_observations_board_pool =
            (mrcal_point3_t*)PyArray_DATA(observations_board);
        mrcal_observation_board_t c_observations_board[Nobservations_board];
        fill_c_observations_board(  // output
            c_observations_board,
            // input
            Nobservations_board,
            indices_frame_camintrinsics_camextrinsics
        );

        // Is contiguous; I made sure above
        mrcal_point3_t* c_observations_point_pool =
            (mrcal_point3_t*)PyArray_DATA(observations_point);
        mrcal_observation_point_t c_observations_point[Nobservations_point];
        fill_c_observations_point(  // output
            c_observations_point,
            // input
            Nobservations_point,
            indices_point_camintrinsics_camextrinsics
        );

        int Nobservations_point_triangulated =
            PyArray_DIMS(observations_point_triangulated)[0];
        mrcal_observation_point_triangulated_t
            c_observations_point_triangulated[Nobservations_point_triangulated];
        if (fill_c_observations_point_triangulated(
                c_observations_point_triangulated,
                observations_point_triangulated,
                &mrcal_lensmodel,
                c_intrinsics,
                indices_point_triangulated_camintrinsics_camextrinsics
            ) < 0) {
            goto done;
        }

        mrcal_problem_selections_t problem_selections =
            CONSTRUCT_PROBLEM_SELECTIONS();

        mrcal_problem_constants_t problem_constants = {
            .point_min_range = point_min_range,
            .point_max_range = point_max_range
        };

        int Nmeasurements = mrcal_num_measurements(
            Nobservations_board,
            Nobservations_point,
            c_observations_point_triangulated,
            Nobservations_point_triangulated,
            calibration_object_width_n,
            calibration_object_height_n,
            Ncameras_intrinsics,
            Ncameras_extrinsics,
            Nframes,
            Npoints,
            Npoints_fixed,
            problem_selections,
            &mrcal_lensmodel
        );

        int Nintrinsics_state = mrcal_num_intrinsics_optimization_params(
            problem_selections,
            &mrcal_lensmodel
        );

        // input
        int* c_imagersizes = PyArray_DATA(imagersizes);

        int Nstate = mrcal_num_states(
            Ncameras_intrinsics,
            Ncameras_extrinsics,
            Nframes,
            Npoints,
            Npoints_fixed,
            Nobservations_board,
            problem_selections,
            &mrcal_lensmodel
        );

        // both optimize() and optimizer_callback() use this
        b_packed_final = (PyArrayObject*)
            PyArray_SimpleNew(1, ((npy_intp[]){Nstate}), NPY_DOUBLE);
        double* c_b_packed_final = PyArray_DATA(b_packed_final);

        x_final = (PyArrayObject*)
            PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements}), NPY_DOUBLE);
        double* c_x_final = PyArray_DATA(x_final);

        if (optimizemode == OPTIMIZEMODE_OPTIMIZE) {
            // we're wrapping mrcal_optimize()
            const int Npoints_fromBoards = Nobservations_board *
                                           calibration_object_width_n *
                                           calibration_object_height_n;

            mrcal_stats_t stats = mrcal_optimize(
                c_b_packed_final,
                Nstate * sizeof(double),
                c_x_final,
                Nmeasurements * sizeof(double),
                c_intrinsics,
                c_extrinsics,
                c_frames,
                c_points,
                c_calobject_warp,

                Ncameras_intrinsics,
                Ncameras_extrinsics,
                Nframes,
                Npoints,
                Npoints_fixed,

                c_observations_board,
                c_observations_point,
                Nobservations_board,
                Nobservations_point,

                c_observations_point_triangulated,
                Nobservations_point_triangulated,

                c_observations_board_pool,
                c_observations_point_pool,

                &mrcal_lensmodel,
                c_imagersizes,
                problem_selections,
                &problem_constants,

                calibration_object_spacing,
                calibration_object_width_n,
                calibration_object_height_n,
                verbose,

                false
            );

            if (stats.rms_reproj_error__pixels < 0.0) {
                // Error! I throw an exception
                BARF("mrcal.optimize() failed!");
                goto done;
            }

            pystats = PyDict_New();
            if (pystats == NULL) {
                BARF("PyDict_New() failed!");
                goto done;
            }
#define MRCAL_STATS_ITEM_POPULATE_DICT(type, name, pyconverter) \
    {                                                           \
        PyObject* obj = pyconverter((type)stats.name);          \
        if (obj == NULL) {                                      \
            BARF("Couldn't make PyObject for '" #name "'");     \
            goto done;                                          \
        }                                                       \
                                                                \
        if (0 != PyDict_SetItemString(pystats, #name, obj)) {   \
            BARF("Couldn't add to stats dict '" #name "'");     \
            Py_DECREF(obj);                                     \
            goto done;                                          \
        }                                                       \
    }
            MRCAL_STATS_ITEM(MRCAL_STATS_ITEM_POPULATE_DICT);

            if (0 != PyDict_SetItemString(
                         pystats,
                         "b_packed",
                         (PyObject*)b_packed_final
                     )) {
                BARF("Couldn't add to stats dict 'b_packed'");
                goto done;
            }
            if (0 != PyDict_SetItemString(pystats, "x", (PyObject*)x_final)) {
                BARF("Couldn't add to stats dict 'x'");
                goto done;
            }

            result = pystats;
            Py_INCREF(result);
        } else if (optimizemode == OPTIMIZEMODE_CALLBACK ||
                   optimizemode == OPTIMIZEMODE_DRTRRP_DB) {
            int N_j_nonzero = _mrcal_num_j_nonzero(
                Nobservations_board,
                Nobservations_point,
                c_observations_point_triangulated,
                Nobservations_point_triangulated,
                calibration_object_width_n,
                calibration_object_height_n,
                Ncameras_intrinsics,
                Ncameras_extrinsics,
                Nframes,
                Npoints,
                Npoints_fixed,
                c_observations_board,
                c_observations_point,
                problem_selections,
                &mrcal_lensmodel
            );
            cholmod_sparse Jt = {
                .nrow = Nstate,
                .ncol = Nmeasurements,
                .nzmax = N_j_nonzero,
                .stype = 0,
                .itype = CHOLMOD_INT,
                .xtype = CHOLMOD_REAL,
                .dtype = CHOLMOD_DOUBLE,
                .sorted = 1,
                .packed = 1
            };

            if (!no_jacobian) {
                // above I made sure that no_jacobian was false if
                // !no_factorization
                P = (PyArrayObject*)PyArray_SimpleNew(
                    1,
                    ((npy_intp[]){Nmeasurements + 1}),
                    NPY_INT32
                );
                I = (PyArrayObject*)PyArray_SimpleNew(
                    1,
                    ((npy_intp[]){N_j_nonzero}),
                    NPY_INT32
                );
                X = (PyArrayObject*)PyArray_SimpleNew(
                    1,
                    ((npy_intp[]){N_j_nonzero}),
                    NPY_DOUBLE
                );
                Jt.p = PyArray_DATA(P);
                Jt.i = PyArray_DATA(I);
                Jt.x = PyArray_DATA(X);
            }

            if (!mrcal_optimizer_callback(  // out
                    c_b_packed_final,
                    Nstate * sizeof(double),
                    c_x_final,
                    Nmeasurements * sizeof(double),
                    no_jacobian ? NULL : &Jt,

                    // in
                    c_intrinsics,
                    c_extrinsics,
                    c_frames,
                    c_points,
                    c_calobject_warp,

                    Ncameras_intrinsics,
                    Ncameras_extrinsics,
                    Nframes,
                    Npoints,
                    Npoints_fixed,

                    c_observations_board,
                    c_observations_point,
                    Nobservations_board,
                    Nobservations_point,

                    c_observations_point_triangulated,
                    Nobservations_point_triangulated,

                    c_observations_board_pool,
                    c_observations_point_pool,

                    &mrcal_lensmodel,
                    c_imagersizes,
                    problem_selections,
                    &problem_constants,

                    calibration_object_spacing,
                    calibration_object_width_n,
                    calibration_object_height_n,
                    verbose
                )) {
                BARF("mrcal_optimizer_callback() failed!'");
                goto done;
            }

            if (optimizemode == OPTIMIZEMODE_CALLBACK) {
                if (no_factorization) {
                    factorization = Py_None;
                    Py_INCREF(factorization);
                } else {
                    // above I made sure that no_jacobian was false if
                    // !no_factorization
                    factorization =
                        CHOLMOD_factorization_from_cholmod_sparse(&Jt);
                    if (factorization == NULL) {
                        // Couldn't compute factorization. I don't barf, but set
                        // the factorization to None
                        factorization = Py_None;
                        Py_INCREF(factorization);
                        PyErr_Clear();
                    }
                }

                if (no_jacobian) {
                    jacobian = Py_None;
                    Py_INCREF(jacobian);
                } else {
                    jacobian = csr_from_cholmod_sparse(
                        (PyObject*)P,
                        (PyObject*)I,
                        (PyObject*)X
                    );
                    if (jacobian == NULL) {
                        // reuse the existing error
                        goto done;
                    }
                }

                result = PyTuple_Pack(
                    4,
                    (PyObject*)b_packed_final,
                    (PyObject*)x_final,
                    jacobian,
                    factorization
                );
            } else {
                // OPTIMIZEMODE_DRTRRP_DB
                const int state_index_frame0 = mrcal_state_index_frames(
                    0,
                    Ncameras_intrinsics,
                    Ncameras_extrinsics,
                    Nframes,
                    Npoints,
                    Npoints_fixed,
                    Nobservations_board,
                    problem_selections,
                    &mrcal_lensmodel
                );
                const int state_index_point0 = mrcal_state_index_points(
                    0,
                    Ncameras_intrinsics,
                    Ncameras_extrinsics,
                    Nframes,
                    Npoints,
                    Npoints_fixed,
                    Nobservations_board,
                    problem_selections,
                    &mrcal_lensmodel
                );
                const int state_index_calobject_warp0 =
                    mrcal_state_index_calobject_warp(
                        Ncameras_intrinsics,
                        Ncameras_extrinsics,
                        Nframes,
                        Npoints,
                        Npoints_fixed,
                        Nobservations_board,
                        problem_selections,
                        &mrcal_lensmodel
                    );

                // _mrcal_drt_ref_refperturbed__dbpacked() returns an array of
                // shape (6,Nstate_noi_noe). I eventually want to use each of
                // its rows to solve a linear system using the big cholesky
                // factorization: factorization.solve_xt_JtJ_bt(K). This uses
                // CHOLMOD internally. CHOLMOD has no good API interface to use
                // a subset of the state vector for its RHS (Nstate_noi_noe
                // instead of Nstate). I can pass in a sparsity pattern, but
                // that feels like it wouldn't win me anything. So I construct
                // and use a full K, filling the unused entries with 0
                PyObject* K =
                    PyArray_ZEROS(2, ((npy_intp[]){6, Nstate}), NPY_DOUBLE, 0);
                if (K == NULL) {
                    BARF("Couldn't allocate K");
                    goto done;
                }
                if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)K)) {
                    BARF("New array K should be contiguous");
                    Py_DECREF(K);
                    goto done;
                }

                const npy_intp* strides = PyArray_STRIDES((PyArrayObject*)K);

                if (!_mrcal_drt_ref_refperturbed__dbpacked(  // output
                        state_index_frame0 >= 0
                            ? &((double*)(PyArray_DATA((PyArrayObject*)K))
                              )[state_index_frame0]
                            : NULL,
                        (int)strides[0],
                        (int)strides[1],

                        state_index_point0 >= 0
                            ? &((double*)(PyArray_DATA((PyArrayObject*)K))
                              )[state_index_point0]
                            : NULL,
                        (int)strides[0],
                        (int)strides[1],

                        state_index_calobject_warp0 >= 0
                            ? &((double*)(PyArray_DATA((PyArrayObject*)K))
                              )[state_index_calobject_warp0]
                            : NULL,
                        (int)strides[0],
                        (int)strides[1],

                        c_b_packed_final,
                        Nstate * sizeof(double),
                        &Jt,

                        Ncameras_intrinsics,
                        Ncameras_extrinsics,
                        Nframes,
                        Npoints,
                        Npoints_fixed,
                        Nobservations_board,
                        Nobservations_point,
                        &mrcal_lensmodel,
                        problem_selections,

                        calibration_object_width_n,
                        calibration_object_height_n
                    )) {
                    BARF("_mrcal_drt_ref_refperturbed__dbpacked() failed");
                    Py_DECREF(K);
                    goto done;
                }

                result = K;
            }
        } else {
            BARF(
                "ERROR: Unknown optimizemode=%d. Giving up",
                (int)optimizemode
            );
            goto done;
        }
    }

done:
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY);
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(FREE_PYARRAY);

    Py_XDECREF(b_packed_final);
    Py_XDECREF(x_final);
    Py_XDECREF(pystats);
    Py_XDECREF(P);
    Py_XDECREF(I);
    Py_XDECREF(X);
    Py_XDECREF(factorization);
    Py_XDECREF(jacobian);

    RESET_SIGINT();
    return result;
}

static PyObject* optimizer_callback(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    return _optimize(OPTIMIZEMODE_CALLBACK, args, kwargs);
}
static PyObject* optimize(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    return _optimize(OPTIMIZEMODE_OPTIMIZE, args, kwargs);
}
static PyObject* drt_ref_refperturbed__dbpacked(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    return _optimize(OPTIMIZEMODE_DRTRRP_DB, args, kwargs);
}

// The state_index_... python functions don't need the full data but many of
// them do need to know the dimensionality of the data. Thus these can take the
// same arguments as optimizer_callback(). OR in lieu of that, the dimensions
// can be passed-in explicitly with arguments
//
// If both are given, the explicit arguments take precedence. If neither are
// given, I assume 0.
//
// This means that the arguments that are required in optimizer_callback() are
// only optional here
//
// The callbacks return the Python object that will be returned. A callback
// should indicate an error by calling PyErr_...() as usual. If a callback
// returns NULL without setting an error, we return None from Python
typedef PyObject*(callback_state_index_t)(int i,
                                          int Ncameras_intrinsics,
                                          int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints,
                                          int Npoints_fixed,
                                          int Nobservations_board,
                                          int Nobservations_point,
                                          int calibration_object_width_n,
                                          int calibration_object_height_n,
                                          const PyArrayObject*
                                              indices_frame_camintrinsics_camextrinsics,
                                          const PyArrayObject*
                                              indices_point_camintrinsics_camextrinsics,
                                          const PyArrayObject*
                                              indices_point_triangulated_camintrinsics_camextrinsics,
                                          const PyArrayObject*
                                              observations_point,
                                          const mrcal_lensmodel_t* lensmodel,
                                          mrcal_problem_selections_t
                                              problem_selections);
#define STATE_INDEX_GENERIC(f, ...) \
    state_index_generic(callback_##f, #f, __VA_ARGS__)
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning \
    "triangulated-solve: more kwargs for the triangulated-solve measurements? Look in state_index_generic()"
#endif
static PyObject* state_index_generic(
    callback_state_index_t cb,
    const char* called_function,
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    bool need_lensmodel,
    const char* argname
) {
    // This is VERY similar to _pack_unpack_state(). Please consolidate
    // Also somewhat similar to _optimize()

    PyObject* result = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    int i = -1;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes = -1;
    int Npoints = -1;
    int Nobservations_board = -1;
    int Nobservations_point = -1;

    char* keywords[] = {
        (char*)argname,
        "Ncameras_intrinsics",
        "Ncameras_extrinsics",
        "Nframes",
        "Npoints",
        "Nobservations_board",
        "Nobservations_point",
        OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
            OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST) NULL
    };

    // needs to be big-enough to store the largest-possible called_function
#define CALLED_FUNCTION_BUFFER \
    "123456789012345678901234567890123456789012345678901234567890"
    char arg_string[] =
        "i"
        "|$"  // everything is kwarg-only and optional. I apply logic down the
              // line to get what I need
        "iiiiii" OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
            OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE
            ) ":mrcal." CALLED_FUNCTION_BUFFER;
    if (strlen(CALLED_FUNCTION_BUFFER) < strlen(called_function)) {
        BARF(
            "CALLED_FUNCTION_BUFFER too small for '%s'. This is a a bug",
            called_function
        );
        goto done;
    }
    arg_string[strlen(arg_string) - strlen(CALLED_FUNCTION_BUFFER)] = '\0';
    strcat(arg_string, called_function);

    if (argname != NULL) {
        if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                arg_string,
                keywords,

                &i,
                &Ncameras_intrinsics,
                &Ncameras_extrinsics,
                &Nframes,
                &Npoints,
                &Nobservations_board,
                &Nobservations_point,
                OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                    OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL
            )) {
            goto done;
        }
    } else {
        if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,

                // skip the initial "i". There is no "argname" here
                &arg_string[1],
                &keywords[1],

                &Ncameras_intrinsics,
                &Ncameras_extrinsics,
                &Nframes,
                &Npoints,
                &Nobservations_board,
                &Nobservations_point,
                OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                    OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL
            )) {
            goto done;
        }
    }
#undef CALLED_FUNCTION_BUFFER

    mrcal_lensmodel_t mrcal_lensmodel = {};

    if (need_lensmodel) {
        if (lensmodel == NULL) {
            BARF("The 'lensmodel' argument is required");
            goto done;
        }
        if (!parse_lensmodel_from_arg(&mrcal_lensmodel, lensmodel)) {
            goto done;
        }
    }

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if (Ncameras_intrinsics < 0) {
        Ncameras_intrinsics =
            IS_NULL(intrinsics) ? 0 : PyArray_DIMS(intrinsics)[0];
    }
    if (Ncameras_extrinsics < 0) {
        Ncameras_extrinsics = IS_NULL(extrinsics_rt_fromref)
                                  ? 0
                                  : PyArray_DIMS(extrinsics_rt_fromref)[0];
    }
    if (Nframes < 0) {
        Nframes =
            IS_NULL(frames_rt_toref) ? 0 : PyArray_DIMS(frames_rt_toref)[0];
    }
    if (Npoints < 0) {
        Npoints = IS_NULL(points) ? 0 : PyArray_DIMS(points)[0];
    }
    if (Nobservations_board < 0) {
        Nobservations_board = IS_NULL(observations_board)
                                  ? 0
                                  : PyArray_DIMS(observations_board)[0];
    }
    if (Nobservations_point < 0) {
        Nobservations_point = IS_NULL(observations_point)
                                  ? 0
                                  : PyArray_DIMS(observations_point)[0];
    }

    int calibration_object_height_n = -1;
    int calibration_object_width_n = -1;
    if (Nobservations_board > 0) {
        calibration_object_height_n = PyArray_DIMS(observations_board)[1];
        calibration_object_width_n = PyArray_DIMS(observations_board)[2];
    }

    mrcal_problem_selections_t problem_selections =
        CONSTRUCT_PROBLEM_SELECTIONS();

    result =
        cb(i,
           Ncameras_intrinsics,
           Ncameras_extrinsics,
           Nframes,
           Npoints,
           Npoints_fixed,
           Nobservations_board,
           Nobservations_point,
           calibration_object_width_n,
           calibration_object_height_n,
           indices_frame_camintrinsics_camextrinsics,
           indices_point_camintrinsics_camextrinsics,
           indices_point_triangulated_camintrinsics_camextrinsics,
           observations_point,
           &mrcal_lensmodel,
           problem_selections);

    // If an error is set I return it. result SHOULD be NULL, but if it isn't, I
    // release it.
    if (result != NULL && PyErr_Occurred()) {
        Py_DECREF(result);
        result = NULL;
    }
    // A callback returning NULL without setting an error indicates that we
    // should return None
    else if (result == NULL && !PyErr_Occurred()) {
        result = Py_None;
        Py_INCREF(result);
    }

    if (result == NULL) {
        // error
        goto done;
    }

done:
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY);
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY);

    return result;
}

static PyObject* callback_state_index_intrinsics(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_state_index_intrinsics(
        i,
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );

    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_intrinsics(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        state_index_intrinsics,
        self,
        args,
        kwargs,
        true,
        "icam_intrinsics"
    );
}

static PyObject* callback_num_states_intrinsics(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_num_states_intrinsics(
        Ncameras_intrinsics,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_intrinsics(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_states_intrinsics,
        self,
        args,
        kwargs,
        true,
        NULL
    );
}

static PyObject* callback_state_index_extrinsics(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_state_index_extrinsics(
        i,
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_extrinsics(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        state_index_extrinsics,
        self,
        args,
        kwargs,
        true,
        "icam_extrinsics"
    );
}

static PyObject* callback_num_states_extrinsics(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index =
        mrcal_num_states_extrinsics(Ncameras_extrinsics, problem_selections);
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_extrinsics(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_states_extrinsics,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_state_index_frames(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_state_index_frames(
        i,
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_frames(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        state_index_frames,
        self,
        args,
        kwargs,
        true,
        "iframe"
    );
}

static PyObject* callback_num_states_frames(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_num_states_frames(Nframes, problem_selections);
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_frames(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_states_frames,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_state_index_points(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_state_index_points(
        i,
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_points(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        state_index_points,
        self,
        args,
        kwargs,
        true,
        "i_point"
    );
}

static PyObject* callback_num_states_points(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index =
        mrcal_num_states_points(Npoints, Npoints_fixed, problem_selections);
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_points(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_states_points,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_state_index_calobject_warp(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_state_index_calobject_warp(
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_calobject_warp(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        state_index_calobject_warp,
        self,
        args,
        kwargs,
        true,
        NULL
    );
}

static PyObject* callback_num_states_calobject_warp(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_num_states_calobject_warp(
        problem_selections,
        Nobservations_board
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_calobject_warp(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_states_calobject_warp,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_num_states(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_num_states(
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(num_states, self, args, kwargs, true, NULL);
}

static PyObject* callback_num_intrinsics_optimization_params(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index =
        mrcal_num_intrinsics_optimization_params(problem_selections, lensmodel);
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_intrinsics_optimization_params(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_intrinsics_optimization_params,
        self,
        args,
        kwargs,
        true,
        NULL
    );
}

static PyObject* callback_measurement_index_boards(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = -1;

    if (calibration_object_width_n > 0 && calibration_object_height_n > 0) {
        index = mrcal_measurement_index_boards(
            i,
            Nobservations_board,
            Nobservations_point,
            calibration_object_width_n,
            calibration_object_height_n
        );
    }
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* measurement_index_boards(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        measurement_index_boards,
        self,
        args,
        kwargs,
        false,
        "i_observation_board"
    );
}

static PyObject* callback_num_measurements_boards(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = 0;

    if (calibration_object_width_n > 0 && calibration_object_height_n > 0) {
        index = mrcal_num_measurements_boards(
            Nobservations_board,
            calibration_object_width_n,
            calibration_object_height_n
        );
    }
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_boards(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_measurements_boards,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_measurement_index_points(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_measurement_index_points(
        i,
        Nobservations_board,
        Nobservations_point,
        calibration_object_width_n,
        calibration_object_height_n
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* measurement_index_points(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        measurement_index_points,
        self,
        args,
        kwargs,
        false,
        "i_observation_point"
    );
}

static PyObject* callback_measurement_index_points_triangulated(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    // VERY similar to callback_num_measurements_points_triangulated() and
    // callback_measurement_index_regularization() and maybe others. Please
    // consolidate
    int N = 0;
    if (!IS_NULL(indices_point_triangulated_camintrinsics_camextrinsics)) {
        N = PyArray_DIM(
            indices_point_triangulated_camintrinsics_camextrinsics,
            0
        );
    }

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        N <= 0 ? 0
               : fill_c_observations_point_triangulated(
                     c_observations_point_triangulated,
                     NULL,
                     NULL,
                     NULL,
                     indices_point_triangulated_camintrinsics_camextrinsics
                 );
    if (Nobservations_point_triangulated < 0) {
        BARF("Error parsing triangulated points");
        return NULL;
    }

    int index = mrcal_measurement_index_points_triangulated(
        i,
        Nobservations_board,
        Nobservations_point,
        c_observations_point_triangulated,
        Nobservations_point_triangulated,
        calibration_object_width_n,
        calibration_object_height_n
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* measurement_index_points_triangulated(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        measurement_index_points_triangulated,
        self,
        args,
        kwargs,
        false,
        "i_point_triangulated"
    );
}

static PyObject* callback_num_measurements_points(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_num_measurements_points(Nobservations_point);
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_points(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_measurements_points,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_num_measurements_points_triangulated(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    // VERY similar to callback_measurement_index_regularization(). Please
    // consolidate
    int N = 0;
    if (!IS_NULL(indices_point_triangulated_camintrinsics_camextrinsics)) {
        N = PyArray_DIM(
            indices_point_triangulated_camintrinsics_camextrinsics,
            0
        );
    }

    if (N == 0) {
        return PyLong_FromLong(0);
    }

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        fill_c_observations_point_triangulated(
            c_observations_point_triangulated,
            NULL,
            NULL,
            NULL,
            indices_point_triangulated_camintrinsics_camextrinsics
        );
    if (Nobservations_point_triangulated < 0) {
        BARF("Error parsing triangulated points");
        return NULL;
    }
    int index = mrcal_num_measurements_points_triangulated(
        c_observations_point_triangulated,
        Nobservations_point_triangulated
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_points_triangulated(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_measurements_points_triangulated,
        self,
        args,
        kwargs,
        false,
        NULL
    );
}

static PyObject* callback_measurement_index_regularization(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    // VERY similar to callback_num_measurements_points_triangulated(). Please
    // consolidate
    int N = 0;
    if (!IS_NULL(indices_point_triangulated_camintrinsics_camextrinsics)) {
        N = PyArray_DIM(
            indices_point_triangulated_camintrinsics_camextrinsics,
            0
        );
    }

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        N <= 0 ? 0
               : fill_c_observations_point_triangulated(
                     c_observations_point_triangulated,
                     NULL,
                     NULL,
                     NULL,
                     indices_point_triangulated_camintrinsics_camextrinsics
                 );
    if (Nobservations_point_triangulated < 0) {
        BARF("Error parsing triangulated points");
        return NULL;
    }
    int index = mrcal_measurement_index_regularization(
        c_observations_point_triangulated,
        Nobservations_point_triangulated,
        calibration_object_width_n,
        calibration_object_height_n,
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        Nobservations_point,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* measurement_index_regularization(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        measurement_index_regularization,
        self,
        args,
        kwargs,
        true,
        NULL
    );
}

static PyObject* callback_num_measurements_regularization(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    int index = mrcal_num_measurements_regularization(
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_regularization(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_measurements_regularization,
        self,
        args,
        kwargs,
        true,
        NULL
    );
}

static PyObject* callback_num_measurements(
    int i,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning \
    "triangulated-solve: add tests to the num_measurements_..., state_index_... ..."
#endif

    mrcal_observation_point_triangulated_t* observations_point_triangulated =
        NULL;
    int Nobservations_point_triangulated = 0;

    int N = 0;

    if (indices_point_triangulated_camintrinsics_camextrinsics != NULL) {
        N = PyArray_DIM(
            indices_point_triangulated_camintrinsics_camextrinsics,
            0
        );
    } else {
        // No triangulated points. No error. I have N = 0 in this path
    }

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    if (N > 0) {
        Nobservations_point_triangulated =
            fill_c_observations_point_triangulated(
                c_observations_point_triangulated,
                NULL,
                NULL,
                NULL,
                indices_point_triangulated_camintrinsics_camextrinsics
            );
        if (Nobservations_point_triangulated < 0) {
            BARF("Error parsing triangulated points");
            return NULL;
        }
        observations_point_triangulated = c_observations_point_triangulated;
    }
    int index = mrcal_num_measurements(
        Nobservations_board,
        Nobservations_point,
        observations_point_triangulated,
        Nobservations_point_triangulated,
        calibration_object_width_n,
        calibration_object_height_n,
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        problem_selections,
        lensmodel
    );
    if (index >= 0) {
        return PyLong_FromLong(index);
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        num_measurements,
        self,
        args,
        kwargs,
        true,
        NULL
    );
}

static PyObject* callback_corresponding_icam_extrinsics(
    int icam_intrinsics,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: barf if we have any triangulated points"
#endif

    if (icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics) {
        BARF(
            "The given icam_intrinsics=%d is out of bounds. Must be >= 0 and < "
            "%d",
            icam_intrinsics,
            Ncameras_intrinsics
        );
        return NULL;
    }

    int icam_extrinsics;

    if (Nobservations_board > 0 &&
        indices_frame_camintrinsics_camextrinsics == NULL) {
        BARF(
            "Have Nobservations_board > 0, but "
            "indices_frame_camintrinsics_camextrinsics == NULL. Some required "
            "arguments missing?"
        );
        return NULL;
    }
    mrcal_observation_board_t c_observations_board[Nobservations_board];
    fill_c_observations_board(  // output
        c_observations_board,
        // input
        Nobservations_board,
        indices_frame_camintrinsics_camextrinsics
    );

    if (Nobservations_point > 0) {
        if (indices_point_camintrinsics_camextrinsics == NULL) {
            BARF(
                "Have Nobservations_point > 0, but "
                "indices_point_camintrinsics_camextrinsics == NULL. Some "
                "required arguments missing?"
            );
            return NULL;
        }
        if (observations_point == NULL) {
            BARF(
                "Have Nobservations_point > 0, but observations_point == NULL. "
                "Some required arguments missing?"
            );
            return NULL;
        }
    }
    mrcal_observation_point_t c_observations_point[Nobservations_point];
    fill_c_observations_point(  // output
        c_observations_point,
        // input
        Nobservations_point,
        indices_point_camintrinsics_camextrinsics
    );

    if (!mrcal_corresponding_icam_extrinsics(
            &icam_extrinsics,

            icam_intrinsics,
            Ncameras_intrinsics,
            Ncameras_extrinsics,
            Nobservations_board,
            c_observations_board,
            Nobservations_point,
            c_observations_point
        )) {
        BARF("Error calling mrcal_corresponding_icam_extrinsics()");
        return NULL;
    }

    return PyLong_FromLong(icam_extrinsics);
}
static PyObject* corresponding_icam_extrinsics(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        corresponding_icam_extrinsics,
        self,
        args,
        kwargs,
        false,
        "icam_intrinsics"
    );
}

static PyObject* callback_decode_observation_indices_points_triangulated(
    int imeasurement,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    int Nobservations_board,
    int Nobservations_point,
    int calibration_object_width_n,
    int calibration_object_height_n,
    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
    const PyArrayObject* observations_point,
    const mrcal_lensmodel_t* lensmodel,
    mrcal_problem_selections_t problem_selections
) {
    if (indices_point_triangulated_camintrinsics_camextrinsics == NULL) {
        BARF("No triangulated points in this solve. Nothing to decode");
        return NULL;
    }

    int N =
        PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    if (N <= 0) {
        BARF("No triangulated points in this solve. Nothing to decode");
        return NULL;
    }

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        fill_c_observations_point_triangulated(
            c_observations_point_triangulated,
            NULL,
            NULL,
            NULL,
            indices_point_triangulated_camintrinsics_camextrinsics
        );
    if (Nobservations_point_triangulated < 0) {
        BARF("Error parsing triangulated points");
        return NULL;
    }

    mrcal_observation_point_triangulated_t* observations_point_triangulated =
        c_observations_point_triangulated;

    int iobservation0;
    int iobservation1;
    int iobservation_point0;
    int Nobservations_this_point;
    int Nmeasurements_this_point;
    int ipoint;

    bool result = mrcal_decode_observation_indices_points_triangulated(
        &iobservation0,
        &iobservation1,
        &iobservation_point0,
        &Nobservations_this_point,
        &Nmeasurements_this_point,
        &ipoint,

        imeasurement,
        observations_point_triangulated,
        Nobservations_point_triangulated
    );
    if (!result) {
        BARF("Error decoding indices");
        return NULL;
    }

    return Py_BuildValue(
        "{sisisisisisi}",
        "iobservation0",
        iobservation0,
        "iobservation1",
        iobservation1,
        "iobservation_point0",
        iobservation_point0,
        "Nobservations_this_point",
        Nobservations_this_point,
        "Nmeasurements_this_point",
        Nmeasurements_this_point,
        "ipoint",
        ipoint
    );
}
static PyObject* decode_observation_indices_points_triangulated(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return STATE_INDEX_GENERIC(
        decode_observation_indices_points_triangulated,
        self,
        args,
        kwargs,
        false,
        "imeasurement"
    );
}

static PyObject* _pack_unpack_state(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    bool pack
) {
    // This is VERY similar to state_index_generic(). Please consolidate
    PyObject* result = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    PyArrayObject* b = NULL;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes = -1;
    int Npoints = -1;
    int Nobservations_board = -1;
    int Nobservations_point = -1;

    char* keywords[] = {
        "b",
        "Ncameras_intrinsics",
        "Ncameras_extrinsics",
        "Nframes",
        "Npoints",
        "Nobservations_board",
        "Nobservations_point",
        OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
            OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST) NULL
    };

#define UNPACK_STATE "unpack_state"
    char arg_string[] =
        "O&"
        "|$"  // everything is kwarg-only and optional. I apply logic down the
              // line to get what I need
        "iiiiii" OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
            OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE) ":mrcal." UNPACK_STATE;
    if (pack) {
        arg_string[strlen(arg_string) - strlen(UNPACK_STATE)] = '\0';
        strcat(arg_string, "pack_state");
    }
#undef UNPACK_STATE

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            arg_string,
            keywords,

            PyArray_Converter,
            &b,
            &Ncameras_intrinsics,
            &Ncameras_extrinsics,
            &Nframes,
            &Npoints,
            &Nobservations_board,
            &Nobservations_point,
            OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL
        )) {
        goto done;
    }

    if (lensmodel == NULL) {
        BARF("The 'lensmodel' argument is required");
        goto done;
    }

    mrcal_lensmodel_t mrcal_lensmodel;
    if (!parse_lensmodel_from_arg(&mrcal_lensmodel, lensmodel)) {
        goto done;
    }

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if (Ncameras_intrinsics < 0) {
        Ncameras_intrinsics =
            IS_NULL(intrinsics) ? 0 : PyArray_DIMS(intrinsics)[0];
    }
    if (Ncameras_extrinsics < 0) {
        Ncameras_extrinsics = IS_NULL(extrinsics_rt_fromref)
                                  ? 0
                                  : PyArray_DIMS(extrinsics_rt_fromref)[0];
    }
    if (Nframes < 0) {
        Nframes =
            IS_NULL(frames_rt_toref) ? 0 : PyArray_DIMS(frames_rt_toref)[0];
    }
    if (Npoints < 0) {
        Npoints = IS_NULL(points) ? 0 : PyArray_DIMS(points)[0];
    }
    if (Nobservations_board < 0) {
        Nobservations_board = IS_NULL(observations_board)
                                  ? 0
                                  : PyArray_DIMS(observations_board)[0];
    }
    if (Nobservations_point < 0) {
        Nobservations_point = IS_NULL(observations_point)
                                  ? 0
                                  : PyArray_DIMS(observations_point)[0];
    }

    mrcal_problem_selections_t problem_selections =
        CONSTRUCT_PROBLEM_SELECTIONS();

    if (PyArray_TYPE(b) != NPY_DOUBLE) {
        BARF("The given array MUST have values of type 'float'");
        goto done;
    }

    if (!PyArray_IS_C_CONTIGUOUS(b)) {
        BARF("The given array MUST be a C-style contiguous array");
        goto done;
    }

    int ndim = PyArray_NDIM(b);
    npy_intp* dims = PyArray_DIMS(b);
    if (ndim < 1) {
        BARF("The given array MUST have at least one dimension");
        goto done;
    }

    int Nstate = mrcal_num_states(
        Ncameras_intrinsics,
        Ncameras_extrinsics,
        Nframes,
        Npoints,
        Npoints_fixed,
        Nobservations_board,
        problem_selections,
        &mrcal_lensmodel
    );

    if (dims[ndim - 1] != Nstate) {
        BARF(
            "The given array MUST have last dimension of size Nstate=%d; "
            "instead got %ld",
            Nstate,
            dims[ndim - 1]
        );
        goto done;
    }

    double* x = (double*)PyArray_DATA(b);
    if (pack) {
        for (int i = 0; i < PyArray_SIZE(b) / Nstate; i++) {
            mrcal_pack_solver_state_vector(
                x,
                Ncameras_intrinsics,
                Ncameras_extrinsics,
                Nframes,
                Npoints,
                Npoints_fixed,
                Nobservations_board,
                problem_selections,
                &mrcal_lensmodel
            );
            x = &x[Nstate];
        }
    } else {
        for (int i = 0; i < PyArray_SIZE(b) / Nstate; i++) {
            mrcal_unpack_solver_state_vector(
                x,
                Ncameras_intrinsics,
                Ncameras_extrinsics,
                Nframes,
                Npoints,
                Npoints_fixed,
                Nobservations_board,
                problem_selections,
                &mrcal_lensmodel
            );
            x = &x[Nstate];
        }
    }

    Py_INCREF(Py_None);
    result = Py_None;

done:
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY);
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY);

    Py_XDECREF(b);
    return result;
}
static PyObject* pack_state(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return _pack_unpack_state(self, args, kwargs, true);
}
static PyObject* unpack_state(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs
) {
    return _pack_unpack_state(self, args, kwargs, false);
}

// LENSMODEL_ONE_ARGUMENTS followed by these
#define RECTIFIED_RESOLUTION_ARGUMENTS(_) \
    _(R_cam0_rect0,                       \
      PyArrayObject*,                     \
      NULL,                               \
      "O&",                               \
      PyArray_Converter COMMA,            \
      R_cam0_rect0,                       \
      NPY_DOUBLE,                         \
      {3 COMMA 3})
static bool rectified_resolution_validate_args(
    RECTIFIED_RESOLUTION_ARGUMENTS(ARG_LIST_DEFINE) void* dummy
    __attribute__((unused))
) {
    RECTIFIED_RESOLUTION_ARGUMENTS(CHECK_LAYOUT);
    return true;
done:
    return false;
}

static PyObject* _rectified_resolution(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    PyObject* result = NULL;

    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, );
    RECTIFIED_RESOLUTION_ARGUMENTS(ARG_DEFINE);

    // input and output
    double pixels_per_deg_az;
    double pixels_per_deg_el;

    // input
    mrcal_lensmodel_t mrcal_lensmodel;
    mrcal_point2_t azel_fov_deg;
    mrcal_point2_t azel0_deg;
    char* rectification_model_string;
    mrcal_lensmodel_t rectification_model;

    char* keywords[] = {
        LENSMODEL_ONE_ARGUMENTS(NAMELIST, )
            RECTIFIED_RESOLUTION_ARGUMENTS(NAMELIST) "az_fov_deg",
        "el_fov_deg",
        "az0_deg",
        "el0_deg",
        "pixels_per_deg_az",
        "pixels_per_deg_el",
        "rectification_model",
        NULL
    };
    // This function is internal, so EVERYTHING is required
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            LENSMODEL_ONE_ARGUMENTS(PARSECODE, )
                RECTIFIED_RESOLUTION_ARGUMENTS(PARSECODE
                ) "dddddds:mrcal.rectified_resolution",

            keywords,

            LENSMODEL_ONE_ARGUMENTS(PARSEARG, )
                    RECTIFIED_RESOLUTION_ARGUMENTS(PARSEARG) &
                azel_fov_deg.x,
            &azel_fov_deg.y,
            &azel0_deg.x,
            &azel0_deg.y,
            &pixels_per_deg_az,
            &pixels_per_deg_el,
            &rectification_model_string
        )) {
        goto done;
    }

    if (!lensmodel_one_validate_args(
            &mrcal_lensmodel,
            LENSMODEL_ONE_ARGUMENTS(
                ARG_LIST_CALL,
            ) true /* DO check the layout */
        )) {
        goto done;
    }

    if (!parse_lensmodel_from_arg(
            &rectification_model,
            rectification_model_string
        )) {
        goto done;
    }

    if (!rectified_resolution_validate_args(
            RECTIFIED_RESOLUTION_ARGUMENTS(ARG_LIST_CALL) NULL
        )) {
        goto done;
    }

    if (!mrcal_rectified_resolution(
            &pixels_per_deg_az,
            &pixels_per_deg_el,

            // input
            &mrcal_lensmodel,
            PyArray_DATA(intrinsics),
            &azel_fov_deg,
            &azel0_deg,
            PyArray_DATA(R_cam0_rect0),
            rectification_model.type
        )) {
        BARF("mrcal_rectified_resolution() failed!");
        goto done;
    }

    result = Py_BuildValue("(dd)", pixels_per_deg_az, pixels_per_deg_el);

done:

    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, );
    RECTIFIED_RESOLUTION_ARGUMENTS(FREE_PYARRAY);

    return result;
}

// LENSMODEL_ONE_ARGUMENTS followed by these
#define RECTIFIED_SYSTEM_ARGUMENTS(_) \
    _(rt_cam0_ref,                    \
      PyArrayObject*,                 \
      NULL,                           \
      "O&",                           \
      PyArray_Converter COMMA,        \
      rt_cam0_ref,                    \
      NPY_DOUBLE,                     \
      {6})                            \
    _(rt_cam1_ref,                    \
      PyArrayObject*,                 \
      NULL,                           \
      "O&",                           \
      PyArray_Converter COMMA,        \
      rt_cam1_ref,                    \
      NPY_DOUBLE,                     \
      {6})
static bool rectified_system_validate_args(
    RECTIFIED_SYSTEM_ARGUMENTS(ARG_LIST_DEFINE) void* dummy
    __attribute__((unused))
) {
    RECTIFIED_SYSTEM_ARGUMENTS(CHECK_LAYOUT);
    return true;
done:
    return false;
}

static PyObject* _rectified_system(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    PyObject* result = NULL;

    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, 0);
    RECTIFIED_SYSTEM_ARGUMENTS(ARG_DEFINE);

    // output
    unsigned int imagersize_rectified[2];
    PyArrayObject* fxycxy_rectified = NULL;
    PyArrayObject* rt_rect0_ref = NULL;
    double baseline;

    // input and output
    double pixels_per_deg_az;
    double pixels_per_deg_el;
    mrcal_point2_t azel_fov_deg;
    mrcal_point2_t azel0_deg;

    // input
    mrcal_lensmodel_t mrcal_lensmodel0;
    char* rectification_model_string;
    mrcal_lensmodel_t rectification_model;

    bool az0_deg_autodetect = false;
    bool el0_deg_autodetect = false;
    bool az_fov_deg_autodetect = false;
    bool el_fov_deg_autodetect = false;

    fxycxy_rectified =
        (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){4}), NPY_DOUBLE);
    if (NULL == fxycxy_rectified) {
        BARF("Couldn't allocate fxycxy_rectified");
        goto done;
    }
    rt_rect0_ref =
        (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){6}), NPY_DOUBLE);
    if (NULL == rt_rect0_ref) {
        BARF("Couldn't allocate rt_rect0_ref");
        goto done;
    }

    char* keywords[] = {
        LENSMODEL_ONE_ARGUMENTS(NAMELIST, 0)
            RECTIFIED_SYSTEM_ARGUMENTS(NAMELIST) "az_fov_deg",
        "el_fov_deg",
        "az0_deg",
        "el0_deg",
        "pixels_per_deg_az",
        "pixels_per_deg_el",
        "rectification_model",
        NULL
    };
    // This function is internal, so EVERYTHING is required
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            LENSMODEL_ONE_ARGUMENTS(PARSECODE, 0)
                RECTIFIED_SYSTEM_ARGUMENTS(PARSECODE
                ) "dddddds:mrcal.rectified_system",

            keywords,

            LENSMODEL_ONE_ARGUMENTS(PARSEARG, 0)
                    RECTIFIED_SYSTEM_ARGUMENTS(PARSEARG) &
                azel_fov_deg.x,
            &azel_fov_deg.y,
            &azel0_deg.x,
            &azel0_deg.y,
            &pixels_per_deg_az,
            &pixels_per_deg_el,
            &rectification_model_string
        )) {
        goto done;
    }

    if (azel0_deg.x > 1e6) {
        az0_deg_autodetect = true;
    }

    if (!lensmodel_one_validate_args(
            &mrcal_lensmodel0,
            LENSMODEL_ONE_ARGUMENTS(
                ARG_LIST_CALL,
                0
            ) true /* DO check the layout */
        )) {
        goto done;
    }

    if (!parse_lensmodel_from_arg(
            &rectification_model,
            rectification_model_string
        )) {
        goto done;
    }

    if (!rectified_system_validate_args(RECTIFIED_SYSTEM_ARGUMENTS(ARG_LIST_CALL
        ) NULL)) {
        goto done;
    }

    if (!mrcal_rectified_system(  // output
            imagersize_rectified,
            PyArray_DATA(fxycxy_rectified),
            PyArray_DATA(rt_rect0_ref),
            &baseline,

            // input, output
            &pixels_per_deg_az,
            &pixels_per_deg_el,

            // input, output
            &azel_fov_deg,
            &azel0_deg,

            // input
            &mrcal_lensmodel0,
            PyArray_DATA(intrinsics0),
            PyArray_DATA(rt_cam0_ref),
            PyArray_DATA(rt_cam1_ref),
            rectification_model.type,
            az0_deg_autodetect,
            el0_deg_autodetect,
            az_fov_deg_autodetect,
            el_fov_deg_autodetect
        )) {
        BARF("mrcal_rectified_system() failed!");
        goto done;
    }

    result = Py_BuildValue(
        "(ddiiOOddddd)",
        pixels_per_deg_az,
        pixels_per_deg_el,
        imagersize_rectified[0],
        imagersize_rectified[1],
        fxycxy_rectified,
        rt_rect0_ref,
        baseline,
        azel_fov_deg.x,
        azel_fov_deg.y,
        azel0_deg.x,
        azel0_deg.y
    );

done:

    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, 0);
    RECTIFIED_SYSTEM_ARGUMENTS(FREE_PYARRAY);

    Py_XDECREF(fxycxy_rectified);
    Py_XDECREF(rt_rect0_ref);

    return result;
}

// LENSMODEL_ONE_ARGUMENTS followed by these
#define RECTIFICATION_MAPS_ARGUMENTS(_) \
    _(r_cam0_ref,                       \
      PyArrayObject*,                   \
      NULL,                             \
      "O&",                             \
      PyArray_Converter COMMA,          \
      r_cam0_ref,                       \
      NPY_DOUBLE,                       \
      {3})                              \
    _(r_cam1_ref,                       \
      PyArrayObject*,                   \
      NULL,                             \
      "O&",                             \
      PyArray_Converter COMMA,          \
      r_cam1_ref,                       \
      NPY_DOUBLE,                       \
      {3})                              \
    _(r_rect0_ref,                      \
      PyArrayObject*,                   \
      NULL,                             \
      "O&",                             \
      PyArray_Converter COMMA,          \
      r_rect0_ref,                      \
      NPY_DOUBLE,                       \
      {3})                              \
    _(rectification_maps,               \
      PyArrayObject*,                   \
      NULL,                             \
      "O&",                             \
      PyArray_Converter COMMA,          \
      rectification_maps,               \
      NPY_FLOAT,                        \
      {2 COMMA - 1 COMMA - 1 COMMA 2})

static bool rectification_maps_validate_args(
    RECTIFICATION_MAPS_ARGUMENTS(ARG_LIST_DEFINE) void* dummy
    __attribute__((unused))
) {
    RECTIFICATION_MAPS_ARGUMENTS(CHECK_LAYOUT);
    return true;
done:
    return false;
}

static PyObject* _rectification_maps(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    PyObject* result = NULL;

    unsigned int imagersize_rectified[2];

    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, 0);
    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, 1);
    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, _rectified);
    RECTIFICATION_MAPS_ARGUMENTS(ARG_DEFINE);

    // input
    mrcal_lensmodel_t mrcal_lensmodel0;
    mrcal_lensmodel_t mrcal_lensmodel1;
    mrcal_lensmodel_t mrcal_lensmodel_rectified;

    char* keywords[] = {LENSMODEL_ONE_ARGUMENTS(NAMELIST, 0)
                            LENSMODEL_ONE_ARGUMENTS(NAMELIST, 1)
                                LENSMODEL_ONE_ARGUMENTS(NAMELIST, _rectified)
                                    RECTIFICATION_MAPS_ARGUMENTS(NAMELIST) NULL
    };
    // This function is internal, so EVERYTHING is required
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            LENSMODEL_ONE_ARGUMENTS(PARSECODE, 0)
                LENSMODEL_ONE_ARGUMENTS(PARSECODE, 1)
                    LENSMODEL_ONE_ARGUMENTS(PARSECODE, _rectified)
                        RECTIFICATION_MAPS_ARGUMENTS(PARSECODE
                        ) ":mrcal.rectification_maps",

            keywords,

            LENSMODEL_ONE_ARGUMENTS(PARSEARG, 0)
                LENSMODEL_ONE_ARGUMENTS(PARSEARG, 1)
                    LENSMODEL_ONE_ARGUMENTS(PARSEARG, _rectified)
                        RECTIFICATION_MAPS_ARGUMENTS(PARSEARG) NULL
        )) {
        goto done;
    }

    if (!lensmodel_one_validate_args(
            &mrcal_lensmodel0,
            LENSMODEL_ONE_ARGUMENTS(
                ARG_LIST_CALL,
                0
            ) true /* DO check the layout */
        )) {
        goto done;
    }
    if (!lensmodel_one_validate_args(
            &mrcal_lensmodel1,
            LENSMODEL_ONE_ARGUMENTS(
                ARG_LIST_CALL,
                1
            ) true /* DO check the layout */
        )) {
        goto done;
    }
    if (!lensmodel_one_validate_args(
            &mrcal_lensmodel_rectified,
            LENSMODEL_ONE_ARGUMENTS(
                ARG_LIST_CALL,
                _rectified
            ) true /* DO check the layout */
        )) {
        goto done;
    }

    if (!rectification_maps_validate_args(
            RECTIFICATION_MAPS_ARGUMENTS(ARG_LIST_CALL) NULL
        )) {
        goto done;
    }

    // rectification_maps has shape (Ncameras=2, Nel, Naz, Nxy=2)
    imagersize_rectified[0] = PyArray_DIMS(rectification_maps)[2];
    imagersize_rectified[1] = PyArray_DIMS(rectification_maps)[1];

    if (!mrcal_rectification_maps(  // output
            PyArray_DATA(rectification_maps),

            // input
            &mrcal_lensmodel0,
            PyArray_DATA(intrinsics0),
            PyArray_DATA(r_cam0_ref),

            &mrcal_lensmodel1,
            PyArray_DATA(intrinsics1),
            PyArray_DATA(r_cam1_ref),

            mrcal_lensmodel_rectified.type,
            PyArray_DATA(intrinsics_rectified),
            imagersize_rectified,
            PyArray_DATA(r_rect0_ref)
        )) {
        BARF("mrcal_rectification_maps() failed!");
        goto done;
    }

    Py_INCREF(Py_None);
    result = Py_None;

done:

    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, 0);
    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, 1);
    RECTIFICATION_MAPS_ARGUMENTS(FREE_PYARRAY);

    return result;
}

static bool callback_sensor_link_C(
    const uint16_t idx_to,
    const uint16_t idx_from,
    void* cookie
) {
    PyObject* callback_sensor_link = (PyObject*)cookie;

    PyObject* py_idx_to = NULL;
    PyObject* py_idx_from = NULL;
    PyObject* result = NULL;

    py_idx_to = PyLong_FromLong(idx_to);
    if (py_idx_to == NULL) {
        goto done;
    }

    py_idx_from = PyLong_FromLong(idx_from);
    if (py_idx_from == NULL) {
        goto done;
    }

    result = PyObject_CallFunctionObjArgs(
        callback_sensor_link,
        py_idx_to,
        py_idx_from,
        NULL
    );

done:
    Py_XDECREF(py_idx_to);
    Py_XDECREF(py_idx_from);

    if (result == NULL) {
        return false;
    }

    Py_DECREF(result);
    return true;
}
static PyObject* traverse_sensor_links(
    PyObject* NPY_UNUSED(self),
    PyObject* args,
    PyObject* kwargs
) {
    PyObject* result = NULL;

    int Nsensors = 0;
    PyArrayObject* connectivity_matrix = NULL;
    PyObject* callback_sensor_link = NULL;

    char* keywords[] = {"connectivity_matrix", "callback_sensor_link", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "$O&O:mrcal.traverse_sensor_links",
            keywords,
            PyArray_Converter,
            &connectivity_matrix,
            &callback_sensor_link
        )) {
        goto done;
    }

    if (PyArray_NDIM(connectivity_matrix) != 2) {
        BARF("The connectivity_matrix must have 2 dimensions");
        goto done;
    }
    Nsensors = PyArray_DIMS(connectivity_matrix)[1];

    if (!_check_layout(
            "connectivity_matrix",
            connectivity_matrix,
            NPY_UINT16,
            "NPY_UINT16",
            (int[]){Nsensors, Nsensors},
            2,
            "{Nsensors,Nsensors}",
            false
        )) {
        goto done;
    }

    if (!PyCallable_Check(callback_sensor_link)) {
        BARF("callback_sensor_link is not callable");
        goto done;
    }

    if (Nsensors > UINT16_MAX) {
        BARF("Nsensors=%d doesn't fit into a uint16_t", Nsensors);
        goto done;
    }

    // Arguments are good. Let's massage them to do the right thing

    {
        // We reconstruct just the upper triangle of the connectivity_matrix
        uint16_t connectivity_matrix_upper[Nsensors * (Nsensors - 1) / 2];
        int k = 0;
        for (int i = 0; i < Nsensors; i++) {
            for (int j = i + 1; j < Nsensors; j++) {
                connectivity_matrix_upper[k++] =
                    *(uint16_t*)PyArray_GETPTR2(connectivity_matrix, i, j);
            }
        }

        if (!mrcal_traverse_sensor_links(
                Nsensors,
                connectivity_matrix_upper,
                &callback_sensor_link_C,
                callback_sensor_link
            )) {
            if (!PyErr_Occurred()) {
                BARF("mrcal_traverse_sensor_links() failed");
            }
            goto done;
        }
    }

    Py_INCREF(Py_None);
    result = Py_None;

done:

    Py_XDECREF(connectivity_matrix);
    return result;
}

static const char state_index_intrinsics_docstring[] = R"(
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
)";
static const char state_index_extrinsics_docstring[] = R"(
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

)";

static const char state_index_frames_docstring[] = R"(
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

)";

static const char state_index_points_docstring[] = R"(
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

)";

static const char state_index_calobject_warp_docstring[] = R"(
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

)";

static const char num_states_intrinsics_docstring[] = R"(
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

)";

static const char num_states_extrinsics_docstring[] = R"(
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

)";

static const char num_states_frames_docstring[] = R"(
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

)";

static const char num_states_points_docstring[] = R"(
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

)";

static const char num_states_calobject_warp_docstring[] = R"(
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

)";

static const char pack_state_docstring[] = R"(
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

)";

static const char unpack_state_docstring[] = R"(
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

)";

static const char measurement_index_boards_docstring[] = R"(
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

)";

static const char measurement_index_points_docstring[] = R"(
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

)";

static const char measurement_index_points_triangulated_docstring[] = R"(
NOT DONE YET; fill this in
)";

static const char measurement_index_regularization_docstring[] = R"(
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

)";

static const char num_measurements_boards_docstring[] = R"(
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

)";

static const char num_measurements_points_docstring[] = R"(
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

)";

static const char num_measurements_points_triangulated_docstring[] = R"(
NOT DONE YET; fill this in
)";

static const char num_measurements_regularization_docstring[] = R"(
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

)";

static const char num_measurements_docstring[] = R"(
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

)";

static const char corresponding_icam_extrinsics_docstring[] = R"(
Return the icam_extrinsics corresponding to a given icam_intrinsics

SYNOPSIS

    m = mrcal.cameramodel('xxx.cameramodel')

    optimization_inputs = m.optimization_inputs()

    icam_intrinsics = m.icam_intrinsics()

    icam_extrinsics = \
        mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                            **optimization_inputs)

    if icam_extrinsics >= 0:
        extrinsics_rt_fromref_at_calibration_time = \
            optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics]
    else:
        extrinsics_rt_fromref_at_calibration_time = \
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

)";

static const char decode_observation_indices_points_triangulated_docstring[] =
    R"(
NOT DONE YET; fill this in
)";

static const char num_states_docstring[] = R"(
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

)";

static const char num_intrinsics_optimization_params_docstring[] = R"(
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

)";

static const char optimize_docstring[] = R"(
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

)";

static const char optimizer_callback_docstring[] = R"(
Call the optimization callback function

SYNOPSIS

    model               = mrcal.cameramodel('xxx.cameramodel')

    optimization_inputs = model.optimization_inputs()

    b_packed,x,J_packed,factorization = \
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

)";

static const char drt_ref_refperturbed__dbpacked_docstring[] = R"(
write this
)";

static const char lensmodel_metadata_and_config_docstring[] = R"(
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


)";

static const char lensmodel_num_params_docstring[] = R"(
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

)";

static const char supported_lensmodels_docstring[] = R"(
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

)";

static const char knots_for_splined_models_docstring[] = R"(
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

)";

static const char _rectified_resolution_docstring[] = R"(
Compute the resolution to be used for the rectified system

This is an internal function. You probably want mrcal.rectified_resolution(). See the
docs for that function for details.

)";

static const char _rectified_system_docstring[] = R"(
Build rectified models for stereo rectification

This is an internal function. You probably want mrcal.rectified_system(). See the
docs for that function for details.

)";

static const char _rectification_maps_docstring[] = R"(
Construct image transformation maps to make rectified images

This is an internal function. You probably want mrcal.rectification_maps(). See
the docs for that function for details.

)";

static const char traverse_sensor_links_docstring[] = R"(
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

    mrcal.traverse_sensor_links( \
        connectivity_matrix  = connectivity_matrix,
        callback_sensor_link = lambda idx_to, idx_from: \
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

)";

static PyMethodDef methods[] = {
    PYMETHODDEF_ENTRY(, optimize, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, optimizer_callback, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        drt_ref_refperturbed__dbpacked,
        METH_VARARGS | METH_KEYWORDS
    ),

    PYMETHODDEF_ENTRY(, state_index_intrinsics, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, state_index_extrinsics, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, state_index_frames, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, state_index_points, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        state_index_calobject_warp,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(, num_states_intrinsics, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, num_states_extrinsics, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, num_states_frames, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, num_states_points, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        num_states_calobject_warp,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(, num_states, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        num_intrinsics_optimization_params,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(, pack_state, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, unpack_state, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, measurement_index_boards, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, measurement_index_points, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        measurement_index_points_triangulated,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(
        ,
        measurement_index_regularization,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(, num_measurements_boards, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, num_measurements_points, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        num_measurements_points_triangulated,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(
        ,
        num_measurements_regularization,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(, num_measurements, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(
        ,
        corresponding_icam_extrinsics,
        METH_VARARGS | METH_KEYWORDS
    ),
    PYMETHODDEF_ENTRY(
        ,
        decode_observation_indices_points_triangulated,
        METH_VARARGS | METH_KEYWORDS
    ),

    PYMETHODDEF_ENTRY(, lensmodel_metadata_and_config, METH_VARARGS),
    PYMETHODDEF_ENTRY(, lensmodel_num_params, METH_VARARGS),
    PYMETHODDEF_ENTRY(, supported_lensmodels, METH_NOARGS),
    PYMETHODDEF_ENTRY(, knots_for_splined_models, METH_VARARGS),

    PYMETHODDEF_ENTRY(, _rectified_resolution, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, _rectified_system, METH_VARARGS | METH_KEYWORDS),
    PYMETHODDEF_ENTRY(, _rectification_maps, METH_VARARGS | METH_KEYWORDS),

    PYMETHODDEF_ENTRY(, traverse_sensor_links, METH_VARARGS | METH_KEYWORDS),
    {}
};
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: fill in the new xxxx.docstring"
#endif

static void _init_bindings_common(
    PyObject* module
) {
    Py_INCREF(&CHOLMOD_factorization_type);
    PyModule_AddObject(
        module,
        "CHOLMOD_factorization",
        (PyObject*)&CHOLMOD_factorization_type
    );
}

#define MODULE_DOCSTRING                                                       \
    "Low-level routines for core mrcal operations\n"                           \
    "\n"                                                                       \
    "This is the written-in-C Python extension module that underlies the "     \
    "routines in\n"                                                            \
    "mrcal.h. Most of the functions in this module (those prefixed with "      \
    "\"_\") are\n"                                                             \
    "not meant to be called directly, but have Python wrappers that should "   \
    "be used\n"                                                                \
    "instead.\n"                                                               \
    "\n"                                                                       \
    "All functions are exported into the mrcal module. So you can call these " \
    "via\n"                                                                    \
    "mrcal._mrcal.fff() or mrcal.fff(). The latter is preferred.\n"

static struct PyModuleDef module_def =
    {PyModuleDef_HEAD_INIT, "bindings", MODULE_DOCSTRING, -1, methods};

PyMODINIT_FUNC PyInit_bindings(
    void
) {
    if (PyType_Ready(&CHOLMOD_factorization_type) < 0) {
        return NULL;
    }

    PyObject* module = PyModule_Create(&module_def);

    _init_bindings_common(module);
    import_array();

    return module;
}
