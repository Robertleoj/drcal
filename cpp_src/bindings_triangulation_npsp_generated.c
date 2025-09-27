// THIS IS A GENERATED FILE. DO NOT MODIFY WITH CHANGES YOU WANT TO KEEP
// Generated on 2025-09-27 12:26:24 with   nps_pywrap_bindings/triangulation-genpywrap.py


#define FUNCTIONS(_) \
  _(_triangulate_geometric, "Internal geometric triangulation routine\n\nThis is the internals for drcal.triangulate_geometric(get_gradients = False). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulate_geometric_withgrad\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_geometric_withgrad, "Internal geometric triangulation routine (with gradients)\n\nThis is the internals for drcal.triangulate_geometric(get_gradients = True). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the gradients-returning function. The internal function that\n  skips those is _triangulate_geometric\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_l1, "Internal Lee-Civera L1 triangulation routine\n\nThis is the internals for drcal.triangulate_leecivera_l1(get_gradients = False). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulate_leecivera_l1_withgrad\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_l1_withgrad, "Internal Lee-Civera L1 triangulation routine (with gradients)\n\nThis is the internals for drcal.triangulate_leecivera_l1(get_gradients = True). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the gradients-returning function. The internal function that\n  skips those is _triangulate_leecivera_l1\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_linf, "Internal Lee-Civera L-infinity triangulation routine\n\nThis is the internals for drcal.triangulate_leecivera_linf(get_gradients = False). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulate_leecivera_linf_withgrad\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_linf_withgrad, "Internal Lee-Civera L-infinity triangulation routine (with gradients)\n\nThis is the internals for drcal.triangulate_leecivera_linf(get_gradients = True). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the gradients-returning function. The internal function that\n  skips those is _triangulate_leecivera_linf\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_mid2, "Internal Lee-Civera Mid2 triangulation routine\n\nThis is the internals for drcal.triangulate_leecivera_mid2(get_gradients = False). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulate_leecivera_mid2_withgrad\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_mid2_withgrad, "Internal Lee-Civera Mid2 triangulation routine (with gradients)\n\nThis is the internals for drcal.triangulate_leecivera_mid2(get_gradients = True). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the gradients-returning function. The internal function that\n  skips those is _triangulate_leecivera_mid2\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_wmid2, "Internal Lee-Civera wMid2 triangulation routine\n\nThis is the internals for drcal.triangulate_leecivera_wmid2(get_gradients = False). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulate_leecivera_wmid2_withgrad\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_leecivera_wmid2_withgrad, "Internal Lee-Civera wMid2 triangulation routine (with gradients)\n\nThis is the internals for drcal.triangulate_leecivera_wmid2(get_gradients = True). As a\nuser, please call THAT function, and see the docs for that function. The\ndifferences:\n\n- This is just the gradients-returning function. The internal function that\n  skips those is _triangulate_leecivera_wmid2\n\nA higher-level function drcal.triangulate() is also available for higher-level\nanalysis.\n\n") \
  _(_triangulate_lindstrom, "Internal lindstrom's triangulation routine\n\nThis is the internals for drcal.triangulate_lindstrom(). As a user, please call\nTHAT function, and see the docs for that function. The differences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulate_lindstrom_withgrad\n\n") \
  _(_triangulate_lindstrom_withgrad, "Internal lindstrom's triangulation routine\n\nThis is the internals for drcal.triangulate_lindstrom(). As a user, please call\nTHAT function, and see the docs for that function. The differences:\n\n- This is just the gradient-returning function. The internal function that skips those\n  is _triangulate_lindstrom\n\n") \
  _(_triangulated_error, "Internal triangulation routine used in the optimization loop\n\nThis is the internals for drcal.triangulated_error(). As a user, please call\nTHAT function, and see the docs for that function. The differences:\n\n- This is just the no-gradients function. The internal function that returns\n  gradients is _triangulated_error_withgrad\n\n") \
  _(_triangulated_error_withgrad, "Internal triangulation routine used in the optimization loop\n\nThis is the internals for drcal.triangulated_error(). As a user, please call\nTHAT function, and see the docs for that function. The differences:\n\n- This is just the gradient-returning function. The internal function that skips those\n  is _triangulated_error\n\n")


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_module_header.c
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>

// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)

static
bool parse_dim_for_one_arg(// input and output

                           // so-far-seen named dimensions. Initially these are
                           // <0 to indicate that they're unknown. As the
                           // broadcasting rules determine the values of these,
                           // the values are stored here (>= 0), and checked for
                           // consistency
                           npy_intp* dims_named,

                           // so-far-seen broadcasted dimensions. Initially
                           // these are 1 to indicate that these are compatible
                           // with anything. As non-1 values are seen, those are
                           // stored here (> 1), and checked for consistency
                           npy_intp* dims_extra,

                           // input
                           const int Ndims_extra,
                           const int Ndims_extra_inputs_only,
                           const char* arg_name,
                           const int Ndims_extra_var,
                           const npy_intp* dims_want, const int Ndims_want,
                           const npy_intp* dims_var,  const int Ndims_var,
                           const bool is_output)
{
    // MAKE SURE THE PROTOTYPE DIMENSIONS MATCH (the trailing dimensions)
    //
    // Loop through the dimensions. Set the dimensionality of any new named
    // argument to whatever the current argument has. Any already-known
    // argument must match
    for( int i_dim=-1;
         i_dim >= -Ndims_want;
         i_dim--)
    {
        int i_dim_want = i_dim + Ndims_want;
        int dim_want   = dims_want[i_dim_want];

        int i_dim_var = i_dim + Ndims_var;
        // if we didn't get enough dimensions, use dim=1
        int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

        if(dim_want < 0)
        {
            // This is a named dimension. These can have any value, but
            // ALL dimensions of the same name must thave the SAME value
            // EVERYWHERE
            if(dims_named[-dim_want-1] < 0)
                dims_named[-dim_want-1] = dim_var;

            dim_want = dims_named[-dim_want-1];
        }

        // The prototype dimension (named or otherwise) now has a numeric
        // value. Make sure it matches what I have
        if(dim_want != dim_var)
        {
            if(dims_want[i_dim_want] < 0)
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s': prototype says dimension %d (named dimension %d) has length %d, but got %d",
                             arg_name,
                             i_dim, (int)dims_want[i_dim_want],
                             dim_want,
                             dim_var);
            else
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s': prototype says dimension %d has length %d, but got %d",
                             arg_name,
                             i_dim,
                             dim_want,
                             dim_var);
            return false;
        }
    }

    // I now know that this argument matches the prototype. I look at the
    // extra dimensions to broadcast, and make sure they match with the
    // dimensions I saw previously

    // MAKE SURE THE BROADCASTED DIMENSIONS MATCH (the leading dimensions)
    //
    // This argument has Ndims_extra_var dimensions above the prototype (may be
    // <0 if there're implicit leading length-1 dimensions at the start). The
    // current dimensions to broadcast must match

    // outputs may be bigger than the inputs (this will result in multiple
    // identical copies in each slice), but may not be smaller. I check that
    // existing extra dimensions are sufficiently large. And then I check to
    // make sure we have enough extra dimensions
    if(is_output)
    {
        for( int i_dim=-1;
             i_dim >= -Ndims_extra_var;
             i_dim--)
        {
            const int i_dim_var = i_dim - Ndims_want + Ndims_var;
            // if we didn't get enough dimensions, use dim=1
            const int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

            const int i_dim_extra = i_dim + Ndims_extra;

            if(dim_var < dims_extra[i_dim_extra])
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Output '%s' dimension %d (broadcasted dimension %d) too small. Inputs have length %d but this output has length %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             (int)dims_extra[i_dim_extra],
                             dim_var);
                return false;
            }
        }

        // I look through extra dimensions above what this output has to make
        // sure that the output array is big-enough to hold all the output. I
        // only care about the broadcasted slices defined by the input. Because
        // I don't NEED to store all the duplicates created by the output-only
        // broadcasting
        for( int i_dim=-Ndims_extra_var-1;
             i_dim >= -Ndims_extra_inputs_only;
             i_dim--)
        {
            const int i_dim_extra = i_dim + Ndims_extra;

            // What if this test passes, but a subsequent output increases
            // dims_extra[i_dim_extra] so that this would have failed? That is
            // OK. Extra dimensions in the outputs do not create new and
            // different results, and I don't need to make sure I have room to
            // store duplicates
            if(dims_extra[i_dim_extra] > 1)
            {
                // This dimension was set, but this array has a DIFFERENT value
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) is too small: this dimension of this output is too small to hold the broadcasted results of size %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             (int)dims_extra[i_dim_extra]);
                return false;
            }
        }
    }


    for( int i_dim=-1;
         i_dim >= -Ndims_extra_var;
         i_dim--)
    {
        const int i_dim_var = i_dim - Ndims_want + Ndims_var;
        // if we didn't get enough dimensions, use dim=1
        const int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

        const int i_dim_extra = i_dim + Ndims_extra;


        if (dim_var != 1)
        {
            if(i_dim_extra < 0)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) i_dim_extra<0: %d. This shouldn't happen. There's a bug in the implicit-leading-dimension logic. Please report",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             i_dim_extra);
                return false;
            }

            // I have a new value for this dimension
            if( dims_extra[i_dim_extra] == 1)
                // This dimension wasn't set yet; I set it
                dims_extra[i_dim_extra] = dim_var;
            else if(dims_extra[i_dim_extra] != dim_var)
            {
                // This dimension was set, but this array has a DIFFERENT value
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) mismatch. Previously saw length %d, but here have length %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             (int)dims_extra[i_dim_extra],
                             dim_var);
                return false;
            }
        }
    }
    return true;
}


#include "drcal.h"


///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_module_header.c
///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_geometric

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_geometric__cookie_t;

static
bool ___triangulate_geometric__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_geometric__cookie_t* cookie __attribute__((unused)))
{

                      return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output(__ivar0) (*(ctype__output*)(data_slice__output + (__ivar0)*strides_slice__output[0]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_geometric__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_geometric__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                *(drcal_point3_t*)data_slice__output =
                  drcal_triangulate_geometric(NULL, NULL, NULL,
                                           v0, v1, t01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_geometric__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_geometric__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_geometric(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_geometric__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_geometric__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_geometric__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_geometric",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulate_geometric__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_geometric__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_geometric__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_geometric__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_geometric__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_geometric__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_geometric_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__output3(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output3; i++)                               \
      if(dims_full__output3[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output3; i--)                       \
      {                                                                 \
          if(strides_slice__output3[i+Ndims_slice__output3] != sizeof_element__output3*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output3' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output3[i+Ndims_slice__output3];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output3()              _CHECK_CONTIGUOUS__output3(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output3() _CHECK_CONTIGUOUS__output3(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__output3() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__output3() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_geometric_withgrad__cookie_t;

static
bool ___triangulate_geometric_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_geometric_withgrad__cookie_t* cookie __attribute__((unused)))
{

                               return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0(__ivar0) (*(ctype__output0*)(data_slice__output0 + (__ivar0)*strides_slice__output0[0]))
#define ctype__output1 npy_float64
#define item__output1(__ivar0,__ivar1) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]+ (__ivar1)*strides_slice__output1[1]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0,__ivar1) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]+ (__ivar1)*strides_slice__output2[1]))
#define ctype__output3 npy_float64
#define item__output3(__ivar0,__ivar1) (*(ctype__output3*)(data_slice__output3 + (__ivar0)*strides_slice__output3[0]+ (__ivar1)*strides_slice__output3[1]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_geometric_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data_slice__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_geometric_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                drcal_point3_t* dm_dv0  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* dm_dv1  = (drcal_point3_t*)data_slice__output2;
                drcal_point3_t* dm_dt01 = (drcal_point3_t*)data_slice__output3;

                *(drcal_point3_t*)data_slice__output0 =
                  drcal_triangulate_geometric( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__output3
#undef ctype__output3
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2) \
  _(output3)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_geometric_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_geometric_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_geometric_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_geometric_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_geometric_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_geometric_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_geometric_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output0[1] = {3};
    const npy_intp PROTOTYPE_output1[2] = {3,3};
    const npy_intp PROTOTYPE_output2[2] = {3,3};
    const npy_intp PROTOTYPE_output3[2] = {3,3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(4);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 4);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         4);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 4 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         4, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5,t6, i)                   \
        else if( __pywrap___triangulate_geometric_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,t6,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_geometric_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_geometric_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_geometric_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_geometric_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_geometric_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS_AND_SETERROR__output3
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_l1

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_l1__cookie_t;

static
bool ___triangulate_leecivera_l1__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_l1__cookie_t* cookie __attribute__((unused)))
{

                      return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output(__ivar0) (*(ctype__output*)(data_slice__output + (__ivar0)*strides_slice__output[0]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_l1__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_l1__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                *(drcal_point3_t*)data_slice__output =
                  drcal_triangulate_leecivera_l1(NULL, NULL, NULL,
                                           v0, v1, t01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_l1__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_l1__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_l1(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_l1__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_l1__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_l1__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_l1",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulate_leecivera_l1__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_l1__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_l1__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_l1__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_l1__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_l1__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_l1_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__output3(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output3; i++)                               \
      if(dims_full__output3[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output3; i--)                       \
      {                                                                 \
          if(strides_slice__output3[i+Ndims_slice__output3] != sizeof_element__output3*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output3' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output3[i+Ndims_slice__output3];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output3()              _CHECK_CONTIGUOUS__output3(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output3() _CHECK_CONTIGUOUS__output3(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__output3() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__output3() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_l1_withgrad__cookie_t;

static
bool ___triangulate_leecivera_l1_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_l1_withgrad__cookie_t* cookie __attribute__((unused)))
{

                               return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0(__ivar0) (*(ctype__output0*)(data_slice__output0 + (__ivar0)*strides_slice__output0[0]))
#define ctype__output1 npy_float64
#define item__output1(__ivar0,__ivar1) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]+ (__ivar1)*strides_slice__output1[1]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0,__ivar1) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]+ (__ivar1)*strides_slice__output2[1]))
#define ctype__output3 npy_float64
#define item__output3(__ivar0,__ivar1) (*(ctype__output3*)(data_slice__output3 + (__ivar0)*strides_slice__output3[0]+ (__ivar1)*strides_slice__output3[1]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_l1_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data_slice__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_l1_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                drcal_point3_t* dm_dv0  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* dm_dv1  = (drcal_point3_t*)data_slice__output2;
                drcal_point3_t* dm_dt01 = (drcal_point3_t*)data_slice__output3;

                *(drcal_point3_t*)data_slice__output0 =
                  drcal_triangulate_leecivera_l1( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__output3
#undef ctype__output3
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2) \
  _(output3)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_l1_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_l1_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_l1_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_l1_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_l1_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_l1_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_l1_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output0[1] = {3};
    const npy_intp PROTOTYPE_output1[2] = {3,3};
    const npy_intp PROTOTYPE_output2[2] = {3,3};
    const npy_intp PROTOTYPE_output3[2] = {3,3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(4);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 4);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         4);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 4 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         4, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5,t6, i)                   \
        else if( __pywrap___triangulate_leecivera_l1_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,t6,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_l1_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_l1_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_l1_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_l1_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_l1_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS_AND_SETERROR__output3
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_linf

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_linf__cookie_t;

static
bool ___triangulate_leecivera_linf__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_linf__cookie_t* cookie __attribute__((unused)))
{

                      return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output(__ivar0) (*(ctype__output*)(data_slice__output + (__ivar0)*strides_slice__output[0]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_linf__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_linf__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                *(drcal_point3_t*)data_slice__output =
                  drcal_triangulate_leecivera_linf(NULL, NULL, NULL,
                                           v0, v1, t01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_linf__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_linf__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_linf(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_linf__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_linf__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_linf__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_linf",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulate_leecivera_linf__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_linf__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_linf__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_linf__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_linf__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_linf__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_linf_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__output3(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output3; i++)                               \
      if(dims_full__output3[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output3; i--)                       \
      {                                                                 \
          if(strides_slice__output3[i+Ndims_slice__output3] != sizeof_element__output3*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output3' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output3[i+Ndims_slice__output3];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output3()              _CHECK_CONTIGUOUS__output3(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output3() _CHECK_CONTIGUOUS__output3(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__output3() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__output3() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_linf_withgrad__cookie_t;

static
bool ___triangulate_leecivera_linf_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_linf_withgrad__cookie_t* cookie __attribute__((unused)))
{

                               return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0(__ivar0) (*(ctype__output0*)(data_slice__output0 + (__ivar0)*strides_slice__output0[0]))
#define ctype__output1 npy_float64
#define item__output1(__ivar0,__ivar1) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]+ (__ivar1)*strides_slice__output1[1]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0,__ivar1) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]+ (__ivar1)*strides_slice__output2[1]))
#define ctype__output3 npy_float64
#define item__output3(__ivar0,__ivar1) (*(ctype__output3*)(data_slice__output3 + (__ivar0)*strides_slice__output3[0]+ (__ivar1)*strides_slice__output3[1]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_linf_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data_slice__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_linf_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                drcal_point3_t* dm_dv0  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* dm_dv1  = (drcal_point3_t*)data_slice__output2;
                drcal_point3_t* dm_dt01 = (drcal_point3_t*)data_slice__output3;

                *(drcal_point3_t*)data_slice__output0 =
                  drcal_triangulate_leecivera_linf( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__output3
#undef ctype__output3
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2) \
  _(output3)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_linf_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_linf_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_linf_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_linf_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_linf_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_linf_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_linf_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output0[1] = {3};
    const npy_intp PROTOTYPE_output1[2] = {3,3};
    const npy_intp PROTOTYPE_output2[2] = {3,3};
    const npy_intp PROTOTYPE_output3[2] = {3,3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(4);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 4);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         4);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 4 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         4, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5,t6, i)                   \
        else if( __pywrap___triangulate_leecivera_linf_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,t6,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_linf_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_linf_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_linf_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_linf_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_linf_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS_AND_SETERROR__output3
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_mid2

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_mid2__cookie_t;

static
bool ___triangulate_leecivera_mid2__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_mid2__cookie_t* cookie __attribute__((unused)))
{

                      return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output(__ivar0) (*(ctype__output*)(data_slice__output + (__ivar0)*strides_slice__output[0]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_mid2__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_mid2__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                *(drcal_point3_t*)data_slice__output =
                  drcal_triangulate_leecivera_mid2(NULL, NULL, NULL,
                                           v0, v1, t01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_mid2__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_mid2__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_mid2(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_mid2__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_mid2__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_mid2__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_mid2",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulate_leecivera_mid2__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_mid2__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_mid2__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_mid2__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_mid2__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_mid2__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_mid2_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__output3(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output3; i++)                               \
      if(dims_full__output3[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output3; i--)                       \
      {                                                                 \
          if(strides_slice__output3[i+Ndims_slice__output3] != sizeof_element__output3*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output3' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output3[i+Ndims_slice__output3];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output3()              _CHECK_CONTIGUOUS__output3(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output3() _CHECK_CONTIGUOUS__output3(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__output3() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__output3() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_mid2_withgrad__cookie_t;

static
bool ___triangulate_leecivera_mid2_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_mid2_withgrad__cookie_t* cookie __attribute__((unused)))
{

                               return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0(__ivar0) (*(ctype__output0*)(data_slice__output0 + (__ivar0)*strides_slice__output0[0]))
#define ctype__output1 npy_float64
#define item__output1(__ivar0,__ivar1) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]+ (__ivar1)*strides_slice__output1[1]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0,__ivar1) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]+ (__ivar1)*strides_slice__output2[1]))
#define ctype__output3 npy_float64
#define item__output3(__ivar0,__ivar1) (*(ctype__output3*)(data_slice__output3 + (__ivar0)*strides_slice__output3[0]+ (__ivar1)*strides_slice__output3[1]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_mid2_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data_slice__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_mid2_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                drcal_point3_t* dm_dv0  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* dm_dv1  = (drcal_point3_t*)data_slice__output2;
                drcal_point3_t* dm_dt01 = (drcal_point3_t*)data_slice__output3;

                *(drcal_point3_t*)data_slice__output0 =
                  drcal_triangulate_leecivera_mid2( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__output3
#undef ctype__output3
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2) \
  _(output3)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_mid2_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_mid2_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_mid2_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_mid2_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_mid2_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_mid2_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_mid2_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output0[1] = {3};
    const npy_intp PROTOTYPE_output1[2] = {3,3};
    const npy_intp PROTOTYPE_output2[2] = {3,3};
    const npy_intp PROTOTYPE_output3[2] = {3,3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(4);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 4);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         4);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 4 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         4, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5,t6, i)                   \
        else if( __pywrap___triangulate_leecivera_mid2_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,t6,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_mid2_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_mid2_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_mid2_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_mid2_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_mid2_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS_AND_SETERROR__output3
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_wmid2

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_wmid2__cookie_t;

static
bool ___triangulate_leecivera_wmid2__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_wmid2__cookie_t* cookie __attribute__((unused)))
{

                      return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output(__ivar0) (*(ctype__output*)(data_slice__output + (__ivar0)*strides_slice__output[0]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_wmid2__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_wmid2__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                *(drcal_point3_t*)data_slice__output =
                  drcal_triangulate_leecivera_wmid2(NULL, NULL, NULL,
                                           v0, v1, t01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_wmid2__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_wmid2__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_wmid2(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_wmid2__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_wmid2__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_wmid2__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_wmid2",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulate_leecivera_wmid2__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_wmid2__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_wmid2__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_wmid2__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_wmid2__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_wmid2__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_leecivera_wmid2_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__output3(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output3; i++)                               \
      if(dims_full__output3[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output3; i--)                       \
      {                                                                 \
          if(strides_slice__output3[i+Ndims_slice__output3] != sizeof_element__output3*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output3' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output3[i+Ndims_slice__output3];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output3()              _CHECK_CONTIGUOUS__output3(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output3() _CHECK_CONTIGUOUS__output3(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__output3() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__output3() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulate_leecivera_wmid2_withgrad__cookie_t;

static
bool ___triangulate_leecivera_wmid2_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulate_leecivera_wmid2_withgrad__cookie_t* cookie __attribute__((unused)))
{

                               return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0(__ivar0) (*(ctype__output0*)(data_slice__output0 + (__ivar0)*strides_slice__output0[0]))
#define ctype__output1 npy_float64
#define item__output1(__ivar0,__ivar1) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]+ (__ivar1)*strides_slice__output1[1]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0,__ivar1) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]+ (__ivar1)*strides_slice__output2[1]))
#define ctype__output3 npy_float64
#define item__output3(__ivar0,__ivar1) (*(ctype__output3*)(data_slice__output3 + (__ivar0)*strides_slice__output3[0]+ (__ivar1)*strides_slice__output3[1]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulate_leecivera_wmid2_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data_slice__output3 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulate_leecivera_wmid2_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                drcal_point3_t* dm_dv0  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* dm_dv1  = (drcal_point3_t*)data_slice__output2;
                drcal_point3_t* dm_dt01 = (drcal_point3_t*)data_slice__output3;

                *(drcal_point3_t*)data_slice__output0 =
                  drcal_triangulate_leecivera_wmid2( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__output3
#undef ctype__output3
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2) \
  _(output3)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_leecivera_wmid2_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_leecivera_wmid2_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_leecivera_wmid2_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_leecivera_wmid2_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_leecivera_wmid2_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_leecivera_wmid2_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_leecivera_wmid2_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output0[1] = {3};
    const npy_intp PROTOTYPE_output1[2] = {3,3};
    const npy_intp PROTOTYPE_output2[2] = {3,3};
    const npy_intp PROTOTYPE_output3[2] = {3,3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(4);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 4);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         4);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 4 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         4, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5,t6, i)                   \
        else if( __pywrap___triangulate_leecivera_wmid2_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,t6,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_leecivera_wmid2_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_leecivera_wmid2_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_leecivera_wmid2_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_leecivera_wmid2_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_leecivera_wmid2_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS_AND_SETERROR__output3
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_lindstrom

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0_local(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0_local; i++)                               \
      if(dims_full__v0_local[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0_local; i--)                       \
      {                                                                 \
          if(strides_slice__v0_local[i+Ndims_slice__v0_local] != sizeof_element__v0_local*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0_local' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0_local[i+Ndims_slice__v0_local];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0_local()              _CHECK_CONTIGUOUS__v0_local(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0_local() _CHECK_CONTIGUOUS__v0_local(true)


#define _CHECK_CONTIGUOUS__v1_local(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1_local; i++)                               \
      if(dims_full__v1_local[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1_local; i--)                       \
      {                                                                 \
          if(strides_slice__v1_local[i+Ndims_slice__v1_local] != sizeof_element__v1_local*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1_local' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1_local[i+Ndims_slice__v1_local];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1_local()              _CHECK_CONTIGUOUS__v1_local(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1_local() _CHECK_CONTIGUOUS__v1_local(true)


#define _CHECK_CONTIGUOUS__Rt01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__Rt01; i++)                               \
      if(dims_full__Rt01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__Rt01; i--)                       \
      {                                                                 \
          if(strides_slice__Rt01[i+Ndims_slice__Rt01] != sizeof_element__Rt01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'Rt01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__Rt01[i+Ndims_slice__Rt01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__Rt01()              _CHECK_CONTIGUOUS__Rt01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__Rt01() _CHECK_CONTIGUOUS__Rt01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0_local() && CHECK_CONTIGUOUS__v1_local() && CHECK_CONTIGUOUS__Rt01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0_local() && CHECK_CONTIGUOUS_AND_SETERROR__v1_local() && CHECK_CONTIGUOUS_AND_SETERROR__Rt01()

typedef struct {  } ___triangulate_lindstrom__cookie_t;

static
bool ___triangulate_lindstrom__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0_local __attribute__((unused)),
  const npy_intp* dims_full__v0_local __attribute__((unused)),
  const npy_intp* strides_full__v0_local __attribute__((unused)),
  const int Ndims_slice__v0_local __attribute__((unused)),
  const npy_intp* dims_slice__v0_local __attribute__((unused)),
  const npy_intp* strides_slice__v0_local __attribute__((unused)),
  npy_intp sizeof_element__v0_local __attribute__((unused)),
  void* data__v0_local __attribute__((unused)),
  const int Ndims_full__v1_local __attribute__((unused)),
  const npy_intp* dims_full__v1_local __attribute__((unused)),
  const npy_intp* strides_full__v1_local __attribute__((unused)),
  const int Ndims_slice__v1_local __attribute__((unused)),
  const npy_intp* dims_slice__v1_local __attribute__((unused)),
  const npy_intp* strides_slice__v1_local __attribute__((unused)),
  npy_intp sizeof_element__v1_local __attribute__((unused)),
  void* data__v1_local __attribute__((unused)),
  const int Ndims_full__Rt01 __attribute__((unused)),
  const npy_intp* dims_full__Rt01 __attribute__((unused)),
  const npy_intp* strides_full__Rt01 __attribute__((unused)),
  const int Ndims_slice__Rt01 __attribute__((unused)),
  const npy_intp* dims_slice__Rt01 __attribute__((unused)),
  const npy_intp* strides_slice__Rt01 __attribute__((unused)),
  npy_intp sizeof_element__Rt01 __attribute__((unused)),
  void* data__Rt01 __attribute__((unused)),
  ___triangulate_lindstrom__cookie_t* cookie __attribute__((unused)))
{

            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output(__ivar0) (*(ctype__output*)(data_slice__output + (__ivar0)*strides_slice__output[0]))
#define ctype__v0_local npy_float64
#define item__v0_local(__ivar0) (*(ctype__v0_local*)(data_slice__v0_local + (__ivar0)*strides_slice__v0_local[0]))
#define ctype__v1_local npy_float64
#define item__v1_local(__ivar0) (*(ctype__v1_local*)(data_slice__v1_local + (__ivar0)*strides_slice__v1_local[0]))
#define ctype__Rt01 npy_float64
#define item__Rt01(__ivar0,__ivar1) (*(ctype__Rt01*)(data_slice__Rt01 + (__ivar0)*strides_slice__Rt01[0]+ (__ivar1)*strides_slice__Rt01[1]))

static
bool ___triangulate_lindstrom__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0_local __attribute__((unused)),
  const npy_intp* dims_full__v0_local __attribute__((unused)),
  const npy_intp* strides_full__v0_local __attribute__((unused)),
  const int Ndims_slice__v0_local __attribute__((unused)),
  const npy_intp* dims_slice__v0_local __attribute__((unused)),
  const npy_intp* strides_slice__v0_local __attribute__((unused)),
  npy_intp sizeof_element__v0_local __attribute__((unused)),
  void* data_slice__v0_local __attribute__((unused)),
  const int Ndims_full__v1_local __attribute__((unused)),
  const npy_intp* dims_full__v1_local __attribute__((unused)),
  const npy_intp* strides_full__v1_local __attribute__((unused)),
  const int Ndims_slice__v1_local __attribute__((unused)),
  const npy_intp* dims_slice__v1_local __attribute__((unused)),
  const npy_intp* strides_slice__v1_local __attribute__((unused)),
  npy_intp sizeof_element__v1_local __attribute__((unused)),
  void* data_slice__v1_local __attribute__((unused)),
  const int Ndims_full__Rt01 __attribute__((unused)),
  const npy_intp* dims_full__Rt01 __attribute__((unused)),
  const npy_intp* strides_full__Rt01 __attribute__((unused)),
  const int Ndims_slice__Rt01 __attribute__((unused)),
  const npy_intp* dims_slice__Rt01 __attribute__((unused)),
  const npy_intp* strides_slice__Rt01 __attribute__((unused)),
  npy_intp sizeof_element__Rt01 __attribute__((unused)),
  void* data_slice__Rt01 __attribute__((unused)),
  ___triangulate_lindstrom__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0_local;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1_local;
                const drcal_point3_t* Rt01= (const drcal_point3_t*)data_slice__Rt01;

                *(drcal_point3_t*)data_slice__output =
                  drcal_triangulate_lindstrom(NULL,NULL,NULL,
                                              v0, v1, Rt01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0_local
#undef ctype__v0_local
#undef item__v1_local
#undef ctype__v1_local
#undef item__Rt01
#undef ctype__Rt01
#define ARGUMENTS(_) \
  _(v0_local) \
  _(v1_local) \
  _(Rt01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_lindstrom__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_lindstrom__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_lindstrom(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_lindstrom__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_lindstrom__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_lindstrom__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_lindstrom",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0_local[1] = {3};
    const npy_intp PROTOTYPE_v1_local[1] = {3};
    const npy_intp PROTOTYPE_Rt01[2] = {4,3};
    const npy_intp PROTOTYPE_output[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulate_lindstrom__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_lindstrom__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_lindstrom__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_lindstrom__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_lindstrom__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_lindstrom__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0_local
#undef CHECK_CONTIGUOUS__v0_local
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0_local
#undef _CHECK_CONTIGUOUS__v1_local
#undef CHECK_CONTIGUOUS__v1_local
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1_local
#undef _CHECK_CONTIGUOUS__Rt01
#undef CHECK_CONTIGUOUS__Rt01
#undef CHECK_CONTIGUOUS_AND_SETERROR__Rt01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulate_lindstrom_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__output3(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output3; i++)                               \
      if(dims_full__output3[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output3; i--)                       \
      {                                                                 \
          if(strides_slice__output3[i+Ndims_slice__output3] != sizeof_element__output3*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output3' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output3[i+Ndims_slice__output3];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output3()              _CHECK_CONTIGUOUS__output3(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output3() _CHECK_CONTIGUOUS__output3(true)


#define _CHECK_CONTIGUOUS__v0_local(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0_local; i++)                               \
      if(dims_full__v0_local[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0_local; i--)                       \
      {                                                                 \
          if(strides_slice__v0_local[i+Ndims_slice__v0_local] != sizeof_element__v0_local*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0_local' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0_local[i+Ndims_slice__v0_local];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0_local()              _CHECK_CONTIGUOUS__v0_local(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0_local() _CHECK_CONTIGUOUS__v0_local(true)


#define _CHECK_CONTIGUOUS__v1_local(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1_local; i++)                               \
      if(dims_full__v1_local[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1_local; i--)                       \
      {                                                                 \
          if(strides_slice__v1_local[i+Ndims_slice__v1_local] != sizeof_element__v1_local*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1_local' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1_local[i+Ndims_slice__v1_local];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1_local()              _CHECK_CONTIGUOUS__v1_local(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1_local() _CHECK_CONTIGUOUS__v1_local(true)


#define _CHECK_CONTIGUOUS__Rt01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__Rt01; i++)                               \
      if(dims_full__Rt01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__Rt01; i--)                       \
      {                                                                 \
          if(strides_slice__Rt01[i+Ndims_slice__Rt01] != sizeof_element__Rt01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'Rt01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__Rt01[i+Ndims_slice__Rt01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__Rt01()              _CHECK_CONTIGUOUS__Rt01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__Rt01() _CHECK_CONTIGUOUS__Rt01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__output3() && CHECK_CONTIGUOUS__v0_local() && CHECK_CONTIGUOUS__v1_local() && CHECK_CONTIGUOUS__Rt01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__output3() && CHECK_CONTIGUOUS_AND_SETERROR__v0_local() && CHECK_CONTIGUOUS_AND_SETERROR__v1_local() && CHECK_CONTIGUOUS_AND_SETERROR__Rt01()

typedef struct {  } ___triangulate_lindstrom_withgrad__cookie_t;

static
bool ___triangulate_lindstrom_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data__output3 __attribute__((unused)),
  const int Ndims_full__v0_local __attribute__((unused)),
  const npy_intp* dims_full__v0_local __attribute__((unused)),
  const npy_intp* strides_full__v0_local __attribute__((unused)),
  const int Ndims_slice__v0_local __attribute__((unused)),
  const npy_intp* dims_slice__v0_local __attribute__((unused)),
  const npy_intp* strides_slice__v0_local __attribute__((unused)),
  npy_intp sizeof_element__v0_local __attribute__((unused)),
  void* data__v0_local __attribute__((unused)),
  const int Ndims_full__v1_local __attribute__((unused)),
  const npy_intp* dims_full__v1_local __attribute__((unused)),
  const npy_intp* strides_full__v1_local __attribute__((unused)),
  const int Ndims_slice__v1_local __attribute__((unused)),
  const npy_intp* dims_slice__v1_local __attribute__((unused)),
  const npy_intp* strides_slice__v1_local __attribute__((unused)),
  npy_intp sizeof_element__v1_local __attribute__((unused)),
  void* data__v1_local __attribute__((unused)),
  const int Ndims_full__Rt01 __attribute__((unused)),
  const npy_intp* dims_full__Rt01 __attribute__((unused)),
  const npy_intp* strides_full__Rt01 __attribute__((unused)),
  const int Ndims_slice__Rt01 __attribute__((unused)),
  const npy_intp* dims_slice__Rt01 __attribute__((unused)),
  const npy_intp* strides_slice__Rt01 __attribute__((unused)),
  npy_intp sizeof_element__Rt01 __attribute__((unused)),
  void* data__Rt01 __attribute__((unused)),
  ___triangulate_lindstrom_withgrad__cookie_t* cookie __attribute__((unused)))
{

            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0(__ivar0) (*(ctype__output0*)(data_slice__output0 + (__ivar0)*strides_slice__output0[0]))
#define ctype__output1 npy_float64
#define item__output1(__ivar0,__ivar1) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]+ (__ivar1)*strides_slice__output1[1]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0,__ivar1) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]+ (__ivar1)*strides_slice__output2[1]))
#define ctype__output3 npy_float64
#define item__output3(__ivar0,__ivar1,__ivar2) (*(ctype__output3*)(data_slice__output3 + (__ivar0)*strides_slice__output3[0]+ (__ivar1)*strides_slice__output3[1]+ (__ivar2)*strides_slice__output3[2]))
#define ctype__v0_local npy_float64
#define item__v0_local(__ivar0) (*(ctype__v0_local*)(data_slice__v0_local + (__ivar0)*strides_slice__v0_local[0]))
#define ctype__v1_local npy_float64
#define item__v1_local(__ivar0) (*(ctype__v1_local*)(data_slice__v1_local + (__ivar0)*strides_slice__v1_local[0]))
#define ctype__Rt01 npy_float64
#define item__Rt01(__ivar0,__ivar1) (*(ctype__Rt01*)(data_slice__Rt01 + (__ivar0)*strides_slice__Rt01[0]+ (__ivar1)*strides_slice__Rt01[1]))

static
bool ___triangulate_lindstrom_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__output3 __attribute__((unused)),
  const npy_intp* dims_full__output3 __attribute__((unused)),
  const npy_intp* strides_full__output3 __attribute__((unused)),
  const int Ndims_slice__output3 __attribute__((unused)),
  const npy_intp* dims_slice__output3 __attribute__((unused)),
  const npy_intp* strides_slice__output3 __attribute__((unused)),
  npy_intp sizeof_element__output3 __attribute__((unused)),
  void* data_slice__output3 __attribute__((unused)),
  const int Ndims_full__v0_local __attribute__((unused)),
  const npy_intp* dims_full__v0_local __attribute__((unused)),
  const npy_intp* strides_full__v0_local __attribute__((unused)),
  const int Ndims_slice__v0_local __attribute__((unused)),
  const npy_intp* dims_slice__v0_local __attribute__((unused)),
  const npy_intp* strides_slice__v0_local __attribute__((unused)),
  npy_intp sizeof_element__v0_local __attribute__((unused)),
  void* data_slice__v0_local __attribute__((unused)),
  const int Ndims_full__v1_local __attribute__((unused)),
  const npy_intp* dims_full__v1_local __attribute__((unused)),
  const npy_intp* strides_full__v1_local __attribute__((unused)),
  const int Ndims_slice__v1_local __attribute__((unused)),
  const npy_intp* dims_slice__v1_local __attribute__((unused)),
  const npy_intp* strides_slice__v1_local __attribute__((unused)),
  npy_intp sizeof_element__v1_local __attribute__((unused)),
  void* data_slice__v1_local __attribute__((unused)),
  const int Ndims_full__Rt01 __attribute__((unused)),
  const npy_intp* dims_full__Rt01 __attribute__((unused)),
  const npy_intp* strides_full__Rt01 __attribute__((unused)),
  const int Ndims_slice__Rt01 __attribute__((unused)),
  const npy_intp* dims_slice__Rt01 __attribute__((unused)),
  const npy_intp* strides_slice__Rt01 __attribute__((unused)),
  npy_intp sizeof_element__Rt01 __attribute__((unused)),
  void* data_slice__Rt01 __attribute__((unused)),
  ___triangulate_lindstrom_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0_local;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1_local;
                const drcal_point3_t* Rt01= (const drcal_point3_t*)data_slice__Rt01;

                drcal_point3_t* dm_dv0  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* dm_dv1  = (drcal_point3_t*)data_slice__output2;
                drcal_point3_t* dm_dRt01= (drcal_point3_t*)data_slice__output3;

                *(drcal_point3_t*)data_slice__output0 =
                  drcal_triangulate_lindstrom(dm_dv0, dm_dv1, dm_dRt01,
                                              v0, v1, Rt01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__output3
#undef ctype__output3
#undef item__v0_local
#undef ctype__v0_local
#undef item__v1_local
#undef ctype__v1_local
#undef item__Rt01
#undef ctype__Rt01
#define ARGUMENTS(_) \
  _(v0_local) \
  _(v1_local) \
  _(Rt01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2) \
  _(output3)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulate_lindstrom_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulate_lindstrom_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulate_lindstrom_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulate_lindstrom_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulate_lindstrom_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulate_lindstrom_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulate_lindstrom_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0_local[1] = {3};
    const npy_intp PROTOTYPE_v1_local[1] = {3};
    const npy_intp PROTOTYPE_Rt01[2] = {4,3};
    const npy_intp PROTOTYPE_output0[1] = {3};
    const npy_intp PROTOTYPE_output1[2] = {3,3};
    const npy_intp PROTOTYPE_output2[2] = {3,3};
    const npy_intp PROTOTYPE_output3[3] = {3,4,3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(4);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 4);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         4);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 4 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         4, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5,t6, i)                   \
        else if( __pywrap___triangulate_lindstrom_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,t6,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulate_lindstrom_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulate_lindstrom_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulate_lindstrom_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulate_lindstrom_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulate_lindstrom_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS__output3
#undef CHECK_CONTIGUOUS_AND_SETERROR__output3
#undef _CHECK_CONTIGUOUS__v0_local
#undef CHECK_CONTIGUOUS__v0_local
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0_local
#undef _CHECK_CONTIGUOUS__v1_local
#undef CHECK_CONTIGUOUS__v1_local
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1_local
#undef _CHECK_CONTIGUOUS__Rt01
#undef CHECK_CONTIGUOUS__Rt01
#undef CHECK_CONTIGUOUS_AND_SETERROR__Rt01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulated_error

#define _CHECK_CONTIGUOUS__output(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output; i++)                               \
      if(dims_full__output[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output; i--)                       \
      {                                                                 \
          if(strides_slice__output[i+Ndims_slice__output] != sizeof_element__output*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output[i+Ndims_slice__output];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output()              _CHECK_CONTIGUOUS__output(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output() _CHECK_CONTIGUOUS__output(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulated_error__cookie_t;

static
bool ___triangulated_error__validate(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulated_error__cookie_t* cookie __attribute__((unused)))
{

            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output npy_float64
#define item__output() (*(ctype__output*)(data_slice__output ))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulated_error__0__slice(
  const int Ndims_full__output __attribute__((unused)),
  const npy_intp* dims_full__output __attribute__((unused)),
  const npy_intp* strides_full__output __attribute__((unused)),
  const int Ndims_slice__output __attribute__((unused)),
  const npy_intp* dims_slice__output __attribute__((unused)),
  const npy_intp* strides_slice__output __attribute__((unused)),
  npy_intp sizeof_element__output __attribute__((unused)),
  void* data_slice__output __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulated_error__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                *(double*)data_slice__output =
                  _drcal_triangulated_error(NULL,NULL,
                                            v0, v1, t01);
                return true;

}
#undef item__output
#undef ctype__output
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulated_error__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulated_error__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulated_error(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulated_error__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulated_error__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulated_error__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulated_error",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output[0] = {};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3, i)                   \
        else if( __pywrap___triangulated_error__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulated_error__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulated_error__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulated_error__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulated_error__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulated_error__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS__output
#undef CHECK_CONTIGUOUS_AND_SETERROR__output
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c
///////// for function   _triangulated_error_withgrad

#define _CHECK_CONTIGUOUS__output0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output0; i++)                               \
      if(dims_full__output0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output0; i--)                       \
      {                                                                 \
          if(strides_slice__output0[i+Ndims_slice__output0] != sizeof_element__output0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output0[i+Ndims_slice__output0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output0()              _CHECK_CONTIGUOUS__output0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output0() _CHECK_CONTIGUOUS__output0(true)


#define _CHECK_CONTIGUOUS__output1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output1; i++)                               \
      if(dims_full__output1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output1; i--)                       \
      {                                                                 \
          if(strides_slice__output1[i+Ndims_slice__output1] != sizeof_element__output1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output1[i+Ndims_slice__output1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output1()              _CHECK_CONTIGUOUS__output1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output1() _CHECK_CONTIGUOUS__output1(true)


#define _CHECK_CONTIGUOUS__output2(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__output2; i++)                               \
      if(dims_full__output2[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__output2; i--)                       \
      {                                                                 \
          if(strides_slice__output2[i+Ndims_slice__output2] != sizeof_element__output2*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'output2' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__output2[i+Ndims_slice__output2];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__output2()              _CHECK_CONTIGUOUS__output2(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__output2() _CHECK_CONTIGUOUS__output2(true)


#define _CHECK_CONTIGUOUS__v0(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v0; i++)                               \
      if(dims_full__v0[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v0; i--)                       \
      {                                                                 \
          if(strides_slice__v0[i+Ndims_slice__v0] != sizeof_element__v0*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v0' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v0[i+Ndims_slice__v0];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v0()              _CHECK_CONTIGUOUS__v0(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v0() _CHECK_CONTIGUOUS__v0(true)


#define _CHECK_CONTIGUOUS__v1(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__v1; i++)                               \
      if(dims_full__v1[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__v1; i--)                       \
      {                                                                 \
          if(strides_slice__v1[i+Ndims_slice__v1] != sizeof_element__v1*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 'v1' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__v1[i+Ndims_slice__v1];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__v1()              _CHECK_CONTIGUOUS__v1(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__v1() _CHECK_CONTIGUOUS__v1(true)


#define _CHECK_CONTIGUOUS__t01(seterror)                             \
({                                                                      \
  bool result     = true;                                               \
  bool have_dim_0 = false;                                              \
  /* If I have no data, just call the thing contiguous. This is useful */ \
  /* because np.ascontiguousarray doesn't set contiguous alignment */   \
  /* for empty arrays */                                                \
  for(int i=0; i<Ndims_full__t01; i++)                               \
      if(dims_full__t01[i] == 0)                                     \
      {                                                                 \
          result     = true;                                            \
          have_dim_0 = true;                                            \
          break;                                                        \
      }                                                                 \
                                                                        \
  if(!have_dim_0)                                                       \
  {                                                                     \
      int Nelems_slice = 1;                                             \
      for(int i=-1; i>=-Ndims_slice__t01; i--)                       \
      {                                                                 \
          if(strides_slice__t01[i+Ndims_slice__t01] != sizeof_element__t01*Nelems_slice) \
          {                                                             \
              result = false;                                           \
              if(seterror)                                              \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Variable 't01' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
              break;                                                    \
          }                                                             \
          Nelems_slice *= dims_slice__t01[i+Ndims_slice__t01];    \
      }                                                                 \
  }                                                                     \
  result;                                                               \
})
#define CHECK_CONTIGUOUS__t01()              _CHECK_CONTIGUOUS__t01(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__t01() _CHECK_CONTIGUOUS__t01(true)


#define CHECK_CONTIGUOUS_ALL() CHECK_CONTIGUOUS__output0() && CHECK_CONTIGUOUS__output1() && CHECK_CONTIGUOUS__output2() && CHECK_CONTIGUOUS__v0() && CHECK_CONTIGUOUS__v1() && CHECK_CONTIGUOUS__t01()
#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() CHECK_CONTIGUOUS_AND_SETERROR__output0() && CHECK_CONTIGUOUS_AND_SETERROR__output1() && CHECK_CONTIGUOUS_AND_SETERROR__output2() && CHECK_CONTIGUOUS_AND_SETERROR__v0() && CHECK_CONTIGUOUS_AND_SETERROR__v1() && CHECK_CONTIGUOUS_AND_SETERROR__t01()

typedef struct {  } ___triangulated_error_withgrad__cookie_t;

static
bool ___triangulated_error_withgrad__validate(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data__output2 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data__t01 __attribute__((unused)),
  ___triangulated_error_withgrad__cookie_t* cookie __attribute__((unused)))
{

            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();
}

#define ctype__output0 npy_float64
#define item__output0() (*(ctype__output0*)(data_slice__output0 ))
#define ctype__output1 npy_float64
#define item__output1(__ivar0) (*(ctype__output1*)(data_slice__output1 + (__ivar0)*strides_slice__output1[0]))
#define ctype__output2 npy_float64
#define item__output2(__ivar0) (*(ctype__output2*)(data_slice__output2 + (__ivar0)*strides_slice__output2[0]))
#define ctype__v0 npy_float64
#define item__v0(__ivar0) (*(ctype__v0*)(data_slice__v0 + (__ivar0)*strides_slice__v0[0]))
#define ctype__v1 npy_float64
#define item__v1(__ivar0) (*(ctype__v1*)(data_slice__v1 + (__ivar0)*strides_slice__v1[0]))
#define ctype__t01 npy_float64
#define item__t01(__ivar0) (*(ctype__t01*)(data_slice__t01 + (__ivar0)*strides_slice__t01[0]))

static
bool ___triangulated_error_withgrad__0__slice(
  const int Ndims_full__output0 __attribute__((unused)),
  const npy_intp* dims_full__output0 __attribute__((unused)),
  const npy_intp* strides_full__output0 __attribute__((unused)),
  const int Ndims_slice__output0 __attribute__((unused)),
  const npy_intp* dims_slice__output0 __attribute__((unused)),
  const npy_intp* strides_slice__output0 __attribute__((unused)),
  npy_intp sizeof_element__output0 __attribute__((unused)),
  void* data_slice__output0 __attribute__((unused)),
  const int Ndims_full__output1 __attribute__((unused)),
  const npy_intp* dims_full__output1 __attribute__((unused)),
  const npy_intp* strides_full__output1 __attribute__((unused)),
  const int Ndims_slice__output1 __attribute__((unused)),
  const npy_intp* dims_slice__output1 __attribute__((unused)),
  const npy_intp* strides_slice__output1 __attribute__((unused)),
  npy_intp sizeof_element__output1 __attribute__((unused)),
  void* data_slice__output1 __attribute__((unused)),
  const int Ndims_full__output2 __attribute__((unused)),
  const npy_intp* dims_full__output2 __attribute__((unused)),
  const npy_intp* strides_full__output2 __attribute__((unused)),
  const int Ndims_slice__output2 __attribute__((unused)),
  const npy_intp* dims_slice__output2 __attribute__((unused)),
  const npy_intp* strides_slice__output2 __attribute__((unused)),
  npy_intp sizeof_element__output2 __attribute__((unused)),
  void* data_slice__output2 __attribute__((unused)),
  const int Ndims_full__v0 __attribute__((unused)),
  const npy_intp* dims_full__v0 __attribute__((unused)),
  const npy_intp* strides_full__v0 __attribute__((unused)),
  const int Ndims_slice__v0 __attribute__((unused)),
  const npy_intp* dims_slice__v0 __attribute__((unused)),
  const npy_intp* strides_slice__v0 __attribute__((unused)),
  npy_intp sizeof_element__v0 __attribute__((unused)),
  void* data_slice__v0 __attribute__((unused)),
  const int Ndims_full__v1 __attribute__((unused)),
  const npy_intp* dims_full__v1 __attribute__((unused)),
  const npy_intp* strides_full__v1 __attribute__((unused)),
  const int Ndims_slice__v1 __attribute__((unused)),
  const npy_intp* dims_slice__v1 __attribute__((unused)),
  const npy_intp* strides_slice__v1 __attribute__((unused)),
  npy_intp sizeof_element__v1 __attribute__((unused)),
  void* data_slice__v1 __attribute__((unused)),
  const int Ndims_full__t01 __attribute__((unused)),
  const npy_intp* dims_full__t01 __attribute__((unused)),
  const npy_intp* strides_full__t01 __attribute__((unused)),
  const int Ndims_slice__t01 __attribute__((unused)),
  const npy_intp* dims_slice__t01 __attribute__((unused)),
  const npy_intp* strides_slice__t01 __attribute__((unused)),
  npy_intp sizeof_element__t01 __attribute__((unused)),
  void* data_slice__t01 __attribute__((unused)),
  ___triangulated_error_withgrad__cookie_t* cookie __attribute__((unused)))
{

                const drcal_point3_t* v0  = (const drcal_point3_t*)data_slice__v0;
                const drcal_point3_t* v1  = (const drcal_point3_t*)data_slice__v1;
                const drcal_point3_t* t01 = (const drcal_point3_t*)data_slice__t01;

                drcal_point3_t* derr_dv1  = (drcal_point3_t*)data_slice__output1;
                drcal_point3_t* derr_dt01 = (drcal_point3_t*)data_slice__output2;

                *(double*)data_slice__output0 =
                  _drcal_triangulated_error(derr_dv1, derr_dt01,
                                            v0, v1, t01);
                return true;

}
#undef item__output0
#undef ctype__output0
#undef item__output1
#undef ctype__output1
#undef item__output2
#undef ctype__output2
#undef item__v0
#undef ctype__v0
#undef item__v1
#undef ctype__v1
#undef item__t01
#undef ctype__t01
#define ARGUMENTS(_) \
  _(v0) \
  _(v1) \
  _(t01)

#define OUTPUTS(_) \
  _(output0) \
  _(output1) \
  _(output2)

#define ARG_DEFINE(     name) PyArrayObject* __py__ ## name = NULL;
#define ARGLIST_DECLARE(name) PyArrayObject* __py__ ## name,
#define ARGLIST_CALL(   name) __py__ ## name,

#define ARGLIST_SELECTED_TYPENUM_PTR_DECLARE(name) int* selected_typenum__ ## name,
#define ARGLIST_SELECTED_TYPENUM_PTR_CALL(   name) &selected_typenum__ ## name,


#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


static
bool __pywrap___triangulated_error_withgrad__next(int* idims, const npy_intp* Ndims, int N)
{
    for(int i = N-1; i>=0; i--)
    {
        if(++idims[i] < Ndims[i])
            return true;
        idims[i] = 0;
    }
    return false;
}

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
bool __pywrap___triangulated_error_withgrad__type_matches(
                  ARGUMENTS(TYPE_MATCHES_ARGLIST)
                  OUTPUTS(  TYPE_MATCHES_ARGLIST)
                  ARGUMENTS(ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_DECLARE)
                  OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_DECLARE)
                  int dummy __attribute__((unused)) )
{

#define SET_SELECTED_TYPENUM_OUTPUT(name) *selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
    && ( __py__ ## name == NULL ||                              \
      (PyObject*)__py__ ## name == Py_None ||                   \
      PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

    if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
    {
        /* all arguments match this typeset! */
        OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
        return true;
    }
    return false;
}
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


static
PyObject* __pywrap___triangulated_error_withgrad(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    ___triangulated_error_withgrad__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    ___triangulated_error_withgrad__cookie_t* cookie = &_cookie;

    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) ___triangulated_error_withgrad__cookie_t* cookie __attribute__((unused)));


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    ;

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O"  ":bindings_triangulation_npsp._triangulated_error_withgrad",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

    const npy_intp PROTOTYPE_v0[1] = {3};
    const npy_intp PROTOTYPE_v1[1] = {3};
    const npy_intp PROTOTYPE_t01[1] = {3};
    const npy_intp PROTOTYPE_output0[0] = {};
    const npy_intp PROTOTYPE_output1[1] = {3};
    const npy_intp PROTOTYPE_output2[1] = {3};
    int Ndims_named = 0;
;

    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New(3);
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", 3);
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         3);
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != 3 )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         3, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
;

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function = NULL;

#define TYPESETS(_) \
        _(12,12,12,12,12,12,0)
#define TYPESET_MATCHES(t0,t1,t2,t3,t4,t5, i)                   \
        else if( __pywrap___triangulated_error_withgrad__type_matches                \
                 (                                                      \
                             t0,t1,t2,t3,t4,t5,                 \
                             ARGUMENTS(ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_CALL)                    \
                             OUTPUTS(  ARGLIST_SELECTED_TYPENUM_PTR_CALL) \
                             0 /* dummy; unused */                      \
                 )                                                      \
               )                                                        \
        {                                                               \
            /* matched */                                               \
            slice_function = ___triangulated_error_withgrad__ ## i ## __slice;       \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64)\n"
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         "  (inputs: float64,float64,float64   outputs: float64,float64,float64)\n");
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        npy_intp        __nbytes__     ## name = -1;                    \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            __nbytes__  ## name = PyArray_NBYTES (__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }


        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);

        const int Ndims_extra_inputs_only = Ndims_extra;

        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(/* input and output */            \
                                      dims_named, dims_extra,           \
                                                                        \
                                      /* input */                       \
                                      Ndims_extra,                      \
                                      Ndims_extra_inputs_only,          \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output;

        is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = __nbytes__ ## name; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! ___triangulated_error_withgrad__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          cookie) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "___triangulated_error_withgrad__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        do
        {
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  cookie) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError,
                                 "___triangulated_error_withgrad__slice failed!");
                goto done;
            }

        } while(__pywrap___triangulated_error_withgrad__next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef ARGLIST_DECLARE
#undef ARGLIST_CALL
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
#undef _CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS__output0
#undef CHECK_CONTIGUOUS_AND_SETERROR__output0
#undef _CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS__output1
#undef CHECK_CONTIGUOUS_AND_SETERROR__output1
#undef _CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS__output2
#undef CHECK_CONTIGUOUS_AND_SETERROR__output2
#undef _CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS__v0
#undef CHECK_CONTIGUOUS_AND_SETERROR__v0
#undef _CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS__v1
#undef CHECK_CONTIGUOUS_AND_SETERROR__v1
#undef _CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS__t01
#undef CHECK_CONTIGUOUS_AND_SETERROR__t01

#undef CHECK_CONTIGUOUS_ALL
#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_function_generic.c


///////// {{{{{{{{{ /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_module_footer_generic.c
#define PYMETHODDEF_ENTRY(name,docstring)       \
    { #name,                                    \
      (PyCFunction)__pywrap__ ## name,          \
      METH_VARARGS | METH_KEYWORDS,             \
      docstring },

static PyMethodDef methods[] =
    { FUNCTIONS(PYMETHODDEF_ENTRY)
      {}
    };

#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC initbindings_triangulation_npsp(void)
{
    Py_InitModule3("bindings_triangulation_npsp", methods, "Internal triangulation routines\n\nThis is the written-in-C Python extension module that underlies the\ntriangulation routines. The user-facing functions are available in\ndrcal.triangulation module in drcal/triangulation.py\n\nAll functions are exported into the drcal module. So you can call these via\ndrcal._triangulation_npsp.fff() or drcal.fff(). The latter is preferred.\n\n");
    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "bindings_triangulation_npsp", "Internal triangulation routines\n\nThis is the written-in-C Python extension module that underlies the\ntriangulation routines. The user-facing functions are available in\ndrcal.triangulation module in drcal/triangulation.py\n\nAll functions are exported into the drcal module. So you can call these via\ndrcal._triangulation_npsp.fff() or drcal.fff(). The latter is preferred.\n\n",
     -1,
     methods
    };

PyMODINIT_FUNC PyInit_bindings_triangulation_npsp(void)
{
    PyObject* module = PyModule_Create(&module_def);
    import_array();
    return module;
}

#endif

///////// }}}}}}}}} /home/robert/projects/drcal/.venv/lib/python3.13/site-packages/pywrap-templates/pywrap_module_footer_generic.c
