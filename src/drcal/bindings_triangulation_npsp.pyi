"""
Internal triangulation routines

This is the written-in-C Python extension module that underlies the
triangulation routines. The user-facing functions are available in
mrcal.triangulation module in mrcal/triangulation.py

All functions are exported into the mrcal module. So you can call these via
mrcal._triangulation_npsp.fff() or mrcal.fff(). The latter is preferred.

"""
from __future__ import annotations
__all__: list[str] = list()
def _triangulate_geometric(*args, **kwargs):
    """
    Internal geometric triangulation routine
    
    This is the internals for mrcal.triangulate_geometric(get_gradients = False). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulate_geometric_withgrad
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_geometric_withgrad(*args, **kwargs):
    """
    Internal geometric triangulation routine (with gradients)
    
    This is the internals for mrcal.triangulate_geometric(get_gradients = True). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the gradients-returning function. The internal function that
      skips those is _triangulate_geometric
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_l1(*args, **kwargs):
    """
    Internal Lee-Civera L1 triangulation routine
    
    This is the internals for mrcal.triangulate_leecivera_l1(get_gradients = False). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulate_leecivera_l1_withgrad
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_l1_withgrad(*args, **kwargs):
    """
    Internal Lee-Civera L1 triangulation routine (with gradients)
    
    This is the internals for mrcal.triangulate_leecivera_l1(get_gradients = True). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the gradients-returning function. The internal function that
      skips those is _triangulate_leecivera_l1
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_linf(*args, **kwargs):
    """
    Internal Lee-Civera L-infinity triangulation routine
    
    This is the internals for mrcal.triangulate_leecivera_linf(get_gradients = False). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulate_leecivera_linf_withgrad
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_linf_withgrad(*args, **kwargs):
    """
    Internal Lee-Civera L-infinity triangulation routine (with gradients)
    
    This is the internals for mrcal.triangulate_leecivera_linf(get_gradients = True). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the gradients-returning function. The internal function that
      skips those is _triangulate_leecivera_linf
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_mid2(*args, **kwargs):
    """
    Internal Lee-Civera Mid2 triangulation routine
    
    This is the internals for mrcal.triangulate_leecivera_mid2(get_gradients = False). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulate_leecivera_mid2_withgrad
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_mid2_withgrad(*args, **kwargs):
    """
    Internal Lee-Civera Mid2 triangulation routine (with gradients)
    
    This is the internals for mrcal.triangulate_leecivera_mid2(get_gradients = True). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the gradients-returning function. The internal function that
      skips those is _triangulate_leecivera_mid2
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_wmid2(*args, **kwargs):
    """
    Internal Lee-Civera wMid2 triangulation routine
    
    This is the internals for mrcal.triangulate_leecivera_wmid2(get_gradients = False). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulate_leecivera_wmid2_withgrad
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_leecivera_wmid2_withgrad(*args, **kwargs):
    """
    Internal Lee-Civera wMid2 triangulation routine (with gradients)
    
    This is the internals for mrcal.triangulate_leecivera_wmid2(get_gradients = True). As a
    user, please call THAT function, and see the docs for that function. The
    differences:
    
    - This is just the gradients-returning function. The internal function that
      skips those is _triangulate_leecivera_wmid2
    
    A higher-level function mrcal.triangulate() is also available for higher-level
    analysis.
    
    """
def _triangulate_lindstrom(*args, **kwargs):
    """
    Internal lindstrom's triangulation routine
    
    This is the internals for mrcal.triangulate_lindstrom(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulate_lindstrom_withgrad
    
    """
def _triangulate_lindstrom_withgrad(*args, **kwargs):
    """
    Internal lindstrom's triangulation routine
    
    This is the internals for mrcal.triangulate_lindstrom(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-returning function. The internal function that skips those
      is _triangulate_lindstrom
    
    """
def _triangulated_error(*args, **kwargs):
    """
    Internal triangulation routine used in the optimization loop
    
    This is the internals for mrcal.triangulated_error(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the no-gradients function. The internal function that returns
      gradients is _triangulated_error_withgrad
    
    """
def _triangulated_error_withgrad(*args, **kwargs):
    """
    Internal triangulation routine used in the optimization loop
    
    This is the internals for mrcal.triangulated_error(). As a user, please call
    THAT function, and see the docs for that function. The differences:
    
    - This is just the gradient-returning function. The internal function that skips those
      is _triangulated_error
    
    """
