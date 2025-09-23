#pragma once

// drcal_cameramodel_converter is a "converter" function that can be used with
// "O&" conversions in PyArg_ParseTupleAndKeywords() calls. Can interpret either
// path strings or drcal.cameramodel objects as drcal_cameramodel_t structures

#include <Python.h>
#include "drcal.h"

int drcal_cameramodel_converter(
    PyObject* py_model,
    drcal_cameramodel_t** model
);
