// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <stdbool.h>

#include "drcal-types.h"

// THESE ARE NOT A PART OF THE EXTERNAL API. Exported for the drcal python
// wrapper only

// These models have no precomputed data
typedef struct {
} drcal_LENSMODEL_PINHOLE__precomputed_t;
typedef struct {
} drcal_LENSMODEL_STEREOGRAPHIC__precomputed_t;
typedef struct {
} drcal_LENSMODEL_LONLAT__precomputed_t;
typedef struct {
} drcal_LENSMODEL_LATLON__precomputed_t;
typedef struct {
} drcal_LENSMODEL_OPENCV4__precomputed_t;
typedef struct {
} drcal_LENSMODEL_OPENCV5__precomputed_t;
typedef struct {
} drcal_LENSMODEL_OPENCV8__precomputed_t;
typedef struct {
} drcal_LENSMODEL_OPENCV12__precomputed_t;

// The splined stereographic models configuration parameters can be used to
// compute the segment size. I cache this computation
typedef struct {
    // The distance between adjacent knots (1 segment) is u_per_segment =
    // 1/segments_per_u
    double segments_per_u;
} drcal_LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed_t;

typedef struct {
    bool ready;
    union {
#define PRECOMPUTED_STRUCT(s, n) drcal_##s##__precomputed_t s##__precomputed;
        drcal_LENSMODEL_LIST(PRECOMPUTED_STRUCT);
#undef PRECOMPUTED_STRUCT
    };
} drcal_projection_precomputed_t;

void _drcal_project_internal_opencv(  // outputs
    drcal_point2_t* q,
    drcal_point3_t* dq_dp,          // may be NULL
    double* dq_dintrinsics_nocore,  // may be NULL

    // inputs
    const drcal_point3_t* p,
    int N,
    const double* intrinsics,
    int Nintrinsics
);
bool _drcal_project_internal(  // out
    drcal_point2_t* q,

    // Stored as a row-first array of shape (N,2,3). Each
    // trailing ,3 dimension element is a drcal_point3_t
    drcal_point3_t* dq_dp,
    // core, distortions concatenated. Stored as a row-first
    // array of shape (N,2,Nintrinsics). This is a DENSE array.
    // High-parameter-count lens models have very sparse
    // gradients here, and the internal project() function
    // returns those sparsely. For now THIS function densifies
    // all of these
    double* dq_dintrinsics,

    // in
    const drcal_point3_t* p,
    int N,
    const drcal_lensmodel_t* lensmodel,
    // core, distortions concatenated
    const double* intrinsics,

    int Nintrinsics,
    const drcal_projection_precomputed_t* precomputed
);
void _drcal_precompute_lensmodel_data(
    drcal_projection_precomputed_t* precomputed,
    const drcal_lensmodel_t* lensmodel
);
bool _drcal_unproject_internal(  // out
    drcal_point3_t* out,

    // in
    const drcal_point2_t* q,
    int N,
    const drcal_lensmodel_t* lensmodel,
    // core, distortions concatenated
    const double* intrinsics,
    const drcal_projection_precomputed_t* precomputed
);

// Report the number of non-zero entries in the optimization jacobian
int _drcal_num_j_nonzero(
    int Nobservations_board,
    int Nobservations_point,

    // May be NULL if we don't have any of these
    const drcal_observation_point_triangulated_t*
        observations_point_triangulated,
    int Nobservations_point_triangulated,

    int calibration_object_width_n,
    int calibration_object_height_n,
    int Ncameras_intrinsics,
    int Ncameras_extrinsics,
    int Nframes,
    int Npoints,
    int Npoints_fixed,
    const drcal_observation_board_t* observations_board,
    const drcal_observation_point_t* observations_point,
    drcal_problem_selections_t problem_selections,
    const drcal_lensmodel_t* lensmodel
);
