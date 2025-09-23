#!/usr/bin/env python3

r"""Tests the stereo routines"""

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL drcal since that's what I'm testing
sys.path[:0] = (f"{testdir}/..",)
import drcal
import scipy.interpolate
import testutils


model0 = drcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
model1 = drcal.cameramodel(model0)

for lensmodel in ("LENSMODEL_LATLON", "LENSMODEL_PINHOLE"):
    # I create geometries to test. First off, a vanilla geometry for left-right stereo
    rt01 = np.array((0, 0, 0, 3.0, 0, 0))
    model1.extrinsics_rt_toref(drcal.compose_rt(model0.extrinsics_rt_toref(), rt01))

    az_fov_deg = 90
    el_fov_deg = 50
    models_rectified = drcal.rectified_system(
        (model0, model1),
        az_fov_deg=az_fov_deg,
        el_fov_deg=el_fov_deg,
        pixels_per_deg_az=-1.0 / 8.0,
        pixels_per_deg_el=-1.0 / 4.0,
        rectification_model=lensmodel,
    )
    az0 = 0.0
    el0 = 0.0

    try:
        drcal.stereo._validate_models_rectified(models_rectified)
        testutils.confirm(True, msg=f"Generated models pass validation ({lensmodel})")
    except:
        testutils.confirm(False, msg=f"Generated models pass validation ({lensmodel})")

    Rt_cam0_rect = drcal.compose_Rt(
        model0.extrinsics_Rt_fromref(), models_rectified[0].extrinsics_Rt_toref()
    )
    Rt01_rectified = drcal.compose_Rt(
        models_rectified[0].extrinsics_Rt_fromref(),
        models_rectified[1].extrinsics_Rt_toref(),
    )

    testutils.confirm_equal(
        models_rectified[0].intrinsics()[0],
        lensmodel,
        msg=f"model0 has the right lensmodel ({lensmodel})",
    )
    testutils.confirm_equal(
        models_rectified[1].intrinsics()[0],
        lensmodel,
        msg=f"model1 has the right lensmodel ({lensmodel})",
    )

    testutils.confirm_equal(
        Rt_cam0_rect,
        drcal.identity_Rt(),
        msg=f"vanilla stereo has a vanilla geometry ({lensmodel})",
    )

    testutils.confirm_equal(
        Rt01_rectified[3, 0],
        nps.mag(rt01[3:]),
        msg=f"vanilla stereo: baseline ({lensmodel})",
    )

    Naz, Nel = models_rectified[0].imagersize()

    q0 = np.array(((Naz - 1.0) / 2.0, (Nel - 1.0) / 2.0))
    v0 = drcal.unproject(q0, *models_rectified[0].intrinsics(), normalize=True)

    if lensmodel == "LENSMODEL_LATLON":
        v0_rect = drcal.unproject_latlon(np.array((az0, el0)))
        # already normalized
        testutils.confirm_equal(
            v0_rect,
            v0,
            msg=f"vanilla stereo: az0,el0 represent the same point ({lensmodel})",
        )
    else:
        v0_rect = drcal.unproject_pinhole(np.array((np.tan(az0), np.tan(el0))))
        v0_rect /= nps.mag(v0_rect)
        testutils.confirm_equal(
            v0_rect,
            v0,
            msg=f"vanilla stereo: az0,el0 represent the same point ({lensmodel})",
            eps=1e-3,
        )

    dq0x = np.array((1e-1, 0))
    dq0y = np.array((0, 1e-1))
    v0x = drcal.unproject(q0 + dq0x, *models_rectified[0].intrinsics())
    v0y = drcal.unproject(q0 + dq0y, *models_rectified[0].intrinsics())
    dthx = np.arccos(nps.inner(v0x, v0) / np.sqrt(nps.norm2(v0x) * nps.norm2(v0)))
    dthy = np.arccos(nps.inner(v0y, v0) / np.sqrt(nps.norm2(v0y) * nps.norm2(v0)))
    pixels_per_rad_az_rect = nps.mag(dq0x) / dthx
    pixels_per_rad_el_rect = nps.mag(dq0y) / dthy

    q0_cam0 = drcal.project(
        drcal.rotate_point_R(Rt_cam0_rect[:3, :], v0), *model0.intrinsics()
    )
    q0x_cam0 = drcal.project(
        drcal.rotate_point_R(Rt_cam0_rect[:3, :], v0x), *model0.intrinsics()
    )
    q0y_cam0 = drcal.project(
        drcal.rotate_point_R(Rt_cam0_rect[:3, :], v0y), *model0.intrinsics()
    )
    pixels_per_rad_az_cam0 = nps.mag(q0x_cam0 - q0_cam0) / dthx
    pixels_per_rad_el_cam0 = nps.mag(q0y_cam0 - q0_cam0) / dthy

    testutils.confirm_equal(
        pixels_per_rad_az_rect * 8.0,
        pixels_per_rad_az_cam0,
        msg=f"vanilla stereo: az pixel density ({lensmodel})",
        relative=True,
        eps=1e-2,
    )

    testutils.confirm_equal(
        pixels_per_rad_el_rect * 4.0,
        pixels_per_rad_el_cam0,
        msg=f"vanilla stereo: el pixel density ({lensmodel})",
        relative=True,
        eps=1e-2,
    )

    v0 = drcal.unproject(
        np.array((0, (Nel - 1.0) / 2.0)), *models_rectified[0].intrinsics()
    )
    v1 = drcal.unproject(
        np.array((Naz - 1, (Nel - 1.0) / 2.0)), *models_rectified[0].intrinsics()
    )
    az_fov_deg_observed = (
        np.arccos(nps.inner(v0, v1) / (nps.mag(v0) * nps.mag(v1))) * 180.0 / np.pi
    )
    testutils.confirm_equal(
        az_fov_deg_observed,
        az_fov_deg,
        msg=f"vanilla stereo: az_fov ({lensmodel})",
        eps=0.5,
    )

    v0 = drcal.unproject(
        np.array(
            (
                (Naz - 1.0) / 2.0,
                0,
            )
        ),
        *models_rectified[0].intrinsics(),
    )
    v0[0] = 0  # el_fov applies at the stereo center only
    v1 = drcal.unproject(
        np.array(
            (
                (Naz - 1.0) / 2.0,
                Nel - 1,
            )
        ),
        *models_rectified[0].intrinsics(),
    )
    v1[0] = 0
    el_fov_deg_observed = (
        np.arccos(nps.inner(v0, v1) / (nps.mag(v0) * nps.mag(v1))) * 180.0 / np.pi
    )
    testutils.confirm_equal(
        el_fov_deg_observed,
        el_fov_deg,
        msg=f"vanilla stereo: el_fov ({lensmodel})",
        eps=0.5,
    )

    ############### Complex geometry
    # Left-right stereo, with sizeable rotation and position fuzz.
    # I especially make sure there's a forward/back shift
    rt01 = np.array((0.1, 0.2, 0.05, 3.0, 0.2, 1.0))
    model1.extrinsics_rt_toref(drcal.compose_rt(model0.extrinsics_rt_toref(), rt01))
    el0_deg = 10.0
    models_rectified = drcal.rectified_system(
        (model0, model1),
        az_fov_deg=az_fov_deg,
        el_fov_deg=el_fov_deg,
        el0_deg=el0_deg,
        pixels_per_deg_az=-1.0 / 8.0,
        pixels_per_deg_el=-1.0 / 4.0,
        rectification_model=lensmodel,
    )
    el0 = el0_deg * np.pi / 180.0
    # az0 is the "forward" direction of the two cameras, relative to the
    # baseline vector
    baseline = rt01[3:] / nps.mag(rt01[3:])
    # "forward" for each of the two cameras, in the cam0 coord system
    forward0 = np.array((0, 0, 1.0))
    forward1 = drcal.rotate_point_r(rt01[:3], np.array((0, 0, 1.0)))
    forward01 = forward0 + forward1
    az0 = np.arcsin(nps.inner(forward01, baseline) / nps.mag(forward01))

    try:
        drcal.stereo._validate_models_rectified(models_rectified)
        testutils.confirm(True, msg=f"Generated models pass validation ({lensmodel})")
    except:
        testutils.confirm(False, msg=f"Generated models pass validation ({lensmodel})")

    Rt_cam0_rect = drcal.compose_Rt(
        model0.extrinsics_Rt_fromref(), models_rectified[0].extrinsics_Rt_toref()
    )
    Rt01_rectified = drcal.compose_Rt(
        models_rectified[0].extrinsics_Rt_fromref(),
        models_rectified[1].extrinsics_Rt_toref(),
    )

    # I visualized the geometry, and confirmed that it is correct. The below array
    # is the correct-looking geometry
    #
    # Rt_cam0_ref   = model0.extrinsics_Rt_fromref()
    # Rt_rect_ref  = drcal.compose_Rt( drcal.invert_Rt(Rt_cam0_rect),
    #                                  Rt_cam0_ref )
    # rt_rect_ref  = drcal.rt_from_Rt(Rt_rect_ref)
    # drcal.show_geometry( [ model0, model1, rt_rect_ref ],
    #                      cameranames = ( "camera0", "camera1", "stereo" ),
    #                      show_calobjects = False,
    #                      wait            = True )
    # print(repr(Rt_cam0_rect))

    testutils.confirm_equal(
        Rt_cam0_rect,
        np.array(
            [
                [0.9467916, -0.08500675, -0.31041828],
                [0.06311944, 0.99480206, -0.07990489],
                [0.3155972, 0.05605985, 0.94723582],
                [0.0, -0.0, -0.0],
            ]
        ),
        msg=f"complex stereo geometry ({lensmodel})",
    )

    testutils.confirm_equal(
        Rt01_rectified[3, 0],
        nps.mag(rt01[3:]),
        msg=f"complex stereo: baseline ({lensmodel})",
    )

    Naz, Nel = models_rectified[0].imagersize()

    q0 = np.array(((Naz - 1.0) / 2.0, (Nel - 1.0) / 2.0))
    v0 = drcal.unproject(q0, *models_rectified[0].intrinsics(), normalize=True)

    if lensmodel == "LENSMODEL_LATLON":
        v0_rect = drcal.unproject_latlon(np.array((az0, el0)))
        # already normalized
        testutils.confirm_equal(
            v0_rect,
            v0,
            msg=f"complex stereo: az0,el0 represent the same point ({lensmodel})",
        )
    else:
        v0_rect = drcal.unproject_pinhole(np.array((np.tan(az0), np.tan(el0))))
        v0_rect /= nps.mag(v0_rect)
        testutils.confirm_equal(
            v0_rect,
            v0,
            msg=f"complex stereo: az0,el0 represent the same point ({lensmodel})",
            eps=1e-3,
        )

    dq0x = np.array((1e-1, 0))
    dq0y = np.array((0, 1e-1))
    v0x = drcal.unproject(q0 + dq0x, *models_rectified[0].intrinsics())
    v0y = drcal.unproject(q0 + dq0y, *models_rectified[0].intrinsics())
    dthx = np.arccos(nps.inner(v0x, v0) / np.sqrt(nps.norm2(v0x) * nps.norm2(v0)))
    dthy = np.arccos(nps.inner(v0y, v0) / np.sqrt(nps.norm2(v0y) * nps.norm2(v0)))
    pixels_per_rad_az_rect = nps.mag(dq0x) / dthx
    pixels_per_rad_el_rect = nps.mag(dq0y) / dthy

    q0_cam0 = drcal.project(
        drcal.rotate_point_R(Rt_cam0_rect[:3, :], v0), *model0.intrinsics()
    )
    q0x_cam0 = drcal.project(
        drcal.rotate_point_R(Rt_cam0_rect[:3, :], v0x), *model0.intrinsics()
    )
    q0y_cam0 = drcal.project(
        drcal.rotate_point_R(Rt_cam0_rect[:3, :], v0y), *model0.intrinsics()
    )
    pixels_per_rad_az_cam0 = nps.mag(q0x_cam0 - q0_cam0) / dthx
    pixels_per_rad_el_cam0 = nps.mag(q0y_cam0 - q0_cam0) / dthy

    testutils.confirm_equal(
        pixels_per_rad_az_rect * 8.0,
        pixels_per_rad_az_cam0,
        msg=f"complex stereo: az pixel density ({lensmodel})",
        relative=True,
        eps=1e-2,
    )

    testutils.confirm_equal(
        pixels_per_rad_el_rect * 4.0,
        pixels_per_rad_el_cam0,
        msg=f"complex stereo: el pixel density ({lensmodel})",
        relative=True,
        eps=1e-2,
    )

    v0 = drcal.unproject(
        np.array((0, (Nel - 1.0) / 2.0)), *models_rectified[0].intrinsics()
    )
    v1 = drcal.unproject(
        np.array((Naz - 1, (Nel - 1.0) / 2.0)), *models_rectified[0].intrinsics()
    )
    az_fov_deg_observed = (
        np.arccos(nps.inner(v0, v1) / (nps.mag(v0) * nps.mag(v1))) * 180.0 / np.pi
    )
    testutils.confirm_equal(
        az_fov_deg_observed,
        az_fov_deg,
        msg=f"complex stereo: az_fov ({lensmodel})",
        eps=1.0,
    )

    v0 = drcal.unproject(
        np.array(
            (
                (Naz - 1.0) / 2.0,
                0,
            )
        ),
        *models_rectified[0].intrinsics(),
    )
    v0[0] = 0  # el_fov applies at the stereo center only
    v1 = drcal.unproject(
        np.array(
            (
                (Naz - 1.0) / 2.0,
                Nel - 1,
            )
        ),
        *models_rectified[0].intrinsics(),
    )
    v1[0] = 0
    el_fov_deg_observed = (
        np.arccos(nps.inner(v0, v1) / (nps.mag(v0) * nps.mag(v1))) * 180.0 / np.pi
    )
    testutils.confirm_equal(
        el_fov_deg_observed,
        el_fov_deg,
        msg=f"complex stereo: el_fov ({lensmodel})",
        eps=0.5,
    )

    # I examine points somewhere in space. I make sure the rectification maps
    # transform it properly. And I compute what its az,el and disparity would have
    # been, and I check the geometric functions
    pcam0 = np.array(((1.0, 2.0, 12.0), (-4.0, 3.0, 12.0)))

    qcam0 = drcal.project(pcam0, *model0.intrinsics())

    pcam1 = drcal.transform_point_rt(drcal.invert_rt(rt01), pcam0)
    qcam1 = drcal.project(pcam1, *model1.intrinsics())

    prect0 = drcal.transform_point_Rt(drcal.invert_Rt(Rt_cam0_rect), pcam0)
    prect1 = prect0 - Rt01_rectified[3, :]
    qrect0 = drcal.project(prect0, *models_rectified[0].intrinsics())
    qrect1 = drcal.project(prect1, *models_rectified[1].intrinsics())

    Naz, Nel = models_rectified[0].imagersize()

    row = np.arange(Naz, dtype=float)
    col = np.arange(Nel, dtype=float)

    rectification_maps = drcal.rectification_maps((model0, model1), models_rectified)

    interp_rectification_map0x = scipy.interpolate.RectBivariateSpline(
        row, col, nps.transpose(rectification_maps[0][..., 0])
    )
    interp_rectification_map0y = scipy.interpolate.RectBivariateSpline(
        row, col, nps.transpose(rectification_maps[0][..., 1])
    )
    interp_rectification_map1x = scipy.interpolate.RectBivariateSpline(
        row, col, nps.transpose(rectification_maps[1][..., 0])
    )
    interp_rectification_map1y = scipy.interpolate.RectBivariateSpline(
        row, col, nps.transpose(rectification_maps[1][..., 1])
    )

    if lensmodel == "LENSMODEL_LATLON":
        qcam0_from_map = nps.transpose(
            nps.cat(
                interp_rectification_map0x(*nps.transpose(qrect0), grid=False),
                interp_rectification_map0y(*nps.transpose(qrect0), grid=False),
            )
        )
        qcam1_from_map = nps.transpose(
            nps.cat(
                interp_rectification_map1x(*nps.transpose(qrect1), grid=False),
                interp_rectification_map1y(*nps.transpose(qrect1), grid=False),
            )
        )

    else:
        qcam0_from_map = nps.transpose(
            nps.cat(
                interp_rectification_map0x(*nps.transpose(qrect0), grid=False),
                interp_rectification_map0y(*nps.transpose(qrect0), grid=False),
            )
        )
        qcam1_from_map = nps.transpose(
            nps.cat(
                interp_rectification_map1x(*nps.transpose(qrect1), grid=False),
                interp_rectification_map1y(*nps.transpose(qrect1), grid=False),
            )
        )

    testutils.confirm_equal(
        qcam0_from_map,
        qcam0,
        eps=1e-1,
        msg=f"rectification map for camera 0 points ({lensmodel})",
    )
    testutils.confirm_equal(
        qcam1_from_map,
        qcam1,
        eps=1e-1,
        msg=f"rectification map for camera 1 points ({lensmodel})",
    )

    # same point, so we should have the same el
    testutils.confirm_equal(
        qrect0[:, 1],
        qrect1[:, 1],
        msg=f"elevations of the same observed point match ({lensmodel})",
    )

    disparity = qrect0[:, 0] - qrect1[:, 0]
    r = drcal.stereo_range(
        disparity,
        models_rectified,
        qrect0=qrect0,
    )

    testutils.confirm_equal(
        r, nps.mag(pcam0), msg=f"stereo_range reports the right thing ({lensmodel})"
    )

    r = drcal.stereo_range(
        disparity[0],
        models_rectified,
        qrect0=qrect0[0],
    )
    testutils.confirm_equal(
        r,
        nps.mag(pcam0[0]),
        msg=f"stereo_range (1-element array) reports the right thing ({lensmodel})",
        eps=2e-6,
    )

    r = drcal.stereo_range(
        float(disparity[0]),
        models_rectified,
        qrect0=qrect0[0],
    )
    testutils.confirm_equal(
        r,
        float(nps.mag(pcam0[0])),
        msg=f"stereo_range (scalar) reports the right thing ({lensmodel})",
        eps=2e-6,
    )

    disparity = qrect0[:, 0] - qrect1[:, 0]
    p = drcal.stereo_unproject(
        disparity,
        models_rectified,
        qrect0=qrect0,
    )
    testutils.confirm_equal(
        p, prect0, msg=f"stereo_unproject reports the right thing ({lensmodel})"
    )

    p = drcal.stereo_unproject(
        float(disparity[0]),
        models_rectified,
        qrect0=qrect0[0],
    )
    testutils.confirm_equal(
        p,
        prect0[0],
        msg=f"stereo_unproject (scalar) reports the right thing ({lensmodel})",
    )


testutils.finish()
