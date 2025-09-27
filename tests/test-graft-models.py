#!/usr/bin/env python3

r"""Test of the drcal-graft-models tool

The basic usage of the tool is simple, but it also supports a non-trivial mode
where it applies an implied transformation. I make sure to test that here

"""

import sys
import numpy as np
import numpysane as nps
import os
import subprocess

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL drcal since that's what I'm testing
sys.path[:0] = (f"{testdir}/..",)
import drcal
import testutils


import tempfile
import atexit
import shutil

workdir = tempfile.mkdtemp()


def cleanup():
    global workdir
    try:
        shutil.rmtree(workdir)
        workdir = None
    except:
        pass


atexit.register(cleanup)


rt_0r = np.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
rt_1r = np.array((0.8, -0.01, -0.2, 2.0, -4.0, 2.0))

imagersize = np.array((2000, 1000), dtype=int)

cxy_center = (imagersize - 1.0) / 2.0
pitch_y = 100.0
cxy_pitched = cxy_center + np.array((0.0, pitch_y))

# very long lenses. I want rotation to look very much like panning
fxycxy0 = nps.glue(np.array((50000.0, 50000.0)), cxy_pitched, axis=-1)
fxycxy1 = nps.glue(np.array((50000.0, 50000.0)), cxy_center, axis=-1)

model0 = drcal.cameramodel(
    intrinsics=("LENSMODEL_PINHOLE", fxycxy0),
    imagersize=imagersize,
    extrinsics_rt_fromref=rt_0r,
)

model1 = drcal.cameramodel(
    intrinsics=("LENSMODEL_PINHOLE", fxycxy1),
    imagersize=imagersize,
    extrinsics_rt_fromref=rt_1r,
)

filename0 = f"{workdir}/model0.cameramodel"
filename1 = f"{workdir}/model1.cameramodel"

model0.write(filename0)
model1.write(filename1)

# Basic test. Combine intrinsics and extrinsics without fitting any extra
# transform
out = subprocess.check_output(
    (f"{testdir}/../drcal-graft-models", filename0, filename1),
    encoding="ascii",
    stderr=subprocess.DEVNULL,
)

filename01 = f"{workdir}/model01.cameramodel"
with open(filename01, "w") as f:
    print(out, file=f)

model01 = drcal.cameramodel(filename01)

testutils.confirm_equal(
    model01.intrinsics()[1],
    model0.intrinsics()[1],
    msg="Basic grafted intrinsics match",
    eps=1.0e-6,
)
testutils.confirm_equal(
    model01.extrinsics_rt_fromref(),
    model1.extrinsics_rt_fromref(),
    msg="Basic grafted extrinsics match",
    eps=1.0e-6,
)


# More complicated test. I want to compensate for the different intrinsics with
# modified extrinsics such that the old-intrinsics and new-intrinsics project
# world points to the same place
out = subprocess.check_output(
    (f"{testdir}/../drcal-graft-models", "--radius", "-1", filename0, filename1),
    encoding="ascii",
    stderr=subprocess.DEVNULL,
)

filename01_compensated = f"{workdir}/model01_compensated.cameramodel"
with open(filename01_compensated, "w") as f:
    print(out, file=f)

model01_compensated = drcal.cameramodel(filename01_compensated)

p1 = np.array((11.0, 17.0, 10000.0))
pref = drcal.transform_point_rt(model1.extrinsics_rt_toref(), p1)

q = drcal.project(
    drcal.transform_point_rt(model1.extrinsics_rt_fromref(), pref), *model1.intrinsics()
)
q_compensated = drcal.project(
    drcal.transform_point_rt(model01_compensated.extrinsics_rt_fromref(), pref),
    *model01_compensated.intrinsics(),
)

testutils.confirm_equal(
    q_compensated, q, msg="Compensated projection ended up in the same place", eps=0.1
)

testutils.finish()
