#!/usr/bin/env python3

"""A script to build the C++ backend and install the bindings into the source tree."""

import sys
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

from fire import Fire

from build_config import CMAKE_FLAGS

BUILD_DIR = "build"

# TARGET_NAME = "bindings.cpython-313-x86_64-linux-gnu.so"
TARGETS = [
    "bindings.so",
    "bindings_npsp.so",
    "bindings_poseutils_npsp.so",
    "bindings_triangulation_npsp.so",
]
MODULES = [
    "bindings",
    "bindings_npsp",
    "bindings_poseutils_npsp",
    "bindings_triangulation_npsp",
]


LIB_DIR = Path("./build")
DEST_DIR = Path("./src/drcal/")


def _python_exe() -> str:
    return sys.executable


def build(debug: bool) -> None:
    """(Re)build the C++ backend."""
    build_path = Path("build")
    build_path.mkdir(exist_ok=True)

    python_executable = _python_exe()
    print(f"Using python executable {python_executable}")

    compile_cmd = [
        "cmake",
        "-B",
        str(build_path),
        "-G",
        "Ninja",
        *CMAKE_FLAGS,
        f"-DPython3_EXECUTABLE={python_executable}",
        "-DPython3_FIND_VIRTUALENV=ONLY",
    ]

    if debug:
        compile_cmd.append("-DCMAKE_BUILD_TYPE=Debug")
    else:
        compile_cmd.append("-DCMAKE_BUILD_TYPE=Release")

    subprocess.run(
        compile_cmd,
        check=True,
    )

    subprocess.run(["ninja", "-C", str(build_path)])

    for target in TARGETS:
        (DEST_DIR / target).unlink(missing_ok=True)
        (DEST_DIR / target).symlink_to((LIB_DIR / target).resolve())

    os.chdir(Path("src/drcal"))

    env = os.environ.copy()
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"

    for module in MODULES:
        print(f"Generating stubs for {module}")
        subprocess.run(
            [
                "pybind11-stubgen",
                module,
                "--numpy-array-remove-parameters",
                "-o",
                ".",
            ],
            env=env,
            check=True,
        )


def clean() -> None:
    """Clean the build folder and remove the symlink, if any."""
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


def clean_build() -> None:
    """First clean and then build."""
    clean()
    build(False)


def debug_build():
    build(True)


def clean_debug_build():
    clean()
    build(True)


def check_in_repo() -> None:
    """Check that we are executing this from repo root."""
    assert Path(".git").exists(), "This command should run in repo root."


if __name__ == "__main__":
    check_in_repo()

    Fire(
        {
            "build": partial(build, False),
            "clean": clean,
            "clean_build": clean_build,
            "debug_build": debug_build,
            "clean_debug_build": clean_debug_build,
        }
    )
