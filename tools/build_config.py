import platform
import os

print(platform.system())

VCPKG_ROOT = os.environ.get("VCPKG_ROOT")


shared_flags = [
    '-DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld"',
    '-DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld"',
    '-DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"',
    # f"-DCMAKE_TOOLCHAIN_FILE={VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
]

if platform.system() == "Darwin":
    CPP_COMPILER_PATH = "/opt/homebrew/opt/llvm/bin/clang++"
    C_COMPILER_PATH = "/opt/homebrew/opt/llvm/bin/clang"

    CMAKE_FLAGS = [
        *shared_flags,
        f"-DCMAKE_C_COMPILER={C_COMPILER_PATH}",
        f"-DCMAKE_CXX_COMPILER={CPP_COMPILER_PATH}",
    ]
elif platform.system() == "Linux":
    CPP_COMPILER_PATH = "/usr/bin/g++"
    C_COMPILER_PATH = "/usr/bin/gcc"

    CMAKE_FLAGS = [
        *shared_flags,
        f"-DCMAKE_C_COMPILER={C_COMPILER_PATH}",
        f"-DCMAKE_CXX_COMPILER={CPP_COMPILER_PATH}",
    ]
