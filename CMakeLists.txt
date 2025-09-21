cmake_minimum_required(VERSION 3.15)
# If no toolchain set, try to pick it up from the env
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain file")
  endif()
endif()


project(drcal LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

if(APPLE)
    set(INSTALL_PATH "@loader_path")
else()
    set(INSTALL_PATH "$ORIGIN")
endif()


find_package(CHOLMOD CONFIG REQUIRED)
find_package(freeimage CONFIG REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

set(NUMPY_INCLUDE_DIR "${Python3_NumPy_INCLUDE_DIRS}")

Python3_add_library(
    bindings MODULE 

    cpp_src/mrcal-pywrap.c
    cpp_src/mrcal.c
    cpp_src/uncertainty.c
    cpp_src/image.c
    cpp_src/stereo.c
    cpp_src/traverse-sensor-links.c
    cpp_src/cahvore.cc
    cpp_src/opencv.c
    cpp_src/poseutils.c
    cpp_src/poseutils-opencv.c
    cpp_src/dogleg.c
    cpp_src/poseutils-uses-autodiff.cc
    cpp_src/triangulation.cc
    cpp_src/heap.cc
)

Python3_add_library(
    bindings_npsp MODULE 

    cpp_src/bindings_npsp_generated.c
    cpp_src/mrcal.c
    cpp_src/stereo.c
    cpp_src/cahvore.cc
    cpp_src/opencv.c
    cpp_src/poseutils.c
    cpp_src/poseutils-opencv.c
    cpp_src/dogleg.c
    cpp_src/poseutils-uses-autodiff.cc
    cpp_src/triangulation.cc
)

Python3_add_library(
    bindings_poseutils_npsp MODULE 

    cpp_src/bindings_poseutils_npsp_generated.c
    cpp_src/mrcal.c
    cpp_src/stereo.c
    cpp_src/cahvore.cc
    cpp_src/opencv.c
    cpp_src/poseutils.c
    cpp_src/poseutils-opencv.c
    cpp_src/dogleg.c
    cpp_src/poseutils-uses-autodiff.cc
    cpp_src/triangulation.cc
)

Python3_add_library(
    bindings_triangulation_npsp MODULE 

    cpp_src/bindings_triangulation_npsp_generated.c
    cpp_src/mrcal.c
    cpp_src/stereo.c
    cpp_src/cahvore.cc
    cpp_src/opencv.c
    cpp_src/poseutils.c
    cpp_src/poseutils-opencv.c
    cpp_src/dogleg.c
    cpp_src/poseutils-uses-autodiff.cc
    cpp_src/triangulation.cc
)

target_include_directories(bindings PRIVATE
  ${NUMPY_INCLUDE_DIR}
)

target_include_directories(bindings_npsp PRIVATE
  ${NUMPY_INCLUDE_DIR}
)

target_include_directories(bindings_poseutils_npsp PRIVATE
  ${NUMPY_INCLUDE_DIR}
)

target_include_directories(bindings_triangulation_npsp PRIVATE
  ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(
    bindings 
    
    PRIVATE 
        SuiteSparse::CHOLMOD_static
        freeimage::FreeImage
        freeimage::FreeImagePlus
        LAPACK::LAPACK
        BLAS::BLAS
)

target_link_libraries(
    bindings_npsp
    
    PRIVATE 
        SuiteSparse::CHOLMOD_static
        freeimage::FreeImage
        freeimage::FreeImagePlus
        LAPACK::LAPACK
        BLAS::BLAS
)

target_link_libraries(
    bindings_poseutils_npsp
    
    PRIVATE 
        SuiteSparse::CHOLMOD_static
        freeimage::FreeImage
        freeimage::FreeImagePlus
        LAPACK::LAPACK
        BLAS::BLAS
)

target_link_libraries(
    bindings_triangulation_npsp
    
    PRIVATE 
        SuiteSparse::CHOLMOD_static
        freeimage::FreeImage
        freeimage::FreeImagePlus
        LAPACK::LAPACK
        BLAS::BLAS
)

install(TARGETS bindings DESTINATION drcal)
install(TARGETS bindings_npsp DESTINATION drcal)
install(TARGETS bindings_poseutils_npsp DESTINATION drcal)
install(TARGETS bindings_triangulation_npsp DESTINATION drcal)