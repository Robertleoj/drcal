cmake_minimum_required(VERSION 3.15)
# If no toolchain set, try to pick it up from the env
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain file")
  endif()
endif()


project(drcal LANGUAGES CXX)

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

# add_subdirectory(external/SuiteSparse/CHOLMOD)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find installed pybind11
# find_package(pybind11 REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, using FetchContent to grab it")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        stable
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Build the Python module
pybind11_add_module(bindings cpp_src/bindings.cpp)

target_link_libraries(bindings PRIVATE SuiteSparse::CHOLMOD_static)

install(TARGETS bindings DESTINATION drcal)