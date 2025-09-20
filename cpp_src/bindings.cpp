#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(
    bindings,
    m
) {
    m.doc() = "Bindings to drcal";

    m.def(
        "example_add",
        [](int a, int b) {
            return a + b;
        },
        py::arg("a"),
        py::arg("b")
    );
}