# drcal: a sanity-preserving fork of mrcal.

Mrcal is amazing work, and I want to be able to `pip install` it.
I don't want to jump through any hoops to use it as a runtime dependency, and I don't like hacks.

This is my attempt to make it usable in a sane software stack.

# TODO

- Put the generation of the `numpysane` bindings into the build script
- Remove all dependency on `numpysane`
- Convert bindings to `pybind11` or `nanobind`
- Make the python code pass
  - type checking
  - `ruff check`

* Put on PyPi, using `cibuildwheel` for automatic wheel building.
* Salvage the tests
