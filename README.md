# liesym

Lie Algebras using Sympy and backend powered by Rust's pyO3 and ndarray

## Overview

In an effort to supply python with the same computer algebra software (CAS)
capabilities, [SymPy](https://github.com/sympy/sympy) was written. This python
library is well written and allows an open source alternative to proprietary
choices like Mathematica/WolframLanguage and Maple. Due to the nature of
how SymPy was written, certain symbolic calculation can be extremely unoptimized
in python. Even using numpy could offer little speed ups as it is not geared
towards rational numbers (fractions). Sympy does currently offer a `liealgebras`
module, but due to the performance limitations, certain tradeoffs had to be
made such as locking the basis for the classic lie algebras in favor of speed.
This is a fair trade off, but would require anyone using a different basis
to hand calculate the representations of the algebra all over again.
An alternative to solve this problem would be to use a compiled
backend that supports generics (and isn't a pain to build with python).

Rust has good python binding support through [py03](https://github.com/PyO3/pyo3)
and allows easy communication through numpy using [rust-numpy](https://github.com/PyO3/rust-numpy) as well as numpy like api inside rust using [ndarray](https://github.com/rust-ndarray/ndarray).

## Repo Layout

If you are new to how python and rust are tied together with py03, below
is a simple top level layout of this repository. When the rust code is
build into a lib binary, it is put into into `./liesym` so it can be
imported in `./liesym/__init__.py`

```
.
├── Cargo.toml          # rust config file
├── README.md           # you are here
├── liesym              # python module
├── notebooks           # jupyter notebooks
├── setup.cfg           # pytest config
├── setup.py            # python setuptools file
├── src                 # rust source code
├── tests               # python tests
└── ...                 # other stuff
```

