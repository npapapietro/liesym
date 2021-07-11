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
and allows easy communication through numpy using [rust-numpy](https://github.com/PyO3/rust-numpy)
as well as numpy like api inside rust using [ndarray](https://github.com/rust-ndarray/ndarray).

## Install

```bash
pip install liesym
```

## Examples

See also example [notebook](notebooks/Example.ipynb)

```python
import liesym as ls
from sympy import Matrix
```

### Cartan Matrix


```python
A3 = ls.A(3)
print(A3.cartan_matrix)
```


    Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])



### Positive Roots


```python
A3.positive_roots
```




    [Matrix([[1, 0, 1]]),
     Matrix([[-1, 1, 1]]),
     Matrix([[1, 1, -1]]),
     Matrix([[-1, 2, -1]]),
     Matrix([[0, -1, 2]]),
     Matrix([[2, -1, 0]])]



### Simple Roots


```python
A3.simple_roots
```




    [Matrix([[1, -1, 0, 0]]), Matrix([[0, 1, -1, 0]]), Matrix([[0, 0, 1, -1]])]



### Fundamental Weights


```python
A3.fundamental_weights # Orthogonal Basis
```




    [Matrix([[3/4, -1/4, -1/4, -1/4]]),
     Matrix([[1/2, 1/2, -1/2, -1/2]]),
     Matrix([[1/4, 1/4, 1/4, -3/4]])]



### Dimension of representation
  


```python

print("Dim | Rep (Omega)")
print("---------")
for i in A3.fundamental_weights:
    print(" ", A3.dim(i), "|", A3.to_omega(i))
```

    Dim | Rep (Omega)
    ---------
      4 | Matrix([[1, 0, 0]])
      6 | Matrix([[0, 1, 0]])
      4 | Matrix([[0, 0, 1]])


### Name of rep

Commonly in literature (especially physics), names of the reps are the dimension rather than the matrix rep.


```python
A3.dim_name(Matrix([[0, 0, 1]]))
```




$\displaystyle \bar{4}$




```python
A3.irrep_lookup(r"\bar{4}")
```




$\displaystyle \left[\begin{matrix}0 & 0 & 1\end{matrix}\right]$



### Tensor product decomps

The decomp of irreps from a product of irreps


```python
results = A3.tensor_product_decomposition([
    Matrix([[1,0,0]]),
    Matrix([[1,0,0]]),
])

for i in results:
    print("Rep:", A3.to_omega(i),"Dim Name:", A3.dim_name(i))
```

    Rep: Matrix([[0, 1, 0]]) Dim Name: 6
    Rep: Matrix([[2, 0, 0]]) Dim Name: \bar{10}



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
├── src                 # rust source code
├── tests               # python tests
└── ...                 # other stuff
```

## Docs

Read the documentation at https://npapapietro.github.io/liesym/
