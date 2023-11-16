from liesym import algebras
from ._base import LieGroup


class F4(LieGroup):
    def __new__(cls):
        return super().__new__(cls, "F4", 4)

    def __init__(self):
        self._algebra = algebras.F4()
        super().__init__()


class G2(LieGroup):
    def __new__(cls):
        return super().__new__(cls, "G2", 2)

    def __init__(self):
        self._algebra = algebras.G2()
        super().__init__()


class E(LieGroup):
    def __new__(cls, n: int):
        obj = super().__new__(cls, "E", n)
        obj._algebra = algebras.E(n)
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__()
