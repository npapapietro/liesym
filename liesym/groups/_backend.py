from typing import List, Protocol

from sympy import Matrix

from liesym import _BranchingRuleBackend, _LieGroupBackend


class _BranchingRuleBackendWrapper:
    def __init__(self, backend):
        self.backend = backend

    def projection_matrix(self) -> Matrix:
        return Matrix(self.backend.projection_matrix())

    def maximal_subalgebras(self):
        return self.backend.maximal_subalgebras()

    def __str__(self) -> str:
        return self.backend.__str__()

    def __repr__(self) -> str:
        return self.backend.__repr__()


BranchingRule = _BranchingRuleBackendWrapper


class _LieGroupBackendWrapped:
    def __init__(self, group_type: str, dim: int) -> None:
        self.backend = _LieGroupBackend(group_type, dim)

    def maximal_subalgebras(self) -> List[BranchingRule]:
        return [
            _BranchingRuleBackendWrapper(x) for x in self.backend.maximal_subalgebras()
        ]
