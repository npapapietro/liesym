from sympy import Symbol, pi, I, exp

from liesym import Z

def test_z():
    z5 = Z(5)

    g = z5.generators()

    assert [1] == z5.product(*g)

    assert [Symbol("Z_3")] == z5.sym_product("Z_1", "Z_2")

    assert [(Symbol("Z_2"), exp(4*I*pi/5))] == z5.sym_product("Z_3", "Z_4", as_tuple=True)