__doc__ = """Gets around the circular import issue"""

def get_algebra(name: str):
    letter = name[0].lower()
    num = int(name[1:])
    if letter == 'a':
        import liesym.algebras._classic as classic
        return classic.A(num)
    if letter == 'b':
        import liesym.algebras._classic as classic
        return classic.B(num)
    if letter == 'c':
        import liesym.algebras._classic as classic
        return classic.C(num)
    if letter == 'd':
        import liesym.algebras._classic as classic
        return classic.D(num)
    if letter == 'e':
        import liesym.algebras._exceptionals as exp
        return exp.E(num)
    if letter == 'f':
        import liesym.algebras._exceptionals as exp
        return exp.F4()
    if letter == 'g':
        import liesym.algebras._exceptionals as exp
        return exp.G2()