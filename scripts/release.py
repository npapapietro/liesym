import toml
from os.path import join, abspath, dirname
import sys
import re

root = abspath(join(dirname(__file__), ".."))


def update_cargo(ver):
    path = join(root, "Cargo.toml")
    raw = toml.load(path)
    raw['package']['version'] = ver
    with open(path, "w") as f:
        toml.dump(raw, f)


def update_pyproject(ver):
    path = join(root, "pyproject.toml")
    raw = toml.load(path)
    raw['tools']['poetry']['version'] = ver
    with open(path, "w") as f:
        toml.dump(raw, f)


def update_setuppy(ver):
    path = join(root, "setup.py")
    with open(path) as f:
        raw = f.read()
    pat = r"version\=\'(\d+\.\d+\.\d+)\'\,"
    raw = re.sub(pat, "version=\'" + ver + "\',", raw)
    with open(path, "w") as f:
        f.write(raw)


def update_sphinx(ver):
    path = join(root, "docs", "source", "conf.py")
    with open(path) as f:
        raw = f.read()
    pat = r"release\s\=\s\'(\d+\.\d+\.\d+)\'"
    raw = re.sub(pat, "release = \'" + ver + "\'", raw)
    with open(path, "w") as f:
        f.write(raw)


if __name__ == "__main__":
    ver = sys.argv[1]
    update_cargo(ver)
    update_pyproject(ver)
    update_setuppy(ver)
    update_sphinx(ver)
