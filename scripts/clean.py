import shutil as sh
targets = [
    "target",
    "liesym.egg-info",
    "build",
    ".eggs",
    ".pytest_cache"
]

for t in targets:
    sh.rmtree(t, ignore_errors=True)
