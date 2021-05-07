#!/bin/python
import os
import subprocess
import shutil

DIR = "/app"

if not os.path.isdir(DIR):
    os.mkdir(DIR)


def tex(x): return (
    r"""\documentclass[convert={density=1000,size=1080x800,outext=.png}]{standalone}
\usepackage{tikz}
\usepackage{dynkin-diagrams}
\begin{document}
\dynkin """ + x + r"""{}
\end{document}"""
)


def compile(t):
    file_ = os.path.join(DIR, "type_" + t + ".tex")
    print("Starting type", t)
    with open(file_, "w") as f:
        f.write(tex(t))

    process = subprocess.Popen(["pdflatex", "-shell-escape", file_],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=DIR)

    # wait for the process to terminate
    process.communicate()

    
    shutil.copy(os.path.join(DIR,"type_" + t + ".png"), "/app/outputs/type_" + t + ".png")


for t in ["A", "B", "C", "D", "E6", "E7", "E8", "G2", "F4"]:
    compile(t)
