import json
import argparse

from . import *

parser = argparse.ArgumentParser("Liesym")
parser.add_argument("--group", "-g", required=True, type=str, help="Name of group")
parser.add_argument("--dim", "-d", required=True, type=int, help="Dim of group")
parser.add_argument("--output-format")