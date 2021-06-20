from __future__ import unicode_literals
from typing import List

import sys
import numpy as np


def print_color(color: int, bright: int = 1, outfile=sys.stdout) -> None:
    if bright:
        outfile.write("\x1b[3{};1m".format(color))
    else:
        outfile.write("\x1b[3{}m".format(color))


def print_reset(outfile=sys.stdout) -> None:
    outfile.write("\x1b[0m")


def print_up(n: int, outfile=sys.stdout) -> None:
    for _ in range(n):
        outfile.write("\033[A")


def print_down(n: int, outfile=sys.stdout) -> None:
    for _ in range(n):
        outfile.write("\033[B")


def print_forward(n: int, outfile=sys.stdout) -> None:
    for _ in range(n):
        outfile.write("\033[C")


def print_back(n: int, outfile=sys.stdout) -> None:
    for _ in range(n):
        outfile.write("\033[D")


def print_puyo(color: int, outfile=sys.stdout) -> None:
    print_color(
        (color % 7) + 1,
        bright=(1 + color // 7) % 2,
        outfile=outfile,
    )
    outfile.write("\u25cf ")


def permute(seq: np.ndarray, permutation: List[int]):
    """
    Permute a sequence in-place according to a list of indices
    """
    temp = seq[:]
    for where, to in enumerate(permutation):
        seq[to] = temp[where]
