"""
mytools - A collection of python tools used in my HI filament stacking simulation work, arxiv:2411.03988. 
"""

__version__ = "0.1.0"
__author__ = "dyliu <dyliu0312@gmail.com>"


from . import (
    bins,
    calculation,
    constant,
    data,
    estimate,
    halo,
    plot,
    stack,
)

__all__ = [
    "bins",
    "calculation",
    "constant",
    "data",
    "estimate",
    "halo",
    "plot",
    "stack",
]