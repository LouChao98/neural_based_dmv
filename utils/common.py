"""data type helper and constants"""

from typing import Sequence
import cupy as cp
import numpy as np
import torch

# float
cpf = cp.float32
npf = np.float32

# int
cpi = cp.int32
npi = np.int32

# zeros


def cpizeros(*args, **kwargs):
    return cp.zeros(*args, **kwargs, dtype=cpi)


def cpfzeros(*args, **kwargs):
    return cp.zeros(*args, **kwargs, dtype=cpf)


def npizeros(*args, **kwargs):
    return np.zeros(*args, **kwargs, dtype=npi)


def npfzeros(*args, **kwargs):
    return np.zeros(*args, **kwargs, dtype=npf)

# fill


def cpifull(*args, **kwargs):
    return cp.full(*args, **kwargs, dtype=cpi)


def cpffull(*args, **kwargs):
    return cp.full(*args, **kwargs, dtype=cpf)


def npifull(*args, **kwargs):
    return np.full(*args, **kwargs, dtype=npi)


def npffull(*args, **kwargs):
    return np.full(*args, **kwargs, dtype=npf)

# empty


def cpiempty(*args, **kwargs):
    return cp.empty(*args, **kwargs, dtype=cpi)


def cpfempty(*args, **kwargs):
    return cp.empty(*args, **kwargs, dtype=cpf)


def npiempty(*args, **kwargs):
    return np.empty(*args, **kwargs, dtype=npi)


def npfempty(*args, **kwargs):
    return np.empty(*args, **kwargs, dtype=npf)


"""asarray func
only handle following conditions:
1. np.ndarray
2. cp.ndarray
3. a list of np.ndarray or cp.ndarray, like [[np.ndarray, ..], [np.ndarray, ...]]
4. a list of int, float, numpy element or cupy element.
in condition 3 and 4, auto dtype is determined only by the first item in a flatterned
"""


def cpasarray(a, dtype=None):
    if dtype is None:
        if isinstance(a, cp.ndarray):
            is_int = cp.issubdtype(a.dtype, cp.integer)
        elif isinstance(a, np.ndarray):
            is_int = np.issubdtype(a.dtype, np.integer)
        elif isinstance(a, Sequence):
            _a = a[0]
            while isinstance(_a, Sequence):
                _a = _a[0]
            is_int = type(_a) == int \
                or cp.issubdtype(_a.dtype, cp.integer) \
                or np.issubdtype(_a.dtype, np.integer)
        else:
            raise ValueError(f"Cannot deal with type {type(a)}")
        dtype = cpi if is_int else cpf
    return cp.asarray(a, dtype)


def npasarray(a, dtype=None):
    if dtype is None:
        if isinstance(a, cp.ndarray):
            is_int = cp.issubdtype(a.dtype, cp.integer)
            a = a.get()
        elif isinstance(a, np.ndarray):
            is_int = np.issubdtype(a.dtype, np.integer)
        elif isinstance(a, Sequence):
            _a = a[0]
            while isinstance(_a, Sequence):
                _a = _a[0]
            is_int = type(_a) == int \
                or cp.issubdtype(_a.dtype, cp.integer) \
                or np.issubdtype(_a.dtype, np.integer)
        else:
            raise ValueError(f"Cannot deal with type {type(a)}")
        dtype = npi if is_int else npf
    return np.asarray(a, dtype)


# constant
STOP = 0  # decision
GO = 1  # decision
NOCHILD = 0  # valence
HASCHILD = 1  # valence

DEBUG = True
