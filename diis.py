import itertools as it
import functools as ft
import numpy as np
import scipy.linalg as spla


# Public
def extrapolate(p_series, r_series):
    assert len(p_series) == len(r_series) >= 2
    a = a_matrix(r_series)
    b = b_vector(len(r_series))
    x = spla.solve(a, b)
    c = x[:-1]
    return _linear_combination(p_series, c)


def a_matrix(r_series):
    n = len(r_series)
    a = np.zeros((n + 1, n + 1))
    r_pairs = it.combinations_with_replacement(r_series, r=2)
    a[np.triu_indices(n)] = tuple(it.starmap(_vdot, r_pairs))
    a[np.tril_indices(n, k=-1)] = a.T[np.tril_indices(n, k=-1)]
    a[n, range(n)] = a[range(n), n] = -1
    return a


def b_vector(n):
    b = np.zeros((n + 1,))
    b[n] = -1
    return b


# Private
def _vdot(p1, p2):
    if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return np.vdot(p1, p2)
    try:
        return sum(it.starmap(np.vdot, zip(p1, p2)))
    except:
        raise Exception("Could not align array collections for vdot.")


def _linear_combination(p_series, weights):
    assert len(p_series) is len(weights)
    weight = ft.partial(np.tensordot, weights, axes=(0, 0))
    return tuple(map(weight, zip(*p_series)))


def _main():
    extrapolate((np.random.rand(4),), (np.eye(4),))
    colls = [(np.arange(5), np.ones((3, 3)))] * 3
    scalars = [1, 2, 3]
    print(_linear_combination(colls, scalars))

if __name__ == "__main__":
    _main()
