import numpy as np


def spinorb(array, pairs):
    pairs = iter(pairs)
    try:
        pair = next(pairs)
    except StopIteration:
        return array
    array = np.moveaxis(array, pair, (-2, -1))
    array = np.kron(np.eye(2), array)
    array = np.moveaxis(array, (-2, -1), pair)
    return spinorb(array, pairs)
