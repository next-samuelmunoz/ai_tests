

from itertools import count
import math

import numpy as np


def make_batches_all(x, y, batch_size, shuffle=False):
    """Generator over the data for an epoch.

    PARAMETERS:
      x: np.array
        Samples
      x: np.array or None
        Labels
      batch_size: int
      shuffle: bool
        Wether or not to shuffle the data. Sampling with no replacement.

    Strategy:
      - Iterate over all the data. 1 epoch.
      - Pad last batch with random samples from data.

    NOTE: to make it stocastic, you should shuffle it before.
    """
    if y!=None:
        assert(len(x)==len(y))
    else:
        b_y = None
    i = np.arange(len(x))  # Data indices
    if shuffle:
        np.random.shuffle(i)
    n_batches = range(math.ceil(len(x)/batch_size))
    for i_batch in n_batches:
        b_x = x[i_batch*batch_size : (i_batch+1)*batch_size]
        if y!=None:
            b_y = y[i_batch*batch_size : (i_batch+1)*batch_size]
        if batch_size > len(b_x):  # Pad to fill the last batch
            indices = np.random.randint(low=0, high=len(x), size=batch_size-len(x))
            b_x = np.concatenate(
                [b_x, np.take(b_x, indices, axis=0)],
                axis=0
            )
            if y!=None:
                b_y = np.concatenate(
                    [b_y, np.take(b_y, indices, axis=0)],
                    axis=0
                )
        yield (i_batch, b_x, b_y)


def make_batches_random(x, y, batch_size, stop_after_epoch=False):
    """Generator over a dataset.
    Sampling with replacement

    PARAMETERS:
      x: np.array
        Samples
      x: np.array or None
        Labels
      batch_size: int
      stop_after_epoch: bool
        Exhaust the generator after an epoch or generate batches for ever.

    Strategy:
      - Generate a batch by sampling randomly over the data with no replacement.
      - The generator ends in an epoch.
      - It is NOT guaranteed to visit all the data.
    """
    if y!=None:
        assert(len(x)==len(y))
    if stop_after_epoch:
        batches = range(math.ceil(len(x)/batch_size))
    else:
        batches = count()
    for i_batch in batches:
            indices = np.random.randint(low=0, high=len(x), size=batch_size)
            yield (
                i_batch,
                np.take(x, indices, axis=0),
                np.take(y, indices, axis=0) if y!=None else None
            )
