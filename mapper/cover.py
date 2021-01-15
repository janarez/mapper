import numpy as np
from functools import partial

class Cover:
    """
    Function that partitions numbers to overlapping intervals.
    Returns list of intervals (each interval is a pair of numbers - bounds).
    """
    def __init__(self, cover_function='linear', **kwargs):
        if cover_function == 'linear':
            self._cover_function = partial(
                self._linear,
                bins = kwargs.get('bins', 5),
                overlap = kwargs.get('overlap', 0.25)
            )
        else:
            raise ValueError(f'Argument `cover_function` must be one of: "linear".')


    def __call__(self, numbers):
        return self._cover_function(numbers)


    def _linear(self, numbers, bins, overlap):
        """
        Partitions into `n` bins with `p` percentage overlap.
        """
        # Get interval bounds.
        start = min(numbers)
        end = max(numbers)

        # Compute size and overlap of each bin.
        bin_size = (end - start) / bins
        bin_overlap = overlap * bin_size

        # Compute intervals.
        bin_intervals = [[start + (bin_size - bin_overlap) * i,
                 start + bin_size * (i + 1)]
                for i in range(bins)]

        return bin_intervals
