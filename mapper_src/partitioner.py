import numpy as np

class Partitioner:
    """
    Function that partitions numbers to overlapping intervals.
    Returns list of intervals (each interval is a pair of numbers - bounds).
    """

    def __call__(self, numbers):
        return np.array([min(numbers), max(numbers)]) # one interval


class BinPartitioner(Partitioner):
    """
    Partitions into `n` bins with `p` percentage overlap.
    """

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self, numbers):
        # Get interval bounds.
        start = min(numbers)
        end = max(numbers)

        # Compute size and overlap of each bin.
        bin_size = (end - start) / self.n
        bin_overlap = self.p * bin_size

        # Compute intervals.
        bins = [[start + (bin_size - bin_overlap) * i,
                 start + bin_size * (i + 1)]
                for i in range(self.n)]
        return np.array(bins)
