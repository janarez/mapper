import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster

class Mapper:
    def __init__(self):
        self.filter = Filter()
        self.partitioner = Partitioner()
        self.clustering = Clustering()

    def process(self, vertices):
        """
        Processes `vertices` and saves the results for further analysis.
        """

        self.vertices = vertices
        self.numbers = self.filter(vertices)
        self.intervals = self.partitioner(self.numbers)

        # Cluster each interval.
        self.vertex_nodes = np.zeros(len(vertices)) # index of node for each vertex
        self.node_count = 0
        for start, end in self.intervals:
            # Get vertices in this interval.
            indices = self.find_interval_vertices(start, end)
            interval_vertices = vertices[indices]

            # Cluster vertices in this interval.
            local_nodes = self.clustering(interval_vertices)

            # Append to global node list.
            for local_node, vertex_index in zip(local_nodes, indices):
                # Shift local node index to make it unique among global nodes.
                global_node_index = self.node_count + local_node
                self.vertex_nodes[vertex_index] = global_node_index
            self.node_count += max(local_nodes) + 1

    def find_interval_vertices(self, start, end):
        "Finds indices of vertices that lie in the specified interval."

        def f(x):
            "Sorts by number."
            _, n = x
            return start <= n and n <= end

        def s(x):
            "Selects index."
            return x[0]

        return np.array(list(map(s, filter(f, enumerate(self.numbers)))))

    def plot_vertices(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], c=self.numbers)
        ax.set_aspect('equal')
        fig.suptitle(f'Vertices ({len(self.vertices)})')
        plt.colorbar(sc)
        plt.show()

    def plot_intervals(self):
        fig = plt.figure()
        axs = fig.subplots(len(self.intervals), sharex=True)
        n = plt.Normalize(min(self.numbers), max(self.numbers))
        for ax, (start, end) in zip(reversed(axs), self.intervals):
            indices = self.find_interval_vertices(start, end)
            interval_vertices = self.vertices[indices]
            interval_numbers = self.numbers[indices]

            ax.scatter(
                interval_vertices[:, 0], interval_vertices[:, 1],
                c=interval_numbers,
                norm=n
            )
        fig.suptitle(f'Intervals ({len(self.intervals)})')
        plt.show()

    def plot_clusters(self):
        fig = plt.figure()
        axs = fig.subplots(len(self.intervals), sharex=True)
        for ax, (start, end) in zip(reversed(axs), self.intervals):
            indices = self.find_interval_vertices(start, end)
            interval_vertices = self.vertices[indices]
            local_nodes = self.clustering(interval_vertices)

            ax.scatter(
                interval_vertices[:, 0], interval_vertices[:, 1],
                c=local_nodes
            )
        fig.suptitle(f'Clusters ({self.node_count})')
        plt.show()

class Filter:
    """
    Function from vertices to some number (e.g., distance).
    """

    def __call__(self, vertices):
        return np.zeros(vertices.shape[0])

class Partitioner:
    """
    Function that partitions numbers to overlapping intervals.
    Returns list of intervals (each interval is a pair of numbers - bounds).
    """

    def __call__(self, numbers):
        return np.array([min(numbers), max(numbers)]) # one interval

class Clustering:
    """
    Function clustering vertices to nodes.
    Returns list of node indices for each vertex.
    """

    def __call__(self, vertices):
        return np.zeros(len(vertices)) # one node

class HeightFilter(Filter):
    """
    Uses last coordinate of vertices.
    """

    def __call__(self, vertices):
        return vertices[:, -1]

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

class SciKitClustering(Clustering):
    """
    Uses existing clustering implementation from `scikit-learn`.
    """

    def __init__(self, cluster):
        self.cluster = cluster

    def __call__(self, vertices):
        self.cluster.fit(vertices)
        return self.cluster.labels_

def noisy_circle(points=100, noise=0.1):
    """
    Generates points of a noisy circle (with radius 1).
    """

    # Generate circle coordinates.
    d = np.linspace(0, 2 * np.pi, points, endpoint=False)
    x = np.cos(d)
    y = np.sin(d)

    # Add Gaussian noise.
    x = np.random.normal(x, noise)
    y = np.random.normal(y, noise)

    return np.vstack((x, y)).T

vertices = noisy_circle()

mapper = Mapper()
mapper.filter = HeightFilter()
mapper.partitioner = BinPartitioner(5, 0.25)
mapper.clustering = SciKitClustering(sklearn.cluster.AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=2
))

mapper.process(vertices)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
