import numpy as np

class Mapper:
    def __init__(self):
        self.filter = Filter()
        self.partitioner = Partitioner()
        self.clustering = Clustering()

    def process(self, vertices):
        """
        Processes `vertices` and saves the results for further analysis.
        """

        self.numbers = self.filter(vertices)
        self.intervals = self.partitioner(self.numbers)

        # Cluster each interval.
        self.vertex_nodes = np.zeros(len(vertices)) # index of node for each vertex
        self.node_count = 0
        for start, end in self.intervals:
            # Get vertices in this interval.
            def f(x):
                "Sorts by number."
                _, n = x
                return start <= n and n <= end
            def s(x):
                "Selects index."
                return x[0]
            indices = np.array(list(map(s, filter(f, enumerate(self.numbers)))))
            interval_vertices = vertices[indices]

            # Cluster vertices in this interval.
            local_nodes = self.clustering(interval_vertices)

            # Append to global node list.
            for vertex_index in indices:
                # Shift local node index to make it unique among global nodes.
                global_node_index = self.node_count + local_nodes[vertex_index]
                self.vertex_nodes[vertex_index] = global_node_index
            self.node_count += max(local_nodes) + 1

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
