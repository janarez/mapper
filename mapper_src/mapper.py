from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mapper_src.filter import Filter
from mapper_src.clustering import Clustering
from mapper_src.cover import Cover
import operator


class Mapper:
    def __init__(self, **kwargs):
        self.filter = Filter(kwargs.get('filter_function', 'last_coordinate'))
        self.cover = Cover(kwargs.get('cover_function', 'linear'), **kwargs)
        self.clustering = Clustering(kwargs.get('clustering_function', 'agglomerative'))
        self._clustering_distance = kwargs.get('distance', 2)

    def fit(self, vertices):
        """
        Fits `vertices` by assigning each to a cluster within partitioned cover.
        """

        self.vertices = vertices
        self.numbers = self.filter(vertices)
        self.intervals = self.cover(self.numbers)

        # Sort intervals and numbers so that we assign them in order into partitions.
        self.numbers, self.vertices = zip(*sorted(zip(self.numbers, self.vertices), key=lambda t: t[0]))
        self.intervals.sort(key=lambda t: (t[0], t[1]))
        self._assign_partitions()

        for indices in self.partitions.values():

            # Cluster vertices in this partion.
            convert_indices = operator.itemgetter(*indices)
            local_nodes = self.clustering(convert_indices(self.vertices) , self._clustering_distance)

            # # Append to global node list.
            # for local_node, vertex_index in zip(local_nodes, indices):
            #     # Shift local node index to make it unique among global nodes.
            #     global_node_index = self.node_count + local_node
            #     self.vertex_nodes[vertex_index] = global_node_index
            # self.node_count += max(local_nodes) + 1

    def _assign_partitions(self):
        "Distributes vertices to cover partitions given by intervals."
        self.partitions = defaultdict(set)
        
        min_start = 0
        unused = 0

        for i, n in enumerate(self.numbers):
            min_start += unused
            unused = 0
            for offset, (a, b) in enumerate(self.intervals[min_start:]):
                if a <= n and b >= n:
                    self.partitions[min_start+offset].add(i)
                elif a > n:
                    break   # Too large.
                elif b < n:
                    unused += 1 # Too low.
        

    def plot_vertices(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(np.array(self.vertices)[:, 0], np.array(self.vertices)[:, 1], c=self.numbers)
        ax.set_aspect('equal')
        fig.suptitle(f'Vertices ({len(self.vertices)})')
        plt.colorbar(sc)
        plt.show()

    def plot_intervals(self):
        fig = plt.figure()
        ax = fig.subplots(len(self.intervals), sharex=True)
        n = plt.Normalize(min(self.numbers), max(self.numbers))
        for i, indices in self.partitions.items():
            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            interval_numbers = convert_indices(self.numbers)

            ax[len(self.intervals) - i - 1].scatter(
                np.array(interval_vertices)[:, 0], 
                np.array(interval_vertices)[:, 1],
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
            local_nodes = self.clustering(interval_vertices, self._clustering_distance)

            ax.scatter(
                interval_vertices[:, 0], interval_vertices[:, 1],
                c=local_nodes
            )
        fig.suptitle(f'Clusters ({self.node_count})')
        plt.show()




