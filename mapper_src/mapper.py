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

        # Collect all clusters - a.k.a graph nodes 
        # and their intersections - a.k.a graph edges.
        node_index = 0
        self.nodes = {}
        self.node_vertices = []
        prev_clusters = []
        prev_node_index = 0

        for indices in self.partitions.values():
            
            # Cluster vertices in this partion.
            convert_indices = operator.itemgetter(*indices)
            _, interval_clusters, cluster_centers = self.clustering(convert_indices(self.vertices) , self._clustering_distance)

            # Find connections.
            for cluster_a, cluster_a_center in zip(interval_clusters, cluster_centers):
                self.nodes[node_index] = []
                self.node_vertices.append(cluster_a_center)
                # Counting on sorted 1D intervals.
                for offset, cluster_b in enumerate(prev_clusters):
                    if len(cluster_a & cluster_b) > 0:
                        self.nodes[node_index].append(prev_node_index+offset)
                node_index += 1

            prev_clusters = interval_clusters
            prev_node_index = node_index - len(interval_clusters)

        return self.nodes
        
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

            ax[len(self.intervals)-i-1].scatter(
                np.array(interval_vertices)[:, 0], 
                np.array(interval_vertices)[:, 1],
                c=interval_numbers,
                norm=n
            )
        fig.suptitle(f'Intervals ({len(self.intervals)})')
        plt.show()

    def plot_clusters(self):
        fig = plt.figure()
        ax = fig.subplots(len(self.intervals), sharex=True)

        for i, indices in self.partitions.items():
            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            labels, _, cluster_centers = self.clustering(interval_vertices, self._clustering_distance)

            ax[len(self.intervals)-i-1].scatter(
                np.array(interval_vertices)[:, 0], 
                np.array(interval_vertices)[:, 1],
                c=labels
            )
            ax[len(self.intervals)-i-1].plot(*zip(*cluster_centers), 'ro', markersize=10)

        fig.suptitle(f'Clusters (COUNT HERE)')
        plt.show()

    def plot_graph(self):
        fig, ax = plt.subplots()
        plt.xlim([np.min(np.array(self.vertices)[:,0]), np.max(np.array(self.vertices)[:,0])])
        plt.ylim([np.min(np.array(self.vertices)[:,1]), np.max(np.array(self.vertices)[:,1])])

        # Plot the resulting triangulation.
        # Vertices.
        plt.plot(*zip(*self.node_vertices), 'ko', markersize=10)
        for a, a_neighors in self.nodes.items():
            print(a, a_neighors)
            for b in a_neighors:
                vertex_a = self.node_vertices[a]
                vertex_b = self.node_vertices[b]
                plt.plot(
                    [vertex_a[0], vertex_b[0]],
                    [vertex_a[1], vertex_b[1]],
                    color='b', 
                    linewidth=2,
                    zorder=0
                )

        fig.suptitle(f'Mapper graph')
        plt.show()




