from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator
import gudhi

from mapper_src.filter import Filter
from mapper_src.clustering import Clustering
from mapper_src.cover import Cover


class Mapper:
    def __init__(self, **kwargs):
        self.filter = Filter(**kwargs)
        self.cover = Cover(**kwargs)
        self.clustering = Clustering(**kwargs)

        # Coordinate for subplot splitting.
        self._coordinate = kwargs.get('coordinate', -1)


    def fit(self, vertices):
        """
        Fits `vertices` by assigning each to a cluster within partitioned cover.
        """

        self.vertices = vertices
        self.numbers = self.filter(vertices)
        self.intervals = self.cover(self.numbers)
        self.n_intervals = len(self.intervals)

        self._initialise_plotting()

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
            _, interval_clusters, cluster_centers = self.clustering(convert_indices(self.vertices), indices)

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

        self._compute_persistent_homology()

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

    def _initialise_plotting(self):
        self._plot_box_aspect = (
            np.ptp(np.array(self.vertices)[:, 0]),
            np.ptp(np.array(self.vertices)[:, 1]),
            np.ptp(np.array(self.vertices)[:, 2])
        )

        self._plot_lim = [
            (np.min(np.array(self.vertices)[:, 0]), np.max(np.array(self.vertices)[:, 0])),
            (np.min(np.array(self.vertices)[:, 1]), np.max(np.array(self.vertices)[:, 1])),
            (np.min(np.array(self.vertices)[:, 2]), np.max(np.array(self.vertices)[:, 2]))
        ]

    def _limit_axis(self, ax):
        if self._coordinate != 0:
            ax.set_xlim3d(*self._plot_lim[0])
        if self._coordinate != 1:
            ax.set_ylim3d(*self._plot_lim[1])
        if self._coordinate != 2 or self._coordinate != -1:
            ax.set_ylim3d(*self._plot_lim[2])


    def _compute_persistent_homology(self):
        "Computes persistence homology of the graph."

        self.rips = gudhi.RipsComplex(
            points=self.node_vertices,
            max_edge_length=40
        )

        self.st = self.rips.create_simplex_tree(
            max_dimension=2
        )

        # Compute persistence diagram.
        self.diag = self.st.persistence(
            homology_coeff_field=2,
            min_persistence=0
        )


    def plot_vertices_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(np.array(self.vertices)[:, 0], np.array(self.vertices)[:, 1], np.array(self.vertices)[:, 2], c=self.numbers)
        ax.set_box_aspect(self._plot_box_aspect)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(sc)
        plt.show()


    def plot_vertices(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(np.array(self.vertices)[:, 0], np.array(self.vertices)[:, 1], c=self.numbers)
        ax.set_aspect('equal')
        fig.suptitle(f'Vertices ({len(self.vertices)})')
        plt.colorbar(sc)
        plt.show()


    def plot_intervals_3d(self):
        fig = plt.figure(figsize=plt.figaspect(self.n_intervals))
        n = plt.Normalize(min(self.numbers), max(self.numbers))

        for i, indices in self.partitions.items():
            ax = fig.add_subplot(self.n_intervals, 1, self.n_intervals-i, projection='3d')
            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            interval_numbers = convert_indices(self.numbers)

            ax.scatter(
                np.array(interval_vertices)[:, 0],
                np.array(interval_vertices)[:, 1],
                np.array(interval_vertices)[:, 2],
                c=interval_numbers,
                norm=n
            )

            self._limit_axis(ax)
            ax.set_box_aspect(self._plot_box_aspect)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        plt.show()


    def plot_intervals(self):
        fig = plt.figure()
        ax = fig.subplots(self.n_intervals, sharex=True)
        n = plt.Normalize(min(self.numbers), max(self.numbers))
        for i, indices in self.partitions.items():
            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            interval_numbers = convert_indices(self.numbers)

            ax[self.n_intervals-i-1].scatter(
                np.array(interval_vertices)[:, 0],
                np.array(interval_vertices)[:, 1],
                c=interval_numbers,
                norm=n
            )
        fig.suptitle(f'Intervals ({self.n_intervals})')
        plt.show()

    def plot_clusters_3d(self):
        fig = plt.figure(figsize=plt.figaspect(self.n_intervals))
        cluster_count = 0

        for i, indices in self.partitions.items():
            ax = fig.add_subplot(self.n_intervals, 1, self.n_intervals-i, projection='3d')

            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            labels, _, cluster_centers = self.clustering(interval_vertices, indices)

            # ax.scatter(
            #     np.array(interval_vertices)[:, 0],
            #     np.array(interval_vertices)[:, 1],
            #     np.array(interval_vertices)[:, 2],
            #     c=labels
            # )
            ax.scatter(*zip(*cluster_centers), 'ro')

            ax.set_box_aspect(self._plot_box_aspect)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            cluster_count += len(cluster_centers)

        fig.suptitle(f'Clusters {cluster_count}')
        plt.show()


    def plot_clusters(self):
        fig = plt.figure()
        ax = fig.subplots(self.n_intervals, sharex=True)
        cluster_count = 0

        for i, indices in self.partitions.items():
            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            labels, _, cluster_centers = self.clustering(interval_vertices, indices)

            ax[self.n_intervals-i-1].scatter(
                np.array(interval_vertices)[:, 0],
                np.array(interval_vertices)[:, 1],
                c=labels
            )
            ax[self.n_intervals-i-1].plot(*zip(*cluster_centers), 'ro', markersize=10)
            cluster_count += len(cluster_centers)

        fig.suptitle(f'Clusters {cluster_count}')
        plt.show()


    def plot_graph_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*zip(*self.node_vertices))
        ax.set_box_aspect(

        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.suptitle(f'Mapper [V = {len(self.node_vertices)}, E = {sum([len(e) for e in self.node_vertices])}]')
        plt.show()


    def plot_graph(self):
        fig, ax = plt.subplots()
        plt.xlim([np.min(np.array(self.vertices)[:,0]), np.max(np.array(self.vertices)[:,0])])
        plt.ylim([np.min(np.array(self.vertices)[:,1]), np.max(np.array(self.vertices)[:,1])])

        # Vertices.
        plt.plot(*zip(*self.node_vertices), 'ko', markersize=10)
        for a, a_neighors in self.nodes.items():
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

    def plot_persistence_homology(self):
        gudhi.plot_persistence_barcode(self.diag, legend=True)
        plt.show()

        gudhi.plot_persistence_diagram(self.diag, legend=True)
        plt.show()
