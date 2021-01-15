from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator
import gudhi
import networkx as nx

from .filter import Filter
from .clustering import Clustering
from .cover import Cover


class Mapper:
    """Mapper algorithm.

    Parameters
    ----------
    filter_function : {"by_coordinate", "distance_from_point"},
        default="by_coordinate" Filtering function to use.

    clustering_function : {"tomato", "agglomerative"}, default="tomato"
        Clustering function to use.

    bins : int, default=5 The number of bins of cover of the `filter_function`
        range.

    overlap : float, default=0.25 Bins overlap within cover.

    coordinate : int, default=-1 Coordinate to filter by if `filter_function` is
        "by_coordinate".

    point : array-like of shape (3,), default=[0,0,0] Point to filter by if
        `filter_function` is "distance_from_point".

    linkage : {"ward", "complete", "average", "single"}, default="ward" Which
        linkage criterion to use for `clustering_function` "agglomerative".

    distance : float, default=None The distance function as interpreted by
        selected `clustering_function`:

        - "tomato" the `merge_threshold` parameter. If `distance` equals None, then it
        is automatically set to `sys.maxsize` for optimal results.
        - "agglomerative" the `distance_threshold` parameter.

    Methods
    --------
    fit(X) : Fits the mapper from array of 3D points.

    plot_vertices : Plots the initial input points colored by filter value.
    plot_intervals : Plots the partitioned input points.
    plot_clusters : Plots the clusters within each partition.
    plot_graph : Plots the mapper graph inside original 3D space.
    plot_graph_in_plane :  Plots the mapper graph embedded into plane.
    plot_persistence_homology(type="filter") : Plots the persistence homology of mapper simplex
    constructed either from the filter function map or as a Rips complex from cluster centres
    positions. Use type : {"filter", "Rips"}.

    Examples
    --------
    >>> from mapper import Mapper
    >>> import numpy as np
    >>> n = 100
    >>> d = np.linspace(0, 2 * np.pi, n, endpoint=False)
    >>> points = np.vstack((np.cos(d), np.sin(d), np.zeros(n))).T
    >>> mapper = Mapper(coordinate=1)
    >>> mapper.fit(points)
    >>> mapper.plot_graph_in_plane()
    """
    def __init__(self, **kwargs):
        self.filter = Filter(**kwargs)
        self.cover = Cover(**kwargs)
        self.clustering = Clustering(**kwargs)

        # Coordinate for subplot splitting.
        self._coordinate = self.filter.coordinate


    def fit(self, vertices):
        """
        Fits `vertices` by assigning each to a cluster within partitioned cover.
        """

        self.vertices = vertices
        self.numbers = self.filter(vertices)
        self.intervals = self.cover(self.numbers)
        self.n_intervals = len(self.intervals)

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

        self._initialise_plotting()
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

        self._norm = plt.Normalize(min(self.numbers), max(self.numbers))

        self._node_numbers = self.filter(np.array(self.node_vertices))

        self._sc = None

    def _limit_axis(self, ax):
        if self._coordinate != 0:
            ax.set_xlim3d(*self._plot_lim[0])
        if self._coordinate != 1:
            ax.set_ylim3d(*self._plot_lim[1])
        if not (self._coordinate == 2 or self._coordinate == -1):
            ax.set_zlim3d(*self._plot_lim[2])


    def _compute_persistent_homology(self):
        "Computes persistence homology of the graph."

        # # Compute Rips filtration on vertices.
        # self._rips = gudhi.RipsComplex(
        #     points=self.node_vertices,
        #     max_edge_length=40
        # )
        # self._st = self._rips.create_simplex_tree(
        #     max_dimension=2
        # )

        self._st = gudhi.SimplexTree()

        # Add artificial triangle.
        n = len(self.nodes)
        self._st.insert([0, n, n+1], filtration=self._node_numbers[0])

        # Add vertices.
        for i, n in enumerate(self._node_numbers):
            self._st.insert([i], filtration=n)

        # Add edges.
        for a, neighbors in self.nodes.items():
            for b in neighbors:
                f = max(self._node_numbers[a], self._node_numbers[b])
                self._st.insert([a, b], filtration=f)

        # Compute persistence diagram.
        self.diag = self._st.persistence(
            homology_coeff_field=2,
            min_persistence=0
        )


    def plot_vertices(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self._sc = ax.scatter(
            np.array(self.vertices)[:, 0],
            np.array(self.vertices)[:, 1],
            np.array(self.vertices)[:, 2],
            c=self.numbers,
            norm=self._norm
        )
        ax.set_box_aspect(self._plot_box_aspect)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.suptitle(f'Vertices ({len(self.vertices)})')
        plt.colorbar(self._sc)
        plt.show()


    def plot_intervals(self):
        fig = plt.figure(figsize=plt.figaspect(1 / self.n_intervals))

        for i, indices in self.partitions.items():
            ax = fig.add_subplot(1, self.n_intervals, i + 1, projection='3d')
            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            interval_numbers = convert_indices(self.numbers)

            ax.scatter(
                np.array(interval_vertices)[:, 0],
                np.array(interval_vertices)[:, 1],
                np.array(interval_vertices)[:, 2],
                c=interval_numbers,
                norm=self._norm
            )

            self._limit_axis(ax)
            ax.set_box_aspect(self._plot_box_aspect)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        fig.suptitle(f'Intervals ({self.n_intervals})')
        plt.show()


    def plot_clusters(self):
        fig = plt.figure(figsize=plt.figaspect(1 / self.n_intervals))
        cluster_count = 0

        for i, indices in self.partitions.items():
            ax = fig.add_subplot(1, self.n_intervals, i+1, projection='3d')

            convert_indices = operator.itemgetter(*indices)
            interval_vertices = convert_indices(self.vertices)
            labels, _, cluster_centers = self.clustering(interval_vertices, indices)

            ax.scatter(
                np.array(interval_vertices)[:, 0],
                np.array(interval_vertices)[:, 1],
                np.array(interval_vertices)[:, 2],
                c=labels
            )
            ax.scatter(*zip(*cluster_centers), 'o', s=100, c='r')

            self._limit_axis(ax)
            ax.set_box_aspect(self._plot_box_aspect)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            cluster_count += len(cluster_centers)

        fig.suptitle(f'Clusters ({cluster_count})')
        plt.show()


    def plot_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot vertices.
        ax.scatter(
            *zip(*self.node_vertices),
            s=50,
            c=self._node_numbers,
            norm=self._norm
        )
        ax.set_box_aspect(self._plot_box_aspect)

        # Plot edges.
        for a, a_neighors in self.nodes.items():
            for b in a_neighors:
                vertex_a = self.node_vertices[a]
                vertex_b = self.node_vertices[b]
                ax.plot(
                    [vertex_a[0], vertex_b[0]],
                    [vertex_a[1], vertex_b[1]],
                    [vertex_a[2], vertex_b[2]],
                    color='k',
                    linewidth=2,
                    zorder=0
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.suptitle(f'Mapper [V = {len(self.node_vertices)}, E = {sum([len(e) for e in self.nodes.values()])}]')
        if self._sc is not None:
            plt.colorbar(self._sc)
        plt.show()


    def plot_graph_in_plane(self, *, seed=None):

        # Initialize NetworkX graph.
        g = nx.Graph()
        for i in range(len(self.node_vertices)):
            g.add_node(i)
        for a, a_neighors in self.nodes.items():
            for b in a_neighors:
                g.add_edge(a, b)

        # Put graph in plane.
        pos = nx.spring_layout(g, seed=seed)

        nx.draw_networkx_nodes(
            g, pos,
            node_color=self._node_numbers,
            vmin=self._norm.vmin,
            vmax=self._norm.vmax
        )
        nx.draw_networkx_edges(g, pos)
        if self._sc is not None:
            plt.colorbar(self._sc)
        plt.title(f'Mapper [V = {len(self.node_vertices)}, E = {sum([len(e) for e in self.nodes.values()])}]')
        plt.plot()


    def plot_persistence_homology(self, type='filter'):
        gudhi.plot_persistence_barcode(self.diag, legend=True)
        plt.show()

        gudhi.plot_persistence_diagram(self.diag, legend=True)
        plt.show()
