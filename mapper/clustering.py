import operator
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from gudhi.clustering.tomato import Tomato

class Clustering:
    """
    Function clustering vertices to nodes.
    Returns list of node indices for each vertex.
    """
    def __init__(self, clustering_function='tomato', **kwargs):
        self._clustering_function = clustering_function
        if clustering_function == 'agglomerative':
            self._clustering = AgglomerativeClustering(
                n_clusters=None,
                compute_full_tree=True,
                linkage=kwargs.get('linkage', 'ward'),
            )
            self._distance_param = 'distance_threshold'

        elif clustering_function == 'tomato':
            self._show_diagram = kwargs.get('cluster_plot', False)
        else:
            raise ValueError(f'Argument `clustering_function` must be one of: "agglomerative", "tomato".')

        # If not None clustering is only run for this single value.
        self._distance = kwargs.get('distance', None)
        # Else clusters up to `max_k` are tested.
        self._max_k = kwargs.get('max_k', 10)


    def __call__(self, vertices, indices):
        if self._clustering_function == 'tomato':
            return self._gudhi_clustering(vertices, indices)
        else:
            return self._scikit_clustering(vertices, indices)


    def _gudhi_clustering(self, vertices, indices):
        # We must recreate the object as it saves fitting information from past fits.
        self._clustering = Tomato(merge_threshold=self._distance)
        self._clustering.fit(vertices)

        if self._show_diagram:
            self._clustering.plot_diagram()
            n_clusters = int(input('Set optimal number of clusters: '))
            self._clustering.n_clusters_ = n_clusters

        return self._assign_vertices_to_clusters(vertices, indices)


    def _scikit_clustering(self, vertices, indices):
        self._clustering.set_params(**{self._distance_param : self._distance})

        if self._distance is None:
            # Find optimal number of clusters by silhouette score.
            opt_sh = 0.6

            opt_k = 1
            for k in range(2, self._max_k+1):
                self._clustering.n_clusters = k
                labels = self._clustering.fit_predict(vertices)
                sh_score = silhouette_score(vertices, labels)

                if sh_score > opt_sh:
                    opt_sh = sh_score
                    opt_k = k

            self._clustering.n_clusters = opt_k
            self._clustering.fit(vertices)
        else:
            self._clustering.fit(vertices)

        return self._assign_vertices_to_clusters(vertices, indices)


    def _assign_vertices_to_clusters(self, vertices, indices):
        labels = self._clustering.labels_

        label_indices = [set() for _ in range(self._clustering.n_clusters_)]
        cluster_vertices = [list() for _ in range(self._clustering.n_clusters_)]
        # Divide indices by labels.
        for index, (i, l) in enumerate(zip(indices, labels)):
            label_indices[l].add(i)
            cluster_vertices[l].append(vertices[index])

        # Calculate center of each cluster.
        cluster_centers = [(np.mean(np.array(c)[:, 0]), np.mean(np.array(c)[:, 1]), np.mean(np.array(c)[:, 2])) for c in cluster_vertices]

        return labels, label_indices, cluster_centers
