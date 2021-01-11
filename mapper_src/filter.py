import numpy as np

class Filter:
    """
    Implements function filter: X -> R.
    """
    def __init__(self, filter_function='by_coordinate', **kwargs):
        self._filter_function = filter_function
        self.coordinate = kwargs.get('coordinate', -1) if filter_function == 'by_coordinate' else None
        self.point = kwargs.get('point', np.array([0, 0, 0]))

    def __call__(self, vertices):
        if self._filter_function == 'by_coordinate':
            return self._by_coordinate(vertices)
        elif self._filter_function == 'distance_from_origin':
            return self._distance_from_origin(vertices)
        elif self._filter_function == 'distance_from_point':
            return self._distance_from_point(vertices)
        else:
            raise ValueError(f'Argument `filter_function` must be one of: "by_coordinate", "distance_from_origin", "distance_from_point".')

    def _by_coordinate(self, vertices):
        """
        Uses the specified coordinate (default is last) of vertices.
        """
        return vertices[:, self.coordinate]


    def _distance_from_origin(self, vertices):
        """
        Returns euclidean distance from origin.
        """
        return np.linalg.norm(vertices, axis=-1)


    def _distance_from_point(self, vertices):
        """
        Returns euclidean distance from `point`.
        """
        return np.linalg.norm(vertices - self.point, axis=-1)
