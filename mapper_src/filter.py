import numpy as np

class Filter:
    """
    Implements function filter: X -> R.
    """
    def __init__(self, filter_function):
        self._filter_function = filter_function

    def __call__(self, vertices):
        if self._filter_function == 'last_coordinate':
            return self._last_coordinate(vertices)
        elif self._filter_function == 'distance_from_origin':
            return self._distance_from_origin(vertices)
        else:
            raise ValueError(f'Argument `filter_function` must be one of: "last_coordinate", "distance_from_origin".')

    def _last_coordinate(self, vertices):
        """
        Uses last coordinate of vertices.
        """
        return vertices[:, -1]

    def _distance_from_origin(self, vertices):
        """
        Returns euclidean distance from origin.
        """
        return np.linalg.norm(vertices, axis=-1)
