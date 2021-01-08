class Filter:
    """
    Implements function filter: X -> R.
    """
    def __init__(self, filter_function='by_coordinate', **kwargs):
        self._filter_function = filter_function
        if filter_function == 'by_coordinate':
            self._coordinate = kwargs.get('coordinate', -1)

    def __call__(self, vertices):
        if self._filter_function == 'by_coordinate':
            return self._by_coordinate(vertices)
        else:
            raise ValueError(f'Argument `filter_function` must be one of: "by_coordinate".')

    def _by_coordinate(self, vertices):
        """
        Uses last coordinate of vertices.
        """
        return vertices[:, self._coordinate]
