from distutils.log import warn
from typing import Callable
from .GraphonAbstract import GraphonAbstract


class Graphon(GraphonAbstract):
    def __init__(self, function: Callable, scaled=False) -> None:

        self.graphon_function = function
        super().__init__(scaled)

    def graphon_function_builder(self) -> Callable:
        return self.graphon_function

    def __add__(self, other):
        function = lambda x, y: 0.5 * self.graphon_function(x, y) + 0.5 * other.graphon_function(
            x, y
        )
        return Graphon(function)
