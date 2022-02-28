from distutils.log import warn
from multiprocessing.sharedctypes import Value
from typing import Callable
from .GraphonAbstract import GraphonAbstract
from scipy.integrate import dblquad

class Graphon(GraphonAbstract):
    """
    basic Graphon only composed of a single function
    """
    def __init__(self, function: Callable, scaled=False) -> None:
        """
        Initialize a non scaled graphon defined only by its function f(x,y) (the function should be between 0 and 1) ? 
        Args:
            function (Callable): graphon function f(x,y)
        """

        self.graphon_function = function
        super().__init__(initial_rho= self.integral(), scaled=scaled)

    def graphon_function_builder(self) -> Callable:
        return self.graphon_function

    def integral(self):
        return dblquad(self.graphon_function, 0, 1, lambda x: 0, lambda x: 1)[0]
    

    def __add__(self, other):
        function = lambda x, y: 0.5 * self.graphon_function(x, y) + 0.5 * other.graphon_function(
            x, y
        )
        return Graphon(function)
