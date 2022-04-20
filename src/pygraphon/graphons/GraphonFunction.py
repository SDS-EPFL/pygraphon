from typing import Callable

from scipy.integrate import dblquad

from pygraphon.utils.utils_func import copy_func

from .GraphonAbstract import GraphonAbstract


class Graphon(GraphonAbstract):
    """
    basic Graphon only composed of a single function
    """

    def __init__(
        self, function: Callable, scaled=True, check=True, initial_rho: float = None
    ) -> None:
        """
        Initialize a non scaled graphon defined only by its function f(x,y) (the function should be between 0 and 1) ?
        Args:
            function (Callable): graphon function f(x,y)
        """
        self.integral_value = None
        self.graphon_function = function
        super().__init__(initial_rho=initial_rho, scaled=scaled, check=check)

    def correct_graphon_integral(self):
        """Normalize the graphon function to 1 if possible.

        Raises:
            ValueError: if graphon integrates to 0.
        """
        integral = self.integral()
        if integral == 0:
            raise ValueError("graphon integrates to 0, cannot be automatically corrected")
        # keep copy of old graphon function to be able to rescale it
        # without having infinite recursion
        self.old_func = copy_func(self.graphon_function)
        self.graphon_function = lambda x, y: self.old_func(x, y) / integral

    def graphon_function_builder(self) -> Callable:
        """Builder for the graphon function f(x,y).

        Returns:
            Callable: graphon function
        """
        return self.graphon_function

    def integral(self, force_recompute=False) -> float:
        """Compute the integral of the graphon.

        Args:
            force_recompute (bool, optional): If False, check for cached value. Defaults to False.

        Returns:
            float: integral value of the graphon.
        """
        if self.integral_value is None:
            self.integral_value = (
                2 * dblquad(self.graphon_function, 0, 1, lambda x: 0, lambda x: x)[0]
            )
        return self.integral_value

    def __add__(self, other):
        """Addition and normalize twp graphons.

        Args:
            other (Graphon): other graphon to add.
        """

        def _function(x, y):
            return 0.5 * self.graphon_function(x, y) + 0.5 * other.graphon_function(x, y)

        return Graphon(_function, check=False)
