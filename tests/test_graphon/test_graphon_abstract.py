# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import pytest


def test_not_able_to_instantiate():
    """Test that it is not possible to instantiate an abstract class."""
    from pygraphon.graphons.GraphonAbstract import GraphonAbstract

    with pytest.raises(TypeError):
        GraphonAbstract()
