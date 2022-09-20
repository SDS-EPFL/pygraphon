# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import numpy as np
import pytest


def test_not_able_to_instantiate_abstract_class():
    """Test that it is not possible to instantiate an abstract class."""
    from pygraphon.graphons.GraphonAbstract import GraphonAbstract

    with pytest.raises(TypeError):
        GraphonAbstract()


def test_step_graphon_normalized():
    """Test that the step graphon is normalized."""
    from pygraphon.graphons.StepGraphon import StepGraphon

    step_graphon = StepGraphon(graphon=np.array([[0.8, 0.2], [0.2, 0.8]]), bandwidthHist=0.5)
    assert step_graphon.integral() == 1
