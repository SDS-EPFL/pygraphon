# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import unittest

from pygraphon.graphons.GraphonAbstract import GraphonAbstract


class TestInitialization(unittest.TestCase):
    """Trivially test a version."""

    def test_not_able_to_instantiate_abstract_class(self):
        """Test the version is a string."""
        with self.assertRaises(TypeError):
            _ = GraphonAbstract()
