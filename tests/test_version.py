# -*- coding: utf-8 -*-

"""Trivial version test."""

import unittest

from src.pygraphon.version import get_version


class TestVersion(unittest.TestCase):
    """Trivially test a version."""

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_version_type(self):
        """Test the version is a string.

        This is only meant to be an example test.
        """
        version = get_version()
        self.assertIsInstance(version, str)
