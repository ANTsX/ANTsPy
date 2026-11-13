import sys
import os
import argparse
import unittest
import warnings
import contextlib
from functools import wraps
from itertools import product
from copy import deepcopy
from numpy import testing as nptest

def run_tests():
    import faulthandler
    faulthandler.enable()
    unittest.main(verbosity=2)


def assert_same_image(actual, expected, rtol=1e-6, atol=1e-6):
    assert actual.shape == expected.shape
    assert actual.components == expected.components
    assert actual.pixeltype == expected.pixeltype
    nptest.assert_allclose(actual.origin, expected.origin)
    nptest.assert_allclose(actual.spacing, expected.spacing)
    nptest.assert_allclose(actual.direction, expected.direction)
    nptest.assert_allclose(actual.numpy(), expected.numpy(), rtol=rtol, atol=atol)


def assert_same_image_space(actual, expected, rtol=1e-8, atol=1e-8):
    assert actual.shape == expected.shape
    assert actual.components == expected.components
    nptest.assert_allclose(actual.origin, expected.origin, rtol=rtol, atol=atol)
    nptest.assert_allclose(actual.spacing, expected.spacing, rtol=rtol, atol=atol)
    nptest.assert_allclose(actual.direction, expected.direction, rtol=rtol, atol=atol)