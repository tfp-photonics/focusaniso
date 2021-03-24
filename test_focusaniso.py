import pytest

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import focusaniso as fa

EPS = 1e-7
EPSSQ = 1e-14

def test_det3():
    m = np.array([[1, 2, Polynomial(3)], [4, Polynomial([5, 6]), 7], [Polynomial([0, -1]), 2, 0]])
    assert fa._det3(m) == Polynomial([10, 1, 18])

def test_prop_mat():
    assert np.sum(
        np.abs(fa.prop_mat([0.5, -1, 0, 1j], np.pi) - np.diag([-1j, -1, 1, np.exp(np.pi)]))
    ) <= EPSSQ

def test_dyn_mat():
    ps = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    qs = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9], [-10, -11, -12]]
    res = [[0, 3, 6, 9], [-2, -5, -8, -11], [1, 4, 7, 10], [-1, -4, -7, -10]]
    assert np.array_equal(fa.dyn_mat(ps, qs), res)

def test_poynting():
    assert 0

def test_findpairs():
    assert 0

def test_multiplicity_no_pair():
    assert 0

def test_multiplicity_one_pair():
    assert 0

def test_multiplicity_two_pairs():
    assert 0

def test_eigenmodes_degenerate():
    assert 0

def test_eigenmodes_nondegenerate():
    assert 0

def test_sort_modes_error():
    assert 0

def test_sort_modes_evanescent():
    assert 0

def test_sort_modes_1():
    assert 0

def test_sort_modes_2():
    assert 0

class TestGrid2D:
    def test_constructor_2d():
        assert 0

    def test_constructor():
        assert 0

    def test_constructor_error():
        assert 0

    def test_coord_to_polar():
        assert 0

    def test_coord_to_cartesian():
        assert 0

    def test_integrate_polar():
        # scalar weight
        assert 0

    def test_integrate_cartesian():
        # grid wight
        assert 0

class TestSpectrum:
    def test_constructor_error():
        assert 0

    def test_focus():
        assert 0

    def test_setspace_to_polar():
        assert 0

    def test_setspace_to_cartesian():
        assert 0

    def test_field():
        assert 0

class TestMaterial:
    def test_constructor_error():
        assert 0

    def test_constructor_scalar():
        assert 0

    def test_modes():
        assert 0

class TestStack:
    def test_constructor_error():
        assert 0

    def test_constructor():
        assert 0

    def test_modes():
        assert 0

    def test_solve():
        assert 0
