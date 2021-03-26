import pytest

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import focusaniso as fa

EPS = 1e-7
EPSSQ = 1e-14


def test_det3():
    m = np.array(
        [[1, 2, Polynomial(3)], [4, Polynomial([5, 6]), 7], [Polynomial([0, -1]), 2, 0]]
    )
    assert fa._det3(m) == Polynomial([10, 1, 18])


def test_prop_mat():
    assert (
        np.sum(
            np.abs(
                fa.prop_mat([0.5, -1, 0, 1j], np.pi)
                - np.diag([-1j, -1, 1, np.exp(np.pi)])
            )
        )
        <= EPSSQ
    )


def test_dyn_mat():
    ps = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    qs = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9], [-10, -11, -12]]
    res = [[0, 3, 6, 9], [-2, -5, -8, -11], [1, 4, 7, 10], [-1, -4, -7, -10]]
    assert np.array_equal(fa.dyn_mat(ps, qs), res)


def test_poynting():
    ps = [[1, 2, 3], [1, 1j, 0]]
    qs = [[4, 5, 6], [1j, -1j, 2]]
    assert np.array_equal(fa.poynting(ps, qs), [[-1.5, 3, -1.5], [0, -1, -0.5]])


def test_findpairs():
    res = fa._findpairs([1, 3, 1 + 1j, -4])
    expect = {(0, 2): 1, (1, 3): 7}
    assert np.all(
        [k == m and i == j for (k, i), (m, j) in zip(res.items(), expect.items())]
    )


def test_multiplicity_no_pair():
    res, mult = fa._multiplicity([3, 1, 0, 4])
    assert np.array_equal(res, [3, 4, 1, 0]) and np.array_equal(mult, [1, 1, 1, 1])


def test_multiplicity_one_pair():
    res, mult = fa._multiplicity([1, 1, 0, 4], tol=0)
    assert np.array_equal(res, [1, 0, 4]) and np.array_equal(mult, [2, 1, 1])


def test_multiplicity_two_pairs():
    res, mult = fa._multiplicity([4 + 2e-6j, 1, 1, 4])
    assert np.array_equal(res, [1, 4 + 1e-6j]) and np.array_equal(mult, [2, 2])


def test_evalpolymat():
    m = np.array(
        [
            [Polynomial([4, 0, -1]), 0, 0],
            [0, Polynomial([1, 0, -1]), 0],
            [Polynomial([1, 1]), 0, 4],
        ]
    )
    expect = [[0, 0, 0], [0, -3, 0], [3, 0, 4]]
    assert np.array_equal(fa._evalpolymat(m, 2), expect)


def test_eigenmodes_degenerate():
    eq = np.array(
        [[Polynomial([4, 0, -1]), 0, 0], [0, Polynomial([4, 0, -1]), 0], [0, 0, 4]]
    )
    kzs, _ = fa._eigenmodes(eq)  # degenerate polarizations are not well defined
    assert np.sum(np.abs(kzs - np.array([2, 2, -2, -2]))) <= EPSSQ


def test_eigenmodes_nondegenerate():
    m = np.diag([Polynomial([4, 0 , -1]), Polynomial([1, 0, -1]), 1])#
    kzs, ps = fa._eigenmodes(m)
    assert (
        np.sum(kzs - np.array([-2, -1, 1, 2])) <= EPSSQ
        and np.sum(np.abs(ps) - np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])) <= EPSSQ
    )


def test_sort_modes_error():
    with pytest.raises(Exception):
        fa._sort_modes(np.zeros(4), np.zeros((4, 3)), np.zeros((4, 3)))


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
