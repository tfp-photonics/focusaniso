import itertools
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

import numpy as np
from numpy.polynomial.polynomial import Polynomial


class Grid2D:
    def __init__(self, grid, x, y, coord='cartesian'):
        # todo: tests
        coord = self._test_coord(coord)
        self.grid = grid
        if x.ndim <= 1 and y.ndim <= 1:
            self.pos = list(np.meshgrid(x, y))
        else:
            self.pos = [x, y]
        self._coord = coord
        self._origcoord = coord

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, new):
        new = self._test_coord(new)
        if new == self._coord:
            return

        if new == 'polar':
            self.pos = [np.hypot(*self.pos), np.mod(np.arctan2(self.pos[1], self.pos[0]), 2 * np.pi)]
        else:
            self.pos = [self.pos[0] * np.cos(self.pos[1]), self.pos[0] * np.sin(self.pos[1])]
        self._coord = new

    def integrate(self, weights=1):
        weights = np.array(weights)
        # logging.debug(weights)
        if weights.ndim == 2:
            weights = weights.reshape(weights.shape + (1,))
        self.coord = self._origcoord
        if self._coord == 'cartesian':
            xdiff = np.diff(self.pos[0], prepend=self.pos[0][0,0], append=self.pos[0][0,-1], axis=1)
            ydiff = np.diff(self.pos[1], prepend=self.pos[1][0,0], append=self.pos[1][-1,0], axis=0)
            measure = 0.25 * (xdiff[:, 1:] + xdiff[:, :-1]) * (ydiff[1:, :] + ydiff[:-1, :])
        else:
            rhosq = self.pos[0] * self.pos[0]
            rhomeasure = np.zeros_like(rhosq)
            rhomeasure[:, 1:-1] = rhosq[:,2:] - rhosq[:,:-2]
            rhomeasure[:, 0] = rhosq[:, 1] - rhosq[:, 0]
            rhomeasure[:, -1] = rhosq[:, -1] - rhosq[:, -2]
            phidiff = np.diff(self.pos[1], prepend=self.pos[1][-1,0] - 2 * np.pi, append=self.pos[1][0,0] + 2 * np.pi, axis=0)
            measure = 0.125 * rhomeasure * (phidiff[1:, :] + phidiff[:-1, :])
        # logging.debug(measure)
        measure = measure.reshape(measure.shape + (1,))
        return np.sum(self.grid * weights * measure, axis=(0, 1))

    def _test_coord(self, c):
        c = c.lower()
        if c not in ('polar', 'cartesian'):
            raise ValueError("todo")
        return c


class Spectrum(Grid2D):
    def __init__(self, grid, x, y, k0, coord='cartesian', space='real'):
        super().__init__(grid, x, y, coord=coord)
        self.k0 = k0
        self._space = self._test_space(space)

    def focus(self, f, ts=1, tp=1, n1=1, n2=1):
        # todo: test space, etc.
        oldcoord = self.coord
        self.coord = 'polar'

        rho, phi = self.pos
        m = np.zeros(self.grid.shape + (3,), np.complex)
        cp, sp = np.cos(phi), np.sin(phi)
        st = rho / f
        ct = np.sqrt(1 - st * st)

        # pref = (
        #     np.sqrt(ct * n1 / n2) # energy conservation on projection
        #     * (-1j) * f * np.exp(1j * self.k0 * n2 * f) / (2 * np.pi * self.k0 * n2 * ct) # go to angular far field
        # )
        pref = (
            np.sqrt(ct * n1 / n2) # energy conservation on projection
            * (1j) * f * np.exp(-1j * self.k0 * n2 * f) / (2 * np.pi * self.k0 * n2 * ct) # go to angular far field
        ) # todo
        m[:, :, 0, 0] = (ts * sp * sp + tp * cp * cp * ct) * pref
        m[:, :, 1, 0] = m[:, :, 0, 1] = (-ts + tp * ct) * cp * sp * pref
        m[:, :, 1, 1] = (ts * cp * cp + tp * sp * sp * ct) * pref
        # m[:, :, 2, 0] = tp * cp * st * pref
        # m[:, :, 2, 1] = tp * sp * st * pref
        m[:, :, 2, 0] = -tp * cp * st * pref
        m[:, :, 2, 1] = -tp * sp * st * pref # todo

        res = m @ self.grid.reshape(self.grid.shape + (1,))
        self.grid = res.reshape(self.grid.shape)
        self.coord = oldcoord

        if self.coord == 'cartesian':
            # self.pos[0] = -self.pos[0][:, ::-1] * self.k0 * n2 / f
            # self.pos[1] = -self.pos[1][::-1, :] * self.k0 * n2 / f
            self.pos[0] = self.pos[0] * self.k0 * n2 / f
            self.pos[1] = self.pos[1] * self.k0 * n2 / f # todo
        else:
            self.pos[0] = self.pos[0] * self.k0 * n2 / f
            # self.pos[1] = self.pos[1] + np.pi todo

        self._space = 'angular'

    def _test_space(self, c):
        c = c.lower()
        if c  not in ('real', 'angular'):
            raise ValueError("todo")
        return c

    @property
    def space(self):
        return self._space

    def setspace(self, new, xs, ys, coord='cartesian'):
        new = self._test_space(new)
        if new == self._space:
            raise ValueError("TODO")
        if new == 'real':
            sign = 1
            pref = 1
        else:
            sign = -1
            pref = 1 / (4 * np.pi * np.pi)
        # print(sign)
        self.coord = self._origcoord
        if xs.ndim <= 1 and ys.ndim <= 1:
            newpos = list(np.meshgrid(xs, ys))
        else:
            newpos = [xs, ys]
        newgrid = np.zeros(newpos[0].shape + (3,), np.complex)
        for i, j in itertools.product(range(newgrid.shape[0]), range(newgrid.shape[1])):
            if coord == 'cartesian':
                if self.coord == 'cartesian':
                    weights = np.exp(sign * 1j * (newpos[0][i, j] * self.pos[0] + newpos[1][i, j] * self.pos[1]))
                else:
                    weights = np.exp(sign * 1j * self.pos[0] * (newpos[0][i, j] * np.cos(self.pos[1]) + newpos[1][i, j] * np.sin(self.pos[1])))
            else:
                if self.coord == 'cartesian':
                    weights = np.exp(sign * 1j * newpos[0][i, j] * (self.pos[0] * np.cos(newpos[1][i, j]) + self.pos[1] * np.sin(newpos[1][i, j])))
                else:
                    weights = np.exp(sign * 1j * newpos[0][i, j] * self.pos[0] * np.cos(newpos[1][i, j] - self.pos[1]))
            newgrid[i, j, :] = self.integrate(weights)

        self.grid = newgrid * pref
        self.pos = newpos
        self._space = new
        self._coord = coord
        self._origcoord = coord

    def field(self, stack, zs=None, solve=True, modes=True):
        if zs is None:
            zs = stack.zs
        if self._space == 'real':
            raise ValueError("TODO")
        self.coord = 'cartesian'
        if solve:
            stack.solve(self.k0, *self.pos, modes)
        # Forward solution
        amps_in = stack.materials[0].ps[:, :, [0, 2], :].conj() @ self.grid.reshape(self.grid.shape + (1,))
        coeffs = np.zeros((len(stack.zs),) + self.pos[0].shape + (4, 1), np.complex)
        coeffs[-1, :, :, :, :][:, :, [0, 2], :] = stack.m[:, :, [1, 3], :] @ amps_in
        # Backward solution
        for i in range(len(stack.zs) - 1, 0, -1):
            coeffs[i - 1, :, :, :, :] = stack.ts[i, :, :, :, :] @ coeffs[i, :, :, :, :]
        # Field calculation
        coeffs = coeffs.reshape((len(stack.zs),) + self.pos[0].shape + (4,))
        res = []
        for z in zs:
            j = 0
            while z > stack.zs[j] and j < len(stack.zs) - 1: # todo: < or <=
                j += 1
            field = stack.materials[j].ps.transpose((0, 1, 3, 2)) @ (
                np.exp(1j * stack.materials[j].kzs * (z - stack.zs[j])) * coeffs[j, :, :, :]
            ).reshape(stack.materials[j].kzs.shape + (1,))
            field = field.reshape(self.grid.shape)
            spec = Spectrum(field, *self.pos, self.k0, coord=self.coord, space='angular')
            spec._origcoord = self._origcoord
            res.append(spec)
        return res


class Material:
    def __init__(self, epsilon, mu=1):
        # todo: tests, single input
        if np.ndim(epsilon) == 0:
            epsilon = epsilon * np.eye(3)
        if np.ndim(mu) == 0:
            mu = mu * np.eye(3)
        if np.ndim(epsilon) != 2 or np.ndim(mu) != 2:
            raise ValueError("todo")
        self.epsilon = epsilon
        self.mu = mu
        self._invmu = np.linalg.inv(mu)

    def modes(self, k0, kxs, kys):
        # todo: test inputs
        self.kzs = np.zeros(kxs.shape + (4,), np.complex)
        self.ps = np.zeros(kxs.shape + (4, 3), np.complex)
        self.qs = np.zeros(kxs.shape + (4, 3), np.complex)
        for i, j in itertools.product(range(kxs.shape[0]), range(kxs.shape[1])):
            kx = kxs[i, j]
            ky = kys[i, j]
            k_cross = np.array([
                [Polynomial([0]), Polynomial([0, -1]), Polynomial([ky])],
                [Polynomial([0, 1]), Polynomial([0]), Polynomial([-kx])],
                [Polynomial([-ky]), Polynomial([kx]), Polynomial([0])]
            ])
            wave_eq = k_cross @ self._invmu @ k_cross + k0 * k0 * self.epsilon
            kzs, ps = eigenmodes(wave_eq)
            ks = np.zeros((4, 3), np.complex)
            ks[:, 0] = kx
            ks[:, 1] = ky
            ks[:, 2] = kzs
            qs = (self._invmu @ np.cross(ks, ps).reshape((4, 3, 1)) / k0).reshape((4, 3)) # no Z_0
            kzs, ps, qs = sort_modes(kzs, ps, qs)
            self.kzs[i, j, :] = kzs
            self.ps[i, j, :, :] = ps
            self.qs[i, j, :, :] = qs
        return self.kzs, self.ps, self.qs


class Stack:
    def __init__(self, zs, materials):
        # todo: tests
        self.zs = zs
        self.materials = materials

    def modes(self, k0, kxs, kys):
        # todo: is angular?
        # todo: is cartesian?
        for m in set(self.materials):
            m.modes(k0, kxs, kys)
        return self


    def solve(self, k0, kxs, kys, modes=True):
        # todo: is angular?
        # todo: is cartesian?
        if modes:
            self.modes(k0, kxs, kys)
        ds = np.diff(self.zs, prepend=0)
        self.ts = np.zeros((len(self.zs),) + kxs.shape + (4, 4), np.complex)
        self.m = np.zeros(kxs.shape + (4, 2), np.complex)
        for i, j in itertools.product(range(kxs.shape[0]), range(kxs.shape[1])):
            # lastdyn = np.empty((4, 4), np.complex)
            allts = np.empty((4, 4), np.complex)
            for l, d in enumerate(ds):
                prop = prop_mat(self.materials[l].kzs[i, j, :], d)
                dyn = dyn_mat(self.materials[l].ps[i, j, :, :], self.materials[l].qs[i, j, :, :])
                if l == 0:
                    self.ts[0, i, j, :, :] = prop
                    allts = prop
                else:
                    self.ts[l, i, j, :, :] = np.linalg.solve(lastdyn, dyn @ prop)
                    allts = allts @ self.ts[l, i, j, :, :]
                lastdyn = dyn
            denom = 1 / (allts[0, 0] * allts[2, 2] - allts[0, 2] * allts[2, 0])
            self.m[i, j, :, :] = denom * np.array([
                [allts[1, 0] * allts[2, 2] - allts[1, 2] * allts[2, 0], allts[1, 2] * allts[0, 0] - allts[0, 2] * allts[1, 0]],
                [allts[2, 2], -allts[0, 2]],
                [allts[3, 0] * allts[2, 2] - allts[3, 2] * allts[2, 0], allts[3, 2] * allts[0, 0] - allts[0, 2] * allts[3, 0]],
                [-allts[2, 0], allts[0, 0]],
            ])
        return self


def prop_mat(kzs, d):
    return np.diag(np.exp(-1j * kzs * d))

def dyn_mat(ps, qs):
    res = np.empty((4, 4), np.complex)
    res[0, :] = ps[:, 0]
    res[1, :] = qs[:, 1]
    res[2, :] = ps[:, 1]
    res[3, :] = qs[:, 0]
    return res


def sort_modes(kzs, ps, qs):
    poyntings = poynting(ps, qs)
    up = poyntings[:, 2] > 1e-14
    down = poyntings[:, 2] < -1e-14
    unassigned = np.logical_not(np.logical_or(up, down))
    for i, kz in enumerate(kzs):
        if unassigned[i]:
            if np.imag(kz) > 0:
                up[i] = True
            else:
                down[i] = True

    if not np.sum(up) == np.sum(down) == 2:
        raise ValueError("Fix error type: also ambigous propagation")

    overlap = np.abs([[np.dot(pup, pdown.conj()) for pdown in ps[down,:]] for pup in ps[up, :]])
    perm = np.zeros((4,), np.int)
    perm[[0, 2]] = np.nonzero(up)[0]

    if overlap[0, 0] > overlap[0, 1] or overlap[1, 1] > overlap[1, 0]:
        perm[[1, 3]] = np.nonzero(down)[0]
    else:
        perm[[3, 1]] = np.nonzero(down)[0]
    return kzs[perm], ps[perm, :], qs[perm, :]

def poynting(ps, qs):
    return 0.5 * np.real(np.cross(ps, qs.conj()))

def _det3(m):
    return (m[0,0] * (m[1,1] * m[2,2] - m[2,1] * m[1,2])
        -m[1,0] * (m[0,1] * m[2,2] - m[2,1] * m[0,2])
        +m[2,0] * (m[0,1] * m[1,2] - m[1,1] * m[0,2])
    )

def findpairs(xs):
    # only len == 4
    dists = {(i, j): np.abs(x - y) for (i, x), (j, y) in itertools.combinations(enumerate(xs), 2)}
    res = {}
    # Minimal distance
    key = min(dists.keys(), key=(lambda k: dists[k]))
    res[key] = dists[key]
    # Other pair
    l = [0, 1, 2, 3]
    l.remove(key[0])
    l.remove(key[1])
    key = tuple(l)
    res[key] = dists[key]
    return res


def multiplicity(xs, tol=1e-5):
    pairs = findpairs(xs)
    res = []
    mult = []
    for key, val in pairs.items():
        if val < tol:
            res.append(0.5 * (xs[key[0]] + xs[key[1]]))
            mult.append(2)
        else:
            res.append(xs[key[0]])
            mult.append(1)
            res.append(xs[key[1]])
            mult.append(1)
    return res, mult


def evalpolymat(m, x):
    shape = m.shape
    res = np.array([i(x) for i in m.flatten()])
    return res.reshape(shape)


def eigenmodes(eq):
    kzs = _det3(eq).roots()
    kzs_mult = multiplicity(kzs)
    ps = np.zeros((4, 3), np.complex)
    i = 0
    for kz, mult in zip(*kzs_mult):
        sol = evalpolymat(eq, kz)
        _, _, vh = np.linalg.svd(sol)
        for j in range(mult):
            ps[i, :] = vh[-j - 1, :].conj()
            kzs[i] = kz
            i += 1
    return kzs, ps
