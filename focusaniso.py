"""Focusing into and propagation in anisotropic media

This module includes classes around the real and angular spectrum representations of
light in stratified media. The media can have anisotropic permittivity and permeability.
Additionally, the light can be focused by a lens.
"""

import itertools

import numpy as np
from numpy.polynomial.polynomial import Polynomial


class Grid2D:
    """Two-dimensional grid of three-component vectors

    The grid is always stored using cartesian vectors component, such that they are
    position-independent. The positions of the grid can be in polar and cartesian
    coordinates.

    Attributes
    ----------
    grid : (N, M, 3)-array
        Grid with the three-component vectors.
    pos : (N, M) array, (N, M) array
        Meshgrid of the postions, M and N include the x and y or rho and phi components,
        respectively.
    _origcoord : string (private)
        Stores the original coordinate base, used for the integration

    Parameters
    ----------
    grid : (N, M, P)-array
        Grid of three- or two-component vector (depending on P). If P is 2 zeros will be
        added for the third component.
    xs: (N?, M)-array
        Meshgrid of the x (or rho) component or a one-dimensional array
    ys: (N, M?)-array
        Meshgrid of the y (or phi) component or a one-dimensional array
    coord : string, optional
        Coordinate system, either 'cartesian' or 'polar'

    """

    def __init__(self, grid, xs, ys, coord="cartesian"):
        coord = Grid2D._test_coord(coord)
        grid = np.array(grid)
        if grid.shape[2] == 2:
            tmp = grid
            grid = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=grid.dtype)
            grid[:, :, :2] = tmp
        xs = np.array(xs)
        ys = np.array(ys)
        if xs.ndim <= 1 and ys.ndim <= 1:
            xs, ys = np.meshgrid(xs, ys)

        self.grid = grid
        self.pos = [xs, ys]
        self._coord = coord
        self._origcoord = coord

    @property
    def coord(self):
        """Coordinate system

        Either 'cartesian' or 'polar'. Changing this parameter changes the `pos`
        attribute.
        """
        return self._coord

    @coord.setter
    def coord(self, new):
        new = Grid2D._test_coord(new)
        if new == self._coord:
            return
        if new == "polar":
            self.pos = [
                np.hypot(*self.pos),
                np.mod(np.arctan2(self.pos[1], self.pos[0]), 2 * np.pi),
            ]
        else:
            self.pos = [
                self.pos[0] * np.cos(self.pos[1]),
                self.pos[0] * np.sin(self.pos[1]),
            ]
        self._coord = new

    def integrate(self, weights=1):
        """Integrate the grid values with some weights

        Parameters
        ----------
        weights : (N, M)- or (N, M, P)-array or scalar
            weights for each lattice point or each element in the integration of the
            grid values

        Returns
        -------
        (3)-array
            Integration result

        """
        weights = np.array(weights)
        if weights.ndim == 2:
            weights = weights.reshape(weights.shape + (1,))
        self.coord = self._origcoord
        if self._coord == "cartesian":
            xdiff = np.diff(
                self.pos[0],
                prepend=self.pos[0][0, 0],
                append=self.pos[0][0, -1],
                axis=1,
            )
            ydiff = np.diff(
                self.pos[1],
                prepend=self.pos[1][0, 0],
                append=self.pos[1][-1, 0],
                axis=0,
            )
            measure = (
                0.25 * (xdiff[:, 1:] + xdiff[:, :-1]) * (ydiff[1:, :] + ydiff[:-1, :])
            )
        else:
            rhosq = self.pos[0] * self.pos[0]
            rhomeasure = np.zeros_like(rhosq)
            rhomeasure[:, 1:-1] = rhosq[:, 2:] - rhosq[:, :-2]
            rhomeasure[:, 0] = rhosq[:, 1] - rhosq[:, 0]
            rhomeasure[:, -1] = rhosq[:, -1] - rhosq[:, -2]
            phidiff = np.mod(np.diff(
                self.pos[1],
                prepend=self.pos[1][-1, -1] - 2 * np.pi,
                append=self.pos[1][0, -1] + 2 * np.pi,
                axis=0,
            ), 2 * np.pi)
            measure = 0.125 * rhomeasure * (phidiff[1:, :] + phidiff[:-1, :])
        measure = measure.reshape(measure.shape + (1,))
        return np.sum(self.grid * weights * measure, axis=(0, 1))

    @staticmethod
    def _test_coord(c):
        """Test if the string is a legal coordinate system descriptor
        """
        tmp = c
        c = c.lower()
        if c not in ("polar", "cartesian"):
            raise ValueError(f"{tmp} is not a legal coordinate system")
        return c


class Spectrum(Grid2D):
    """Spectrum of light

    A real or an angular spectrum of light inheriting from :class:`Grid2D`.

    Attributes
    ----------
    grid : (N, M, 3) array
        Grid with the three-component vectors.
    pos : (N, M) array, (N, M) array
        Meshgrid of the postions, M and N include the x and y or rho and phi components,
        respectively.
    k0 : scalar
        Wave number of the light (in vacuum)
    _origcoord : string (private)
        Stores the original coordinate base, used for the integration

    Parameters
    ----------
    grid : (N, M, P) array
        Grid of three- or two-component vector (depending on P). If P is 2 zeros will be
        added for the third component.
    xs: (N?, M) array
        Meshgrid of the x (or rho) component or a one-dimensional array
    ys: (N, M?) array
        Meshgrid of the y (or phi) component or a one-dimensional array
    k0 : scalar
        Wave number of the light (in vacuum)
    coord : string, optional
        Coordinate system, either 'cartesian' or 'polar', defaults to 'cartesian'
    space : string, optional
        Either 'real' or 'angular', defaults to 'real'

    """

    def __init__(self, grid, x, y, k0, coord="cartesian", space="real"):
        super().__init__(grid, x, y, coord=coord)
        self.k0 = k0
        self._space = Spectrum._test_space(space)

    def focus(self, f, ts=1, tp=1, n1=1, n2=1):
        """Focus the spectrum at a lens

        The spectrum has to be in real space. The function computes first the refracted
        real spectrum. Then, this field is transformed to the angular spectrum, assuming
        it is far away from the focus. No evanescent modes are included. The numerical
        aperture has to be taken into account by the non-zero entries in the beam.
        Typically, ``rho_max = f * NA / n2`` where ``n2`` is the refractive index
        of the medium focused into. The origin of the spectrum with respect to z is at
        the focal point.

        Arguments
        ---------
        f : scalar
            Focal length
        ts : (N, M)-array or scalar, optional
            Transmission coefficient of s-polarized light through the lens. Can be
            elementwise for each `pos`. Defaults to 1.
        ts : (N, M)-array or scalar, optional
            Transmission coefficient of p-polarized light through the lens. Can be
            elementwise for each `pos`. Defaults to 1.
        n1, n2 : scalar, optional
            Refractive indices of the medium in front of and behind of the lens.
            Defaults to 1.

        """
        if self._space == "angular":
            raise ValueError(
                "Cannot focus angular spectrum, transform to real spectrum first"
            )
        oldcoord = self.coord
        self.coord = "polar"

        rho, phi = self.pos
        m = np.zeros(self.grid.shape + (3,), np.complex)
        cp, sp = np.cos(phi), np.sin(phi)
        st = rho / f
        ct = np.sqrt(1 - st * st)
        pref = (
            np.sqrt(ct * n1 / n2) # energy conservation on projection
            * (-1j) * f
            * np.exp(1j * self.k0 * n2 * f)
            / (2 * np.pi * self.k0 * n2 * ct) # go to angular far field
        )
        m[:, :, 0, 0] = (ts * sp * sp + tp * cp * cp * ct) * pref
        m[:, :, 1, 0] = m[:, :, 0, 1] = (-ts + tp * ct) * cp * sp * pref
        m[:, :, 1, 1] = (ts * cp * cp + tp * sp * sp * ct) * pref
        m[:, :, 2, 0] = -tp * cp * st * pref
        m[:, :, 2, 1] = -tp * sp * st * pref

        res = m @ self.grid.reshape(self.grid.shape + (1,))
        self.grid = res.reshape(self.grid.shape)
        self.coord = oldcoord

        if self.coord == "cartesian":
            self.pos[0] = -self.pos[0][:, ::-1] * self.k0 * n2 / f
            self.pos[1] = -self.pos[1][::-1, :] * self.k0 * n2 / f
        else:
            self.pos[0] = self.pos[0] * self.k0 * n2 / f
            self.pos[1] = self.pos[1] + np.pi
        self._space = "angular"

    @staticmethod
    def _test_space(c):
        """Test if the string is a legal space descriptor
        """
        tmp = c
        c = c.lower()
        if c not in ("real", "angular"):
            raise ValueError(f"{tmp} is not a legal space descriptor")
        return c

    @property
    def space(self):
        """Space of the spectrum

        Either 'real' or 'angular', change with `setspace`
        """
        # Basically just here to protect against tempering with the variable
        return self._space

    def setspace(self, new, xs, ys, coord="cartesian"):
        """Change the space

        Transforms between real and angular space

        Arguments
        ---------
        new : string
            Either 'real' or 'angular'
        xs: (N?, M) array
            Meshgrid of the x (or rho) component or a one-dimensional array
        ys: (N, M?) array
            Meshgrid of the y (or phi) component or a one-dimensional array
        coord : string, optional
            Coordinate system, either 'cartesian' or 'polar', defaults to 'cartesian'
        """
        new = Spectrum._test_space(new)
        if new == self._space:
            raise ValueError(f"Already in {new} space")
        if new == "real":
            sign = 1
            pref = 1
        else:
            sign = -1
            pref = 1 / (4 * np.pi * np.pi)
        self.coord = self._origcoord
        if xs.ndim <= 1 and ys.ndim <= 1:
            newpos = list(np.meshgrid(xs, ys))
        else:
            newpos = [xs, ys]
        newgrid = np.zeros(newpos[0].shape + (3,), np.complex)
        for i, j in itertools.product(range(newgrid.shape[0]), range(newgrid.shape[1])):
            if coord == "cartesian":
                if self.coord == "cartesian":
                    weights = np.exp(
                        sign
                        * 1j
                        * (
                            newpos[0][i, j] * self.pos[0]
                            + newpos[1][i, j] * self.pos[1]
                        )
                    )
                else:
                    weights = np.exp(
                        sign
                        * 1j
                        * self.pos[0]
                        * (
                            newpos[0][i, j] * np.cos(self.pos[1])
                            + newpos[1][i, j] * np.sin(self.pos[1])
                        )
                    )
            else:
                if self.coord == "cartesian":
                    weights = np.exp(
                        sign
                        * 1j
                        * newpos[0][i, j]
                        * (
                            self.pos[0] * np.cos(newpos[1][i, j])
                            + self.pos[1] * np.sin(newpos[1][i, j])
                        )
                    )
                else:
                    weights = np.exp(
                        sign
                        * 1j
                        * newpos[0][i, j]
                        * self.pos[0]
                        * np.cos(newpos[1][i, j] - self.pos[1])
                    )
            newgrid[i, j, :] = self.integrate(weights)

        self.grid = newgrid * pref
        self.pos = newpos
        self._space = new
        self._coord = coord
        self._origcoord = coord

    def field(self, stack, zs=None, solve=True, modes=True):
        """Spectra at the given postions

        The field is assumed to be coming from minus infinity propagating towards plus
        infinity. If the stack has negative z values the spectrum is propagated back to
        that point. The spectrum has to be angular.

        Parameters
        ----------
        stack : Stack
            The stack of materials where the field is defined
        zs : (N,)-array, optional
            The z values where a spectrum shall be returned. Defaults to stack.zs
        solve : bool, optional
            Solve the stack for the postions `pos` of the spectrum. Calculation can be
            omitted if the stack has been already solved for these positions and this
            wave number saving the most time consuming step.
        modes : bool, optional
            If solving, this parameter is passed to `Stack.solve`

        Returns
        -------
        List of Spectra
            Spectra at the postions corresponding to zs
        """
        if zs is None:
            zs = stack.zs
        if self._space == "real":
            raise ValueError(
                "Cannot propagate real spectrum, transform to angular spectrum first"
            )
        oldcoord = self.coord
        self.coord = "cartesian"
        if solve:
            stack.solve(self.k0, *self.pos, modes)
        # Forward solution
        amps_in = stack.materials[0].ps[:, :, [0, 2], :].conj() @ self.grid.reshape(
            self.grid.shape + (1,)
        )
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
            while z >= stack.zs[j] and j < len(stack.zs) - 1:
                j += 1
            field = stack.materials[j].ps.transpose((0, 1, 3, 2)) @ (
                np.exp(1j * stack.materials[j].kzs * (z - stack.zs[j]))
                * coeffs[j, :, :, :]
            ).reshape(stack.materials[j].kzs.shape + (1,))
            field = field.reshape(self.grid.shape)
            spec = Spectrum(
                field, *self.pos, self.k0, coord=self.coord, space="angular"
            )
            spec._origcoord = self._origcoord
            res.append(spec)

        self.coord = oldcoord
        return res


class Material:
    """Material properties

    Attributes
    ----------
    epsilon : (3, 3)-array
        Relative permittivity in cartesian coordinates
    mu : (3, 3)-array
        Relative permeability in cartesian coordinates
    _invmu : (3, 3)-array
        Inverse of mu in cartesian coordinates

    Parameters
    ----------
    epsilon : (3, 3)-array or scalar
        Relative permittivity in cartesian coordinates, if scalar an isotropic medium is
        assumed
    mu : (3, 3)-array, optional
        Relative permeability in cartesian coordinates, if scalar an isotropic medium is
        assumed. Defaults to 1.
    """

    def __init__(self, epsilon, mu=1):
        epsilon = np.array(epsilon)
        mu = np.array(mu)
        if epsilon.ndim == 0:
            epsilon = epsilon * np.eye(3)
        if mu.ndim == 0:
            mu = mu * np.eye(3)
        if epsilon.ndim != 2 or mu.ndim != 2:
            raise ValueError(
                f"Cannot identify material with {epsilon.shape} shaped epsilon and {mu.shape} shaped mu"
            )
        self.epsilon = epsilon
        self.mu = mu
        self._invmu = np.linalg.inv(mu)

    def modes(self, k0, kxs, kys):
        """Mode calculation in the medium

        Arguments
        ---------
        k0 : scalar
            Wave number (in vacuum)
        kxs : (N?, M)-array or scalar
            X component of the wave vector
        kys : (N, M?)-array or scalar
            Y component of the wave vector

        Returns
        ------
        kzs : (N, M, 4)-array
            Z component solution of the wave vector, the first and third solution
            propagate in positive, the other two in negative, z direction
        ps : (N, M, 4, 3)-array
            E-field modes corresponding to the two solutions
        qs : (N, M, 4, 3)-array
            H-field modes corresponding to the two solutions divided by the free space
            impedance (such that `ps` and `qs` have the same units)
        """
        kxs = np.array(kxs)
        kys = np.array(kys)
        if kxs.ndim <= 1 and kys.ndim <= 1:
            kxs, kys = list(np.meshgrid(kxs, kys))

        self.kzs = np.zeros(kxs.shape + (4,), np.complex)
        self.ps = np.zeros(kxs.shape + (4, 3), np.complex)
        self.qs = np.zeros(kxs.shape + (4, 3), np.complex)
        for i, j in itertools.product(range(kxs.shape[0]), range(kxs.shape[1])):
            kx = kxs[i, j]
            ky = kys[i, j]
            k_cross = np.array(
                [
                    [Polynomial([0]), Polynomial([0, -1]), Polynomial([ky])],
                    [Polynomial([0, 1]), Polynomial([0]), Polynomial([-kx])],
                    [Polynomial([-ky]), Polynomial([kx]), Polynomial([0])],
                ]
            )
            wave_eq = k_cross @ self._invmu @ k_cross + k0 * k0 * self.epsilon
            kzs, ps = _eigenmodes(wave_eq)
            ks = np.zeros((4, 3), np.complex)
            ks[:, 0] = kx
            ks[:, 1] = ky
            ks[:, 2] = kzs
            qs = (self._invmu @ np.cross(ks, ps).reshape((4, 3, 1)) / k0).reshape(
                (4, 3)
            )  # no Z_0
            kzs, ps, qs = _sort_modes(kzs, ps, qs)
            self.kzs[i, j, :] = kzs
            self.ps[i, j, :, :] = ps
            self.qs[i, j, :, :] = qs
        return self.kzs, self.ps, self.qs


class Stack:
    """Stack of materials

    Describes a stratified system of materials

    Attributes
    ----------
    zs : list or (N)-array
        Z values of the material interfaces, the corresponding material ends at these
        values
    materials : list of Materials
        Material definitions
    """

    def __init__(self, zs, materials):
        zs = list(zs)
        materials = list(materials)
        if len(zs) != len(materials):
            raise ValueError("Cannot match zs and materials of different length")
        self.zs = zs
        self.materials = materials

    def modes(self, k0, kxs, kys):
        """Mode calculation for all media

        Calls `Material.modes` for each material in the stack.

        Arguments
        ---------
        k0 : scalar
            Wave number (in vacuum)
        kxs : (N?, M)-array or scalar
            X component of the wave vector
        kys : (N, M?)-array or scalar
            Y component of the wave vector

        """
        for m in set(self.materials):
            m.modes(k0, kxs, kys)

    def solve(self, k0, kxs, kys, modes=True):
        """Calculate transfer matrices for the stack

        Arguments
        ---------
        k0 : scalar
            Wave number (in vacuum)
        kxs : (N?, M)-array or scalar
            X component of the wave vector
        kys : (N, M?)-array or scalar
            Y component of the wave vector
        modes : bool, optional
            Call `Material.modes` for each material, otherwise it is assumed, that they
            are already solved for the same wave vector paramters

        Returns
        -------
        ts : (L, N, M, 4, 4)-array
            Transfer matrices, L corresponds to the length of the stack. The first
            matrix does only include a propagation and no interface
        m : (N, M, 4, 2)-array
            Relation of the two input modes with reflected (third index 0 and 2) and
            transmitted (third index 1 and 2) modes
        """
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
                dyn = dyn_mat(
                    self.materials[l].ps[i, j, :, :], self.materials[l].qs[i, j, :, :]
                )
                if l == 0:
                    self.ts[0, i, j, :, :] = prop
                    allts = prop
                else:
                    self.ts[l, i, j, :, :] = np.linalg.solve(lastdyn, dyn @ prop)
                    allts = allts @ self.ts[l, i, j, :, :]
                lastdyn = dyn
            denom = 1 / (allts[0, 0] * allts[2, 2] - allts[0, 2] * allts[2, 0])
            self.m[i, j, :, :] = denom * np.array(
                [
                    [
                        allts[1, 0] * allts[2, 2] - allts[1, 2] * allts[2, 0],
                        allts[1, 2] * allts[0, 0] - allts[0, 2] * allts[1, 0],
                    ],
                    [allts[2, 2], -allts[0, 2]],
                    [
                        allts[3, 0] * allts[2, 2] - allts[3, 2] * allts[2, 0],
                        allts[3, 2] * allts[0, 0] - allts[0, 2] * allts[3, 0],
                    ],
                    [-allts[2, 0], allts[0, 0]],
                ]
            )
        return self.ts, self.m


def prop_mat(kzs, d):
    """Propagtion matrix

    Actually the backwards propagation

    Arguments
    ---------
    kzs : (N)-array or scalar
        Z component of the wave vector
    d : scalar
        Propagation distance

    Returns
    -------
    (N, N)-matrix
        Diagonal matrix of ``exp(-1j * kzs * d)``
    """
    kzs = np.array(kzs)
    return np.diag(np.exp(-1j * kzs * d))


def dyn_mat(ps, qs):
    """Dynamical matrix

    The tangential (x, y) components of the electric and magnetic field modes

    Arguments
    ---------
    ps : (4, 3)-array
        E-field modes
    qs : (4, 3)-array
        H-field modes

    Returns
    -------
    (4, 4)-matrix
        Tangential components as rows
    """
    ps = np.array(ps)
    qs = np.array(qs)
    res = np.empty((4, 4), np.complex)
    res[0, :] = ps[:, 0]
    res[1, :] = qs[:, 1]
    res[2, :] = ps[:, 1]
    res[3, :] = qs[:, 0]
    return res


def _sort_modes(kzs, ps, qs):
    """Sort the modes

    The modes are separated in forward and backward propagating (decaying) modes. The
    forward modes are put at positions 0 and 2, the backward modes in positions 1 and 3.
    Additionally, it is attempted to match the modes, such that 0 and 1 (2 and 3) are
    roughly equal, but this is allowed to silently fail.

    Arguments
    ---------
    kzs : (4)-array
        Z component of the wave vector
    ps : (4, 3)-array
        E-field modes
    qs : (4, 3)-array
        H-field modes

    Returns
    ---------
    kzs : (4)-array
        Z component of the wave vector sorted
    ps : (4, 3)-array
        E-field modes sorted
    qs : (4, 3)-array
        H-field modes sorted
    """
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
        raise Exception("Ambigous propagation direction of the modes")

    overlap = np.abs(
        [[np.dot(pup, pdown.conj()) for pdown in ps[down, :]] for pup in ps[up, :]]
    )
    perm = np.zeros((4,), np.int)
    perm[[0, 2]] = np.nonzero(up)[0]

    if overlap[0, 0] > overlap[0, 1] or overlap[1, 1] > overlap[1, 0]:
        perm[[1, 3]] = np.nonzero(down)[0]
    else:
        perm[[3, 1]] = np.nonzero(down)[0]
    return kzs[perm], ps[perm, :], qs[perm, :]


def poynting(ps, qs):
    """Poynting vector

    Arguments
    ---------
    ps : (..., 3)-array
        E-field modes
    qs : (..., 3)-array
        H-field modes

    Returns
    -------
    (..., 3)-array
        Poynting vectors
    """
    return 0.5 * np.real(np.cross(ps, qs.conj()))


def _det3(m):
    """Determinant of 3-by-3-matrix

    Brute force computation, works with anything that has addition, subtraction and
    multiplication defined, especially numpy.polynomial.polynomial.Polynomial.

    Arguments
    ---------
    m : (3, 3)-array
        Matrix
    """
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2])
        - m[1, 0] * (m[0, 1] * m[2, 2] - m[2, 1] * m[0, 2])
        + m[2, 0] * (m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2])
    )


def _findpairs(xs):
    """Pair the elements of xs

    Picks the two values with the shortest distance (by absolute value) and pairs them.
    Then, pairs the remaining two.

    Arguments
    ---------
    xs : (4)-array

    Returns
    -------
    dict
        The keys are the pair indices as tuple and the value the absolute distance
    """
    dists = {
        (i, j): np.abs(x - y)
        for (i, x), (j, y) in itertools.combinations(enumerate(xs), 2)
    }
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


def _multiplicity(xs, tol=1e-5):
    """Get the multiplicity of the roots

    Maximal multiplicity is 2.

    Arguments
    ---------
    xs : (4)-array

    tol : scalar
        (Non-negative) absolute tolerance to merge two values of xs to one

    Returns
    -------
    res : list
        Values
    mult : list of int
        Multiplicity
    """
    pairs = _findpairs(xs)
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


def _evalpolymat(m, x):
    """Evaluate a matrix of polynomials

    Arguments
    ---------
    m : array
        Array of polynomials
    x : scalar
        Value at which to evaluate the polynomials

    Returns
    -------
    array
        Evaluated matrix
    """
    shape = m.shape
    res = np.array([i(x) for i in m.flatten()])
    return res.reshape(shape)


def _eigenmodes(eq):
    """Eigenmodes of the Helmholtz equation

    The given equation is solved by finding values, where the polynomials give a
    non-trivial kernel. A total of four solutions are expected.

    Arguments
    ---------
    eq : (3, 3)-array
        Matrix of polynomials in kz

    Returns
    -------
    kzs : (4)-array
        Roots of the matrix determinant polynomial
    ps : (4, 3)-array
        Corresponding modes of the kernel

    """
    kzs = _det3(eq).roots()
    kzs_mult = _multiplicity(kzs)
    ps = np.zeros((4, 3), np.complex)
    i = 0
    for kz, mult in zip(*kzs_mult):
        sol = _evalpolymat(eq, kz)
        _, _, vh = np.linalg.svd(sol)
        for j in range(mult):
            ps[i, :] = vh[-j - 1, :].conj()
            kzs[i] = kz
            i += 1
    return kzs, ps
