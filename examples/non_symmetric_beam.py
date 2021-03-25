"""
This example is the focusing of a plane wave with oblique incidence. The resulting plot
shows how the focus is shifted.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt # Needed to plot the results

import focusaniso as fa

# General definitions
na = 0.5
f = 100
k0 = 2 * np.pi
aperture = 50

# Beam definitions
x = np.linspace(-50, 50, 51)
x, y = np.meshgrid(x, x)
beam = (
    np.exp(0.5j * x)
    * np.heaviside(aperture * aperture - x * x - y * y, 1)
    )[:,:,None] * np.array([0, 1])

# Material definitions
zs = [0]
air = fa.Material(1)
stack = fa.Stack([0], [air])


spec = fa.Spectrum(beam, x, y, k0)
spec.focus(f)

nz = 101
nx = 101
z_probe = np.linspace(-10, 10, nz)
x_probe = np.linspace(-10, 10, nx)
efield_sq = np.zeros((nx, nz))

# Calculation

specs_zs = spec.field(stack, z_probe)

for j, s in enumerate(specs_zs):
    tmp = copy.deepcopy(s)
    tmp.setspace('real', x_probe, np.array([0]))
    efield_sq[:, j] = np.sum(np.abs(tmp.grid[0,:,:]), axis=-1)

efield_sq = efield_sq / np.max(efield_sq)

# Plotting
fig, ax = plt.subplots()
ax.pcolormesh(z_probe, x_probe, efield_sq, shading='nearest')
ax.set_xlabel(u'z (µm)')
ax.set_ylabel(u'x (µm)')
ax.set_aspect('equal', adjustable='box')
fig.savefig('non_symmetric_beam.png')
