"""
This example simulates three different polarization and how they are focused in a
birefringent resist.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt # Needed to plot the results

import focusaniso as fa

# General definitions
na = .8
f = 165000 / 25
w = 4750
n2 = 1.506
k0 = 2 * np.pi / 0.790

# Beam definitions
nphi = 72
nrho = 65
rhos = np.linspace(0, f * na / n2, nrho)
phis = np.linspace(0, 2 * np.pi * (1 - 1 / nphi), nphi)
rhos, phis = np.meshgrid(rhos, phis)
beams = []
beams.append((np.exp((-rhos * rhos) / (w * w)))[:,:,None] * np.array([0, 1, 0]))
beams.append((np.exp((-rhos * rhos) / (w * w)))[:,:,None] * np.dstack((-np.sin(phis), np.cos(phis), np.zeros_like(phis))))
beams.append((np.exp((-rhos * rhos) / (w * w)))[:,:,None] * np.array([1, 1j, 0]) * np.sqrt(0.5))

# Material definitions
alpha = np.pi / 4
r = np.array([
    [np.cos(alpha), 0, np.sin(alpha)],
    [0, 1, 0],
    [-np.sin(alpha), 0, np.cos(alpha)],
])
zs = [-270, -100, -100]
oil = fa.Material(n2**2)
glass = fa.Material(1.517**2)
resist = fa.Material(r @ np.diag([1.511**2, 1.511**2, 1.731**2]) @ r.T)
stack = fa.Stack(zs, [oil, glass, resist])

# Expected results
nz = 161
nx = 101
z_probe = np.linspace(-20, 60, nz)
x_probe = np.linspace(-25, 25, nx)
efield_sq = np.zeros((len(beams), nx, nz))

# Calculation
for i, b in enumerate(beams):
    spec = fa.Spectrum(b, rhos, phis, k0, coord='polar')
    spec.focus(f, n2=n2)

    specs_zs = spec.field(stack, z_probe, solve=(i == 0)) # Only solve once

    for j, s in enumerate(specs_zs):
        tmp = copy.deepcopy(s)
        tmp.setspace('real', x_probe, np.array([0]))
        efield_sq[i, :, j] = np.sum(np.abs(tmp.grid[0,:,:]), axis=-1)

efield_sq = efield_sq / np.max(efield_sq)

# Plotting
fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace=1.2)
titles = ['linear', 'azimuthal', 'circular']
for i, ax in enumerate(axs):
    ax.pcolormesh(z_probe, x_probe, efield_sq[i,:,:], shading='nearest')
    ax.set_xlabel(u'z (µm)')
    ax.set_ylabel(u'x (µm)')
    ax.set_title(titles[i])
    ax.set_aspect('equal', adjustable='box')
fig.savefig('birefringent_lacquer.png')
