# focusaniso

A small tool to simulate focusing in anisotropic media

## Installation

You can simply put the file `focusaniso.py` where you want it. The only dependency is
`numpy`.

Alternatively, you can install the module with
`pip install git+https://git.scc.kit.edu/photonics/focusaniso.git`. I'd advise to do
this in a virtual environment. For the documentation and tests use
`pip install -e git+https://git.scc.kit.edu/photonics/focusaniso.git#egg=focusaniso[docs,test]`.
Omit `docs` or `test`, if you only want one of them.

## Getting started

We want to do a small example based on the results of [S. Wang *et al*][1]. For more
examples see the `examples` folder. First, we import the module and define the
properties.

```python
import numpy as np
import focusaniso as fa

k0 = 2 * np.pi # Wavelength is one
f = 60 # Focus length equal to 60 wavelengths
na = 1.4 # Numerical aperture
n2 = 1.518 # Refractive index of the medium focused into

# Those parameters mainly influence the accuracy
nrho = 23 # Radial values
nphi = 72 # Azimuthal values
# The beam is shaped as a ring with constant power between 0.9 * NA and NA with
# circular polarization
rhos, phis = np.meshgrid(
    f * na / n2 * np.linspace(0.9, 1, nrho)),
    np.linspace(0, 2 * np.pi * (1 - 1 / nphi), nphi),
)
beam = np.ones_like(rhos)[:, :, :] * np.array([1, 1j, 0]) * np.sqrt(0.5)
```
Next, we define that this beam is a real space spectrum defined on a polar grid and
then immediatly focus that light.
```python
spec = fa.Spectrum(beam, rhos, phis, k0, coord='polar')
spec.focus(f, n2=n2)
```
Now we turn to the materials, where we have the immersion oil where we focused into, and
a uniaxial crystal with the interface 30 wavelengths in front of the focus.
```python
oil = Material(n2**2)
crystal = Material(np.diag([1.6555**2, 1.6555**2, 1.4851**2]))
stack = Stack([-30, -30], [oil, crystal])
```
We will propagate the focused spectrum in this stack and probe it at a several z values
```python
z_probe = np.linspace(-10, 15, 251)
specs_probe = spec.field(stack, zs)
```
These angular spectra will be converted to the real space at multiple x values, such
that the electric field is known in a x-z-plane.
```python
x_probe = np.linspace(-2, 2, 41)
abs_sq_efield = np.zeros((41, 241))
for i, s in enumerate(specs_probe):
    s.setspace('real', x_probe, 0)
    abs_sq_efield[:, i] = np.sum(np.power(np.abs(s.grid), 2), axis=-1)
```

## Documentation

The documentation is created with sphinx. Run `make html` in the `docs` directory. Then,
html files are found in `docs/_build/html/` with `index.html` being the starting page.
You can also create a pdf file using latex with `make latexpdf` in the `docs` directory,
which is put in `docs/_build/latex/focusaniso.pdf`

## Testing

The tests are done with pytest, using `python -m pytest`. Use the option
`--cov focusaniso` for a coverage report.

[1]: <https://doi.org/10.1364/JOSAA.32.001026>
     "S. Wang, et al., J. Opt. Soc. Am. A 32, 1026-1031 (2015)"
