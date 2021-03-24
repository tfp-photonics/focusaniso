Theory
======

A very short overview over the stuff the code implements.

Real and angular spectrum
-------------------------

In stratified media a useful tool is the angular spectrum. We have a distinguished axis,
which is conventionally chosen as z-axis. The angular spectrum
:math:`\boldsymbol{\underline E}(k_x, k_y; z)` is the Fourier transform of the real
field :math:`\boldsymbol E(x, y; z)` given by

.. math::

    \boldsymbol E(x, y; z)
    = \iint \mathrm d k_x \mathrm d k_y
    \boldsymbol{\underline E}(k_x, k_y; z)
    \mathrm e^{\mathrm i (k_x x + k_y y)}
    \\
    \boldsymbol{\underline E}(k_x, k_y; z)
    = \frac{1}{(2\pi)^2} \iint \mathrm d k_x \mathrm d k_y
    \boldsymbol E(x, y; z)
    \mathrm e^{-\mathrm i (k_x x + k_y y)}

and the relation

.. math::

    \boldsymbol{\underline E}(k_x, k_y; z)
    = \mathrm e^{\mathrm i k_z z}
    \boldsymbol{\underline E}(k_x, k_y; 0)

to propagate the field along the z axis. The important knowledge now is the value of
:math:`k_z`, which in homogeneous isotropic media come from the dispersion relation and
the propagation direction is simply given by its sign.
For anisotropic media, this relation gets more complicated, since the electric field
has to be decomposed into the different modes with different values of :math:`k_z`.


Anisotropic media
-----------------
In a homogeneous anisotropic medium, defined by its relative permittivity tensor
:math:`\epsilon` and relative permeability tensor :math:`\mu` we solve the equation

.. math::

    (\boldsymbol k \times \mu^{-1} \times \boldsymbol k + k_0^2 \epsilon)
    \boldsymbol{\underline E} = 0

using a plane wave ansatz. To find non-trivial solutions means to find values resulting
in a non-trivial kernel. Taking :math:`k_x` and :math:`k_y` as given, this equation has
:math:`k_y` as only free parameter left. The determinant of the term in brackets leads
to a fourth order polynomial equation in :math:`k_y`. Thus, four solutions can be
obtained that correspond to two polarizations each for forward and backward propagation.
The span of the kernel corresponds to those field polarizations. For degenerate modes
the kernel has a dimension corresponding to the multiplicity. We denote the
polarizations with :math:`\boldsymbol p`.

The magnetic field :math:`\boldsymbol H` is then given by the polarization vectors
:math:`\boldsymbol q = \mu^{-1} \boldsymbol k \times \boldsymbol p`. From electric and
magnetic field the Poynting vector can be obtained to distinguish the two principal
directions of propagation. For non-propagating modes the imaginary part of :math:`k_z`
determines the direction.

With the full wave vector at hand, the angular spectrum modes can easily be propagated
in the medium. At an interface between two anisotropic media the tangential components
of the electric and magnetic fields have to be conserved. With this knowledge transfer
matrices between different media can be derived. For the angular spectrum the field
has to be separated into those modes and then be propagated with the according value of
:math:`k_z`.

Focusing
--------

The focusing calculation is based on two steps. First, the light is refracted at a
plane, then, it is converted to an angular spectrum.

For the refraction of the light we work in polar coordinates. The electric field is
decomposed into the radial :math:`\boldsymbol{\hat \rho}` and azimuthal
:math:`\boldsymbol{\hat \varphi}` component, a z component is neglected.
The radial component is refracted onto

.. math::

    \boldsymbol{\hat \vartheta}
    = (\cos\vartheta\cos\varphi, \cos\vartheta\sin\varphi, -\sin\vartheta)^{\mathrm T}

where :math:`\sin\vartheta = \frac{\rho}{f}` and :math:`f` the focal length of the lens.
The azimuthal component is refracted onto itself. Additionally, we need to include
energy conservation at this refraction onto a reference sphere. This adds the prefactor
in

.. math::

    \boldsymbol E_{\text{refr}}(f\vartheta, \varphi)
    =
    \sqrt{\frac{n_1}{n_2} \cos\vartheta}
    (t_p \boldsymbol{\hat \vartheta} \otimes \boldsymbol{\hat \rho}
    + t_s \boldsymbol{\hat \varphi} \otimes \boldsymbol{\hat \varphi})
    \boldsymbol E_{\text{illu}}(\rho, \varphi)\,.

Here, we also added possible transmission coefficients for the two polarizations.

This refracted field is assumed to be far away from the focus. We want to go to an
angular spectrum at the focal point. So, we plug the results into the corresponding
Fourier integral and propagate the field

.. math::

    \boldsymbol E_\text{refr}(x, y)
    = \iint \mathrm d k_x \mathrm d k_y
    \boldsymbol{\underline E}(k_x, k_y; z \ll 0)
    \mathrm e^{\mathrm i (k_x x + k_y y)}
    \\
    = \iint \mathrm d k_x \mathrm d k_y
    \boldsymbol{\underline E}(k_x, k_y; 0)
    \mathrm e^{\mathrm i (k_x x + k_y y - \sqrt(k^2 - k_x^2 - k_y^2) |z|)}

which we evaluate in the limiting case of a far distance from the focal point by the
stationary phase approximation

.. math::

    \boldsymbol E_\text{refr}(x, y)
    = \lim_{kr \rightarrow \infty} \iint_{k_x^2 + ky^2 \leq k^2}
    \mathrm d k_x \mathrm d k_y
    \boldsymbol{\underline E}(k_x, k_y; 0)
    \mathrm e^{\mathrm i k r (\frac{k_x x}{kr} + \frac{k_y x}{kr} - \frac{k_z |z|}{kr})}
    \\
    = 2 \pi \mathrm i \frac{\mathrm e^{\mathrm i k r}}{r} k_z
    \boldsymbol{\underline E}\left(\frac{-k x}{r}, \frac{-k y}{r}; 0\right)\,.

This expression can be inverted to result in

.. math::

    \boldsymbol{\underline E}\left(\frac{k \rho}{f}, \varphi + \pi\right)
    =
    \frac{-\mathrm i f \mathrm e^{-\mathrm i k f}}{2 \pi k_z}
    \boldsymbol E_\text{refr}(\rho, \varphi)

where :math:`r` is replaced by the focal length.
