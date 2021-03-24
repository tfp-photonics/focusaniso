.. focusaniso documentation master file, created by
   sphinx-quickstart on Tue Mar 23 16:14:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to focusaniso's documentation!
======================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

       Module description <focusaniso.rst>
       Theory <theory.rst>

This is a small tool for focusing and propagating light in anisotropic stratified media.
It includes a class `Spectrum` for representing real and angular spectra of light. This
spectrum can be focused, using the approach of Richards and Wolf [1]_.

`Materials` can be defined in a class of this name and solved for modes defined by the
tangential components of the wave vector. A stratified medium of the materials is
defined in a `Stack`, which can be solved using the solutions in the material and
including interface conditions. This part mainly follows the method of Yeh [2]_.

A spectrum can be propagated through a stack returning the field at arbitrary positions.

.. [1] `B. Richards and E. Wolf, Proc. R. Soc. Lond. A 253, 358–379 (1959)
        <https://doi.org/10.1098/rspa.1959.0200>`_

.. [2] `P. Yeh, Surface Science 96, 41-53 (1980)
        <https://doi.org/10.1016/0039-6028(80)90293-9>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
