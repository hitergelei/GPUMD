.. _adp:
.. index::
   single: Angular Dependent Potential

Angular Dependent Potential (ADP)
==================================

:program:`GPUMD` supports the Angular Dependent Potential (ADP), which is an extension of the embedded atom method (:term:`EAM`) that includes angular forces through dipole and quadruple distortions of the local atomic environment.

Potential form
--------------

The ADP is described in detail by Y. Mishin et al. [Mishin2005]_. The total energy of atom :math:`i` is given by:

.. math::
   
   E_i = F_\alpha\left(\sum_{j\neq i} \rho_\beta(r_{ij})\right) + \frac{1}{2} \sum_{j\neq i} \phi_{\alpha\beta}(r_{ij}) + \frac{1}{2} \sum_s (\mu_{is})^2 + \frac{1}{2} \sum_{s,t} (\lambda_{ist})^2 - \frac{1}{6} \nu_i^2

where:

- :math:`F` is the embedding energy which is a function of the atomic electron density :math:`\rho`
- :math:`\phi` is a pair potential interaction
- :math:`\alpha` and :math:`\beta` are the element types of atoms :math:`i` and :math:`j`
- :math:`s` and :math:`t = 1,2,3` refer to the cartesian coordinates

The angular terms are given by:

.. math::
   
   \mu_{is} = \sum_{j\neq i} u_{\alpha\beta}(r_{ij}) r_{ij}^s

.. math::
   
   \lambda_{ist} = \sum_{j\neq i} w_{\alpha\beta}(r_{ij}) r_{ij}^s r_{ij}^t

where :math:`u` and :math:`w` are additional tabulated functions in the ADP potential file, and :math:`\nu_i = \lambda_{ixx} + \lambda_{iyy} + \lambda_{izz}` is the trace of the quadruple tensor.

The :math:`\mu` and :math:`\lambda` terms represent the dipole and quadruple distortions of the local atomic environment which extend the original EAM framework by introducing angular forces.

File format
-----------

The ADP potential file follows the extended DYNAMO setfl format as used in LAMMPS. The file structure is:

- Lines 1-3: comments (ignored)
- Line 4: :attr:`Nelements` :attr:`Element1` :attr:`Element2` ... :attr:`ElementN`
- Line 5: :attr:`Nrho` :attr:`drho` :attr:`Nr` :attr:`dr` :attr:`cutoff`

Following the 5 header lines are :attr:`Nelements` sections, one for each element:

- Line 1: atomic number, mass, lattice constant, lattice type
- embedding function :math:`F(\rho)` (:attr:`Nrho` values)
- density function :math:`\rho(r)` (:attr:`Nr` values)

Following the :attr:`Nelements` sections, :attr:`Nr` values for each pair potential :math:`\phi(r)` array are listed for all :math:`i,j` element pairs. Since these interactions are symmetric (:math:`i,j = j,i`) only :math:`\phi` arrays with :math:`i \geq j` are listed.

The tabulated values for each :math:`\phi` function are listed as :math:`r \times \phi` (in units of eV-Angstroms), since they are for atom pairs.

After the :math:`\phi(r)` arrays, each of the :math:`u(r)` arrays are listed in the same order with the same assumptions of symmetry.

Directly following the :math:`u(r)`, the :math:`w(r)` arrays are listed. Note that :math:`\phi(r)` is the only array tabulated with a scaling by :math:`r`.

Usage
-----

To use an ADP potential in GPUMD::

    potential adp Ta.adp

The first line of the potential file should contain::

    adp

followed by the tabulated data in the extended DYNAMO setfl format.

References
----------

.. [Mishin2005] Y. Mishin, M. J. Mehl, and D. A. Papaconstantopoulos, "Interatomic potentials for monoatomic metals from experimental data and ab initio calculations," Phys. Rev. B 65, 224114 (2002); Y. Mishin et al., "Structural stability and lattice defects in copper: Ab initio, tight-binding, and embedded-atom calculations," Phys. Rev. B 63, 224106 (2001).
