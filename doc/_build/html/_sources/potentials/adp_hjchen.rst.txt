.. _adp:
.. index::
   single: Angular Dependent Potential

Angular Dependent Potential (ADP)
==================================

:program:`GPUMD` supports the Angular Dependent Potential (ADP), which is an extension of the embedded atom method (:term:`EAM`) that includes angular forces through dipole and quadrupole distortions of the local atomic environment.

The ADP was developed to provide a more accurate description of directional bonding and angular forces in metallic systems, particularly for materials where traditional EAM potentials fail to capture the full complexity of atomic interactions. The ADP formalism is especially useful for modeling complex crystal structures, defects, and phase transformations in metals and alloys.

Potential form
--------------

General form
^^^^^^^^^^^^

The ADP is described in detail by Mishin et al. [Mishin2005]_. The total energy of atom :math:`i` is given by:

.. math::
   
   E_i = F_\alpha\left(\sum_{j\neq i} \rho_\beta(r_{ij})\right) + \frac{1}{2} \sum_{j\neq i} \phi_{\alpha\beta}(r_{ij}) + \frac{1}{2} \sum_s (\mu_{is})^2 + \frac{1}{2} \sum_{s,t} (\lambda_{ist})^2 - \frac{1}{6} \nu_i^2

where:

- :math:`F_\alpha` is the embedding energy as a function of the total electron density at atom :math:`i`
- :math:`\rho_\beta(r_{ij})` is the electron density contribution from atom :math:`j` at distance :math:`r_{ij}`
- :math:`\phi_{\alpha\beta}(r_{ij})` is the pair potential interaction between atoms of types :math:`\alpha` and :math:`\beta`
- :math:`\alpha` and :math:`\beta` are the element types of atoms :math:`i` and :math:`j`
- :math:`s` and :math:`t` are indices running over Cartesian coordinates (:math:`x, y, z`)
- :math:`\mu_{is}` is the dipole distortion tensor (3 components)
- :math:`\lambda_{ist}` is the quadrupole distortion tensor (6 independent components)
- :math:`\nu_i` is the trace of the quadrupole tensor

Angular terms
^^^^^^^^^^^^^

The dipole distortion tensor represents the first moment of the local environment:

.. math::
   
   \mu_{is} = \sum_{j\neq i} u_{\alpha\beta}(r_{ij}) r_{ij}^s

where :math:`u_{\alpha\beta}(r)` is a tabulated function and :math:`r_{ij}^s` is the :math:`s`-component of the vector from atom :math:`i` to atom :math:`j`.

The quadrupole distortion tensor represents the second moment of the local environment:

.. math::
   
   \lambda_{ist} = \sum_{j\neq i} w_{\alpha\beta}(r_{ij}) r_{ij}^s r_{ij}^t

where :math:`w_{\alpha\beta}(r)` is another tabulated function. The trace of the quadrupole tensor is:

.. math::
   
   \nu_i = \lambda_{ixx} + \lambda_{iyy} + \lambda_{izz}

The angular terms :math:`\mu` and :math:`\lambda` introduce directional dependence into the potential energy, allowing the ADP to capture angular forces that are absent in the traditional EAM formalism. These terms are essential for accurately modeling materials with complex bonding environments.

Physical interpretation
^^^^^^^^^^^^^^^^^^^^^^^

The ADP energy can be understood as:

1. **EAM contribution**: :math:`F(\rho) + \frac{1}{2}\phi(r)` - Standard many-body and pair interactions
2. **Dipole contribution**: :math:`\frac{1}{2}\sum_s \mu_{is}^2` - Energetic penalty for asymmetric local environment
3. **Quadrupole contribution**: :math:`\frac{1}{2}\sum_{s,t} \lambda_{ist}^2 - \frac{1}{6}\nu_i^2` - Energetic penalty for aspherical distortions

The angular terms effectively measure deviations from ideal local symmetry, with the energy increasing for distorted configurations. This makes the ADP particularly well-suited for modeling:

- Crystal defects (vacancies, interstitials, dislocations)
- Grain boundaries and surfaces
- Phase transformations
- Materials under complex stress states

File format
-----------

General structure
^^^^^^^^^^^^^^^^^

The ADP potential file follows the extended DYNAMO setfl format, which is compatible with LAMMPS and other molecular dynamics codes. The file structure consists of:

**Header section** (lines 1-5):

- Lines 1-3: Comment lines (can contain any text, typically author and date information)
- Line 4: :attr:`Nelements` :attr:`Element1` :attr:`Element2` ... :attr:`ElementN`

  * :attr:`Nelements`: Number of elements in the potential
  * :attr:`Element1`, :attr:`Element2`, etc.: Element symbols (e.g., Cu, Ta, Mo)

- Line 5: :attr:`Nrho` :attr:`drho` :attr:`Nr` :attr:`dr` :attr:`cutoff`

  * :attr:`Nrho`: Number of points in the embedding function :math:`F(\rho)` tabulation
  * :attr:`drho`: Spacing between tabulated :math:`\rho` values
  * :attr:`Nr`: Number of points in the pair potential and density function tabulations
  * :attr:`dr`: Spacing between tabulated :math:`r` values
  * :attr:`cutoff`: Cutoff distance for all functions (in Angstroms)

**Per-element sections** (repeated :attr:`Nelements` times):

Each element section contains:

- Line 1: :attr:`atomic_number` :attr:`mass` :attr:`lattice_constant` :attr:`lattice_type`

  * :attr:`atomic_number`: Atomic number of the element
  * :attr:`mass`: Atomic mass (in amu)
  * :attr:`lattice_constant`: Equilibrium lattice constant (in Angstroms)
  * :attr:`lattice_type`: Crystal structure (e.g., fcc, bcc, hcp)

- Next :attr:`Nrho` values: Embedding function :math:`F(\rho)` 

  * Tabulated values of :math:`F` at :math:`\rho = 0, \Delta\rho, 2\Delta\rho, ..., (N_\rho-1)\Delta\rho`
  * Units: eV

- Next :attr:`Nr` values: Electron density function :math:`\rho(r)`

  * Tabulated values at :math:`r = 0, \Delta r, 2\Delta r, ..., (N_r-1)\Delta r`
  * Units: electron density

**Pair potential section**:

For all element pairs :math:`(i, j)` with :math:`i \geq j` (upper triangular, since :math:`\phi_{ij} = \phi_{ji}`):

- :attr:`Nr` values: Pair potential :math:`\phi_{ij}(r)`

  * Tabulated as :math:`r \times \phi(r)` (scaled by distance)
  * Units: eV·Angstrom
  * Order: (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), etc.

**Dipole function section**:

For all element pairs :math:`(i, j)` with :math:`i \geq j`:

- :attr:`Nr` values: Dipole function :math:`u_{ij}(r)`

  * Tabulated as :math:`u(r)` (NOT scaled by distance)
  * Units: electron density·Angstrom
  * Same ordering as pair potentials

**Quadrupole function section**:

For all element pairs :math:`(i, j)` with :math:`i \geq j`:

- :attr:`Nr` values: Quadrupole function :math:`w_{ij}(r)`

  * Tabulated as :math:`w(r)` (NOT scaled by distance)  
  * Units: electron density·Angstrom²
  * Same ordering as pair potentials

.. note::

   **Important**: Only :math:`\phi(r)` is tabulated with an :math:`r` scaling factor. The functions :math:`u(r)` and :math:`w(r)` are tabulated directly without scaling. This is a critical detail when creating or converting ADP potential files.

Example format
^^^^^^^^^^^^^^

For a two-element system (e.g., Cu-Ta), the file structure would be::

    # Comment line 1: ADP potential for Cu-Ta system
    # Comment line 2: Developed by Author Name
    # Comment line 3: Date: YYYY-MM-DD
    2 Cu Ta
    10000 0.001 10000 0.001 6.5
    
    # Element 1: Cu
    29 63.546 3.615 fcc
    <10000 values of F_Cu(rho)>
    <10000 values of rho_Cu(r)>
    
    # Element 2: Ta
    73 180.948 3.303 bcc
    <10000 values of F_Ta(rho)>
    <10000 values of rho_Ta(r)>
    
    # Pair potentials
    <10000 values of r*phi_CuCu(r)>
    <10000 values of r*phi_TaCu(r)>
    <10000 values of r*phi_TaTa(r)>
    
    # Dipole functions
    <10000 values of u_CuCu(r)>
    <10000 values of u_TaCu(r)>
    <10000 values of u_TaTa(r)>
    
    # Quadrupole functions
    <10000 values of w_CuCu(r)>
    <10000 values of w_TaCu(r)>
    <10000 values of w_TaTa(r)>

The tabulated values are typically written with 5 values per line, separated by whitespace.

Element mapping
^^^^^^^^^^^^^^^

By default, GPUMD maps elements from the potential file to atoms in the simulation based on the order they appear in the structure file. However, you can explicitly specify the element mapping using an extended syntax:

**Default mapping**::

    potential adp Cu_Ta.adp

This maps the first element in the potential file (Cu) to atom type 0, and the second element (Ta) to atom type 1.

**Explicit mapping**::

    potential adp Cu_Ta.adp Cu Ta

This explicitly maps Cu → type 0 and Ta → type 1.

**Reordered mapping**::

    potential adp Cu_Ta.adp Ta Cu

This maps Ta → type 0 and Cu → type 1, effectively swapping the element assignments.

The explicit mapping is particularly useful when:

- Your structure file has atoms in a different order than the potential file
- You want to use a subset of elements from a multi-element potential
- You need to verify which element corresponds to which atom type

.. warning::

   The number of elements specified in the mapping must match the number of atom types in your system. Mismatches will cause GPUMD to report an error during initialization.

Usage
-----

Basic usage
^^^^^^^^^^^

To use an ADP potential in GPUMD, specify it in the :file:`run.in` input file::

    potential adp Ta.adp

The potential file must begin with the line::

    adp

followed by the tabulated data in the extended DYNAMO setfl format as described above.

Single-element system
^^^^^^^^^^^^^^^^^^^^^

For a pure metal system (e.g., pure molybdenum)::

    potential adp Mo.adp

GPUMD will automatically map the Mo element from the potential file to all atoms in the system.

Multi-element system
^^^^^^^^^^^^^^^^^^^^

For a binary alloy (e.g., U-Mo)::

    potential adp U_Mo.adp

By default, the first element in the potential file (U) maps to atom type 0, and the second element (Mo) maps to atom type 1. Ensure your structure file has the correct atom type assignments.

Explicit element mapping::

    potential adp U_Mo.adp U Mo

Or with reversed order::

    potential adp U_Mo.adp Mo U

This is useful when your structure file has atom types in a different order than the potential file.

Implementation details
----------------------

Interpolation method
^^^^^^^^^^^^^^^^^^^^

GPUMD uses **Hermite cubic spline interpolation** for evaluating the tabulated functions :math:`F(\rho)`, :math:`\rho(r)`, :math:`\phi(r)`, :math:`u(r)`, and :math:`w(r)`. 

The Hermite spline method:

- Constructs cubic polynomials between each pair of tabulated points
- Uses finite difference approximations for derivatives at grid points
- Provides smooth, continuous first derivatives across the entire interpolation range
- Offers excellent balance between accuracy and computational efficiency

The spline coefficients are pre-computed during initialization and stored in GPU memory for fast lookup during force calculations.

Neighbor list construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPUMD automatically selects the optimal neighbor list algorithm based on system size and cutoff radius:

- **Cell list method** (O(N)): Used by default for most systems

  * Divides simulation box into cells of size ≥ cutoff radius
  * Efficiently handles systems with :math:`r_c > 0.5 \times L_{\text{box}}`
  * Automatically accounts for periodic boundary conditions
  * Recommended for all production simulations

- **Brute force method** (O(N²)): Automatic fallback for very small systems

  * Used only when box dimensions are too small for cell list
  * Typically activated for systems with < ~100 atoms
  * Less efficient but guarantees correctness for edge cases

Users do not need to specify the neighbor list method - GPUMD selects the optimal algorithm automatically.

CUDA optimization
^^^^^^^^^^^^^^^^^

The ADP implementation in GPUMD is highly optimized for NVIDIA GPUs:

- **Two-pass kernel design**: 
  
  * First pass: Compute electron density, dipole, and quadrupole tensors
  * Second pass: Compute forces and stresses using precomputed quantities

- **Memory coalescing**: All atomic data stored in structure-of-arrays format for optimal memory access patterns

- **Inline functions**: Critical interpolation routines marked with ``__forceinline__`` to reduce function call overhead

- **Restrict pointers**: ``__restrict__`` qualifiers enable compiler optimizations by guaranteeing no pointer aliasing

- **Optimized block size**: Default thread block size of 128 provides good occupancy across different GPU architectures

Performance benchmarks on RTX 4060 Laptop GPU (sm_89) show throughput of ~45,000 atom·steps/second for typical Mo systems with ~10,000 atoms.

Limitations and considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Cutoff radius**: Must be consistent across all tabulated functions in the potential file

- **Tabulation resolution**: Higher :attr:`Nr` and :attr:`Nrho` values increase accuracy but also memory usage

  * Typical values: :attr:`Nr` = :attr:`Nrho` = 5,000 to 10,000
  * Grid spacing should be fine enough to resolve features in :math:`F`, :math:`\phi`, :math:`u`, :math:`w`

- **Boundary conditions**: The implementation assumes fully periodic boundary conditions (PBC) in all three directions

- **Units**: GPUMD uses the metal unit system (Angstrom, eV, amu, ps)

  * Ensure potential file uses consistent units
  * Distances in Angstroms, energies in eV

- **Compatibility**: ADP potential files from LAMMPS can be used directly in GPUMD without modification

Available potentials
--------------------

ADP potentials are available for various metallic systems. Check the :file:`potentials/` directory in the GPUMD distribution for example files.

Published ADP potentials include:

- **Uranium-Molybdenum (U-Mo)**: For nuclear fuel applications [Mishin2005]_
- **Aluminum-Copper (Al-Cu)**: For precipitation studies in aluminum alloys
- **Copper-Tantalum (Cu-Ta)**: For immiscible alloy systems
- **Molybdenum-Tantalum (Mo-Ta)**: For BCC refractory alloys

When using published potentials, always cite the original reference where the potential was developed and validated.

Troubleshooting
---------------

Common issues
^^^^^^^^^^^^^

**"ADP element mapping failed"**

- Check that the number of elements in the potential file matches your system
- Verify element symbols are spelled correctly
- Ensure atom types in structure file match the specified element mapping

**"Potential cutoff exceeds simulation box"**

- The cutoff radius :math:`r_c` must be less than half the shortest box dimension
- Either increase box size or use a potential with smaller cutoff
- For small systems, GPUMD will automatically switch to brute force neighbor list

**Poor energy conservation in NVE simulations**

- Check time step is sufficiently small (typically ≤ 1 fs for metals)
- Verify potential file tabulation has adequate resolution
- Ensure cutoff radius is appropriate for the system

**Unexpected forces or energies**

- Verify potential file format is correct (especially :math:`r\phi` vs :math:`\phi` scaling)
- Check that element mapping is correct
- Compare single-point energy/force calculation with LAMMPS for validation

References
----------

.. [Mishin2005] Y. Mishin, M. J. Mehl, and D. A. Papaconstantopoulos, "Interatomic potentials for monoatomic metals from experimental data and ab initio calculations," Phys. Rev. B 59, 3393 (1999); Y. Mishin, D. Farkas, M. J. Mehl, and D. A. Papaconstantopoulos, "Interatomic potentials for monoatomic metals from experimental data and ab initio calculations," Phys. Rev. B 59, 3393 (1999); Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter, and J. D. Kress, "Structural stability and lattice defects in copper: Ab initio, tight-binding, and embedded-atom calculations," Phys. Rev. B 63, 224106 (2001); Y. Mishin, M. R. Sørensen, and A. F. Voter, "Calculation of point-defect entropy in metals," Philos. Mag. A 81, 2591 (2001); Y. Mishin, M. J. Mehl, and D. A. Papaconstantopoulos, "Phase stability in the Fe–Ni system: Investigation by first-principles calculations and atomistic simulations," Acta Mater. 53, 4029 (2005).

See also
--------

- :ref:`eam` - Embedded Atom Method potential (parent formalism of ADP)
- :ref:`fcp` - Force-Constant Potential (another extension of EAM)
- LAMMPS `pair_style adp documentation <https://docs.lammps.org/pair_adp.html>`_

Example files
-------------

Example ADP potential files and test cases can be found in:

- :file:`examples/adp_U_Mo/` - U-Mo binary system with simulation examples
- :file:`examples/adp_Mo_Ta/` - Mo-Ta system with analysis scripts
- :file:`potentials/` - Collection of validated ADP potential files

These examples include:

- Complete potential files in correct format
- Sample structure files (XYZ format)
- :file:`run.in` input files for GPUMD
- Analysis and visualization scripts (MATLAB, Python)

For questions or issues with ADP potentials in GPUMD, please:

1. Check the examples directory for reference implementations
2. Validate your potential file format against working examples
3. Report issues on the GPUMD GitHub repository: https://github.com/brucefan1983/GPUMD
