.. index::
   single: gpumd output files

Output files
============

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - File name
     - Generating keyword
     - Brief description
     - Output mode
   * - :ref:`thermo.out <thermo_out>`
     - :ref:`dump_thermo <kw_dump_thermo>`
     - Global thermodynamic quantities
     - Append
   * - :ref:`movie.xyz <movie_xyz>`
     - :ref:`dump_position <kw_dump_position>`
     - Trajectory (atomic positions, velocities etc)
     - Append
   * - :ref:`restart.xyz <restart_xyz>`
     - :ref:`dump_restart <kw_dump_restart>`
     - The restart file
     - Overwrite
   * - :ref:`velocity.out <velocity_out>`
     - :ref:`dump_velocity <kw_dump_velocity>`
     - Contains the atomic velocities
     - Append
   * - :ref:`force.out <force_out>`
     - :ref:`dump_force <kw_dump_force>`
     - Contains the atomic forces
     - Append
   * - :ref:`compute.out <compute_out>`
     - :ref:`compute <kw_compute>`
     - Time and space (group) averaged quantities
     - Append
   * - :ref:`hac.out <hac_out>`
     - :ref:`compute_hac <kw_compute_hac>`
     - Thermal conductivity data from the :term:`EMD` method
     - Append
   * - :ref:`kappa.out <kappa_out>`
     - :ref:`compute_hnemd <kw_compute_hnemd>`
     - Thermal conductivity data from the :term:`HNEMD` method
     - Append
   * - :ref:`shc.out <shc_out>`
     - :ref:`compute_shc <kw_compute_shc>`
     - Spectral heat current (:term:`SHC`) data
     - Append
   * - :ref:`heatmode.out <heatmode_out>`
     - :ref:`compute_gkma <kw_compute_gkma>`
     - Modal heat current data from :term:`GKMA` method
     - Append
   * - :ref:`kappamode.out <kappamode_out>`
     - :ref:`compute_hnema <kw_compute_hnema>`
     - Modal thermal conductivity data from :term:`HNEMA` method
     - Append
   * - :ref:`dos.out <dos_out>`, :ref:`mvac.out <mvac_out>`
     - :ref:`compute_dos <kw_compute_dos>`
     - Phonon density of states (:term:`PDOS`) data
     - Append
   * - :ref:`sdc.out <sdc_out>`
     - :ref:`compute_sdc <kw_compute_sdc>`
     - Self-diffusion coefficient (:term:`SDC`) data
     - Append
   * - :ref:`cohesive.out <cohesive_out>`
     - :ref:`compute_cohesive <kw_compute_cohesive>`
     - Cohesive energy curve
     - Overwrite
   * - :ref:`D.out <D_out>`
     - :ref:`compute_phonon <kw_compute_phonon>`
     - Dynamical matrices :math:`D(\boldsymbol{k})` for the input k points
     - Overwrite
   * - :ref:`omega2.out <omega2_out>`
     - :ref:`compute_phonon <kw_compute_phonon>`
     - Phonon frequency squared :math:`\omega^2(\boldsymbol{k})` for the input :math:`\boldsymbol{k}`-points
     - Overwrite

.. toctree::
   :maxdepth: 0
   :caption: Contents

   cohesive_out
   compute_out
   D_out
   dos_out
   force_out
   hac_out
   heatmode_out
   kappa_out
   kappamode_out
   mvac_out
   movie_xyz
   omega2_out
   restart_xyz
   sdc_out
   shc_out
   thermo_out
   velocity_out