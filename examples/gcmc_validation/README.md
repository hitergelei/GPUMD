# GCMC Validation Example

This example demonstrates how to validate the CUDA GCMC implementation against LAMMPS results.

## Files in this directory:

1. **GPUMD files:**
   - `run.in` - GPUMD input for CUDA GCMC simulation
   - `model.xyz` - Initial atomic configuration
   - `nep.txt` - Neural network potential (if using NEP)

2. **LAMMPS reference files:**
   - `lammps_gcmc.in` - LAMMPS GCMC script for comparison
   - `lammps_umbrella.in` - LAMMPS umbrella sampling GCMC
   - `potential.eam` - EAM potential for LAMMPS (if needed)

3. **Analysis scripts:**
   - `compare_results.py` - Python script to compare GPUMD vs LAMMPS
   - `plot_density.py` - Plot density profiles and acceptance ratios
   - `validation_metrics.py` - Calculate validation metrics

## How to run validation:

1. **Run GPUMD simulation:**
   ```bash
   cd examples/gcmc_validation
   ../../src/gpumd
   ```

2. **Run LAMMPS reference:**
   ```bash
   lmp -in lammps_gcmc.in
   ```

3. **Compare results:**
   ```bash
   python compare_results.py
   ```

## Expected validation metrics:

- Density profiles should match within 5%
- Average particle number should agree within 2%
- Acceptance ratios should be similar (±10%)
- Energy distributions should overlap significantly
