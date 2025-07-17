/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Example usage of CUDA GCMC with Umbrella Sampling enhancement
Based on LAMMPS fix_gcmc_umbrella implementation
------------------------------------------------------------------------------*/

#include "mc_ensemble_cuda_gcmc.cuh"
#include "../model/atom.cuh"
#include "../model/box.cuh"
#include "../force/nep.cuh"

void example_umbrella_sampling_adsorption()
{
  printf("=== CUDA GCMC Umbrella Sampling: Gas Adsorption Example ===\n");
  
  // Initialize CUDA GCMC ensemble
  MC_Ensemble_CUDA_GCMC gcmc;
  
  // Configure basic GCMC parameters
  gcmc.set_number_of_steps(100000);
  gcmc.set_sample_interval(100);
  gcmc.set_temperature(300.0); // K
  gcmc.set_pressure(1.0);      // atm
  
  // Set chemical potentials for different species
  std::vector<std::string> species = {"Ar", "N2", "CO2"};
  std::vector<double> mu = {-2.5, -3.2, -4.1}; // kT units
  gcmc.set_chemical_potentials(species, mu);
  
  // Enable umbrella sampling for enhanced adsorption sampling
  gcmc.enable_umbrella_sampling = true;
  gcmc.target_type = 0; // Ar atoms (first species)
  gcmc.umbrella_target_atoms = 50; // Target 50 Ar atoms
  gcmc.umbrella_force_constant = 2.0; // Spring constant k
  gcmc.enable_adaptive_umbrella = true;
  
  // Configure advanced sampling
  gcmc.enable_batch_processing = true;
  gcmc.batch_size_insertion = 32;
  gcmc.batch_size_deletion = 16;
  gcmc.enable_energy_caching = true;
  gcmc.enable_neighbor_optimization = true;
  
  // Set move probabilities
  gcmc.prob_insertion = 0.3;
  gcmc.prob_deletion = 0.3;
  gcmc.prob_displacement = 0.4;
  
  // Configure output
  gcmc.verbose_output = true;
  gcmc.enable_detailed_output = true;
  
  printf("Umbrella sampling parameters:\n");
  printf("  Target atoms: %d\n", gcmc.umbrella_target_atoms);
  printf("  Force constant: %.3f\n", gcmc.umbrella_force_constant);
  printf("  Adaptive tuning: %s\n", gcmc.enable_adaptive_umbrella ? "enabled" : "disabled");
}

void example_umbrella_sampling_binary_mixture()
{
  printf("=== CUDA GCMC Umbrella Sampling: Binary Mixture Example ===\n");
  
  MC_Ensemble_CUDA_GCMC gcmc;
  
  // Binary mixture of CO2 and N2
  std::vector<std::string> species = {"CO2", "N2"};
  std::vector<double> mu = {-3.8, -3.2};
  gcmc.set_chemical_potentials(species, mu);
  
  // Use umbrella sampling to maintain specific CO2:N2 ratio
  gcmc.enable_umbrella_sampling = true;
  gcmc.target_type = 0; // CO2 molecules
  gcmc.umbrella_target_atoms = 25; // Target 25 CO2 molecules
  gcmc.umbrella_force_constant = 1.5;
  
  // Enhanced sampling for better mixing
  gcmc.enable_wang_landau = false; // Use only umbrella sampling
  gcmc.enable_parallel_tempering = false;
  
  gcmc.set_number_of_steps(200000);
  gcmc.set_sample_interval(500);
  gcmc.set_temperature(273.15); // 0°C
  
  printf("Binary mixture umbrella sampling:\n");
  printf("  Species: CO2, N2\n");
  printf("  Target CO2 count: %d\n", gcmc.umbrella_target_atoms);
  printf("  Expected ratio: 1:1 (N2:CO2)\n");
}

void example_umbrella_sampling_multiple_windows()
{
  printf("=== CUDA GCMC Umbrella Sampling: Multiple Windows Example ===\n");
  
  // Multiple independent GCMC simulations with different umbrella targets
  std::vector<int> target_counts = {10, 20, 30, 40, 50, 60};
  std::vector<double> force_constants = {3.0, 2.5, 2.0, 1.5, 1.0, 0.8};
  
  for (size_t i = 0; i < target_counts.size(); ++i) {
    printf("\n--- Window %zu: Target %d atoms ---\n", i+1, target_counts[i]);
    
    MC_Ensemble_CUDA_GCMC gcmc;
    
    // Configure each window
    std::vector<std::string> species = {"Ar"};
    std::vector<double> mu = {-2.0};
    gcmc.set_chemical_potentials(species, mu);
    
    gcmc.enable_umbrella_sampling = true;
    gcmc.target_type = 0;
    gcmc.umbrella_target_atoms = target_counts[i];
    gcmc.umbrella_force_constant = force_constants[i];
    
    gcmc.set_number_of_steps(50000);
    gcmc.set_temperature(300.0);
    
    printf("  Force constant: %.2f\n", force_constants[i]);
    printf("  Expected enhanced sampling around %d atoms\n", target_counts[i]);
  }
  
  printf("\nNote: In practice, run these windows in parallel or sequentially\n");
  printf("      to construct free energy profile F(N) vs atom count N\n");
}

void example_umbrella_sampling_adaptive_tuning()
{
  printf("=== CUDA GCMC Umbrella Sampling: Adaptive Tuning Example ===\n");
  
  MC_Ensemble_CUDA_GCMC gcmc;
  
  std::vector<std::string> species = {"H2O"};
  std::vector<double> mu = {-4.5};
  gcmc.set_chemical_potentials(species, mu);
  
  // Enable adaptive umbrella sampling
  gcmc.enable_umbrella_sampling = true;
  gcmc.enable_adaptive_umbrella = true;
  gcmc.target_type = 0;
  gcmc.umbrella_target_atoms = 100; // Target 100 water molecules
  gcmc.umbrella_force_constant = 1.0; // Initial guess
  
  // Run with monitoring
  gcmc.set_number_of_steps(500000);
  gcmc.set_sample_interval(1000);
  gcmc.verbose_output = true;
  
  printf("Adaptive umbrella sampling features:\n");
  printf("  Initial force constant: %.2f\n", gcmc.umbrella_force_constant);
  printf("  Target atoms: %d\n", gcmc.umbrella_target_atoms);
  printf("  Adaptive tuning: enabled\n");
  printf("  Force constant will adjust based on acceptance rates\n");
  printf("    - Low acceptance (< 10%%): reduce k by 5%%\n");
  printf("    - High acceptance (> 70%%): increase k by 5%%\n");
}

void example_umbrella_bias_energy_calculation()
{
  printf("=== CUDA GCMC Umbrella Sampling: Bias Energy Calculation ===\n");
  
  MC_Ensemble_CUDA_GCMC gcmc;
  
  // Example bias energy calculations
  double k = 2.0;     // Force constant
  int n0 = 50;        // Target atom count
  
  printf("Umbrella bias energy U_bias = 0.5 * k * (N - N0)^2\n");
  printf("Force constant k = %.1f, Target N0 = %d\n\n", k, n0);
  
  printf("Atom count  |  Bias Energy  |  Relative to N0\n");
  printf("------------|---------------|----------------\n");
  
  for (int n = 30; n <= 70; n += 5) {
    double bias_energy = gcmc.calculate_umbrella_bias_energy(n, n0, k);
    printf("     %2d     |    %6.2f     |     %+3d\n", n, bias_energy, n - n0);
  }
  
  printf("\nNote: Bias energy is minimized at target count N0 = %d\n", n0);
  printf("      Higher deviations from N0 result in higher bias energy\n");
  printf("      This biases sampling toward the target atom count\n");
}

int main()
{
  printf("CUDA GCMC with Umbrella Sampling Examples\n");
  printf("=========================================\n\n");
  
  // Run all examples
  example_umbrella_sampling_adsorption();
  printf("\n");
  
  example_umbrella_sampling_binary_mixture();
  printf("\n");
  
  example_umbrella_sampling_multiple_windows();
  printf("\n");
  
  example_umbrella_sampling_adaptive_tuning();
  printf("\n");
  
  example_umbrella_bias_energy_calculation();
  printf("\n");
  
  printf("=== Usage Notes ===\n");
  printf("1. Umbrella sampling enhances sampling around target atom counts\n");
  printf("2. Use multiple windows to construct free energy profiles\n");
  printf("3. Adaptive tuning automatically optimizes force constants\n");
  printf("4. Bias energy follows LAMMPS fix_gcmc_umbrella implementation\n");
  printf("5. Integration with CUDA provides GPU acceleration\n");
  
  return 0;
}
