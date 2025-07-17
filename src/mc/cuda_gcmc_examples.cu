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
Example usage and integration guide for CUDA GCMC implementation in GPUMD.
This file demonstrates how to use the new MC_Ensemble_CUDA_GCMC class.
------------------------------------------------------------------------------*/

#include "mc_ensemble_cuda_gcmc.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"

/*
Example 1: Basic CUDA GCMC setup for a binary alloy system
*/
void example_binary_alloy_gcmc()
{
  // Setup parameters for CUDA GCMC
  const char* params[] = {"gcmc", "cuda", "enhanced"};
  int num_params = 3;
  
  // Monte Carlo parameters
  int num_mc_steps = 10000;
  double max_displacement = 0.2; // Angstroms
  double temperature = 300.0;    // Kelvin
  
  // Species configuration
  std::vector<std::string> species = {"Cu", "Ag"};
  std::vector<int> types = {1, 2};
  std::vector<double> chemical_potentials = {-3.5, -2.8}; // eV
  
  // Create CUDA GCMC ensemble
  MC_Ensemble_CUDA_GCMC gcmc_ensemble(
    params, num_params, num_mc_steps,
    species, types, chemical_potentials,
    max_displacement, temperature
  );
  
  // Enable enhanced sampling methods
  std::vector<std::string> enhanced_methods = {"wang_landau", "umbrella_sampling"};
  gcmc_ensemble.setup_enhanced_sampling(enhanced_methods);
  
  // Configure bias potentials for enhanced sampling
  std::vector<double> bias_params = {1.0, 10.0}; // force constants
  gcmc_ensemble.configure_bias_potentials(bias_params);
  
  // Enable crystallization detection
  gcmc_ensemble.enable_crystallization_detection(true);
  
  printf("Binary alloy CUDA GCMC ensemble initialized successfully!\n");
  printf("Species: Cu (type 1, μ=%.2f eV), Ag (type 2, μ=%.2f eV)\n", 
         chemical_potentials[0], chemical_potentials[1]);
}

/*
Example 2: Multi-component system with pressure coupling
*/
void example_multicomponent_gcmc()
{
  const char* params[] = {"gcmc", "cuda", "multicomponent", "pressure_coupled"};
  int num_params = 4;
  
  int num_mc_steps = 50000;
  double max_displacement = 0.15;
  double temperature = 500.0;
  
  // Multi-component system: Cu-Ag-Au
  std::vector<std::string> species = {"Cu", "Ag", "Au"};
  std::vector<int> types = {1, 2, 3};
  std::vector<double> chemical_potentials = {-3.5, -2.8, -3.1};
  
  MC_Ensemble_CUDA_GCMC gcmc_ensemble(
    params, num_params, num_mc_steps,
    species, types, chemical_potentials,
    max_displacement, temperature
  );
  
  // Set external pressure for NPT-like volume moves
  double external_pressure = 1.0; // GPa
  gcmc_ensemble.set_pressure_coupling(external_pressure);
  
  printf("Multi-component CUDA GCMC with pressure coupling initialized!\n");
  printf("External pressure: %.2f GPa\n", external_pressure);
}

/*
Example 3: Integration with GPUMD MD simulation
*/
void example_hybrid_md_gcmc(Atom& atom, Box& box, std::vector<Group>& groups)
{
  const char* params[] = {"gcmc", "cuda", "hybrid_md"};
  int num_params = 3;
  
  int num_mc_steps = 1000; // MC steps per MD timestep
  double max_displacement = 0.1;
  double temperature = 300.0;
  
  std::vector<std::string> species = {"C"};
  std::vector<int> types = {1};
  std::vector<double> chemical_potentials = {-7.5}; // Carbon chemical potential
  
  MC_Ensemble_CUDA_GCMC gcmc_ensemble(
    params, num_params, num_mc_steps,
    species, types, chemical_potentials,
    max_displacement, temperature
  );
  
  // Example MD-MC hybrid simulation loop
  int md_steps = 10000;
  int mc_frequency = 100; // Perform MC every 100 MD steps
  
  for (int step = 0; step < md_steps; ++step) {
    
    // Regular MD timestep would be performed here
    // md_integrator.compute(step, temperature, atom, box);
    
    // Perform MC moves periodically
    if (step % mc_frequency == 0) {
      gcmc_ensemble.compute(step, temperature, atom, box, groups, -1, -1);
      
      // Print statistics
      if (step % (10 * mc_frequency) == 0) {
        gcmc_ensemble.compute_performance_statistics_cuda();
      }
    }
  }
  
  // Save final checkpoint
  gcmc_ensemble.save_checkpoint_cuda("final_state.gcmc");
  
  printf("Hybrid MD-GCMC simulation completed!\n");
}

/*
Example 4: Advanced GCMC with all features enabled
*/
void example_advanced_gcmc_features()
{
  const char* params[] = {"gcmc", "cuda", "advanced", "full_features"};
  int num_params = 4;
  
  int num_mc_steps = 100000;
  double max_displacement = 0.2;
  double temperature = 400.0;
  
  std::vector<std::string> species = {"Fe", "Ni", "Cr"};
  std::vector<int> types = {1, 2, 3};
  std::vector<double> chemical_potentials = {-8.5, -5.2, -9.3};
  
  MC_Ensemble_CUDA_GCMC gcmc_ensemble(
    params, num_params, num_mc_steps,
    species, types, chemical_potentials,
    max_displacement, temperature
  );
  
  // Enable all enhanced sampling methods
  std::vector<std::string> all_methods = {
    "wang_landau", 
    "umbrella_sampling", 
    "parallel_tempering", 
    "bias_potential"
  };
  gcmc_ensemble.setup_enhanced_sampling(all_methods);
  
  // Configure advanced bias potentials
  std::vector<double> advanced_bias_params = {
    2.0,   // Wang-Landau factor
    15.0,  // Umbrella force constant
    1.2,   // Parallel tempering scaling
    0.5    // Bias potential strength
  };
  gcmc_ensemble.configure_bias_potentials(advanced_bias_params);
  
  // Enable crystallization detection and pressure coupling
  gcmc_ensemble.enable_crystallization_detection(true);
  gcmc_ensemble.set_pressure_coupling(0.1); // 0.1 GPa
  
  printf("Advanced CUDA GCMC with all features enabled!\n");
  printf("Enhanced sampling: Wang-Landau, Umbrella, Parallel Tempering, Bias Potential\n");
  printf("Additional features: Crystallization detection, Pressure coupling\n");
}

/*
Example 5: Benchmarking CUDA GCMC performance
*/
void example_performance_benchmark()
{
  const char* params[] = {"gcmc", "cuda", "benchmark"};
  int num_params = 3;
  
  // Large system for performance testing
  int num_mc_steps = 1000000;
  double max_displacement = 0.15;
  double temperature = 350.0;
  
  std::vector<std::string> species = {"Al", "Cu"};
  std::vector<int> types = {1, 2};
  std::vector<double> chemical_potentials = {-3.36, -3.50};
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  MC_Ensemble_CUDA_GCMC gcmc_ensemble(
    params, num_params, num_mc_steps,
    species, types, chemical_potentials,
    max_displacement, temperature
  );
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  printf("CUDA GCMC Performance Benchmark\n");
  printf("Initialization time: %ld ms\n", duration.count());
  printf("Ready for %d MC steps with %lu species\n", num_mc_steps, species.size());
  
  // Additional performance metrics would be collected during actual simulation
  gcmc_ensemble.compute_performance_statistics_cuda();
}

/*
Main function to run examples
*/
int main()
{
  printf("=== CUDA GCMC Examples for GPUMD ===\n\n");
  
  printf("Example 1: Binary Alloy GCMC\n");
  example_binary_alloy_gcmc();
  printf("\n");
  
  printf("Example 2: Multi-component GCMC with Pressure Coupling\n");
  example_multicomponent_gcmc();
  printf("\n");
  
  printf("Example 4: Advanced GCMC with All Features\n");
  example_advanced_gcmc_features();
  printf("\n");
  
  printf("Example 5: Performance Benchmark\n");
  example_performance_benchmark();
  printf("\n");
  
  printf("=== All examples completed successfully! ===\n");
  
  return 0;
}
