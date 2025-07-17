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
CUDA-accelerated Grand Canonical Monte Carlo (GCMC) ensemble for GPUMD.
Implements full GPU acceleration for insertion, deletion, and displacement moves
with advanced sampling techniques.
------------------------------------------------------------------------------*/

#pragma once
#include "mc_ensemble.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>
#include <random>
#include <fstream>
#include <memory>

class NEP;
class Atom;
class Box;
struct Group;

class MC_Ensemble_CUDA_GCMC : public MC_Ensemble
{
public:
  MC_Ensemble_CUDA_GCMC(
    const char** param,
    int num_param,
    int num_steps_mc_input,
    std::vector<std::string>& species_input,
    std::vector<int>& types_input,
    std::vector<double>& mu_input,
    double max_displacement_input,
    double temperature_input);
  
  virtual ~MC_Ensemble_CUDA_GCMC(void);
  
  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& groups,
    int grouping_method,
    int group_id) override;

  // Core GCMC moves
  void attempt_insertion_cuda(
    Atom& atom, Box& box, std::vector<Group>& groups,
    int grouping_method, int group_id, double temperature, int& num_accepted);
    
  void attempt_deletion_cuda(
    Atom& atom, Box& box, std::vector<Group>& groups,
    int grouping_method, int group_id, double temperature, int& num_accepted);
    
  void attempt_displacement_cuda(
    Atom& atom, Box& box, std::vector<Group>& groups,
    int grouping_method, int group_id, double temperature, int& num_accepted);

  // Advanced GCMC methods
  void attempt_volume_change_cuda(Atom& atom, Box& box, double temperature, int& num_accepted);
  void attempt_cluster_moves_cuda(Atom& atom, Box& box, double temperature, int& num_accepted);
  void attempt_identity_change_cuda(Atom& atom, Box& box, double temperature, int& num_accepted);
  
  // Enhanced sampling methods
  void wang_landau_sampling_cuda(Atom& atom, Box& box, double temperature);
  void umbrella_sampling_cuda(Atom& atom, Box& box, double temperature);
  void parallel_tempering_cuda(Atom& atom, Box& box, double temperature, int& num_accepted);
  void bias_potential_cuda(Atom& atom, Box& box, double& energy_bias);
  
  // Energy calculations
  float calculate_system_energy_cuda(Atom& atom, Box& box);
  float calculate_local_energy_cuda(int atom_index, Atom& atom, Box& box);
  void calculate_energy_difference_cuda(
    int atom_index, int old_type, int new_type, 
    Atom& atom, Box& box, float& energy_diff);
  
  // CUDA utility functions
  void generate_insertion_candidates_cuda(
    Atom& atom, Box& box, int num_candidates, 
    std::vector<float3>& positions, std::vector<int>& species);
  bool check_overlap_cuda(
    float x, float y, float z, Atom& atom, Box& box, 
    float min_distance = 0.5f);
  void update_neighbor_lists_cuda(int atom_index, Atom& atom, Box& box);
  
  // System validation and optimization
  bool validate_system_state_cuda(Atom& atom, Box& box);
  void adaptive_parameter_tuning(double acceptance_rate);
  void balance_chemical_potentials();
  void optimize_sampling_window();
  
  // Statistics and output
  void compute_performance_statistics_cuda();
  void save_checkpoint_cuda(const std::string& filename);
  bool load_checkpoint_cuda(const std::string& filename);
  void print_gcmc_statistics();
  
  // Memory management
  void resize_cuda_arrays(int new_size);
  void initialize_cuda_memory(int initial_size);
  void cleanup_cuda_memory();

  // Thermodynamic property calculations
  void calculate_density_profile(Atom& atom, Box& box);
  void calculate_radial_distribution(Atom& atom, Box& box);
  void calculate_order_parameters(Atom& atom, Box& box);
  
  // Advanced features
  void setup_enhanced_sampling(const std::vector<std::string>& methods);
  void configure_bias_potentials(const std::vector<double>& bias_params);
  void enable_crystallization_detection(bool enable = true);
  void set_pressure_coupling(double external_pressure);

protected:
  // GCMC parameters
  std::vector<std::string> species;
  std::vector<int> types;
  std::vector<double> mu;              // Chemical potentials
  std::vector<double> mu_history;      // Chemical potential evolution
  double max_displacement;
  double temperature_mc;
  double external_pressure;
  
  // Sampling parameters
  double insertion_probability;
  double deletion_probability;
  double displacement_probability;
  double volume_change_probability;
  double cluster_move_probability;
  double identity_change_probability;
  
  // System parameters
  int max_atoms;
  int max_insertions_per_step;
  int max_deletions_per_step;
  double min_distance;
  double overlap_cutoff;
  double energy_cutoff;
  
  // Statistics tracking
  int num_insertions_attempted;
  int num_insertions_accepted;
  int num_deletions_attempted;
  int num_deletions_accepted;
  int num_displacements_attempted;
  int num_displacements_accepted;
  int num_volume_changes_attempted;
  int num_volume_changes_accepted;
  int num_cluster_moves_attempted;
  int num_cluster_moves_accepted;
  int num_identity_changes_attempted;
  int num_identity_changes_accepted;
  
  // Enhanced sampling flags and parameters
  bool enable_wang_landau;
  bool enable_umbrella_sampling;
  bool enable_parallel_tempering;
  bool enable_bias_potential;
  bool enable_crystallization_detection;
  bool enable_pressure_coupling;
  
  double wang_landau_factor;
  double umbrella_force_constant;
  std::vector<double> bias_potential_params;
  std::vector<double> order_parameters;
  
  // GPU memory arrays
  GPU_Vector<float> gpu_candidate_positions_x;
  GPU_Vector<float> gpu_candidate_positions_y;
  GPU_Vector<float> gpu_candidate_positions_z;
  GPU_Vector<int> gpu_candidate_types;
  GPU_Vector<float> gpu_candidate_energies;
  GPU_Vector<bool> gpu_overlap_flags;
  GPU_Vector<int> gpu_neighbor_counts;
  GPU_Vector<int> gpu_neighbor_lists;
  GPU_Vector<float> gpu_local_energies;
  GPU_Vector<bool> gpu_active_atoms;
  GPU_Vector<float> gpu_energy_contributions;
  GPU_Vector<float3> gpu_forces_backup;
  GPU_Vector<double> gpu_virial_backup;
  
  // Working arrays
  std::vector<float> cpu_insertion_energies;
  std::vector<bool> cpu_insertion_accepted;
  std::vector<int> cpu_active_indices;
  std::vector<std::array<double, 3>> cpu_position_backup;
  std::vector<int> cpu_type_backup;
  std::vector<double> cpu_mass_backup;
  
  // Performance optimization
  int cuda_blocks_per_grid;
  int cuda_threads_per_block;
  int batch_size_insertion;
  int batch_size_deletion;
  bool use_fast_energy_calculation;
  bool use_neighbor_list_optimization;
  
  // Random number generation
  std::mt19937 rng_cpu;
  std::uniform_real_distribution<double> uniform_dist;
  std::normal_distribution<double> normal_dist;
  
  // Output and logging
  std::ofstream gcmc_output;
  std::ofstream energy_output;
  std::ofstream statistics_output;
  int output_frequency;
  bool verbose_output;
  
  // System state history
  std::vector<double> acceptance_rate_history;
  std::vector<int> atom_count_history;
  std::vector<double> energy_history;
  std::vector<std::vector<int>> species_count_history;
  
  // Advanced analysis
  std::vector<double> density_profile;
  std::vector<double> radial_distribution;
  std::vector<double> order_parameter_history;
  
private:
  // Internal utility functions
  void initialize_parameters();
  void setup_cuda_kernels();
  void allocate_working_arrays(int size);
  void update_statistics();
  void write_output_files(int step);
  bool check_energy_conservation();
  void handle_cuda_errors(const std::string& operation);
  
  // CUDA kernel launch helpers
  void launch_insertion_kernel(int num_candidates, Atom& atom, Box& box);
  void launch_deletion_kernel(int num_candidates, Atom& atom, Box& box);
  void launch_displacement_kernel(int num_atoms, Atom& atom, Box& box);
  void launch_overlap_check_kernel(Atom& atom, Box& box);
  void launch_energy_calculation_kernel(Atom& atom, Box& box);
  
  // Memory management helpers
  void ensure_sufficient_memory(int required_size);
  void optimize_memory_layout();
  void synchronize_cpu_gpu_data(Atom& atom);
};
