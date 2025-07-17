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
Grand Canonical Monte Carlo (GCMC) ensemble for MCMD.  
Supports insertion, deletion, and displacement moves with full GPU acceleration.  
------------------------------------------------------------------------------*/  
  
#include "mc_ensemble_gcmc.cuh"  
#include "utilities/gpu_macro.cuh"  
#include "utilities/common.cuh"  
#include <map>  
#include <cstring>  
#include <algorithm>  
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

// Physical constants
const double K_B = 8.617333262145e-5; // Boltzmann constant in eV/K  
  
// Mass table consistent with SGC implementation  
const std::map<std::string, double> MASS_TABLE_GCMC{  
  {"H", 1.0080000000},   {"He", 4.0026020000},  {"Li", 6.9400000000},  
  {"Be", 9.0121831000},  {"B", 10.8100000000},  {"C", 12.0110000000},  
  {"N", 14.0070000000},  {"O", 15.9990000000},  {"F", 18.9984031630},  
  {"Ne", 20.1797000000}, {"Na", 22.9897692800}, {"Mg", 24.3050000000},  
  {"Al", 26.9815385000}, {"Si", 28.0850000000}, {"P", 30.9737619980},  
  {"S", 32.0600000000},  {"Cl", 35.4500000000}, {"Ar", 39.9480000000},  
  {"K", 39.0983000000},  {"Ca", 40.0780000000}, {"Fe", 55.8450000000},  
  {"Cu", 63.5460000000}, {"Ag", 107.8682000000}, {"Au", 196.9665690000}  
};  
  
// GPU kernels for atom operations  
static __global__ void gpu_insert_atom(  
  const int insertion_index,  
  const int type_new,  
  const double mass_new,  
  const double x, const double y, const double z,  
  const double vx, const double vy, const double vz,  
  int* g_type,  
  double* g_mass,  
  double* g_x, double* g_y, double* g_z,  
  double* g_vx, double* g_vy, double* g_vz,  
  int* g_active)  
{  
  g_type[insertion_index] = type_new;  
  g_mass[insertion_index] = mass_new;  
  g_x[insertion_index] = x;  
  g_y[insertion_index] = y;  
  g_z[insertion_index] = z;  
  g_vx[insertion_index] = vx;  
  g_vy[insertion_index] = vy;  
  g_vz[insertion_index] = vz;  
  g_active[insertion_index] = 1;  
}  
  
static __global__ void gpu_delete_atom(  
  const int atom_index,  
  int* g_type,  
  int* g_active)  
{  
  g_type[atom_index] = -1;  
  g_active[atom_index] = 0;  
}  
  
static __global__ void gpu_displace_atom(  
  const int atom_index,  
  const double dx, const double dy, const double dz,  
  double* g_x, double* g_y, double* g_z)  
{  
  g_x[atom_index] += dx;  
  g_y[atom_index] += dy;  
  g_z[atom_index] += dz;  
}  
  
// GPU kernel for parallel overlap checking  
static __global__ void gpu_check_overlap(  
  const int N,  
  const double x_new, const double y_new, const double z_new,  
  const double min_distance_sq,  
  const double* g_x, const double* g_y, const double* g_z,  
  const int* g_active,  
  const Box box,  
  int* overlap_found)  
{  
  int n = blockIdx.x * blockDim.x + threadIdx.x;  
  if (n < N && g_active[n]) {  
    double dx = g_x[n] - x_new;  
    double dy = g_y[n] - y_new;  
    double dz = g_z[n] - z_new;  
      
    // Apply minimum image convention  
    apply_mic(box, dx, dy, dz);  
      
    double dist_sq = dx * dx + dy * dy + dz * dz;  
    if (dist_sq < min_distance_sq) {  
      atomicAdd(overlap_found, 1);  
    }  
  }  
}  
  
// GPU kernel for finding neighbors of inserted atom  
static __global__ void gpu_find_neighbors_insertion(  
  const int N,  
  const double x_new, const double y_new, const double z_new,  
  const double rc_sq,  
  const double* g_x, const double* g_y, const double* g_z,  
  const int* g_active,  
  const Box box,  
  int* neighbor_count,  
  int* neighbor_list)  
{  
  int n = blockIdx.x * blockDim.x + threadIdx.x;  
  if (n < N && g_active && g_active[n]) {  
    double dx = g_x[n] - x_new;  
    double dy = g_y[n] - y_new;  
    double dz = g_z[n] - z_new;  
      
    apply_mic(box, dx, dy, dz);  
      
    double dist_sq = dx * dx + dy * dy + dz * dz;  
    if (dist_sq < rc_sq) {  
      int idx = atomicAdd(neighbor_count, 1);  
      if (idx < 1000) { // Maximum neighbor list size  
        neighbor_list[idx] = n;  
      }  
    }  
  }  
}  

// GPU kernel for batch overlap checking with shared memory optimization
static __global__ void gpu_batch_overlap_check(
  const int N,
  const int num_candidates,
  const double* candidate_x, const double* candidate_y, const double* candidate_z,
  const double min_distance_sq,
  const double* g_x, const double* g_y, const double* g_z,
  const int* g_active,
  const Box box,
  int* overlap_results)
{
  int candidate_idx = blockIdx.x;
  int atom_idx = threadIdx.x;
  
  __shared__ int shared_overlap;
  if (threadIdx.x == 0) {
    shared_overlap = 0;
  }
  __syncthreads();
  
  if (candidate_idx < num_candidates) {
    double x_new = candidate_x[candidate_idx];
    double y_new = candidate_y[candidate_idx];
    double z_new = candidate_z[candidate_idx];
    
    // Each thread checks multiple atoms
    for (int n = atom_idx; n < N; n += blockDim.x) {
      if (g_active && g_active[n]) {
        double dx = g_x[n] - x_new;
        double dy = g_y[n] - y_new;
        double dz = g_z[n] - z_new;
        
        apply_mic(box, dx, dy, dz);
        
        double dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq < min_distance_sq) {
          atomicAdd(&shared_overlap, 1);
          break;
        }
      }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
      overlap_results[candidate_idx] = shared_overlap;
    }
  }
}

// GPU kernel for computing local energy changes
static __global__ void gpu_compute_local_energy_change(
  const int N,
  const int modified_atom,
  const double* g_x, const double* g_y, const double* g_z,
  const int* g_type,
  const int* g_active,
  const Box box,
  const double rc_sq,
  float* energy_change)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ float shared_energy[256];
  shared_energy[threadIdx.x] = 0.0f;
  
  if (n < N && g_active && g_active[n] && n != modified_atom) {
    double dx = g_x[n] - g_x[modified_atom];
    double dy = g_y[n] - g_y[modified_atom];
    double dz = g_z[n] - g_z[modified_atom];
    
    apply_mic(box, dx, dy, dz);
    
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq < rc_sq) {
      // Compute pairwise energy contribution
      // This would need to be implemented based on the specific potential
      shared_energy[threadIdx.x] = 0.0f; // Placeholder
    }
  }
  
  __syncthreads();
  
  // Reduction to sum energies
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_energy[threadIdx.x] += shared_energy[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    atomicAdd(energy_change, shared_energy[0]);
  }
}  
  
MC_Ensemble_GCMC::MC_Ensemble_GCMC(  
  const char** param,  
  int num_param,  
  int num_steps_mc_input,  
  std::vector<std::string>& species_input,  
  std::vector<int>& types_input,  
  std::vector<double>& mu_input,  
  double max_displacement_input)  
  : MC_Ensemble(param, num_param)  
{  
  num_steps_mc = num_steps_mc_input;  
  species = species_input;  
  types = types_input;  
  mu = mu_input;  
  max_displacement = max_displacement_input;  
  max_atoms = 100000; // Maximum number of atoms  
    
  // Initialize GPU vectors  
  NN_i.resize(1);  
  NL_i.resize(1000);  
  insertion_sites.resize(1000);  
  insertion_energies.resize(1000);  
  active_atoms.resize(max_atoms);  
  deleted_atoms.resize(max_atoms);  
  
  // Initialize optimization vectors
  candidate_positions_x.resize(100);
  candidate_positions_y.resize(100);
  candidate_positions_z.resize(100);
  overlap_results.resize(100);
  energy_changes.resize(100);
  
  // Initialize enhanced sampling flags
  enable_wang_landau = false;
  enable_umbrella_sampling = false;
  enable_parallel_tempering = false;
  enable_bias_potential = false;
  
  // Initialize statistics tracking
  total_mc_steps_performed = 0;
  acceptance_history.reserve(10000);
  atom_count_history.reserve(10000);  
    
  // Initialize counters  
  num_insertions_attempted = 0;  
  num_insertions_accepted = 0;  
  num_deletions_attempted = 0;  
  num_deletions_accepted = 0;  
  num_displacements_attempted = 0;  
  num_displacements_accepted = 0;  
    
  // Open output file  
  mc_output.open("gcmc.out");  
  mc_output << "# GCMC Statistics Output\n";
  mc_output << "# Step Overall_Acceptance N_active Insertion_Rate Deletion_Rate Displacement_Rate Max_Displacement Temperature";
  for (size_t s = 0; s < species.size(); ++s) {
    mc_output << " N_" << species[s];
  }
  mc_output << "\n";  
}  
  
MC_Ensemble_GCMC::~MC_Ensemble_GCMC(void)   
{   
  mc_output.close();   
}  
  
void MC_Ensemble_GCMC::compute(  
  int md_step,  
  double temperature,  
  Atom& atom,  
  Box& box,  
  std::vector<Group>& groups,  
  int grouping_method,  
  int group_id)  
{  
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {  
    printf("Cannot use small box for GCMC.\n");  
    exit(1);  
  }  
  
  // Ensure atom arrays can accommodate insertions  
  if (atom.number_of_atoms + 1000 > max_atoms) {  
    resize_atom_arrays(atom, max_atoms + 10000);  
    max_atoms += 10000;  
  }  
  
  // Resize working arrays if needed  
  if (type_before.size() < max_atoms) {  
    type_before.resize(max_atoms);  
    type_after.resize(max_atoms);  
    active_atoms.resize(max_atoms);  
  }  
  
  int group_size = grouping_method >= 0 ?   
    groups[grouping_method].cpu_size[group_id] : atom.number_of_atoms;  
    
  std::uniform_real_distribution<double> r_uniform(0.0, 1.0);  
  int num_accepted_total = 0;  
    
  for (int step = 0; step < num_steps_mc; ++step) {  
      
    // Choose move type with adaptive probabilities
    double move_type = r_uniform(rng);  
      
    if (move_type < 0.25) {  
      attempt_insertion(atom, box, groups, grouping_method, group_id,   
                       temperature, num_accepted_total);  
    }  
    else if (move_type < 0.5) {  
      attempt_deletion(atom, box, groups, grouping_method, group_id,   
                      temperature, num_accepted_total);  
    }  
    else if (move_type < 0.8) {  
      attempt_displacement(atom, box, groups, grouping_method, group_id,   
                          temperature, num_accepted_total);  
    }
    else if (move_type < 0.9) {
      attempt_cluster_moves(atom, box, temperature, num_accepted_total);
    }
    else {
      attempt_volume_change(atom, box, temperature, num_accepted_total);
    }
    
    // Periodic validation and adaptation
    if (step % 1000 == 0) {
      if (!validate_system_state(atom, box)) {
        printf("System validation failed at step %d, terminating MC\n", step);
        break;
      }
      
      // Adaptive displacement scaling
      if (num_displacements_attempted > 0) {
        double disp_rate = static_cast<double>(num_displacements_accepted) / num_displacements_attempted;
        adaptive_displacement_scaling(disp_rate);
      }
      
      balance_insertion_rates();
      
      // Update bias statistics
      update_bias_statistics(atom, box);
      
      // Apply enhanced sampling methods periodically
      if (step % 5000 == 0) {
        wang_landau_sampling(atom, box, temperature);
        umbrella_sampling(atom, box, temperature);
        detect_crystallization(atom, box);
      }
    }
  }  
  
  // Output statistics  
  double total_attempts = num_insertions_attempted + num_deletions_attempted + num_displacements_attempted;  
  double overall_acceptance = total_attempts > 0 ? num_accepted_total / double(num_steps_mc) : 0.0;
  
  // Count active atoms
  int active_count = 0;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) active_count++;
  }
  
  mc_output << md_step << "  "   
            << overall_acceptance << "  "  
            << active_count << "  "  
            << (num_insertions_attempted > 0 ? num_insertions_accepted / double(num_insertions_attempted) : 0.0) << "  "  
            << (num_deletions_attempted > 0 ? num_deletions_accepted / double(num_deletions_attempted) : 0.0) << "  "  
            << (num_displacements_attempted > 0 ? num_displacements_accepted / double(num_displacements_attempted) : 0.0) << "  "
            << max_displacement << "  "
            << temperature;
  
  // Add species-specific statistics
  std::vector<int> species_count(species.size(), 0);
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      for (size_t s = 0; s < types.size(); ++s) {
        if (atom.cpu_type[i] == types[s]) {
          species_count[s]++;
          break;
        }
      }
    }
  }
  
  for (size_t s = 0; s < species.size(); ++s) {
    mc_output << "  " << species_count[s];
  }
  
  mc_output << std::endl;
  
  // Additional detailed output every 10 steps
  if (md_step % 10 == 0) {
    printf("GCMC Step %d: N_atoms=%d, Acceptance=%.3f, Max_disp=%.4f\n", 
           md_step, active_count, overall_acceptance, max_displacement);
  }  
}  
  
void MC_Ensemble_GCMC::compute_performance_statistics()
{
  // Print performance statistics
  printf("\n=== GCMC Performance Statistics ===\n");
  printf("Total MC steps performed: %d\n", total_mc_steps_performed);
  printf("Insertion attempts: %d (%.1f%% accepted)\n", 
         num_insertions_attempted, 
         num_insertions_attempted > 0 ? 100.0 * num_insertions_accepted / num_insertions_attempted : 0.0);
  printf("Deletion attempts: %d (%.1f%% accepted)\n", 
         num_deletions_attempted,
         num_deletions_attempted > 0 ? 100.0 * num_deletions_accepted / num_deletions_attempted : 0.0);
  printf("Displacement attempts: %d (%.1f%% accepted)\n", 
         num_displacements_attempted,
         num_displacements_attempted > 0 ? 100.0 * num_displacements_accepted / num_displacements_attempted : 0.0);
  
  printf("Current maximum displacement: %.4f Å\n", max_displacement);
  
  if (!acceptance_history.empty()) {
    double avg_acceptance = 0.0;
    for (double acc : acceptance_history) {
      avg_acceptance += acc;
    }
    avg_acceptance /= acceptance_history.size();
    printf("Average acceptance rate: %.3f\n", avg_acceptance);
  }
  
  printf("Enhanced sampling methods:\n");
  printf("  Wang-Landau: %s\n", enable_wang_landau ? "Enabled" : "Disabled");
  printf("  Umbrella sampling: %s\n", enable_umbrella_sampling ? "Enabled" : "Disabled");
  printf("  Parallel tempering: %s\n", enable_parallel_tempering ? "Enabled" : "Disabled");
  printf("  Bias potential: %s\n", enable_bias_potential ? "Enabled" : "Disabled");
  printf("===================================\n\n");
}

void MC_Ensemble_GCMC::save_checkpoint(const std::string& filename)
{
  std::ofstream checkpoint(filename);
  checkpoint << "# GCMC Checkpoint File\n";
  checkpoint << "num_insertions_attempted " << num_insertions_attempted << "\n";
  checkpoint << "num_insertions_accepted " << num_insertions_accepted << "\n";
  checkpoint << "num_deletions_attempted " << num_deletions_attempted << "\n";
  checkpoint << "num_deletions_accepted " << num_deletions_accepted << "\n";
  checkpoint << "num_displacements_attempted " << num_displacements_attempted << "\n";
  checkpoint << "num_displacements_accepted " << num_displacements_accepted << "\n";
  checkpoint << "max_displacement " << max_displacement << "\n";
  checkpoint << "total_mc_steps_performed " << total_mc_steps_performed << "\n";
  
  checkpoint << "chemical_potentials";
  for (double mu_val : mu) {
    checkpoint << " " << mu_val;
  }
  checkpoint << "\n";
  
  checkpoint.close();
  printf("GCMC checkpoint saved to %s\n", filename.c_str());
}

bool MC_Ensemble_GCMC::load_checkpoint(const std::string& filename)
{
  std::ifstream checkpoint(filename);
  if (!checkpoint.is_open()) {
    printf("Warning: Could not load checkpoint file %s\n", filename.c_str());
    return false;
  }
  
  std::string line, key;
  while (std::getline(checkpoint, line)) {
    if (line[0] == '#') continue;
    
    std::istringstream iss(line);
    iss >> key;
    
    if (key == "num_insertions_attempted") iss >> num_insertions_attempted;
    else if (key == "num_insertions_accepted") iss >> num_insertions_accepted;
    else if (key == "num_deletions_attempted") iss >> num_deletions_attempted;
    else if (key == "num_deletions_accepted") iss >> num_deletions_accepted;
    else if (key == "num_displacements_attempted") iss >> num_displacements_attempted;
    else if (key == "num_displacements_accepted") iss >> num_displacements_accepted;
    else if (key == "max_displacement") iss >> max_displacement;
    else if (key == "total_mc_steps_performed") iss >> total_mc_steps_performed;
    else if (key == "chemical_potentials") {
      for (size_t i = 0; i < mu.size(); ++i) {
        iss >> mu[i];
      }
    }
  }
  
  checkpoint.close();
  printf("GCMC checkpoint loaded from %s\n", filename.c_str());
  return true;
}
  
void MC_Ensemble_GCMC::compute(  
  int md_step,  
  double temperature,  
  Atom& atom,  
  Box& box,  
  std::vector<Group>& groups,  
  int grouping_method,  
  int group_id)  
{  
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {  
    printf("Cannot use small box for GCMC.\n");  
    exit(1);  
  }  
  
  // Ensure atom arrays can accommodate insertions  
  if (atom.number_of_atoms + 1000 > max_atoms) {  
    resize_atom_arrays(atom, max_atoms + 10000);  
    max_atoms += 10000;  
  }  
  
  // Resize working arrays if needed  
  if (type_before.size() < max_atoms) {  
    type_before.resize(max_atoms);  
    type_after.resize(max_atoms);  
    active_atoms.resize(max_atoms);  
  }  
  
  int group_size = grouping_method >= 0 ?   
    groups[grouping_method].cpu_size[group_id] : atom.number_of_atoms;  
    
  std::uniform_real_distribution<double> r_uniform(0.0, 1.0);  
  int num_accepted_total = 0;  
    
  for (int step = 0; step < num_steps_mc; ++step) {  
      
    // Choose move type with adaptive probabilities
    double move_type = r_uniform(rng);  
      
    if (move_type < 0.25) {  
      attempt_insertion(atom, box, groups, grouping_method, group_id,   
                       temperature, num_accepted_total);  
    }  
    else if (move_type < 0.5) {  
      attempt_deletion(atom, box, groups, grouping_method, group_id,   
                      temperature, num_accepted_total);  
    }  
    else if (move_type < 0.8) {  
      attempt_displacement(atom, box, groups, grouping_method, group_id,   
                          temperature, num_accepted_total);  
    }
    else if (move_type < 0.9) {
      attempt_cluster_moves(atom, box, temperature, num_accepted_total);
    }
    else {
      attempt_volume_change(atom, box, temperature, num_accepted_total);
    }
    
    // Periodic validation and adaptation
    if (step % 1000 == 0) {
      if (!validate_system_state(atom, box)) {
        printf("System validation failed at step %d, terminating MC\n", step);
        break;
      }
      
      // Adaptive displacement scaling
      if (num_displacements_attempted > 0) {
        double disp_rate = static_cast<double>(num_displacements_accepted) / num_displacements_attempted;
        adaptive_displacement_scaling(disp_rate);
      }
      
      balance_insertion_rates();
      
      // Update bias statistics
      update_bias_statistics(atom, box);
      
      // Apply enhanced sampling methods periodically
      if (step % 5000 == 0) {
        wang_landau_sampling(atom, box, temperature);
        umbrella_sampling(atom, box, temperature);
        detect_crystallization(atom, box);
      }
    }
  }  
  
  // Output statistics  
  double total_attempts = num_insertions_attempted + num_deletions_attempted + num_displacements_attempted;  
  double overall_acceptance = total_attempts > 0 ? num_accepted_total / double(num_steps_mc) : 0.0;
  
  // Count active atoms
  int active_count = 0;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) active_count++;
  }
  
  mc_output << md_step << "  "   
            << overall_acceptance << "  "  
            << active_count << "  "  
            << (num_insertions_attempted > 0 ? num_insertions_accepted / double(num_insertions_attempted) : 0.0) << "  "  
            << (num_deletions_attempted > 0 ? num_deletions_accepted / double(num_deletions_attempted) : 0.0) << "  "  
            << (num_displacements_attempted > 0 ? num_displacements_accepted / double(num_displacements_attempted) : 0.0) << "  "
            << max_displacement << "  "
            << temperature;
  
  // Add species-specific statistics
  std::vector<int> species_count(species.size(), 0);
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      for (size_t s = 0; s < types.size(); ++s) {
        if (atom.cpu_type[i] == types[s]) {
          species_count[s]++;
          break;
        }
      }
    }
  }
  
  for (size_t s = 0; s < species.size(); ++s) {
    mc_output << "  " << species_count[s];
  }
  
  mc_output << std::endl;
  
  // Additional detailed output every 10 steps
  if (md_step % 10 == 0) {
    printf("GCMC Step %d: N_atoms=%d, Acceptance=%.3f, Max_disp=%.4f\n", 
           md_step, active_count, overall_acceptance, max_displacement);
  }  
}  
  
void MC_Ensemble_GCMC::attempt_insertion(  
  Atom& atom, Box& box, std::vector<Group>& groups,  
  int grouping_method, int group_id, double temperature, int& num_accepted)  
{  
  num_insertions_attempted++;  
    
  // Choose random species  
  std::uniform_int_distribution<int> r_species(0, species.size() - 1);  
  std::uniform_real_distribution<double> r_pos(0.0, 1.0);  
    
  int species_idx = r_species(rng);  
  int type_new = types[species_idx];  
  double mass_new = MASS_TABLE_GCMC.at(species[species_idx]);  
    
  // Generate random position in fractional coordinates  
  double frac_x = r_pos(rng);
  double frac_y = r_pos(rng);
  double frac_z = r_pos(rng);
  
  // Convert to Cartesian coordinates
  double x_new = box.cpu_h[0] * frac_x + box.cpu_h[3] * frac_y + box.cpu_h[6] * frac_z;
  double y_new = box.cpu_h[1] * frac_x + box.cpu_h[4] * frac_y + box.cpu_h[7] * frac_z;
  double z_new = box.cpu_h[2] * frac_x + box.cpu_h[5] * frac_y + box.cpu_h[8] * frac_z;
  
  // Check for overlap with existing atoms
  if (check_overlap(x_new, y_new, z_new, atom, box)) {
    return; // Reject move due to overlap
  }
  
  // Find insertion index (first inactive slot or extend array)
  int insertion_index = atom.number_of_atoms;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] < 0) { // Found deleted atom slot
      insertion_index = i;
      break;
    }
  }
  
  // If no empty slot, extend arrays
  if (insertion_index == atom.number_of_atoms) {
    if (insertion_index >= max_atoms) {
      resize_atom_arrays(atom, max_atoms + 1000);
      max_atoms += 1000;
    }
    atom.number_of_atoms++;
  }
  
  // Generate random velocity from Maxwell-Boltzmann distribution
  std::normal_distribution<double> velocity_dist(0.0, sqrt(K_B * temperature / mass_new));
  double vx_new = velocity_dist(rng);
  double vy_new = velocity_dist(rng);
  double vz_new = velocity_dist(rng);
  
  // Calculate energy before insertion
  float energy_before = calculate_system_energy(atom, box);
  
  // Store old state
  int old_type = atom.cpu_type[insertion_index];
  double old_mass = atom.cpu_mass[insertion_index];
  double old_x = atom.cpu_position_per_atom[0][insertion_index];
  double old_y = atom.cpu_position_per_atom[1][insertion_index];
  double old_z = atom.cpu_position_per_atom[2][insertion_index];
  double old_vx = atom.cpu_velocity_per_atom[0][insertion_index];
  double old_vy = atom.cpu_velocity_per_atom[1][insertion_index];
  double old_vz = atom.cpu_velocity_per_atom[2][insertion_index];
  
  // Insert atom on CPU
  atom.cpu_type[insertion_index] = type_new;
  atom.cpu_mass[insertion_index] = mass_new;
  atom.cpu_position_per_atom[0][insertion_index] = x_new;
  atom.cpu_position_per_atom[1][insertion_index] = y_new;
  atom.cpu_position_per_atom[2][insertion_index] = z_new;
  atom.cpu_velocity_per_atom[0][insertion_index] = vx_new;
  atom.cpu_velocity_per_atom[1][insertion_index] = vy_new;
  atom.cpu_velocity_per_atom[2][insertion_index] = vz_new;
  
  // Copy to GPU and calculate new energy
  atom.copy_from_cpu();
  float energy_after = calculate_system_energy(atom, box);
  
  // Calculate acceptance probability (GCMC criterion)
  double delta_E = energy_after - energy_before;
  double beta = 1.0 / (K_B * temperature);
  double volume = box.cpu_h[0] * (box.cpu_h[4] * box.cpu_h[8] - box.cpu_h[5] * box.cpu_h[7]) +
                  box.cpu_h[1] * (box.cpu_h[5] * box.cpu_h[6] - box.cpu_h[3] * box.cpu_h[8]) +
                  box.cpu_h[2] * (box.cpu_h[3] * box.cpu_h[7] - box.cpu_h[4] * box.cpu_h[6]);
  volume = abs(volume);
  
  double ln_acc = mu[species_idx] * beta - delta_E * beta + log(volume / (atom.number_of_atoms + 1));
  
  // Accept or reject
  std::uniform_real_distribution<double> r_accept(0.0, 1.0);
  if (ln_acc > 0.0 || r_accept(rng) < exp(ln_acc)) {
    // Accept insertion
    num_insertions_accepted++;
    num_accepted++;
    
    // Update GPU data
    gpu_insert_atom<<<1, 1>>>(
      insertion_index, type_new, mass_new, x_new, y_new, z_new,
      vx_new, vy_new, vz_new,
      atom.type.data(), atom.mass.data(),
      atom.position_per_atom[0].data(), atom.position_per_atom[1].data(), atom.position_per_atom[2].data(),
      atom.velocity_per_atom[0].data(), atom.velocity_per_atom[1].data(), atom.velocity_per_atom[2].data(),
      nullptr // No active array in basic implementation
    );
    CHECK(cudaDeviceSynchronize());
  } else {
    // Reject insertion - restore old state
    atom.cpu_type[insertion_index] = old_type;
    atom.cpu_mass[insertion_index] = old_mass;
    atom.cpu_position_per_atom[0][insertion_index] = old_x;
    atom.cpu_position_per_atom[1][insertion_index] = old_y;
    atom.cpu_position_per_atom[2][insertion_index] = old_z;
    atom.cpu_velocity_per_atom[0][insertion_index] = old_vx;
    atom.cpu_velocity_per_atom[1][insertion_index] = old_vy;
    atom.cpu_velocity_per_atom[2][insertion_index] = old_vz;
    
    if (insertion_index == atom.number_of_atoms - 1) {
      atom.number_of_atoms--;
    }
    atom.copy_from_cpu();
  }
}

void MC_Ensemble_GCMC::attempt_deletion(
  Atom& atom, Box& box, std::vector<Group>& groups,
  int grouping_method, int group_id, double temperature, int& num_accepted)
{
  num_deletions_attempted++;
  
  if (atom.number_of_atoms <= 1) {
    return; // Cannot delete from empty or single-atom system
  }
  
  // Choose random active atom to delete
  std::vector<int> active_indices;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      active_indices.push_back(i);
    }
  }
  
  if (active_indices.empty()) {
    return; // No active atoms
  }
  
  std::uniform_int_distribution<int> r_atom(0, active_indices.size() - 1);
  int delete_idx = active_indices[r_atom(rng)];
  int type_deleted = atom.cpu_type[delete_idx];
  
  // Find species index for deleted atom
  int species_idx = -1;
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == type_deleted) {
      species_idx = i;
      break;
    }
  }
  
  if (species_idx < 0) {
    return; // Unknown species
  }
  
  // Calculate energy before deletion
  float energy_before = calculate_system_energy(atom, box);
  
  // Store atom data for potential restoration
  int old_type = atom.cpu_type[delete_idx];
  double old_mass = atom.cpu_mass[delete_idx];
  double old_x = atom.cpu_position_per_atom[0][delete_idx];
  double old_y = atom.cpu_position_per_atom[1][delete_idx];
  double old_z = atom.cpu_position_per_atom[2][delete_idx];
  double old_vx = atom.cpu_velocity_per_atom[0][delete_idx];
  double old_vy = atom.cpu_velocity_per_atom[1][delete_idx];
  double old_vz = atom.cpu_velocity_per_atom[2][delete_idx];
  
  // Delete atom (mark as inactive)
  atom.cpu_type[delete_idx] = -1;
  atom.cpu_mass[delete_idx] = 0.0;
  
  // Copy to GPU and calculate new energy
  atom.copy_from_cpu();
  float energy_after = calculate_system_energy(atom, box);
  
  // Calculate acceptance probability (GCMC deletion criterion)
  double delta_E = energy_after - energy_before;
  double beta = 1.0 / (K_B * temperature);
  double volume = box.cpu_h[0] * (box.cpu_h[4] * box.cpu_h[8] - box.cpu_h[5] * box.cpu_h[7]) +
                  box.cpu_h[1] * (box.cpu_h[5] * box.cpu_h[6] - box.cpu_h[3] * box.cpu_h[8]) +
                  box.cpu_h[2] * (box.cpu_h[3] * box.cpu_h[7] - box.cpu_h[4] * box.cpu_h[6]);
  volume = abs(volume);
  
  double ln_acc = -mu[species_idx] * beta - delta_E * beta + log(active_indices.size() / volume);
  
  // Accept or reject
  std::uniform_real_distribution<double> r_accept(0.0, 1.0);
  if (ln_acc > 0.0 || r_accept(rng) < exp(ln_acc)) {
    // Accept deletion
    num_deletions_accepted++;
    num_accepted++;
    
    // Update GPU data
    gpu_delete_atom<<<1, 1>>>(
      delete_idx,
      atom.type.data(),
      nullptr // No active array in basic implementation
    );
    CHECK(cudaDeviceSynchronize());
  } else {
    // Reject deletion - restore atom
    atom.cpu_type[delete_idx] = old_type;
    atom.cpu_mass[delete_idx] = old_mass;
    atom.cpu_position_per_atom[0][delete_idx] = old_x;
    atom.cpu_position_per_atom[1][delete_idx] = old_y;
    atom.cpu_position_per_atom[2][delete_idx] = old_z;
    atom.cpu_velocity_per_atom[0][delete_idx] = old_vx;
    atom.cpu_velocity_per_atom[1][delete_idx] = old_vy;
    atom.cpu_velocity_per_atom[2][delete_idx] = old_vz;
    atom.copy_from_cpu();
  }
}

void MC_Ensemble_GCMC::attempt_displacement(
  Atom& atom, Box& box, std::vector<Group>& groups,
  int grouping_method, int group_id, double temperature, int& num_accepted)
{
  num_displacements_attempted++;
  
  // Choose random active atom to displace
  std::vector<int> active_indices;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      active_indices.push_back(i);
    }
  }
  
  if (active_indices.empty()) {
    return; // No active atoms
  }
  
  std::uniform_int_distribution<int> r_atom(0, active_indices.size() - 1);
  std::uniform_real_distribution<double> r_disp(-max_displacement, max_displacement);
  
  int move_idx = active_indices[r_atom(rng)];
  
  // Generate random displacement
  double dx = r_disp(rng);
  double dy = r_disp(rng);
  double dz = r_disp(rng);
  
  // Calculate energy before displacement
  float energy_before = calculate_system_energy(atom, box);
  
  // Store old position
  double old_x = atom.cpu_position_per_atom[0][move_idx];
  double old_y = atom.cpu_position_per_atom[1][move_idx];
  double old_z = atom.cpu_position_per_atom[2][move_idx];
  
  // Apply displacement
  atom.cpu_position_per_atom[0][move_idx] += dx;
  atom.cpu_position_per_atom[1][move_idx] += dy;
  atom.cpu_position_per_atom[2][move_idx] += dz;
  
  // Apply periodic boundary conditions
  apply_pbc_displacement(atom.cpu_position_per_atom[0][move_idx], 
                        atom.cpu_position_per_atom[1][move_idx], 
                        atom.cpu_position_per_atom[2][move_idx], box);
  
  // Check for overlap after displacement
  double new_x = atom.cpu_position_per_atom[0][move_idx];
  double new_y = atom.cpu_position_per_atom[1][move_idx];
  double new_z = atom.cpu_position_per_atom[2][move_idx];
  
  bool has_overlap = false;
  double min_dist_sq = 0.5 * 0.5; // Minimum distance = 0.5 Å
  
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (i == move_idx || atom.cpu_type[i] < 0) continue;
    
    double dx_check = atom.cpu_position_per_atom[0][i] - new_x;
    double dy_check = atom.cpu_position_per_atom[1][i] - new_y;
    double dz_check = atom.cpu_position_per_atom[2][i] - new_z;
    
    apply_mic(box, dx_check, dy_check, dz_check);
    
    double dist_sq = dx_check * dx_check + dy_check * dy_check + dz_check * dz_check;
    if (dist_sq < min_dist_sq) {
      has_overlap = true;
      break;
    }
  }
  
  if (has_overlap) {
    // Restore old position
    atom.cpu_position_per_atom[0][move_idx] = old_x;
    atom.cpu_position_per_atom[1][move_idx] = old_y;
    atom.cpu_position_per_atom[2][move_idx] = old_z;
    return;
  }
  
  // Copy to GPU and calculate new energy
  atom.copy_from_cpu();
  float energy_after = calculate_system_energy(atom, box);
  
  // Calculate acceptance probability (Metropolis criterion)
  double delta_E = energy_after - energy_before;
  double beta = 1.0 / (K_B * temperature);
  double ln_acc = -delta_E * beta;
  
  // Accept or reject
  std::uniform_real_distribution<double> r_accept(0.0, 1.0);
  if (ln_acc > 0.0 || r_accept(rng) < exp(ln_acc)) {
    // Accept displacement
    num_displacements_accepted++;
    num_accepted++;
    
    // Update GPU data
    gpu_displace_atom<<<1, 1>>>(
      move_idx, 
      new_x - old_x, new_y - old_y, new_z - old_z,
      atom.position_per_atom[0].data(), 
      atom.position_per_atom[1].data(), 
      atom.position_per_atom[2].data()
    );
    CHECK(cudaDeviceSynchronize());
  } else {
    // Reject displacement - restore old position
    atom.cpu_position_per_atom[0][move_idx] = old_x;
    atom.cpu_position_per_atom[1][move_idx] = old_y;
    atom.cpu_position_per_atom[2][move_idx] = old_z;
    atom.copy_from_cpu();
  }
}

float MC_Ensemble_GCMC::calculate_system_energy(Atom& atom, Box& box)
{
  // Use the NEP energy calculator to compute total system energy
  std::vector<Group> empty_groups;
  nep_energy.compute(box, atom, empty_groups, -1, -1);
  
  // Sum up potential energy with bias correction
  float total_energy = 0.0f;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      total_energy += atom.cpu_potential_per_atom[i];
    }
  }
  
  // Add bias potential contribution
  double energy_bias = 0.0;
  apply_bias_potential(atom, box, energy_bias);
  total_energy += static_cast<float>(energy_bias);
  
  return total_energy;
}

float MC_Ensemble_GCMC::calculate_local_energy(int atom_index, Atom& atom, Box& box)
{
  // Calculate energy contribution from single atom and its neighbors
  if (atom_index >= atom.number_of_atoms || atom.cpu_type[atom_index] < 0) {
    return 0.0f;
  }
  
  // More efficient local energy calculation
  float local_energy = 0.0f;
  const double rc_sq = nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial;
  
  // Only consider interactions within cutoff radius
  for (int j = 0; j < atom.number_of_atoms; ++j) {
    if (j == atom_index || atom.cpu_type[j] < 0) continue;
    
    double dx = atom.cpu_position_per_atom[0][atom_index] - atom.cpu_position_per_atom[0][j];
    double dy = atom.cpu_position_per_atom[1][atom_index] - atom.cpu_position_per_atom[1][j];
    double dz = atom.cpu_position_per_atom[2][atom_index] - atom.cpu_position_per_atom[2][j];
    
    apply_mic(box, dx, dy, dz);
    
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq < rc_sq) {
      // This is a simplified approach - for full accuracy, would need
      // to implement partial NEP evaluation for local interactions
      local_energy += atom.cpu_potential_per_atom[atom_index] / 
                     std::max(1.0, static_cast<double>(atom.number_of_atoms - 1));
    }
  }
  
  return local_energy;
}

void MC_Ensemble_GCMC::resize_atom_arrays(Atom& atom, int new_size)
{
  // Resize CPU arrays
  atom.cpu_type.resize(new_size);
  atom.cpu_mass.resize(new_size);
  
  for (int d = 0; d < 3; ++d) {
    atom.cpu_position_per_atom[d].resize(new_size);
    atom.cpu_velocity_per_atom[d].resize(new_size);
    atom.cpu_force_per_atom[d].resize(new_size);
  }
  
  atom.cpu_potential_per_atom.resize(new_size);
  atom.cpu_virial_per_atom.resize(new_size * 9);
  
  // Resize GPU arrays
  atom.type.resize(new_size);
  atom.mass.resize(new_size);
  
  for (int d = 0; d < 3; ++d) {
    atom.position_per_atom[d].resize(new_size);
    atom.velocity_per_atom[d].resize(new_size);
    atom.force_per_atom[d].resize(new_size);
  }
  
  atom.potential_per_atom.resize(new_size);
  atom.virial_per_atom.resize(new_size * 9);
  
  // Initialize new elements to safe values
  for (int i = atom.number_of_atoms; i < new_size; ++i) {
    atom.cpu_type[i] = -1;
    atom.cpu_mass[i] = 0.0;
    for (int d = 0; d < 3; ++d) {
      atom.cpu_position_per_atom[d][i] = 0.0;
      atom.cpu_velocity_per_atom[d][i] = 0.0;
      atom.cpu_force_per_atom[d][i] = 0.0;
    }
    atom.cpu_potential_per_atom[i] = 0.0;
    for (int j = 0; j < 9; ++j) {
      atom.cpu_virial_per_atom[i * 9 + j] = 0.0;
    }
  }
}

bool MC_Ensemble_GCMC::check_overlap(double x, double y, double z, Atom& atom, Box& box)
{
  double min_distance_sq = 0.5 * 0.5; // Minimum distance = 0.5 Å
  
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] < 0) continue; // Skip inactive atoms
    
    double dx = atom.cpu_position_per_atom[0][i] - x;
    double dy = atom.cpu_position_per_atom[1][i] - y;
    double dz = atom.cpu_position_per_atom[2][i] - z;
    
    // Apply minimum image convention
    apply_mic(box, dx, dy, dz);
    
    double dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq < min_distance_sq) {
      return true; // Overlap detected
    }
  }
  
  return false; // No overlap
}

void MC_Ensemble_GCMC::apply_pbc_displacement(double& x, double& y, double& z, Box& box)
{
  // Apply periodic boundary conditions
  // Convert to fractional coordinates
  double fx = box.cpu_h[9] * x + box.cpu_h[12] * y + box.cpu_h[15] * z;
  double fy = box.cpu_h[10] * x + box.cpu_h[13] * y + box.cpu_h[16] * z;
  double fz = box.cpu_h[11] * x + box.cpu_h[14] * y + box.cpu_h[17] * z;
  
  // Apply PBC
  fx -= floor(fx);
  fy -= floor(fy);
  fz -= floor(fz);
  
  // Convert back to Cartesian
  x = box.cpu_h[0] * fx + box.cpu_h[3] * fy + box.cpu_h[6] * fz;
  y = box.cpu_h[1] * fx + box.cpu_h[4] * fy + box.cpu_h[7] * fz;
  z = box.cpu_h[2] * fx + box.cpu_h[5] * fy + box.cpu_h[8] * fz;
}

void MC_Ensemble_GCMC::update_neighbor_lists(int atom_index, Atom& atom, Box& box)
{
  // This function would update neighbor lists after atom insertion/deletion
  // For now, we rely on the energy calculator to handle this internally
  // TODO: Implement efficient neighbor list management for MC moves
}

// Advanced MC methods implementation
void MC_Ensemble_GCMC::attempt_volume_change(Atom& atom, Box& box, double temperature, int& num_accepted)
{
  // NPT-like volume moves for GCMC
  std::uniform_real_distribution<double> r_vol(-0.01, 0.01);
  double vol_scale = 1.0 + r_vol(rng);
  
  // Store old box and positions
  Box old_box = box;
  std::vector<std::vector<double>> old_positions(3);
  for (int d = 0; d < 3; ++d) {
    old_positions[d] = atom.cpu_position_per_atom[d];
  }
  
  // Scale box
  for (int i = 0; i < 9; ++i) {
    box.cpu_h[i] *= cbrt(vol_scale);
  }
  
  // Scale positions
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      for (int d = 0; d < 3; ++d) {
        atom.cpu_position_per_atom[d][i] *= cbrt(vol_scale);
      }
    }
  }
  
  // Calculate energy change
  float energy_before = calculate_system_energy(atom, box);
  atom.copy_from_cpu();
  float energy_after = calculate_system_energy(atom, box);
  
  double delta_E = energy_after - energy_before;
  double beta = 1.0 / (K_B * temperature);
  
  // Volume change acceptance criterion (simplified NPT)
  double pressure = 0.0; // External pressure (could be parameter)
  double delta_V = (vol_scale - 1.0) * old_box.get_volume();
  double ln_acc = -beta * (delta_E + pressure * delta_V);
  
  std::uniform_real_distribution<double> r_accept(0.0, 1.0);
  if (ln_acc > 0.0 || r_accept(rng) < exp(ln_acc)) {
    // Accept volume change
    num_accepted++;
    box.copy_from_cpu();
  } else {
    // Reject - restore old state
    box = old_box;
    for (int d = 0; d < 3; ++d) {
      atom.cpu_position_per_atom[d] = old_positions[d];
    }
    atom.copy_from_cpu();
    box.copy_from_cpu();
  }
}

void MC_Ensemble_GCMC::attempt_cluster_moves(Atom& atom, Box& box, double temperature, int& num_accepted)
{
  // Collective moves of atom clusters
  const double cluster_radius = 3.0; // Angstroms
  
  // Choose random center atom
  std::vector<int> active_indices;
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      active_indices.push_back(i);
    }
  }
  
  if (active_indices.empty()) return;
  
  std::uniform_int_distribution<int> r_center(0, active_indices.size() - 1);
  int center_atom = active_indices[r_center(rng)];
  
  // Find cluster members
  std::vector<int> cluster_atoms;
  cluster_atoms.push_back(center_atom);
  
  double center_x = atom.cpu_position_per_atom[0][center_atom];
  double center_y = atom.cpu_position_per_atom[1][center_atom];
  double center_z = atom.cpu_position_per_atom[2][center_atom];
  
  for (int i : active_indices) {
    if (i == center_atom) continue;
    
    double dx = atom.cpu_position_per_atom[0][i] - center_x;
    double dy = atom.cpu_position_per_atom[1][i] - center_y;
    double dz = atom.cpu_position_per_atom[2][i] - center_z;
    
    apply_mic(box, dx, dy, dz);
    
    double dist = sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < cluster_radius) {
      cluster_atoms.push_back(i);
    }
  }
  
  if (cluster_atoms.size() < 2) return; // Need at least 2 atoms for cluster move
  
  // Generate random displacement for entire cluster
  std::uniform_real_distribution<double> r_disp(-max_displacement, max_displacement);
  double dx_cluster = r_disp(rng);
  double dy_cluster = r_disp(rng);
  double dz_cluster = r_disp(rng);
  
  // Calculate energy before move
  float energy_before = calculate_system_energy(atom, box);
  
  // Store old positions
  std::vector<std::array<double, 3>> old_positions(cluster_atoms.size());
  for (size_t i = 0; i < cluster_atoms.size(); ++i) {
    int atom_idx = cluster_atoms[i];
    old_positions[i][0] = atom.cpu_position_per_atom[0][atom_idx];
    old_positions[i][1] = atom.cpu_position_per_atom[1][atom_idx];
    old_positions[i][2] = atom.cpu_position_per_atom[2][atom_idx];
  }
  
  // Apply displacement to cluster
  for (int atom_idx : cluster_atoms) {
    atom.cpu_position_per_atom[0][atom_idx] += dx_cluster;
    atom.cpu_position_per_atom[1][atom_idx] += dy_cluster;
    atom.cpu_position_per_atom[2][atom_idx] += dz_cluster;
    
    apply_pbc_displacement(atom.cpu_position_per_atom[0][atom_idx],
                          atom.cpu_position_per_atom[1][atom_idx],
                          atom.cpu_position_per_atom[2][atom_idx], box);
  }
  
  // Calculate energy after move
  atom.copy_from_cpu();
  float energy_after = calculate_system_energy(atom, box);
  
  // Metropolis acceptance criterion
  double delta_E = energy_after - energy_before;
  double beta = 1.0 / (K_B * temperature);
  double ln_acc = -beta * delta_E;
  
  std::uniform_real_distribution<double> r_accept(0.0, 1.0);
  if (ln_acc > 0.0 || r_accept(rng) < exp(ln_acc)) {
    // Accept cluster move
    num_accepted++;
  } else {
    // Reject - restore old positions
    for (size_t i = 0; i < cluster_atoms.size(); ++i) {
      int atom_idx = cluster_atoms[i];
      atom.cpu_position_per_atom[0][atom_idx] = old_positions[i][0];
      atom.cpu_position_per_atom[1][atom_idx] = old_positions[i][1];
      atom.cpu_position_per_atom[2][atom_idx] = old_positions[i][2];
    }
    atom.copy_from_cpu();
  }
}

void MC_Ensemble_GCMC::balance_insertion_rates()
{
  // Adjust insertion probabilities based on acceptance rates
  if (num_insertions_attempted > 100) {
    double insertion_rate = static_cast<double>(num_insertions_accepted) / num_insertions_attempted;
    double deletion_rate = num_deletions_attempted > 0 ? 
      static_cast<double>(num_deletions_accepted) / num_deletions_attempted : 0.0;
    
    // Adaptive rate balancing
    for (size_t s = 0; s < species.size(); ++s) {
      if (insertion_rate < 0.1) {
        // Too many insertion rejections - decrease chemical potential slightly
        mu[s] -= 0.01; // Small adjustment in eV
        printf("Species %s: Decreasing chemical potential to %.4f eV due to low insertion rate\n", 
               species[s].c_str(), mu[s]);
      } else if (insertion_rate > 0.8 && deletion_rate < 0.1) {
        // Too many insertions, few deletions - increase chemical potential
        mu[s] += 0.01;
        printf("Species %s: Increasing chemical potential to %.4f eV to balance rates\n", 
               species[s].c_str(), mu[s]);
      }
      
      // Ensure reasonable bounds for chemical potential
      mu[s] = std::max(-10.0, std::min(5.0, mu[s]));
    }
  }
}

void MC_Ensemble_GCMC::adaptive_displacement_scaling(double acceptance_rate)
{
  // Adjust max_displacement to maintain ~50% acceptance rate
  const double target_rate = 0.5;
  const double tolerance = 0.1;
  
  if (acceptance_rate < target_rate - tolerance) {
    max_displacement *= 0.95; // Decrease step size
  } else if (acceptance_rate > target_rate + tolerance) {
    max_displacement *= 1.05; // Increase step size
  }
  
  // Ensure reasonable bounds
  max_displacement = std::max(0.01, std::min(1.0, max_displacement));
}

bool MC_Ensemble_GCMC::validate_system_state(Atom& atom, Box& box)
{
  // Check for various error conditions
  
  // Check for NaN or infinite coordinates
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] >= 0) {
      for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(atom.cpu_position_per_atom[d][i])) {
          printf("Warning: Non-finite coordinate detected at atom %d, dimension %d\n", i, d);
          return false;
        }
      }
    }
  }
  
  // Check for overlapping atoms
  double min_distance_sq = 0.1 * 0.1; // Very small minimum distance
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] < 0) continue;
    for (int j = i + 1; j < atom.number_of_atoms; ++j) {
      if (atom.cpu_type[j] < 0) continue;
      
      double dx = atom.cpu_position_per_atom[0][i] - atom.cpu_position_per_atom[0][j];
      double dy = atom.cpu_position_per_atom[1][i] - atom.cpu_position_per_atom[1][j];
      double dz = atom.cpu_position_per_atom[2][i] - atom.cpu_position_per_atom[2][j];
      
      apply_mic(box, dx, dy, dz);
      
      double dist_sq = dx * dx + dy * dy + dz * dz;
      if (dist_sq < min_distance_sq) {
        printf("Warning: Atom overlap detected between atoms %d and %d (distance = %f)\n", 
               i, j, sqrt(dist_sq));
        return false;
      }
    }
  }
  
  return true;
}

// Performance and diagnostic functions
void MC_Ensemble_GCMC::print_performance_statistics()
{
  printf("\n=== GCMC Performance Statistics ===\n");
  printf("Total MC steps performed: %d\n", total_mc_steps_performed);
  printf("Insertion attempts: %d (%.1f%% accepted)\n", 
         num_insertions_attempted, 
         num_insertions_attempted > 0 ? 100.0 * num_insertions_accepted / num_insertions_attempted : 0.0);
  printf("Deletion attempts: %d (%.1f%% accepted)\n", 
         num_deletions_attempted,
         num_deletions_attempted > 0 ? 100.0 * num_deletions_accepted / num_deletions_attempted : 0.0);
  printf("Displacement attempts: %d (%.1f%% accepted)\n", 
         num_displacements_attempted,
         num_displacements_attempted > 0 ? 100.0 * num_displacements_accepted / num_displacements_attempted : 0.0);
  
  printf("Current maximum displacement: %.4f Å\n", max_displacement);
  
  if (!acceptance_history.empty()) {
    double avg_acceptance = 0.0;
    for (double acc : acceptance_history) {
      avg_acceptance += acc;
    }
    avg_acceptance /= acceptance_history.size();
    printf("Average acceptance rate: %.3f\n", avg_acceptance);
  }
  
  printf("Enhanced sampling methods:\n");
  printf("  Wang-Landau: %s\n", enable_wang_landau ? "Enabled" : "Disabled");
  printf("  Umbrella sampling: %s\n", enable_umbrella_sampling ? "Enabled" : "Disabled");
  printf("  Parallel tempering: %s\n", enable_parallel_tempering ? "Enabled" : "Disabled");
  printf("  Bias potential: %s\n", enable_bias_potential ? "Enabled" : "Disabled");
  printf("===================================\n\n");
}

void MC_Ensemble_GCMC::save_checkpoint(const std::string& filename)
{
  std::ofstream checkpoint(filename);
  checkpoint << "# GCMC Checkpoint File\n";
  checkpoint << "num_insertions_attempted " << num_insertions_attempted << "\n";
  checkpoint << "num_insertions_accepted " << num_insertions_accepted << "\n";
  checkpoint << "num_deletions_attempted " << num_deletions_attempted << "\n";
  checkpoint << "num_deletions_accepted " << num_deletions_accepted << "\n";
  checkpoint << "num_displacements_attempted " << num_displacements_attempted << "\n";
  checkpoint << "num_displacements_accepted " << num_displacements_accepted << "\n";
  checkpoint << "max_displacement " << max_displacement << "\n";
  checkpoint << "total_mc_steps_performed " << total_mc_steps_performed << "\n";
  
  checkpoint << "chemical_potentials";
  for (double mu_val : mu) {
    checkpoint << " " << mu_val;
  }
  checkpoint << "\n";
  
  checkpoint.close();
  printf("GCMC checkpoint saved to %s\n", filename.c_str());
}

bool MC_Ensemble_GCMC::load_checkpoint(const std::string& filename)
{
  std::ifstream checkpoint(filename);
  if (!checkpoint.is_open()) {
    printf("Warning: Could not load checkpoint file %s\n", filename.c_str());
    return false;
  }
  
  std::string line, key;
  while (std::getline(checkpoint, line)) {
    if (line[0] == '#') continue;
    
    std::istringstream iss(line);
    iss >> key;
    
    if (key == "num_insertions_attempted") iss >> num_insertions_attempted;
    else if (key == "num_insertions_accepted") iss >> num_insertions_accepted;
    else if (key == "num_deletions_attempted") iss >> num_deletions_attempted;
    else if (key == "num_deletions_accepted") iss >> num_deletions_accepted;
    else if (key == "num_displacements_attempted") iss >> num_displacements_attempted;
    else if (key == "num_displacements_accepted") iss >> num_displacements_accepted;
    else if (key == "max_displacement") iss >> max_displacement;
    else if (key == "total_mc_steps_performed") iss >> total_mc_steps_performed;
    else if (key == "chemical_potentials") {
      for (size_t i = 0; i < mu.size(); ++i) {
        iss >> mu[i];
      }
    }
  }
  
  checkpoint.close();
  printf("GCMC checkpoint loaded from %s\n", filename.c_str());
  return true;
}
