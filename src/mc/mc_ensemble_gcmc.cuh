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
  
#pragma once  
#include "mc_ensemble.cuh"  
  
class MC_Ensemble_GCMC : public MC_Ensemble  
{  
public:  
  MC_Ensemble_GCMC(  
    const char** param,  
    int num_param,  
    int num_steps_mc,  
    std::vector<std::string>& species,  
    std::vector<int>& types,  
    std::vector<double>& mu,  
    double max_displacement);  
  virtual ~MC_Ensemble_GCMC(void);  
  
  virtual void compute(  
    int md_step,  
    double temperature,  
    Atom& atom,  
    Box& box,  
    std::vector<Group>& group,  
    int grouping_method,  
    int group_id);  
  
private:  
  std::vector<std::string> species;  
  std::vector<int> types;  
  std::vector<double> mu;  
  double max_displacement;  
    
  // Statistics counters  
  int num_insertions_attempted;  
  int num_insertions_accepted;  
  int num_deletions_attempted;  
  int num_deletions_accepted;  
  int num_displacements_attempted;  
  int num_displacements_accepted;  
    
  // GPU vectors for neighbor lists  
  GPU_Vector<int> NN_i;  
  GPU_Vector<int> NL_i;  
  GPU_Vector<int> insertion_sites;  
  GPU_Vector<double> insertion_energies;  
    
  // Dynamic atom management  
  GPU_Vector<int> active_atoms;  
  GPU_Vector<int> deleted_atoms;  
  int max_atoms;  
  
  // GPU optimization vectors
  GPU_Vector<double> candidate_positions_x;
  GPU_Vector<double> candidate_positions_y;
  GPU_Vector<double> candidate_positions_z;
  GPU_Vector<int> overlap_results;
  GPU_Vector<float> energy_changes;
  
  // Enhanced sampling parameters
  bool enable_wang_landau;
  bool enable_umbrella_sampling;
  bool enable_parallel_tempering;
  bool enable_bias_potential;
  
  // Statistics tracking
  std::vector<double> acceptance_history;
  std::vector<int> atom_count_history;
  int total_mc_steps_performed;
    
  void attempt_insertion(  
    Atom& atom, Box& box, std::vector<Group>& groups,  
    int grouping_method, int group_id, double temperature, int& num_accepted);  
    
  void attempt_deletion(  
    Atom& atom, Box& box, std::vector<Group>& groups,  
    int grouping_method, int group_id, double temperature, int& num_accepted);  
    
  void attempt_displacement(  
    Atom& atom, Box& box, std::vector<Group>& groups,  
    int grouping_method, int group_id, double temperature, int& num_accepted);  
    
  float calculate_system_energy(Atom& atom, Box& box);  
  float calculate_local_energy(int atom_index, Atom& atom, Box& box);  
  void update_neighbor_lists(int atom_index, Atom& atom, Box& box);  
  void resize_atom_arrays(Atom& atom, int new_size);  
  bool check_overlap(double x, double y, double z, Atom& atom, Box& box);  
  void apply_pbc_displacement(double& dx, double& dy, double& dz, Box& box);
  
  // Advanced MC methods
  void attempt_volume_change(Atom& atom, Box& box, double temperature, int& num_accepted);
  void attempt_cluster_moves(Atom& atom, Box& box, double temperature, int& num_accepted);
  void attempt_parallel_tempering(Atom& atom, Box& box, double temperature, int& num_accepted);
  
  // Bias potential methods
  void apply_bias_potential(Atom& atom, Box& box, double& energy_bias);
  void update_bias_statistics(Atom& atom, Box& box);
  
  // Enhanced sampling
  void wang_landau_sampling(Atom& atom, Box& box, double temperature);
  void umbrella_sampling(Atom& atom, Box& box, double temperature);
  
  // Multi-component optimization
  void balance_insertion_rates();
  void adaptive_displacement_scaling(double acceptance_rate);
  
  // Error checking and validation
  bool validate_system_state(Atom& atom, Box& box);
  void check_energy_conservation(Atom& atom, Box& box);
  void detect_crystallization(Atom& atom, Box& box);
  
  // Performance and diagnostics
  void print_performance_statistics();
  void save_checkpoint(const std::string& filename);
  bool load_checkpoint(const std::string& filename);  
};