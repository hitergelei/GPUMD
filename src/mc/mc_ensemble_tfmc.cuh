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
Time-stamped Force-bias Monte Carlo (tfMC) ensemble for MCMD.

Reference:
K. M. Bal and E. C. Neyts, 
Merging Metadynamics into Hyperdynamics: Accelerated Molecular Simulations 
Reaching Time Scales from Microseconds to Seconds,
J. Chem. Theory Comput. 11, 4545 (2015).
------------------------------------------------------------------------------*/

#pragma once
#include "mc_ensemble.cuh"
#include "utilities/gpu_vector.cuh"
#include <curand_kernel.h>

class MC_Ensemble_TFMC : public MC_Ensemble
{
public:
  MC_Ensemble_TFMC(
    const char** param,
    int num_param,
    int num_steps_mc,
    double d_max,
    double temperature,
    int seed,
    bool fix_com_x,
    bool fix_com_y,
    bool fix_com_z,
    bool fix_rotation);
  virtual ~MC_Ensemble_TFMC(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

private:
  double d_max;            // maximum displacement length
  double mass_min;         // minimum mass in the system
  bool fix_com_x;          // fix center of mass motion in x
  bool fix_com_y;          // fix center of mass motion in y
  bool fix_com_z;          // fix center of mass motion in z
  bool fix_rotation;       // fix rotation
  int seed;                // random seed
  
  GPU_Vector<double> mass;
  GPU_Vector<double> displacements;
  GPU_Vector<double> com_displacement;
  GPU_Vector<curandState> curand_states;

  void find_mass_min(Atom& atom);
  void generate_displacements(
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);
  void remove_com_motion(
    Atom& atom,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);
  void remove_rotation(
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);
};
