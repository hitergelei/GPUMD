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
------------------------------------------------------------------------------*/

#include "mc_ensemble_tfmc.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <curand_kernel.h>
#include <cmath>

// Initialize cuRAND states
static __global__ void initialize_curand_states(
  curandState* state,
  int N,
  int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curand_init(seed, n, 0, &state[n]);
  }
}

// Kernel to generate tfMC displacements
static __global__ void generate_tfmc_displacements_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double d_max,
  const double mass_min,
  const double temperature,
  const double* __restrict__ mass,
  const double* __restrict__ force,
  curandState* state,
  double* displacements)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  const int n = group_contents ? group_contents[idx] : idx;

  curandState localState = state[n];
  const double boltz = 8.617343e-5; // eV/K
  const double kbT = boltz * temperature;
  const double mass_n = mass[n];
  const double d_i = d_max * pow(mass_min / mass_n, 0.25);

  // Generate displacement for each dimension
  for (int j = 0; j < 3; j++) {
    double force_j = force[j * N + n];

    double P_acc = 0.0;
    double P_ran = 1.0;
    double gamma = force_j * d_i / (2.0 * kbT);
    double gamma_exp = exp(gamma);
    double gamma_expi = 1.0 / gamma_exp;
    double xi = 0.0;

    // Rejection sampling according to tfMC distribution
    while (P_acc < P_ran) {
      xi = 2.0 * curand_uniform_double(&localState) - 1.0;
      P_ran = curand_uniform_double(&localState);
      
      if (xi < 0.0) {
        P_acc = exp(2.0 * xi * gamma) * gamma_exp - gamma_expi;
        P_acc = P_acc / (gamma_exp - gamma_expi);
      } else if (xi > 0.0) {
        P_acc = gamma_expi - exp(2.0 * xi * gamma) * gamma_expi;
        P_acc = P_acc / (gamma_exp - gamma_expi);
      } else {
        P_acc = 1.0;
      }
    }

    // Store displacement
    displacements[j * N + n] = xi * d_i;
  }

  state[n] = localState;
}

// Kernel to compute COM displacement
static __global__ void compute_com_displacement_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double* __restrict__ mass,
  const double* __restrict__ dx,
  const double* __restrict__ dy,
  const double* __restrict__ dz,
  double* com_dx,
  double* com_dy,
  double* com_dz,
  double* total_mass)
{
  __shared__ double s_com_dx[256];
  __shared__ double s_com_dy[256];
  __shared__ double s_com_dz[256];
  __shared__ double s_mass[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = (group_contents && idx < N) ? group_contents[idx] : idx;

  s_com_dx[tid] = 0.0;
  s_com_dy[tid] = 0.0;
  s_com_dz[tid] = 0.0;
  s_mass[tid] = 0.0;

  if (n < N) {
    double m = mass[n];
    s_com_dx[tid] = m * dx[n];
    s_com_dy[tid] = m * dy[n];
    s_com_dz[tid] = m * dz[n];
    s_mass[tid] = m;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_com_dx[tid] += s_com_dx[tid + offset];
      s_com_dy[tid] += s_com_dy[tid + offset];
      s_com_dz[tid] += s_com_dz[tid + offset];
      s_mass[tid] += s_mass[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(com_dx, s_com_dx[0]);
    atomicAdd(com_dy, s_com_dy[0]);
    atomicAdd(com_dz, s_com_dz[0]);
    atomicAdd(total_mass, s_mass[0]);
  }
}

// Kernel to remove COM motion
static __global__ void remove_com_motion_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const bool fix_x,
  const bool fix_y,
  const bool fix_z,
  const double com_dx,
  const double com_dy,
  const double com_dz,
  double* dx,
  double* dy,
  double* dz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int n = group_contents ? group_contents[idx] : idx;

  if (fix_x) dx[n] -= com_dx;
  if (fix_y) dy[n] -= com_dy;
  if (fix_z) dz[n] -= com_dz;
}

// Kernel to apply displacements
static __global__ void apply_displacements_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double* __restrict__ dx,
  const double* __restrict__ dy,
  const double* __restrict__ dz,
  double* x,
  double* y,
  double* z)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int n = group_contents ? group_contents[idx] : idx;

  x[n] += dx[n];
  y[n] += dy[n];
  z[n] += dz[n];
}

// Kernel to compute center of mass
static __global__ void compute_center_of_mass_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double* __restrict__ mass,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* cm_x,
  double* cm_y,
  double* cm_z,
  double* total_mass)
{
  __shared__ double s_cm_x[256];
  __shared__ double s_cm_y[256];
  __shared__ double s_cm_z[256];
  __shared__ double s_mass[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = (group_contents && idx < N) ? group_contents[idx] : idx;

  s_cm_x[tid] = 0.0;
  s_cm_y[tid] = 0.0;
  s_cm_z[tid] = 0.0;
  s_mass[tid] = 0.0;

  if (n < N) {
    double m = mass[n];
    // Using unwrapped coordinates for accurate COM calculation across PBC
    s_cm_x[tid] = m * x[n];
    s_cm_y[tid] = m * y[n];
    s_cm_z[tid] = m * z[n];
    s_mass[tid] = m;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_cm_x[tid] += s_cm_x[tid + offset];
      s_cm_y[tid] += s_cm_y[tid + offset];
      s_cm_z[tid] += s_cm_z[tid + offset];
      s_mass[tid] += s_mass[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(cm_x, s_cm_x[0]);
    atomicAdd(cm_y, s_cm_y[0]);
    atomicAdd(cm_z, s_cm_z[0]);
    atomicAdd(total_mass, s_mass[0]);
  }
}

// Kernel to compute angular momentum (for rotation removal)
static __global__ void compute_angular_momentum_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double* __restrict__ mass,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const double* __restrict__ dx,
  const double* __restrict__ dy,
  const double* __restrict__ dz,
  const double cm_x,
  const double cm_y,
  const double cm_z,
  double* angmom_x,
  double* angmom_y,
  double* angmom_z)
{
  __shared__ double s_lx[256];
  __shared__ double s_ly[256];
  __shared__ double s_lz[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = (group_contents && idx < N) ? group_contents[idx] : idx;

  s_lx[tid] = 0.0;
  s_ly[tid] = 0.0;
  s_lz[tid] = 0.0;

  if (n < N) {
    double m = mass[n];
    // Using unwrapped coordinates for accurate angular momentum calculation
    // This matches LAMMPS fix_tfmc behavior
    double rx = x[n] - cm_x;
    double ry = y[n] - cm_y;
    double rz = z[n] - cm_z;
    
    s_lx[tid] = m * (ry * dz[n] - rz * dy[n]);
    s_ly[tid] = m * (rz * dx[n] - rx * dz[n]);
    s_lz[tid] = m * (rx * dy[n] - ry * dx[n]);
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_lx[tid] += s_lx[tid + offset];
      s_ly[tid] += s_ly[tid + offset];
      s_lz[tid] += s_lz[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(angmom_x, s_lx[0]);
    atomicAdd(angmom_y, s_ly[0]);
    atomicAdd(angmom_z, s_lz[0]);
  }
}

// Kernel to compute inertia tensor
static __global__ void compute_inertia_tensor_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double* __restrict__ mass,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const double cm_x,
  const double cm_y,
  const double cm_z,
  double* inertia)
{
  __shared__ double s_inertia[256][6]; // xx, yy, zz, xy, xz, yz

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = (group_contents && idx < N) ? group_contents[idx] : idx;

  for (int i = 0; i < 6; i++) {
    s_inertia[tid][i] = 0.0;
  }

  if (n < N) {
    double m = mass[n];
    // Using unwrapped coordinates for accurate inertia tensor calculation
    // This matches LAMMPS fix_tfmc behavior
    double dx = x[n] - cm_x;
    double dy = y[n] - cm_y;
    double dz = z[n] - cm_z;
    
    s_inertia[tid][0] = m * (dy * dy + dz * dz); // Ixx
    s_inertia[tid][1] = m * (dx * dx + dz * dz); // Iyy
    s_inertia[tid][2] = m * (dx * dx + dy * dy); // Izz
    s_inertia[tid][3] = -m * dx * dy;             // Ixy
    s_inertia[tid][4] = -m * dx * dz;             // Ixz
    s_inertia[tid][5] = -m * dy * dz;             // Iyz
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int i = 0; i < 6; i++) {
        s_inertia[tid][i] += s_inertia[tid + offset][i];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int i = 0; i < 6; i++) {
      atomicAdd(&inertia[i], s_inertia[0][i]);
    }
  }
}

// Kernel to remove rotation
static __global__ void remove_rotation_kernel(
  const int N,
  const int* __restrict__ group_contents,
  const double omega_x,
  const double omega_y,
  const double omega_z,
  const double cm_x,
  const double cm_y,
  const double cm_z,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* dx,
  double* dy,
  double* dz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int n = group_contents ? group_contents[idx] : idx;

  // Using unwrapped coordinates for accurate rotation removal
  // This matches LAMMPS fix_tfmc behavior
  double rx = x[n] - cm_x;
  double ry = y[n] - cm_y;
  double rz = z[n] - cm_z;

  // Remove rotation: d -= omega x r
  dx[n] -= (omega_y * rz - omega_z * ry);
  dy[n] -= (omega_z * rx - omega_x * rz);
  dz[n] -= (omega_x * ry - omega_y * rx);
}

MC_Ensemble_TFMC::MC_Ensemble_TFMC(
  const char** param,
  int num_param,
  int num_steps_mc,
  double d_max,
  double temperature,
  int seed,
  bool fix_com_x,
  bool fix_com_y,
  bool fix_com_z,
  bool fix_rotation)
  : MC_Ensemble(param, num_param),
    d_max(d_max),
    fix_com_x(fix_com_x),
    fix_com_y(fix_com_y),
    fix_com_z(fix_com_z),
    fix_rotation(fix_rotation),
    seed(seed),
    mass_min(1.0e10)
{
  this->num_steps_mc = num_steps_mc;
  this->temperature = temperature;

  if (d_max <= 0.0) {
    PRINT_INPUT_ERROR("tfMC displacement length must be > 0");
  }
  if (temperature <= 0.0) {
    PRINT_INPUT_ERROR("tfMC temperature must be > 0");
  }
  if (seed <= 0) {
    PRINT_INPUT_ERROR("tfMC random seed must be > 0");
  }

  // Overwrite the generic header with tfMC-specific one
  mc_output.close();
  mc_output.open("mcmd.out", std::ios::app);
  mc_output << "# tfMC (time-stamped force-bias Monte Carlo)\n";
  mc_output << "# num_MD_steps  temperature  num_MC_trials" << std::endl;

  printf("Time-stamped Force-bias Monte Carlo (tfMC) initialized:\n");
  printf("  Maximum displacement: %g A\n", d_max);
  printf("  Temperature: %g K\n", temperature);
  printf("  Random seed: %d\n", seed);
  printf("  Fix COM motion: x=%d y=%d z=%d\n", fix_com_x, fix_com_y, fix_com_z);
  printf("  Fix rotation: %d\n", fix_rotation);
}

MC_Ensemble_TFMC::~MC_Ensemble_TFMC(void)
{
  // GPU vectors will be automatically cleaned up
}

void MC_Ensemble_TFMC::find_mass_min(Atom& atom)
{
  int N = atom.number_of_atoms;
  std::vector<double> cpu_mass(N);
  
  // Copy mass from GPU to CPU to find minimum
  CHECK(cudaMemcpy(cpu_mass.data(), atom.mass.data(), N * sizeof(double), cudaMemcpyDeviceToHost));

  mass_min = cpu_mass[0];
  for (int n = 1; n < N; n++) {
    if (cpu_mass[n] < mass_min) {
      mass_min = cpu_mass[n];
    }
  }

  printf("  Minimum mass in system: %g amu\n", mass_min);
}

void MC_Ensemble_TFMC::generate_displacements(
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  int grouping_method,
  int group_id)
{
  const int N = (grouping_method >= 0) ? group[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  const int* group_contents_ptr = (grouping_method >= 0) 
    ? (group[grouping_method].contents.data() + group[grouping_method].cpu_size_sum[group_id])
    : nullptr;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;

  // Generate displacements using tfMC algorithm
  generate_tfmc_displacements_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    d_max,
    mass_min,
    temperature,
    mass.data(),
    atom.force_per_atom.data(),
    curand_states.data(),
    displacements.data());
  GPU_CHECK_KERNEL
}

void MC_Ensemble_TFMC::remove_com_motion(
  Atom& atom,
  std::vector<Group>& group,
  int grouping_method,
  int group_id)
{
  if (!fix_com_x && !fix_com_y && !fix_com_z) {
    return;
  }

  const int N = (grouping_method >= 0) ? group[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  const int* group_contents_ptr = (grouping_method >= 0)
    ? (group[grouping_method].contents.data() + group[grouping_method].cpu_size_sum[group_id])
    : nullptr;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;

  // Zero COM displacement array
  CHECK(cudaMemset(com_displacement.data(), 0, 4 * sizeof(double)));

  // Compute COM displacement
  compute_com_displacement_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    mass.data(),
    displacements.data(),
    displacements.data() + atom.number_of_atoms,
    displacements.data() + 2 * atom.number_of_atoms,
    com_displacement.data(),
    com_displacement.data() + 1,
    com_displacement.data() + 2,
    com_displacement.data() + 3);
  GPU_CHECK_KERNEL

  // Copy back to host
  std::vector<double> cpu_com(4);
  com_displacement.copy_to_host(cpu_com.data());

  double total_mass = cpu_com[3];
  if (total_mass > 0.0) {
    cpu_com[0] /= total_mass;
    cpu_com[1] /= total_mass;
    cpu_com[2] /= total_mass;
  } else {
    cpu_com[0] = cpu_com[1] = cpu_com[2] = 0.0;
  }

  // Remove COM motion
  remove_com_motion_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    fix_com_x,
    fix_com_y,
    fix_com_z,
    cpu_com[0],
    cpu_com[1],
    cpu_com[2],
    displacements.data(),
    displacements.data() + atom.number_of_atoms,
    displacements.data() + 2 * atom.number_of_atoms);
  GPU_CHECK_KERNEL
}

void MC_Ensemble_TFMC::remove_rotation(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  int grouping_method,
  int group_id)
{
  if (!fix_rotation) {
    return;
  }

  const int N = (grouping_method >= 0) ? group[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  const int* group_contents_ptr = (grouping_method >= 0)
    ? (group[grouping_method].contents.data() + group[grouping_method].cpu_size_sum[group_id])
    : nullptr;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;

  // Allocate temporary arrays if needed
  static GPU_Vector<double> cm_data(4);    // cm_x, cm_y, cm_z, total_mass
  static GPU_Vector<double> angmom_data(3); // L_x, L_y, L_z
  static GPU_Vector<double> inertia_data(6); // Ixx, Iyy, Izz, Ixy, Ixz, Iyz

  // Zero arrays
  CHECK(cudaMemset(cm_data.data(), 0, 4 * sizeof(double)));
  CHECK(cudaMemset(angmom_data.data(), 0, 3 * sizeof(double)));
  CHECK(cudaMemset(inertia_data.data(), 0, 6 * sizeof(double)));

  // Compute center of mass using unwrapped coordinates (like LAMMPS)
  compute_center_of_mass_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    mass.data(),
    atom.unwrapped_position.data(),
    atom.unwrapped_position.data() + atom.number_of_atoms,
    atom.unwrapped_position.data() + 2 * atom.number_of_atoms,
    cm_data.data(),
    cm_data.data() + 1,
    cm_data.data() + 2,
    cm_data.data() + 3);
  GPU_CHECK_KERNEL

  // Copy CM to host
  std::vector<double> cpu_cm(4);
  cm_data.copy_to_host(cpu_cm.data());
  
  double total_mass = cpu_cm[3];
  if (total_mass > 0.0) {
    cpu_cm[0] /= total_mass;
    cpu_cm[1] /= total_mass;
    cpu_cm[2] /= total_mass;
  } else {
    return; // No mass, no rotation
  }

  double cm_x = cpu_cm[0];
  double cm_y = cpu_cm[1];
  double cm_z = cpu_cm[2];

  // Compute angular momentum using unwrapped coordinates (like LAMMPS)
  // L = sum[m_i * (r_i - r_cm) x d_i]
  compute_angular_momentum_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    mass.data(),
    atom.unwrapped_position.data(),
    atom.unwrapped_position.data() + atom.number_of_atoms,
    atom.unwrapped_position.data() + 2 * atom.number_of_atoms,
    displacements.data(),
    displacements.data() + atom.number_of_atoms,
    displacements.data() + 2 * atom.number_of_atoms,
    cm_x, cm_y, cm_z,
    angmom_data.data(),
    angmom_data.data() + 1,
    angmom_data.data() + 2);
  GPU_CHECK_KERNEL

  // Compute inertia tensor using unwrapped coordinates (like LAMMPS)
  compute_inertia_tensor_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    mass.data(),
    atom.unwrapped_position.data(),
    atom.unwrapped_position.data() + atom.number_of_atoms,
    atom.unwrapped_position.data() + 2 * atom.number_of_atoms,
    cm_x, cm_y, cm_z,
    inertia_data.data());
  GPU_CHECK_KERNEL

  // Copy to host for inversion
  std::vector<double> cpu_angmom(3);
  std::vector<double> cpu_inertia(6);
  angmom_data.copy_to_host(cpu_angmom.data());
  inertia_data.copy_to_host(cpu_inertia.data());

  // Compute angular velocity: omega = I^(-1) * L
  // Inertia tensor is symmetric, stored as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
  double I[3][3];
  I[0][0] = cpu_inertia[0]; // Ixx
  I[1][1] = cpu_inertia[1]; // Iyy
  I[2][2] = cpu_inertia[2]; // Izz
  I[0][1] = I[1][0] = cpu_inertia[3]; // Ixy
  I[0][2] = I[2][0] = cpu_inertia[4]; // Ixz
  I[1][2] = I[2][1] = cpu_inertia[5]; // Iyz

  // Invert 3x3 matrix using cofactor method
  double det = I[0][0] * (I[1][1] * I[2][2] - I[1][2] * I[2][1]) -
               I[0][1] * (I[1][0] * I[2][2] - I[1][2] * I[2][0]) +
               I[0][2] * (I[1][0] * I[2][1] - I[1][1] * I[2][0]);

  if (fabs(det) < 1e-10) {
    // Singular inertia tensor, cannot remove rotation
    return;
  }

  double invI[3][3];
  invI[0][0] = (I[1][1] * I[2][2] - I[1][2] * I[2][1]) / det;
  invI[0][1] = (I[0][2] * I[2][1] - I[0][1] * I[2][2]) / det;
  invI[0][2] = (I[0][1] * I[1][2] - I[0][2] * I[1][1]) / det;
  invI[1][0] = (I[1][2] * I[2][0] - I[1][0] * I[2][2]) / det;
  invI[1][1] = (I[0][0] * I[2][2] - I[0][2] * I[2][0]) / det;
  invI[1][2] = (I[0][2] * I[1][0] - I[0][0] * I[1][2]) / det;
  invI[2][0] = (I[1][0] * I[2][1] - I[1][1] * I[2][0]) / det;
  invI[2][1] = (I[0][1] * I[2][0] - I[0][0] * I[2][1]) / det;
  invI[2][2] = (I[0][0] * I[1][1] - I[0][1] * I[1][0]) / det;

  // omega = I^(-1) * L
  double omega[3];
  omega[0] = invI[0][0] * cpu_angmom[0] + invI[0][1] * cpu_angmom[1] + invI[0][2] * cpu_angmom[2];
  omega[1] = invI[1][0] * cpu_angmom[0] + invI[1][1] * cpu_angmom[1] + invI[1][2] * cpu_angmom[2];
  omega[2] = invI[2][0] * cpu_angmom[0] + invI[2][1] * cpu_angmom[1] + invI[2][2] * cpu_angmom[2];

  // Remove rotation from displacements using unwrapped coordinates (like LAMMPS)
  // d_i -= omega x (r_i - r_cm)
  remove_rotation_kernel<<<grid_size, block_size>>>(
    N,
    group_contents_ptr,
    omega[0], omega[1], omega[2],
    cm_x, cm_y, cm_z,
    atom.unwrapped_position.data(),
    atom.unwrapped_position.data() + atom.number_of_atoms,
    atom.unwrapped_position.data() + 2 * atom.number_of_atoms,
    displacements.data(),
    displacements.data() + atom.number_of_atoms,
    displacements.data() + 2 * atom.number_of_atoms);
  GPU_CHECK_KERNEL
}

void MC_Ensemble_TFMC::compute(
  int md_step,
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  int grouping_method,
  int group_id)
{
  const int N = atom.number_of_atoms;

  // Initialize on first call
  if (mass.size() == 0) {
    mass.resize(N);
    displacements.resize(3 * N);
    com_displacement.resize(4);
    curand_states.resize(N);

    // Copy mass
    mass.copy_from_host(atom.mass.data());

    // Find minimum mass
    find_mass_min(atom);

    // Initialize cuRAND states
    const int block_size = 256;
    const int grid_size = (N - 1) / block_size + 1;
    initialize_curand_states<<<grid_size, block_size>>>(
      curand_states.data(), N, seed);
    GPU_CHECK_KERNEL

    // Initialize unwrapped_position for accurate rotation removal (like LAMMPS)
    if (fix_rotation) {
      if (atom.unwrapped_position.size() == 0) {
        atom.unwrapped_position.resize(3 * N);
        atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
      }
      if (atom.position_temp.size() == 0) {
        atom.position_temp.resize(3 * N);
        atom.position_temp.copy_from_device(atom.position_per_atom.data());
      }
    }
  }

  // Perform MC steps
  for (int step = 0; step < num_steps_mc; step++) {
    // Note: Forces should be computed before calling this function
    // The forces are already available in atom.force_per_atom
    
    // Generate displacements based on forces
    generate_displacements(temperature, atom, box, group, grouping_method, group_id);

    // Remove COM motion if requested
    remove_com_motion(atom, group, grouping_method, group_id);

    // Remove rotation if requested
    remove_rotation(atom, box, group, grouping_method, group_id);

    // Apply displacements
    const int N_group = (grouping_method >= 0) ? group[grouping_method].cpu_size[group_id] : N;
    const int* group_contents_ptr = (grouping_method >= 0)
      ? (group[grouping_method].contents.data() + group[grouping_method].cpu_size_sum[group_id])
      : nullptr;
    const int block_size = 256;
    const int grid_size = (N_group - 1) / block_size + 1;
    apply_displacements_kernel<<<grid_size, block_size>>>(
      N_group,
      group_contents_ptr,
      displacements.data(),
      displacements.data() + N,
      displacements.data() + 2 * N,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + N,
      atom.position_per_atom.data() + 2 * N);
    GPU_CHECK_KERNEL

    // Note: PBC will be applied by the main MD loop after forces are recalculated
  }

  // Output statistics
  if (md_step % 100 == 0) {
    printf("tfMC step %d completed with %d MC trials\n", md_step, num_steps_mc);
    fflush(stdout);  // Force flush output buffer
    // Write to mcmd.out: MD_step, current_temperature, num_MC_trials
    mc_output << md_step << "  " << temperature << "  " << num_steps_mc << std::endl;
    mc_output.flush();  // Force flush file output
  }
}
