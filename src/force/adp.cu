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
The ADP (Angular Dependent Potential) implementation.
This is an extension of the EAM potential that includes angular forces through
dipole and quadruple distortions of the local atomic environment.

Reference: Y. Mishin et al., Acta Mater. 53, 4029 (2005)

The total energy of atom i is:
E_i = F_α(∑_{j≠i} ρ_β(r_ij)) + 1/2 ∑_{j≠i} φ_αβ(r_ij) + 1/2 ∑_s (μ_is)^2 + 1/2 ∑_{s,t} (λ_ist)^2 - 1/6 ν_i^2

Where:
- F is the embedding energy as a function of electron density ρ  
- φ is the pair potential interaction
- μ and λ are dipole and quadruple distortion terms
- ν is the trace of λ tensor (ν = λ_xx + λ_yy + λ_zz)
- s,t = 1,2,3 refer to cartesian coordinates (x,y,z)

The dipole and quadruple terms are calculated as:
μ_is = ∑_{j≠i} u_αβ(r_ij) * r_ij^s
λ_ist = ∑_{j≠i} w_αβ(r_ij) * r_ij^s * r_ij^t

where u and w are tabulated functions in the ADP potential file.

IMPORTANT NOTES BASED ON LAMMPS IMPLEMENTATION:
1. Dipole terms have opposite signs for atom pairs: μ_i += u*r_ij, μ_j -= u*r_ij
2. Quadruple tensor ordering: [λ_xx, λ_yy, λ_zz, λ_yz, λ_xz, λ_xy] (not [xx,yy,zz,xy,xz,yz])
3. Energy: E_quad = 0.5*(λ_xx² + λ_yy² + λ_zz²) + 1.0*(λ_yz² + λ_xz² + λ_xy²) - (1/6)*ν²
------------------------------------------------------------------------------*/

#include "adp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

#define BLOCK_SIZE_FORCE 64
#define MAX_ADP_NEIGHBORS 400

// ADP-specific neighbor compression kernel (remove duplicates, cap length)
__global__ void compress_adp_neighbors(int N, int maxN, int* g_NN, int* g_NL)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int old_count = g_NN[n];
    if (old_count <= 1) return;
    int write = 0;
    int prev = -1;
    for (int i = 0; i < old_count; ++i) {
      int neighbor = g_NL[n + i * N];
      if (neighbor != prev) {
        g_NL[n + write * N] = neighbor;
        prev = neighbor;
        ++write;
        if (write == maxN) break;
      }
    }
    g_NN[n] = write;
  }
}

ADP::ADP(const char* file_potential, const int number_of_atoms)
{
  initialize_adp(file_potential, number_of_atoms);
}

ADP::~ADP(void)
{
  // Nothing special to clean up
}

void ADP::initialize_adp(const char* file_potential, const int number_of_atoms)
{
  read_adp_file(file_potential);
  setup_spline_interpolation();
  
  // Resize GPU vectors
  adp_data.Fp.resize(number_of_atoms);
  adp_data.mu.resize(number_of_atoms * 3);       // 3 components (x,y,z) per atom
  adp_data.lambda.resize(number_of_atoms * 6);   // 6 components (xx,yy,zz,xy,xz,yz) per atom
  adp_data.NN.resize(number_of_atoms);
  adp_data.NL.resize(number_of_atoms * 1024);     // enlarged allocation for neighbor list
  adp_data.cell_count.resize(number_of_atoms);
  adp_data.cell_count_sum.resize(number_of_atoms);
  adp_data.cell_contents.resize(number_of_atoms);
  adp_data.d_F_rho_i_g.resize(number_of_atoms);
  
  // Copy spline coefficients to GPU
  int total_rho_points = adp_data.Nelements * adp_data.nrho;
  int total_r_points = adp_data.Nelements * adp_data.nr;
  int total_pair_points = adp_data.Nelements * (adp_data.Nelements + 1) / 2 * adp_data.nr;
  
  adp_data.F_rho_a_g.resize(total_rho_points);
  adp_data.F_rho_b_g.resize(total_rho_points);
  adp_data.F_rho_c_g.resize(total_rho_points);
  adp_data.F_rho_d_g.resize(total_rho_points);
  
  adp_data.rho_r_a_g.resize(total_r_points);
  adp_data.rho_r_b_g.resize(total_r_points);
  adp_data.rho_r_c_g.resize(total_r_points);
  adp_data.rho_r_d_g.resize(total_r_points);
  
  adp_data.phi_r_a_g.resize(total_pair_points);
  adp_data.phi_r_b_g.resize(total_pair_points);
  adp_data.phi_r_c_g.resize(total_pair_points);
  adp_data.phi_r_d_g.resize(total_pair_points);
  
  adp_data.u_r_a_g.resize(total_pair_points);
  adp_data.u_r_b_g.resize(total_pair_points);
  adp_data.u_r_c_g.resize(total_pair_points);
  adp_data.u_r_d_g.resize(total_pair_points);
  
  adp_data.w_r_a_g.resize(total_pair_points);
  adp_data.w_r_b_g.resize(total_pair_points);
  adp_data.w_r_c_g.resize(total_pair_points);
  adp_data.w_r_d_g.resize(total_pair_points);
  
  // Copy spline data to GPU
  adp_data.F_rho_a_g.copy_from_host(adp_data.F_rho_a.data());
  adp_data.F_rho_b_g.copy_from_host(adp_data.F_rho_b.data());
  adp_data.F_rho_c_g.copy_from_host(adp_data.F_rho_c.data());
  adp_data.F_rho_d_g.copy_from_host(adp_data.F_rho_d.data());
  
  adp_data.rho_r_a_g.copy_from_host(adp_data.rho_r_a.data());
  adp_data.rho_r_b_g.copy_from_host(adp_data.rho_r_b.data());
  adp_data.rho_r_c_g.copy_from_host(adp_data.rho_r_c.data());
  adp_data.rho_r_d_g.copy_from_host(adp_data.rho_r_d.data());
  
  adp_data.phi_r_a_g.copy_from_host(adp_data.phi_r_a.data());
  adp_data.phi_r_b_g.copy_from_host(adp_data.phi_r_b.data());
  adp_data.phi_r_c_g.copy_from_host(adp_data.phi_r_c.data());
  adp_data.phi_r_d_g.copy_from_host(adp_data.phi_r_d.data());
  
  adp_data.u_r_a_g.copy_from_host(adp_data.u_r_a.data());
  adp_data.u_r_b_g.copy_from_host(adp_data.u_r_b.data());
  adp_data.u_r_c_g.copy_from_host(adp_data.u_r_c.data());
  adp_data.u_r_d_g.copy_from_host(adp_data.u_r_d.data());
  
  adp_data.w_r_a_g.copy_from_host(adp_data.w_r_a.data());
  adp_data.w_r_b_g.copy_from_host(adp_data.w_r_b.data());
  adp_data.w_r_c_g.copy_from_host(adp_data.w_r_c.data());
  adp_data.w_r_d_g.copy_from_host(adp_data.w_r_d.data());
  
  printf("ADP potential initialized with %d elements and cutoff = %f Angstrom.\n", 
             adp_data.Nelements, adp_data.rc);
}

void ADP::read_adp_file(const char* file_potential)
{
  std::ifstream input_file(file_potential);
  if (!input_file.is_open()) {
    printf("Error: Cannot open ADP potential file: %s\n", file_potential);
    PRINT_INPUT_ERROR("Cannot open ADP potential file.");
  }
  
  std::string line;
  
  // Skip first 3 comment lines
  for (int i = 0; i < 3; i++) {
    std::getline(input_file, line);
  }
  
  // Line 4: Nelements Element1 Element2 ... ElementN
  std::getline(input_file, line);
  std::istringstream iss4(line);
  iss4 >> adp_data.Nelements;
  adp_data.elements_list.resize(adp_data.Nelements);
  for (int i = 0; i < adp_data.Nelements; i++) {
    iss4 >> adp_data.elements_list[i];
  }
  
  // Line 5: Nrho, drho, Nr, dr, cutoff
  std::getline(input_file, line);
  std::istringstream iss5(line);
  iss5 >> adp_data.nrho >> adp_data.drho >> adp_data.nr >> adp_data.dr >> adp_data.rc;
  
  printf("Reading ADP potential with %d elements: ", adp_data.Nelements);
  for (int i = 0; i < adp_data.Nelements; i++) {
    printf("%s ", adp_data.elements_list[i].c_str());
  }
  printf("\n");
  printf("  nrho = %d, drho = %f\n", adp_data.nrho, adp_data.drho);
  printf("  nr = %d, dr = %f\n", adp_data.nr, adp_data.dr);
  printf("  cutoff = %f\n", adp_data.rc);
  
  rc = adp_data.rc;
  
  // Initialize storage arrays
  adp_data.atomic_number.resize(adp_data.Nelements);
  adp_data.atomic_mass.resize(adp_data.Nelements);
  adp_data.lattice_constant.resize(adp_data.Nelements);
  adp_data.lattice_type.resize(adp_data.Nelements);
  
  adp_data.F_rho.resize(adp_data.Nelements * adp_data.nrho);
  adp_data.rho_r.resize(adp_data.Nelements * adp_data.nr);
  
  // Read element-specific data
  for (int element = 0; element < adp_data.Nelements; element++) {
    // Line: atomic number, mass, lattice constant, lattice type
    std::getline(input_file, line);
    std::istringstream iss_elem(line);
    iss_elem >> adp_data.atomic_number[element] >> adp_data.atomic_mass[element] 
             >> adp_data.lattice_constant[element] >> adp_data.lattice_type[element];
    
    // Read embedding function F(rho) for this element
    int base_rho = element * adp_data.nrho;
    int values_read = 0;
    while (values_read < adp_data.nrho) {
      std::getline(input_file, line);
      std::istringstream iss_f(line);
      double val;
      while (iss_f >> val && values_read < adp_data.nrho) {
        adp_data.F_rho[base_rho + values_read] = val;
        values_read++;
      }
    }
    
    // Read density function rho(r) for this element  
    int base_r = element * adp_data.nr;
    values_read = 0;
    while (values_read < adp_data.nr) {
      std::getline(input_file, line);
      std::istringstream iss_rho(line);
      double val;
      while (iss_rho >> val && values_read < adp_data.nr) {
        adp_data.rho_r[base_r + values_read] = val;
        values_read++;
      }
    }
  }
  
  // Read pairwise interactions phi(r), u(r), w(r)
  int num_pairs = adp_data.Nelements * (adp_data.Nelements + 1) / 2;
  adp_data.phi_r.resize(num_pairs * adp_data.nr);
  adp_data.u_r.resize(num_pairs * adp_data.nr);
  adp_data.w_r.resize(num_pairs * adp_data.nr);
  
  int pair_index = 0;
  for (int i = 0; i < adp_data.Nelements; i++) {
    for (int j = 0; j <= i; j++) {
      int base_pair = pair_index * adp_data.nr;
      
      // Read phi(r) - pair potential (multiplied by r)
      int values_read = 0;
      while (values_read < adp_data.nr) {
        std::getline(input_file, line);
        std::istringstream iss_phi(line);
        double val;
        while (iss_phi >> val && values_read < adp_data.nr) {
          // Store r*phi(r) directly (as in LAMMPS z2r array)
          // This will be used as z2 = r*phi(r) in force calculation
          adp_data.phi_r[base_pair + values_read] = val;
          values_read++;
        }
      }
      
      pair_index++;
    }
  }
  
  // Read u(r) functions
  pair_index = 0;
  for (int i = 0; i < adp_data.Nelements; i++) {
    for (int j = 0; j <= i; j++) {
      int base_pair = pair_index * adp_data.nr;
      
      int values_read = 0;
      while (values_read < adp_data.nr) {
        std::getline(input_file, line);
        std::istringstream iss_u(line);
        double val;
        while (iss_u >> val && values_read < adp_data.nr) {
          adp_data.u_r[base_pair + values_read] = val;
          values_read++;
        }
      }
      
      pair_index++;
    }
  }
  
  // Read w(r) functions
  pair_index = 0;
  for (int i = 0; i < adp_data.Nelements; i++) {
    for (int j = 0; j <= i; j++) {
      int base_pair = pair_index * adp_data.nr;
      
      int values_read = 0;
      while (values_read < adp_data.nr) {
        std::getline(input_file, line);
        std::istringstream iss_w(line);
        double val;
        while (iss_w >> val && values_read < adp_data.nr) {
          adp_data.w_r[base_pair + values_read] = val;
          values_read++;
        }
      }
      
      pair_index++;
    }
  }
  
  input_file.close();
}

void ADP::setup_spline_interpolation()
{
  // Calculate cubic spline coefficients for all tabulated functions
  // This follows the same approach as the EAM_alloy implementation
  
  int total_rho_points = adp_data.Nelements * adp_data.nrho;
  int total_r_points = adp_data.Nelements * adp_data.nr;
  int total_pair_points = adp_data.Nelements * (adp_data.Nelements + 1) / 2 * adp_data.nr;
  
  // Resize spline coefficient arrays
  adp_data.F_rho_a.resize(total_rho_points);
  adp_data.F_rho_b.resize(total_rho_points);
  adp_data.F_rho_c.resize(total_rho_points);
  adp_data.F_rho_d.resize(total_rho_points);
  
  adp_data.rho_r_a.resize(total_r_points);
  adp_data.rho_r_b.resize(total_r_points);
  adp_data.rho_r_c.resize(total_r_points);
  adp_data.rho_r_d.resize(total_r_points);
  
  adp_data.phi_r_a.resize(total_pair_points);
  adp_data.phi_r_b.resize(total_pair_points);
  adp_data.phi_r_c.resize(total_pair_points);
  adp_data.phi_r_d.resize(total_pair_points);
  
  adp_data.u_r_a.resize(total_pair_points);
  adp_data.u_r_b.resize(total_pair_points);
  adp_data.u_r_c.resize(total_pair_points);
  adp_data.u_r_d.resize(total_pair_points);
  
  adp_data.w_r_a.resize(total_pair_points);
  adp_data.w_r_b.resize(total_pair_points);
  adp_data.w_r_c.resize(total_pair_points);
  adp_data.w_r_d.resize(total_pair_points);
  
  // Calculate cubic spline coefficients for accurate interpolation
  // Using natural cubic splines for smooth interpolation
  
  // F_rho splines
  calculate_cubic_spline_coefficients(
    adp_data.F_rho.data(), total_rho_points, adp_data.drho,
    adp_data.F_rho_a.data(), adp_data.F_rho_b.data(),
    adp_data.F_rho_c.data(), adp_data.F_rho_d.data(), adp_data.Nelements, adp_data.nrho);
    
  
  // rho_r splines
  calculate_cubic_spline_coefficients(
    adp_data.rho_r.data(), total_r_points, adp_data.dr,
    adp_data.rho_r_a.data(), adp_data.rho_r_b.data(),
    adp_data.rho_r_c.data(), adp_data.rho_r_d.data(), adp_data.Nelements, adp_data.nr);
  
  // phi_r, u_r, w_r splines
  calculate_cubic_spline_coefficients(
    adp_data.phi_r.data(), total_pair_points, adp_data.dr,
    adp_data.phi_r_a.data(), adp_data.phi_r_b.data(),
    adp_data.phi_r_c.data(), adp_data.phi_r_d.data(), 
    adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
    
  calculate_cubic_spline_coefficients(
    adp_data.u_r.data(), total_pair_points, adp_data.dr,
    adp_data.u_r_a.data(), adp_data.u_r_b.data(),
    adp_data.u_r_c.data(), adp_data.u_r_d.data(),
    adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
    
  calculate_cubic_spline_coefficients(
    adp_data.w_r.data(), total_pair_points, adp_data.dr,
    adp_data.w_r_a.data(), adp_data.w_r_b.data(),
    adp_data.w_r_c.data(), adp_data.w_r_d.data(),
    adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
}

// Calculate cubic spline coefficients for tabulated functions
void ADP::calculate_cubic_spline_coefficients(
  const double* y, int n_total, double dx,
  double* a, double* b, double* c, double* d,
  int n_functions, int n_points)
{
  // Process each function separately (by element or element pair)
  for (int func = 0; func < n_functions; func++) {
    int offset = func * n_points;
    const double* y_func = y + offset;
    double* a_func = a + offset;
    double* b_func = b + offset;
    double* c_func = c + offset;
    double* d_func = d + offset;
    
    // Natural cubic spline implementation
    std::vector<double> h(n_points - 1);
    std::vector<double> alpha(n_points - 1);
    std::vector<double> l(n_points);
    std::vector<double> mu(n_points);
    std::vector<double> z(n_points);
    
    // Step 1: Calculate h and alpha
    for (int i = 0; i < n_points - 1; i++) {
      h[i] = dx;
      if (i > 0) {
        alpha[i] = (3.0 / h[i]) * (y_func[i+1] - y_func[i]) - 
                   (3.0 / h[i-1]) * (y_func[i] - y_func[i-1]);
      }
    }
    
    // Step 2: Solve the tridiagonal system for natural spline
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;
    
    for (int i = 1; i < n_points - 1; i++) {
      l[i] = 2.0 * (h[i-1] + h[i]) - h[i-1] * mu[i-1];
      mu[i] = h[i] / l[i];
      z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
    }
    
    l[n_points-1] = 1.0;
    z[n_points-1] = 0.0;
    c_func[n_points-1] = 0.0;
    
    // Step 3: Back substitution to get coefficients
    for (int j = n_points - 2; j >= 0; j--) {
      c_func[j] = z[j] - mu[j] * c_func[j+1];
      b_func[j] = (y_func[j+1] - y_func[j]) / h[j] - h[j] * (c_func[j+1] + 2.0 * c_func[j]) / 3.0;
      d_func[j] = (c_func[j+1] - c_func[j]) / (3.0 * h[j]);
      a_func[j] = y_func[j];
    }
  }
}

// GPU utility functions for interpolation
__device__ static void interpolate_adp(
  const double* a, const double* b, const double* c, const double* d,
  int index, double x, double& y, double& yp)
{
  // x is already the fractional part from LAMMPS-style calculation
  // Use cubic spline interpolation: y = a + b*x + c*x^2 + d*x^3
  y = a[index] + b[index] * x + c[index] * x * x + d[index] * x * x * x;
  yp = b[index] + 2.0 * c[index] * x + 3.0 * d[index] * x * x;
  // NOTE: yp is dy/dx where x is the fractional coordinate
  // To get dy/dr, we need to multiply by dx/dr = 1/dr in the calling function
}

// Get pair index for element types (0-based) consistent with reading order (i loop outer, j<=i)
__device__ static int get_pair_index(int type1, int type2, int /*Nelements*/)
{
  // Avoid relying on std::max/min inside device; implement explicitly
  int a = (type1 >= type2) ? type1 : type2; // ensure a >= b
  int b = (type1 >= type2) ? type2 : type1;
  // sequence: (0,0)=0; (1,0)=1,(1,1)=2; (2,0)=3,(2,1)=4,(2,2)=5; index = a*(a+1)/2 + b
  return a * (a + 1) / 2 + b;
}

// Calculate density and dipole/quadruple terms
static __global__ void find_force_adp_step1(
  const int N,
  const int N1,
  const int N2,
  const int Nelements,
  const int nrho,
  const double drho,
  const int nr,
  const double dr,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* F_rho_a,
  const double* F_rho_b,
  const double* F_rho_c,
  const double* F_rho_d,
  const double* rho_r_a,
  const double* rho_r_b,
  const double* rho_r_c,
  const double* rho_r_d,
  const double* u_r_a,
  const double* u_r_b,
  const double* u_r_c,
  const double* u_r_d,
  const double* w_r_a,
  const double* w_r_b,
  const double* w_r_c,
  const double* w_r_d,
  double* g_Fp,
  double* g_mu,
  double* g_lambda,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  
  if (n1 < N2) {
    int NN = g_NN[n1];
    if (NN > 400) {
      // Hard guard against neighbor list overflow (allocation is N*400)
      printf("ADP ERROR: NN(%d) > 400 for atom %d. Truncating to 400.\n", NN, n1);
      NN = 400;
    }
    int type1 = g_type[n1];
    
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    
    // Initialize density 
    double rho = 0.0;
    
    // Initialize ADP terms - these will be accumulated from neighbors
    // For dipole: μ_i = Σ_j u(r_ij) * r_ij   (LAMMPS formula)
    // For quadruple: λ_i = Σ_j w(r_ij) * r_ij ⊗ r_ij (tensor product)
    double mu_x = 0.0, mu_y = 0.0, mu_z = 0.0;
    double lambda_xx = 0.0, lambda_yy = 0.0, lambda_zz = 0.0;
    double lambda_xy = 0.0, lambda_xz = 0.0, lambda_yz = 0.0;
    
    // Loop over neighbors
    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int type2 = g_type[n2];
      // LAMMPS uses del = x_i - x_j
      double delx = x1 - g_x[n2];
      double dely = y1 - g_y[n2];
      double delz = z1 - g_z[n2];
      apply_mic(box, delx, dely, delz);
      double d12 = sqrt(delx * delx + dely * dely + delz * delz);
      
      if (d12 < dr * nr && d12 > 1e-12) {
        double r_index = d12 / dr;
        int ir = (int)r_index;
        int rho_base = type2 * nr;
        int pair_base = get_pair_index(type1, type2, Nelements) * nr;
        
        if (ir < nr - 1) {
          // Calculate rho contribution (electron density)
          double rho_val, rho_deriv;
          interpolate_adp(rho_r_a, rho_r_b, rho_r_c, rho_r_d, 
                         rho_base + ir, r_index - ir, rho_val, rho_deriv);
          rho += rho_val;
          
          // DEBUG: Print density contribution for first atom
          if (n1 == 0 && i1 < 3) {
            printf("  Neighbor %d: r=%f, rho_val=%f, rho_total=%f\n", i1, d12, rho_val, rho);
          }
          
          // Calculate u and w values for dipole and quadruple terms
          double u_val, u_deriv, w_val, w_deriv;
          interpolate_adp(u_r_a, u_r_b, u_r_c, u_r_d,
                         pair_base + ir, r_index - ir, u_val, u_deriv);
          interpolate_adp(w_r_a, w_r_b, w_r_c, w_r_d,
                         pair_base + ir, r_index - ir, w_val, w_deriv);
          
          // Dipole terms: mu_i += u(r)*del (del = x_i - x_j)
          mu_x += u_val * delx;
          mu_y += u_val * dely;
          mu_z += u_val * delz;
          // Quadruple tensor components (sign invariant to del sign for squares / symmetric products)
          lambda_xx += w_val * delx * delx;
          lambda_yy += w_val * dely * dely;
          lambda_zz += w_val * delz * delz;
          lambda_xy += w_val * delx * dely;
          lambda_xz += w_val * delx * delz;
          lambda_yz += w_val * dely * delz;
        }
      }
    }
    
    // Calculate embedding energy F(rho) and its derivative
    double rho_index = rho / drho;
    int irho = (int)rho_index;
    int F_base = type1 * nrho;
    
    double F = 0.0, Fp = 0.0;
    
    // CRITICAL FIX: Add proper boundary checking for F(rho) table
    if (rho_index >= 0.0 && irho < nrho - 1) {
      interpolate_adp(F_rho_a, F_rho_b, F_rho_c, F_rho_d,
                     F_base + irho, rho_index - irho, F, Fp);
      // NOTE: Fp from spline interpolation already has correct units
      // No additional scaling needed as spline coefficients include 1/drho scaling
      // Fp /= drho;  // REMOVED: This was causing excessive values
    } else {
      // If outside table range, use boundary values
      if (rho_index <= 0.0) {
        // Use first point
        F = F_rho_a[F_base];  // This should be the value at the first point
        Fp = 0.0;  // Set derivative to zero at boundary
      } else {
        // Use last point  
        F = F_rho_a[F_base + nrho - 1];  // This should be the value at the last point
        Fp = 0.0;  // Set derivative to zero at boundary
      }
    }
    
    // DEBUG: Print some values to diagnose the problem
    if (n1 == 0) {
      printf("DEBUG step1 atom 0: NN=%d, rho=%.6f, F=%.6f, mu=(%.3f,%.3f,%.3f)\n",
             NN, rho, F, mu_x, mu_y, mu_z);
      printf("DEBUG step1 atom 0: lambda=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
             lambda_xx, lambda_yy, lambda_zz, lambda_yz, lambda_xz, lambda_xy);
    }
    
    // Calculate ADP energy contributions following LAMMPS exactly
    // LAMMPS pair_adp.cpp line 263-269:
    // phi += 0.5*(mu[i][0]*mu[i][0]+mu[i][1]*mu[i][1]+mu[i][2]*mu[i][2]);
    // phi += 0.5*(lambda[i][0]*lambda[i][0]+lambda[i][1]*lambda[i][1]+lambda[i][2]*lambda[i][2]);
    // phi += 1.0*(lambda[i][3]*lambda[i][3]+lambda[i][4]*lambda[i][4]+lambda[i][5]*lambda[i][5]);
    // phi -= 1.0/6.0*(lambda[i][0]+lambda[i][1]+lambda[i][2])*(lambda[i][0]+lambda[i][1]+lambda[i][2]);
    
    double mu_squared = mu_x * mu_x + mu_y * mu_y + mu_z * mu_z;
    double lambda_diagonal = lambda_xx * lambda_xx + lambda_yy * lambda_yy + lambda_zz * lambda_zz;
    double lambda_offdiag = lambda_xy * lambda_xy + lambda_xz * lambda_xz + lambda_yz * lambda_yz;
    double nu = lambda_xx + lambda_yy + lambda_zz;
    
    // LAMMPS exact formulation:
    double adp_energy = 0.5 * mu_squared + 0.5 * lambda_diagonal + 1.0 * lambda_offdiag - (nu * nu) / 6.0;
    
    g_pe[n1] += F + adp_energy;
    g_Fp[n1] = Fp;
    
    // Store dipole and quadruple terms - LAMMPS ordering: [xx, yy, zz, yz, xz, xy]
    g_mu[n1] = mu_x;
    g_mu[n1 + N] = mu_y;
    g_mu[n1 + 2 * N] = mu_z;
    
    g_lambda[n1] = lambda_xx;          // index 0: xx
    g_lambda[n1 + N] = lambda_yy;      // index 1: yy  
    g_lambda[n1 + 2 * N] = lambda_zz;  // index 2: zz
    g_lambda[n1 + 3 * N] = lambda_yz;  // index 3: yz
    g_lambda[n1 + 4 * N] = lambda_xz;  // index 4: xz
    g_lambda[n1 + 5 * N] = lambda_xy;  // index 5: xy
  }
}

// Calculate forces
static __global__ void find_force_adp_step2(
  const int N,
  const int N1,
  const int N2,
  const int Nelements,
  const int nr,
  const double dr,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* __restrict__ g_Fp,
  const double* __restrict__ g_mu,
  const double* __restrict__ g_lambda,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* rho_r_a,
  const double* rho_r_b,
  const double* rho_r_c,
  const double* rho_r_d,
  const double* phi_r_a,
  const double* phi_r_b,
  const double* phi_r_c,
  const double* phi_r_d,
  const double* u_r_a,
  const double* u_r_b,
  const double* u_r_c,
  const double* u_r_d,
  const double* w_r_a,
  const double* w_r_b,
  const double* w_r_c,
  const double* w_r_d,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  
  if (n1 < N2) {
    int NN = g_NN[n1];
    if (NN > 400) {
      printf("ADP ERROR: NN(%d) > 400 for atom %d in step2. Truncating to 400.\n", NN, n1);
      NN = 400;
    }
    int type1 = g_type[n1];
    
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    
  double fx = 0.0, fy = 0.0, fz = 0.0;
  double pe = 0.0;
  // Virial tensor components (consistent ordering with other potentials):
  // store: xx, yy, zz, xy, xz, yz, yx, zx, zy
  double s_sxx = 0.0, s_syy = 0.0, s_szz = 0.0;
  double s_sxy = 0.0, s_sxz = 0.0, s_syz = 0.0;
  double s_syx = 0.0, s_szx = 0.0, s_szy = 0.0;
    
    // Get dipole and quadruple terms for atom n1 - LAMMPS ordering
    double mu1_x = g_mu[n1];
    double mu1_y = g_mu[n1 + N];
    double mu1_z = g_mu[n1 + 2 * N];
    
    double lambda1_xx = g_lambda[n1];          // index 0: xx
    double lambda1_yy = g_lambda[n1 + N];      // index 1: yy
    double lambda1_zz = g_lambda[n1 + 2 * N];  // index 2: zz
    double lambda1_yz = g_lambda[n1 + 3 * N];  // index 3: yz
    double lambda1_xz = g_lambda[n1 + 4 * N];  // index 4: xz
    double lambda1_xy = g_lambda[n1 + 5 * N];  // index 5: xy
    
    // Loop over neighbors to calculate forces
    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int type2 = g_type[n2];
      
  double delx = x1 - g_x[n2];
  double dely = y1 - g_y[n2];
  double delz = z1 - g_z[n2];
  apply_mic(box, delx, dely, delz);
  double d12 = sqrt(delx * delx + dely * dely + delz * delz);
      
      if (d12 < dr * nr && d12 > 1e-12) {
        double d12_inv = 1.0 / d12;
        
        double r_index = d12 / dr;
        int ir = (int)r_index;
        
        if (ir < nr - 1) {
          int rho1_base = type1 * nr;
          int rho2_base = type2 * nr;
          int pair_base = get_pair_index(type1, type2, Nelements) * nr;
          
          // Get interpolated values and derivatives
          double rho1_val, rho1_deriv, rho2_val, rho2_deriv;
          double phi_rphi_val, phi_rphi_deriv, u_val, u_deriv, w_val, w_deriv;
          
          interpolate_adp(rho_r_a, rho_r_b, rho_r_c, rho_r_d,
                         rho1_base + ir, r_index - ir, rho1_val, rho1_deriv);
          interpolate_adp(rho_r_a, rho_r_b, rho_r_c, rho_r_d,
                         rho2_base + ir, r_index - ir, rho2_val, rho2_deriv);
          interpolate_adp(phi_r_a, phi_r_b, phi_r_c, phi_r_d,
                         pair_base + ir, r_index - ir, phi_rphi_val, phi_rphi_deriv);
          interpolate_adp(u_r_a, u_r_b, u_r_c, u_r_d,
                         pair_base + ir, r_index - ir, u_val, u_deriv);
          interpolate_adp(w_r_a, w_r_b, w_r_c, w_r_d,
                         pair_base + ir, r_index - ir, w_val, w_deriv);
          
          // NOTE: Our spline interpolation already returns correct derivatives
          // with proper scaling included in the spline coefficients
          // No additional scaling needed
          
          // Convert r*phi(r) back to phi(r) and calculate phi'(r)
          // phi_rphi_val is r*phi(r) from the table
          // phi_rphi_deriv is d(r*phi)/dr from interpolation
          double phi_val = phi_rphi_val / d12;  // phi(r) = r*phi(r) / r
          double phi_deriv = phi_rphi_deriv / d12 - phi_rphi_val / (d12 * d12);  // phi'(r) = d(r*phi)/dr / r - r*phi(r) / r^2
          
          double mu2_x = g_mu[n2];
          double mu2_y = g_mu[n2 + N];
          double mu2_z = g_mu[n2 + 2 * N];
          
          double lambda2_xx = g_lambda[n2];          // index 0: xx
          double lambda2_yy = g_lambda[n2 + N];      // index 1: yy
          double lambda2_zz = g_lambda[n2 + 2 * N];  // index 2: zz
          double lambda2_yz = g_lambda[n2 + 3 * N];  // index 3: yz
          double lambda2_xz = g_lambda[n2 + 4 * N];  // index 4: xz
          double lambda2_xy = g_lambda[n2 + 5 * N];  // index 5: xy
          
          // Calculate force components
          // 1. Pair potential force
          double force_pair = phi_deriv;
          
          // 2. Embedding density force (EAM-like)
          double force_density = g_Fp[n1] * rho2_deriv + g_Fp[n2] * rho1_deriv;
          
          // DEBUG: Print first neighbor's values for atom 0
          if (n1 == 0 && i1 == 0) {
            printf("DEBUG - Atom 0, Neighbor 0:\n");
            printf("  Distance r = %f\n", d12);
            printf("  phi_rphi_val = %f, phi_rphi_deriv = %f\n", phi_rphi_val, phi_rphi_deriv);
            printf("  phi_val = %f, phi_deriv = %f\n", phi_val, phi_deriv);
            printf("  rho1_val = %f, rho1_deriv = %f\n", rho1_val, rho1_deriv);
            printf("  rho2_val = %f, rho2_deriv = %f\n", rho2_val, rho2_deriv);
            printf("  force_pair = %f, force_density = %f\n", force_pair, force_density);
            printf("  Fp[n1] = %f, Fp[n2] = %f\n", g_Fp[n1], g_Fp[n2]);
          }
          
          // 3. Calculate ADP force components following LAMMPS exactly
          // Based on LAMMPS pair_adp.cpp formulation
          
          // LAMMPS uses these exact variable definitions:
          double delmux = mu1_x - mu2_x;
          double delmuy = mu1_y - mu2_y; 
          double delmuz = mu1_z - mu2_z;
          double trdelmu = delmux * delx + delmuy * dely + delmuz * delz;
          
          double sumlamxx = lambda1_xx + lambda2_xx;
          double sumlamyy = lambda1_yy + lambda2_yy;
          double sumlamzz = lambda1_zz + lambda2_zz;
          double sumlamyz = lambda1_yz + lambda2_yz;
          double sumlamxz = lambda1_xz + lambda2_xz;
          double sumlamxy = lambda1_xy + lambda2_xy;
          
          double tradellam = sumlamxx * delx * delx + sumlamyy * dely * dely + 
                            sumlamzz * delz * delz + 2.0 * sumlamxy * delx * dely +
                            2.0 * sumlamxz * delx * delz + 2.0 * sumlamyz * dely * delz;
          double nu = sumlamxx + sumlamyy + sumlamzz;
          
          // LAMMPS ADP force formulation (with del = x_i - x_j)
          double adpx = delmux * u_val + trdelmu * u_deriv * delx * d12_inv +
                       2.0 * w_val * (sumlamxx * delx + sumlamxy * dely + sumlamxz * delz) +
                       w_deriv * delx * d12_inv * tradellam - 
                       (1.0/3.0) * nu * (w_deriv * d12 + 2.0 * w_val) * delx;
          double adpy = delmuy * u_val + trdelmu * u_deriv * dely * d12_inv +
                       2.0 * w_val * (sumlamxy * delx + sumlamyy * dely + sumlamyz * delz) +
                       w_deriv * dely * d12_inv * tradellam -
                       (1.0/3.0) * nu * (w_deriv * d12 + 2.0 * w_val) * dely;
          double adpz = delmuz * u_val + trdelmu * u_deriv * delz * d12_inv +
                       2.0 * w_val * (sumlamxz * delx + sumlamyz * dely + sumlamzz * delz) +
                       w_deriv * delz * d12_inv * tradellam -
                       (1.0/3.0) * nu * (w_deriv * d12 + 2.0 * w_val) * delz;
          
          // LAMMPS applies negative sign to ADP forces (line 372)
          // adpx*=-1.0; adpy*=-1.0; adpz*=-1.0;
          adpx *= -1.0;
          adpy *= -1.0;
          adpz *= -1.0;
          
          // Total force components: fpair scalar = -(phi' + Fp[i]*rho_j' + Fp[j]*rho_i')
          double scalar = -(force_pair + force_density);
          double force_total_x = scalar * delx * d12_inv + adpx;
          double force_total_y = scalar * dely * d12_inv + adpy;
          double force_total_z = scalar * delz * d12_inv + adpz;
          
          fx += force_total_x;
          fy += force_total_y;
          fz += force_total_z;
          
          // Add pair potential energy (half to avoid double counting)
          pe += 0.5 * phi_val;
          
          // DEBUG: Energy contributions for first pair of atom 0
          if (n1 == 0 && i1 == 0) {
            printf("DEBUG step2 atom 0: pair_energy=%.6f, phi_val=%.6f\n", 0.5 * phi_val, phi_val);
          }
          
          // Virial tensor contribution: r_ij (from i to j) times force on i due to j (pairwise style)
          // Following convention used in LJ and other potentials (using force on i):
          // s_sab += r_ab * f_b (with r = r_j - r_i). Currently del = r_i - r_j, so use -del for r_ij.
          double rx = -delx;
          double ry = -dely;
          double rz = -delz;
          double fxi = force_total_x;
          double fyi = force_total_y;
          double fzi = force_total_z;
          s_sxx += rx * fxi;
          s_sxy += rx * fyi;
          s_sxz += rx * fzi;
          s_syx += ry * fxi;
          s_syy += ry * fyi;
          s_syz += ry * fzi;
          s_szx += rz * fxi;
          s_szy += rz * fyi;
          s_szz += rz * fzi;
        }
      }
    }
    
    g_fx[n1] += fx;
    g_fy[n1] += fy;
    g_fz[n1] += fz;
  // Save virial tensor components to global array
  g_virial[n1 + 0 * N] += s_sxx; // xx
  g_virial[n1 + 1 * N] += s_syy; // yy
  g_virial[n1 + 2 * N] += s_szz; // zz
  g_virial[n1 + 3 * N] += s_sxy; // xy
  g_virial[n1 + 4 * N] += s_sxz; // xz
  g_virial[n1 + 5 * N] += s_syz; // yz
  g_virial[n1 + 6 * N] += s_syx; // yx
  g_virial[n1 + 7 * N] += s_szx; // zx
  g_virial[n1 + 8 * N] += s_szy; // zy
    g_pe[n1] += pe;
  }
}

void ADP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
  
  // Find neighbors
  find_neighbor(
    N1,
    N2,
    adp_data.rc,
    box,
    type,
    position_per_atom,
    adp_data.cell_count,
    adp_data.cell_count_sum,
    adp_data.cell_contents,
    adp_data.NN,
    adp_data.NL);
  {
    int blocks = (number_of_atoms + 127) / 128;
    compress_adp_neighbors<<<blocks,128>>>(number_of_atoms, MAX_ADP_NEIGHBORS, adp_data.NN.data(), adp_data.NL.data());
    GPU_CHECK_KERNEL
  }

  // Zero only internal accumulators needed for this step; top-level framework already zeroed force/potential/virial.
  adp_data.mu.fill(0.0);      // size 3N
  adp_data.lambda.fill(0.0);  // size 6N
  adp_data.Fp.fill(0.0);      // size N
  
  // Step 1: Calculate density, embedding energy, and angular terms
  find_force_adp_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,  // This is N parameter
    N1,
    N2,
    adp_data.Nelements,
    adp_data.nrho,
    adp_data.drho,
    adp_data.nr,
    adp_data.dr,
    box,
    adp_data.NN.data(),
    adp_data.NL.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    adp_data.F_rho_a_g.data(),
    adp_data.F_rho_b_g.data(),
    adp_data.F_rho_c_g.data(),
    adp_data.F_rho_d_g.data(),
    adp_data.rho_r_a_g.data(),
    adp_data.rho_r_b_g.data(),
    adp_data.rho_r_c_g.data(),
    adp_data.rho_r_d_g.data(),
    adp_data.u_r_a_g.data(),
    adp_data.u_r_b_g.data(),
    adp_data.u_r_c_g.data(),
    adp_data.u_r_d_g.data(),
    adp_data.w_r_a_g.data(),
    adp_data.w_r_b_g.data(),
    adp_data.w_r_c_g.data(),
    adp_data.w_r_d_g.data(),
    adp_data.Fp.data(),
    adp_data.mu.data(),
    adp_data.lambda.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
  
  // Step 2: Calculate forces
  find_force_adp_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    adp_data.Nelements,
    adp_data.nr,
    adp_data.dr,
    box,
    adp_data.NN.data(),
    adp_data.NL.data(),
    type.data(),
    adp_data.Fp.data(),
    adp_data.mu.data(),
    adp_data.lambda.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    adp_data.rho_r_a_g.data(),
    adp_data.rho_r_b_g.data(),
    adp_data.rho_r_c_g.data(),
    adp_data.rho_r_d_g.data(),
    adp_data.phi_r_a_g.data(),
    adp_data.phi_r_b_g.data(),
    adp_data.phi_r_c_g.data(),
    adp_data.phi_r_d_g.data(),
    adp_data.u_r_a_g.data(),
    adp_data.u_r_b_g.data(),
    adp_data.u_r_c_g.data(),
    adp_data.u_r_d_g.data(),
    adp_data.w_r_a_g.data(),
    adp_data.w_r_b_g.data(),
    adp_data.w_r_c_g.data(),
    adp_data.w_r_d_g.data(),
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
}