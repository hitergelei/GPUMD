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
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <cmath>

// Use the standard GPUMD neighbor list implementation from neighbor.cu

#define BLOCK_SIZE_FORCE 64

// Periodic image encoding helper for neighbor shifts (supports offsets in [-15,15])
constexpr int PBC_SHIFT_BIAS = 15;
constexpr int PBC_SHIFT_PAYLOAD_MASK = 0x7FFF;
constexpr int PBC_SHIFT_FLAG = 1 << 15;

__host__ __device__ inline int encode_neighbor_shift(int sx, int sy, int sz)
{
  if (sx == 0 && sy == 0 && sz == 0) {
    return 0;
  }
  const int bx = sx + PBC_SHIFT_BIAS;
  const int by = sy + PBC_SHIFT_BIAS;
  const int bz = sz + PBC_SHIFT_BIAS;
  return ((bx & 0x1F) | ((by & 0x1F) << 5) | ((bz & 0x1F) << 10)) | PBC_SHIFT_FLAG;
}

__host__ __device__ inline void decode_neighbor_shift(int code, int& sx, int& sy, int& sz)
{
  if (code == 0) {
    sx = sy = sz = 0;
    return;
  }
  int payload = code & PBC_SHIFT_PAYLOAD_MASK;
  sx = ((payload & 0x1F) - PBC_SHIFT_BIAS);
  sy = (((payload >> 5) & 0x1F) - PBC_SHIFT_BIAS);
  sz = (((payload >> 10) & 0x1F) - PBC_SHIFT_BIAS);
}

// Debug dump switch for comparing with LAMMPS setfl parsing
#ifndef ADP_DEBUG_PRINT
#define ADP_DEBUG_PRINT 1
#endif

// Simple O(N^2) neighbor construction for small boxes (avoids duplicates with coarse cell lists)
static __global__ void build_neighbor_ON2(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const double rc2,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  int* __restrict__ NN,
  int* __restrict__ NL,
  int* __restrict__ shift_codes,
  const int max_neighbors)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];

  const double rc = sqrt(rc2);
  const double hx0 = box.cpu_h[0];
  const double hx1 = box.cpu_h[1];
  const double hx2 = box.cpu_h[2];
  const double hy0 = box.cpu_h[3];
  const double hy1 = box.cpu_h[4];
  const double hy2 = box.cpu_h[5];
  const double hz0 = box.cpu_h[6];
  const double hz1 = box.cpu_h[7];
  const double hz2 = box.cpu_h[8];

  int px = 0;
  if (box.pbc_x && box.thickness_x > 0.0) {
    double needed = rc / box.thickness_x - 0.5;
    if (needed > 0.0) {
      px = static_cast<int>(ceil(needed));
    }
  }
  int py = 0;
  if (box.pbc_y && box.thickness_y > 0.0) {
    double needed = rc / box.thickness_y - 0.5;
    if (needed > 0.0) {
      py = static_cast<int>(ceil(needed));
    }
  }
  int pz = 0;
  if (box.pbc_z && box.thickness_z > 0.0) {
    double needed = rc / box.thickness_z - 0.5;
    if (needed > 0.0) {
      pz = static_cast<int>(ceil(needed));
    }
  }

  int count = 0;

  for (int n2 = 0; n2 < N; ++n2) {
    for (int iz = -pz; iz <= pz; ++iz) {
      for (int iy = -py; iy <= py; ++iy) {
        for (int ix = -px; ix <= px; ++ix) {
          if (ix == 0 && iy == 0 && iz == 0 && n2 == n1) continue;

          const double shift_x = ix * hx0 + iy * hx1 + iz * hx2;
          const double shift_y = ix * hy0 + iy * hy1 + iz * hy2;
          const double shift_z = ix * hz0 + iy * hz1 + iz * hz2;

          const double dx = (x[n2] + shift_x) - x1;
          const double dy = (y[n2] + shift_y) - y1;
          const double dz = (z[n2] + shift_z) - z1;
          const double d2 = dx * dx + dy * dy + dz * dz;

          if (d2 < rc2 && d2 > 1.0e-12) {
            if (count < max_neighbors) {
              NL[count * N + n1] = n2;
              shift_codes[count * N + n1] = encode_neighbor_shift(ix, iy, iz);
              ++count;
            }
          }
        }
      }
    }
  }

  NN[n1] = count;
}

ADP::ADP(const char* file_potential, const int number_of_atoms,
         const std::vector<std::string>& options)
{
  parse_options(options);
  initialize_adp(file_potential, number_of_atoms);
}

// Minimal constructor used by existing Force::parse_potential path
ADP::ADP(const char* file_potential, const int number_of_atoms)
  : ADP(file_potential, number_of_atoms, std::vector<std::string>{}) {}

ADP::~ADP(void)
{
  // Nothing special to clean up
}

void ADP::initialize_adp(const char* file_potential, const int number_of_atoms)
{
  read_adp_file(file_potential);
  
  // Build element mapping: from user-specified elements to ADP file elements
  setup_element_mapping();
  
  // Copy element mapping to GPU if needed
  if (!element_mapping.empty()) {
    adp_data.element_mapping_gpu.resize(element_mapping.size());
    adp_data.element_mapping_gpu.copy_from_host(element_mapping.data());
  }
  
  setup_spline_interpolation();
  
  // Resize GPU vectors
  adp_data.Fp.resize(number_of_atoms);
  adp_data.mu.resize(number_of_atoms * 3);       // 3 components (x,y,z) per atom
  adp_data.lambda.resize(number_of_atoms * 6);   // 6 components (xx,yy,zz,xy,xz,yz) per atom
  adp_data.mapped_type.resize(number_of_atoms);  // Type array for element mapping
  adp_data.NN.resize(number_of_atoms);
  adp_data.NL.resize(number_of_atoms * 400);     // consistent with EAM, sufficient for ADP
  adp_data.NL_shift.resize(number_of_atoms * 400);
  adp_data.cell_count.resize(number_of_atoms);
  adp_data.cell_count_sum.resize(number_of_atoms);
  adp_data.cell_contents.resize(number_of_atoms);
  adp_data.dbg_F.resize(number_of_atoms);
  adp_data.dbg_ADP.resize(number_of_atoms);
  adp_data.dbg_PAIR.resize(number_of_atoms);
  
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
  
  printf("Use %d-element ADP potential, rc = %.6f.\n", adp_data.Nelements, adp_data.rc);
  printf("ADP spline mode: %s.\n", use_lammps_spline_ ? "LAMMPS" : "natural");
  printf("ADP neighbor mode: %s.\n", use_linear_neighbor_ ? "O(N) cell list" : "O(N^2) brute force");
}

void ADP::parse_options(const std::vector<std::string>& options)
{
  // Default: LAMMPS-compatible spline and linear neighbor list
  use_lammps_spline_ = true;
  use_linear_neighbor_ = true;
  user_elements.clear();
  
  auto to_lower = [](std::string s){ for (auto& c : s) c = std::tolower(static_cast<unsigned char>(c)); return s; };
  
  // First pass: collect elements (tokens without '=')
  for (const auto& opt : options) {
    if (opt.find('=') == std::string::npos) {
      // This is likely an element name
      user_elements.push_back(opt);
    }
  }
  
  // Second pass: process key=value options
  for (auto opt : options) {
    size_t eq = opt.find('=');
    if (eq == std::string::npos) continue; // skip element names
    
    std::string key = to_lower(opt.substr(0, eq));
    std::string val = to_lower(opt.substr(eq + 1));
    
    if (key == "adp_spline" || key == "spline") {
      if (val == "natural") use_lammps_spline_ = false;
      else if (val == "lammps") use_lammps_spline_ = true;
    } else if (key == "neighbor" || key == "neighbor_list") {
      if (val == "on2" || val == "n2" || val == "brute") use_linear_neighbor_ = false;
      else if (val == "on1" || val == "linear" || val == "cell") use_linear_neighbor_ = true;
    }
  }
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
  adp_data.mass.resize(adp_data.Nelements, 0.0);
  for (int i = 0; i < adp_data.Nelements; i++) {
    iss4 >> adp_data.elements_list[i];
  }
  
  // Line 5: Nrho, drho, Nr, dr, cutoff
  std::getline(input_file, line);
  std::istringstream iss5(line);
  iss5 >> adp_data.nrho >> adp_data.drho >> adp_data.nr >> adp_data.dr >> adp_data.rc;
  
  rc = adp_data.rc;
  
  // Initialize storage arrays
  adp_data.F_rho.resize(adp_data.Nelements * adp_data.nrho);
  adp_data.rho_r.resize(adp_data.Nelements * adp_data.nr);
  
  // Read element-specific data
  for (int element = 0; element < adp_data.Nelements; element++) {
    // Line: atomic number, mass, lattice constant, lattice type
    std::getline(input_file, line);
    // Parse and store mass for debug/consistency (others ignored)
    if (!line.empty()) {
      std::istringstream iss_mass(line);
      double z = 0.0, m = 0.0, a0 = 0.0; 
      std::string lattice;
      iss_mass >> z >> m >> a0 >> lattice;
      adp_data.mass[element] = m;
    }
    
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
  
#if ADP_DEBUG_PRINT
  // Dump a compact summary for cross-check with LAMMPS
  {
    FILE* fp = fopen("adp_debug_setfl_gpumd.txt", "w");
    if (fp) {
      fprintf(fp, "file: %s\n", file_potential);
      fprintf(fp, "nelements=%d nrho=%d drho=%g nr=%d dr=%g cut=%g\n",
              adp_data.Nelements, adp_data.nrho, adp_data.drho,
              adp_data.nr, adp_data.dr, adp_data.rc);
      // per-element F(rho), rho(r)
      for (int i = 0; i < adp_data.Nelements; ++i) {
        fprintf(fp, "element[%d]=%s mass=%g\n",
                i, adp_data.elements_list[i].c_str(), adp_data.mass[i]);
        int mmax = adp_data.nrho < 5 ? adp_data.nrho : 5;
        fprintf(fp, "frho first%d:", mmax);
        for (int m = 0; m < mmax; ++m) {
          fprintf(fp, " %.16g", adp_data.F_rho[i * adp_data.nrho + m]);
        }
        fprintf(fp, "\n");
        mmax = adp_data.nr < 5 ? adp_data.nr : 5;
        fprintf(fp, "rhor first%d:", mmax);
        for (int m = 0; m < mmax; ++m) {
          fprintf(fp, " %.16g", adp_data.rho_r[i * adp_data.nr + m]);
        }
        fprintf(fp, "\n");
      }
      // pair arrays in i>=j order
      int pair_index = 0;
      int mmax = adp_data.nr < 5 ? adp_data.nr : 5;
      for (int i = 0; i < adp_data.Nelements; ++i) {
        for (int j = 0; j <= i; ++j, ++pair_index) {
          int base = pair_index * adp_data.nr;
          fprintf(fp, "z2r (%d,%d) first%d:", i, j, mmax);
          for (int m = 0; m < mmax; ++m) fprintf(fp, " %.16g", adp_data.phi_r[base + m]);
          fprintf(fp, "\n");
          fprintf(fp, "u2r (%d,%d) first%d:", i, j, mmax);
          for (int m = 0; m < mmax; ++m) fprintf(fp, " %.16g", adp_data.u_r[base + m]);
          fprintf(fp, "\n");
          fprintf(fp, "w2r (%d,%d) first%d:", i, j, mmax);
          for (int m = 0; m < mmax; ++m) fprintf(fp, " %.16g", adp_data.w_r[base + m]);
          fprintf(fp, "\n");
        }
      }
      fclose(fp);
    }
  }
#endif
}

void ADP::setup_element_mapping()
{
  // If no user elements specified, use default mapping (all types map to first ADP element)
  if (user_elements.empty()) {
    // Default behavior: assume single element system
    element_mapping.clear();
    element_mapping.push_back(0);  // All atoms map to first element in ADP file
    printf("ADP element mapping: using default (all atoms -> %s)\n", 
           adp_data.elements_list.empty() ? "element_0" : adp_data.elements_list[0].c_str());
    return;
  }
  
  // Build mapping from user elements to ADP file elements
  element_mapping.resize(user_elements.size());
  printf("ADP element mapping: ");
  
  for (size_t i = 0; i < user_elements.size(); i++) {
    element_mapping[i] = -1;  // Invalid initially
    
    // Find matching element in ADP file
    for (int j = 0; j < adp_data.Nelements; j++) {
      if (user_elements[i] == adp_data.elements_list[j]) {
        element_mapping[i] = j;
        break;
      }
    }
    
    if (element_mapping[i] == -1) {
      printf("\nError: User element '%s' not found in ADP file.\n", user_elements[i].c_str());
      printf("Available elements in ADP file: ");
      for (int j = 0; j < adp_data.Nelements; j++) {
        printf("%s ", adp_data.elements_list[j].c_str());
      }
      printf("\n");
      PRINT_INPUT_ERROR("Element mapping failed.");
    }
    
    printf("%s->%s ", user_elements[i].c_str(), adp_data.elements_list[element_mapping[i]].c_str());
  }
  printf("\n");
  
  // Debug: Print some function values to verify correct reading
  if (adp_data.Nelements == 2) {
    printf("Debug: ADP file has 2 elements, checking Mo values...\n");
    int mo_idx = -1;
    for (int i = 0; i < adp_data.Nelements; i++) {
      if (adp_data.elements_list[i] == "Mo") {
        mo_idx = i;
        break;
      }
    }
    if (mo_idx >= 0) {
      printf("Debug: Mo element index in ADP file: %d\n", mo_idx);
      printf("Debug: Mo F[0-2]: %.6f %.6f %.6f\n", 
             adp_data.F_rho[mo_idx * adp_data.nrho],
             adp_data.F_rho[mo_idx * adp_data.nrho + 1],
             adp_data.F_rho[mo_idx * adp_data.nrho + 2]);
      printf("Debug: Mo rho[0-2]: %.6f %.6f %.6f\n",
             adp_data.rho_r[mo_idx * adp_data.nr],
             adp_data.rho_r[mo_idx * adp_data.nr + 1], 
             adp_data.rho_r[mo_idx * adp_data.nr + 2]);
      // Mo-Mo pair should be at index 2 (U-U=0, Mo-U=1, Mo-Mo=2)
      int mo_mo_pair_idx = 2;
      printf("Debug: Mo-Mo phi[0-2]: %.6f %.6f %.6f\n",
             adp_data.phi_r[mo_mo_pair_idx * adp_data.nr],
             adp_data.phi_r[mo_mo_pair_idx * adp_data.nr + 1],
             adp_data.phi_r[mo_mo_pair_idx * adp_data.nr + 2]);
    }
  }
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
  
  if (use_lammps_spline_) {
    // Build coefficients equivalent to LAMMPS pair_adp interpolate() (Hermite with estimated slopes)
    calculate_lammps_like_coefficients(
      adp_data.F_rho.data(), total_rho_points, adp_data.drho,
      adp_data.F_rho_a.data(), adp_data.F_rho_b.data(), adp_data.F_rho_c.data(), adp_data.F_rho_d.data(),
      adp_data.Nelements, adp_data.nrho);
    calculate_lammps_like_coefficients(
      adp_data.rho_r.data(), total_r_points, adp_data.dr,
      adp_data.rho_r_a.data(), adp_data.rho_r_b.data(), adp_data.rho_r_c.data(), adp_data.rho_r_d.data(),
      adp_data.Nelements, adp_data.nr);
    calculate_lammps_like_coefficients(
      adp_data.phi_r.data(), total_pair_points, adp_data.dr,
      adp_data.phi_r_a.data(), adp_data.phi_r_b.data(), adp_data.phi_r_c.data(), adp_data.phi_r_d.data(),
      adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
    calculate_lammps_like_coefficients(
      adp_data.u_r.data(), total_pair_points, adp_data.dr,
      adp_data.u_r_a.data(), adp_data.u_r_b.data(), adp_data.u_r_c.data(), adp_data.u_r_d.data(),
      adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
    calculate_lammps_like_coefficients(
      adp_data.w_r.data(), total_pair_points, adp_data.dr,
      adp_data.w_r_a.data(), adp_data.w_r_b.data(), adp_data.w_r_c.data(), adp_data.w_r_d.data(),
      adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
  } else {
    // Natural cubic splines
    calculate_cubic_spline_coefficients(
      adp_data.F_rho.data(), total_rho_points, adp_data.drho,
      adp_data.F_rho_a.data(), adp_data.F_rho_b.data(),
      adp_data.F_rho_c.data(), adp_data.F_rho_d.data(), adp_data.Nelements, adp_data.nrho);
    calculate_cubic_spline_coefficients(
      adp_data.rho_r.data(), total_r_points, adp_data.dr,
      adp_data.rho_r_a.data(), adp_data.rho_r_b.data(),
      adp_data.rho_r_c.data(), adp_data.rho_r_d.data(), adp_data.Nelements, adp_data.nr);
    calculate_cubic_spline_coefficients(
      adp_data.phi_r.data(), total_pair_points, adp_data.dr,
      adp_data.phi_r_a.data(), adp_data.phi_r_b.data(), adp_data.phi_r_c.data(), adp_data.phi_r_d.data(),
      adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
    calculate_cubic_spline_coefficients(
      adp_data.u_r.data(), total_pair_points, adp_data.dr,
      adp_data.u_r_a.data(), adp_data.u_r_b.data(), adp_data.u_r_c.data(), adp_data.u_r_d.data(),
      adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
    calculate_cubic_spline_coefficients(
      adp_data.w_r.data(), total_pair_points, adp_data.dr,
      adp_data.w_r_a.data(), adp_data.w_r_b.data(), adp_data.w_r_c.data(), adp_data.w_r_d.data(),
      adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
  }
  
  // Output spline coefficients for comparison with LAMMPS
  output_spline_coefficients();
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
    
  a_func[n_points-1] = y_func[n_points-1];
  }
}

// GPU utility functions for interpolation
__device__ static void interpolate_adp(
  const double* a, const double* b, const double* c, const double* d,
  int index, double x_frac, double dx, double& y, double& yp)
{
  // Evaluate natural cubic spline on uniform grid with spacing dx
  // x_frac in [0,1). Physical offset t = x_frac * dx
  double t = x_frac * dx;
  y = a[index] + b[index] * t + c[index] * t * t + d[index] * t * t * t;
  // yp is derivative w.r.t physical variable (drho or dr)
  yp = b[index] + 2.0 * c[index] * t + 3.0 * d[index] * t * t;
}

// Construct LAMMPS-like spline coefficients but stored as (a,b,c,d)
void ADP::calculate_lammps_like_coefficients(
  const double* y, int /*n_total*/, double dx,
  double* a, double* b, double* c, double* d,
  int n_functions, int n_points)
{
  for (int func = 0; func < n_functions; ++func) {
    int off = func * n_points;
    const double* yf = y + off;
    double* af = a + off;
    double* bf = b + off;
    double* cf = c + off;
    double* df = d + off;

    std::vector<double> s(n_points, 0.0);
    // LAMMPS slopes (PairADP::interpolate):
    // s[0]   = f[1] - f[0]
    // s[1]   = 0.5*(f[2] - f[0])
    // s[i]   = ((f[i-2]-f[i+2]) + 8*(f[i+1]-f[i-1]))/12 for i in [2, n-3]
    // s[n-2] = 0.5*(f[n-1] - f[n-3])
    // s[n-1] = f[n-1] - f[n-2]
    if (n_points >= 2) s[0] = yf[1] - yf[0];
    if (n_points >= 3) s[1] = 0.5 * (yf[2] - yf[0]);
    for (int i = 2; i <= n_points - 3; ++i) {
      s[i] = ((yf[i-2] - yf[i+2]) + 8.0 * (yf[i+1] - yf[i-1])) / 12.0;
    }
    if (n_points >= 3) s[n_points - 2] = 0.5 * (yf[n_points - 1] - yf[n_points - 3]);
    if (n_points >= 2) s[n_points - 1] = yf[n_points - 1] - yf[n_points - 2];

    for (int m = 0; m < n_points - 1; ++m) {
      double dy = yf[m+1] - yf[m];
      double c4 = 3.0 * dy - 2.0 * s[m] - s[m+1];
      double c3 = s[m] + s[m+1] - 2.0 * dy;
      af[m] = yf[m];
      bf[m] = s[m] / dx;
      cf[m] = c4 / (dx * dx);
      df[m] = c3 / (dx * dx * dx);
    }
    af[n_points-1] = yf[n_points-1];
    bf[n_points-1] = 0.0;
    cf[n_points-1] = 0.0;
    df[n_points-1] = 0.0;
  }
}

// Output spline coefficients for comparison with LAMMPS
void ADP::output_spline_coefficients()
{
  printf("Writing GPUMD ADP spline coefficients to adp_spline_coeffs_gpumd.txt\n");
  
  FILE* fp = fopen("adp_spline_coeffs_gpumd.txt", "w");
  if (!fp) {
    printf("Warning: Cannot create adp_spline_coeffs_gpumd.txt\n");
    return;
  }
  
  // Output F_rho coefficients
  fprintf(fp, "--- F_rho ---\n");
  for (int elem = 0; elem < adp_data.Nelements; ++elem) {
    fprintf(fp, "Function %d:\n", elem);
    int offset = elem * adp_data.nrho;
    for (int i = 0; i < adp_data.nrho && i < 50; ++i) { // Limit output for readability
      double a = adp_data.F_rho_a[offset + i];
      double b = adp_data.F_rho_b[offset + i];
      double c = adp_data.F_rho_c[offset + i];
      double d = adp_data.F_rho_d[offset + i];
      fprintf(fp, "  pt %d: a=%.15e, b=%.15e, c=%.15e, d=%.15e\n", i, a, b, c, d);
    }
  }
  
  // Output rho_r coefficients  
  fprintf(fp, "\n--- rho_r ---\n");
  for (int elem = 0; elem < adp_data.Nelements; ++elem) {
    fprintf(fp, "Function %d:\n", elem);
    int offset = elem * adp_data.nr;
    for (int i = 0; i < adp_data.nr && i < 50; ++i) { // Limit output for readability
      double a = adp_data.rho_r_a[offset + i];
      double b = adp_data.rho_r_b[offset + i];
      double c = adp_data.rho_r_c[offset + i];
      double d = adp_data.rho_r_d[offset + i];
      fprintf(fp, "  pt %d: a=%.15e, b=%.15e, c=%.15e, d=%.15e\n", i, a, b, c, d);
    }
  }
  
  // Output phi_r coefficients
  fprintf(fp, "\n--- phi_r ---\n");
  int pair_idx = 0;
  for (int i = 0; i < adp_data.Nelements; ++i) {
    for (int j = 0; j <= i; ++j) {
      fprintf(fp, "Function %d (pair %d-%d):\n", pair_idx, i, j);
      int offset = pair_idx * adp_data.nr;
      for (int k = 0; k < adp_data.nr && k < 50; ++k) { // Limit output for readability
        double a = adp_data.phi_r_a[offset + k];
        double b = adp_data.phi_r_b[offset + k];
        double c = adp_data.phi_r_c[offset + k];
        double d = adp_data.phi_r_d[offset + k];
        fprintf(fp, "  pt %d: a=%.15e, b=%.15e, c=%.15e, d=%.15e\n", k, a, b, c, d);
      }
      pair_idx++;
    }
  }
  
  // Output u_r coefficients
  fprintf(fp, "\n--- u_r ---\n");
  pair_idx = 0;
  for (int i = 0; i < adp_data.Nelements; ++i) {
    for (int j = 0; j <= i; ++j) {
      fprintf(fp, "Function %d (pair %d-%d):\n", pair_idx, i, j);
      int offset = pair_idx * adp_data.nr;
      for (int k = 0; k < adp_data.nr && k < 50; ++k) { // Limit output for readability
        double a = adp_data.u_r_a[offset + k];
        double b = adp_data.u_r_b[offset + k];
        double c = adp_data.u_r_c[offset + k];
        double d = adp_data.u_r_d[offset + k];
        fprintf(fp, "  pt %d: a=%.15e, b=%.15e, c=%.15e, d=%.15e\n", k, a, b, c, d);
      }
      pair_idx++;
    }
  }
  
  // Output w_r coefficients
  fprintf(fp, "\n--- w_r ---\n");
  pair_idx = 0;
  for (int i = 0; i < adp_data.Nelements; ++i) {
    for (int j = 0; j <= i; ++j) {
      fprintf(fp, "Function %d (pair %d-%d):\n", pair_idx, i, j);
      int offset = pair_idx * adp_data.nr;
      for (int k = 0; k < adp_data.nr && k < 50; ++k) { // Limit output for readability
        double a = adp_data.w_r_a[offset + k];
        double b = adp_data.w_r_b[offset + k];
        double c = adp_data.w_r_c[offset + k];
        double d = adp_data.w_r_d[offset + k];
        fprintf(fp, "  pt %d: a=%.15e, b=%.15e, c=%.15e, d=%.15e\n", k, a, b, c, d);
      }
      pair_idx++;
    }
  }
  
  fclose(fp);
  printf("GPUMD spline coefficients written successfully.\n");
}

// GPU kernel to map user element types to ADP file element indices
__global__ void map_element_types(
  const int N,
  const int* g_type_user,
  int* g_type_mapped,
  const int* g_element_mapping,
  const int num_user_elements)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int user_type = g_type_user[i] - 1;  // Convert from 1-based to 0-based
    if (user_type >= 0 && user_type < num_user_elements) {
      g_type_mapped[i] = g_element_mapping[user_type];
    } else {
      g_type_mapped[i] = 0;  // Default to first element if invalid type
    }
  }
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
  const double rc,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_shift,
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
  double* g_pe,
  double* g_dbg_F,
  double* g_dbg_ADP)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  
  if (n1 < N2) {
    int NN = g_NN[n1];
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
      if (g_shift != nullptr) {
        int sx, sy, sz;
        const int code = g_shift[n1 + N * i1];
        decode_neighbor_shift(code, sx, sy, sz);
        if (code != 0) {
          const double shift_x = sx * box.cpu_h[0] + sy * box.cpu_h[1] + sz * box.cpu_h[2];
          const double shift_y = sx * box.cpu_h[3] + sy * box.cpu_h[4] + sz * box.cpu_h[5];
          const double shift_z = sx * box.cpu_h[6] + sy * box.cpu_h[7] + sz * box.cpu_h[8];
          delx -= shift_x;
          dely -= shift_y;
          delz -= shift_z;
        }
      } else {
        apply_mic(box, delx, dely, delz);
      }
      double d12 = sqrt(delx * delx + dely * dely + delz * delz);
      
      if (d12 < rc && d12 > 1e-12) {
        // LAMMPS clamped indexing
        double pp = d12 / dr + 1.0;
        int m = (int)pp; if (m < 1) m = 1; if (m > nr - 1) m = nr - 1;
        double frac = pp - m; if (frac > 1.0) frac = 1.0;
        int ir = m - 1;
        // In ADP (LAMMPS setfl), density contribution to atom i uses rho(r) of neighbor j's element type
        int rho_base = type2 * nr;
        int pair_base = get_pair_index(type1, type2, Nelements) * nr;
        
        // Calculate rho contribution (electron density)
        double rho_val, rho_deriv;
        interpolate_adp(rho_r_a, rho_r_b, rho_r_c, rho_r_d, 
                       rho_base + ir, frac, dr, rho_val, rho_deriv);
        rho += rho_val;
        
        // Calculate u and w values for dipole and quadruple terms.
        // In the ADP setfl format, the tabulated u2r and w2r arrays already
        // correspond to u(r) and w(r) directly (only the z2 array is scaled by r).
        double u2r_val, u2r_deriv, w2r_val, w2r_deriv;
        interpolate_adp(u_r_a, u_r_b, u_r_c, u_r_d,
                       pair_base + ir, frac, dr, u2r_val, u2r_deriv);
        interpolate_adp(w_r_a, w_r_b, w_r_c, w_r_d,
                       pair_base + ir, frac, dr, w2r_val, w2r_deriv);
        (void)u2r_deriv;
        (void)w2r_deriv;

          // ADP setfl convention: u(r) and w(r) are stored directly
          const double u_val = u2r_val;
          const double w_val = w2r_val;
          
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
    
    // Calculate embedding energy F(rho) and its derivative
    // Follow LAMMPS indexing/clamping exactly:
    // p = rho*rdrho + 1; m = int(p); m = clamp(m, 1, nrho-1); p -= m; p = min(p,1)
    int F_base = type1 * nrho;
    double pp = rho / drho + 1.0;
    int m = (int)pp;
    if (m < 1) m = 1;
    if (m > nrho - 1) m = nrho - 1;
    double frac = pp - m;
    if (frac > 1.0) frac = 1.0;
    int irho = m - 1; // convert to 0-based interval index
    double F = 0.0, Fp = 0.0;
    interpolate_adp(F_rho_a, F_rho_b, F_rho_c, F_rho_d,
                   F_base + irho, frac, drho, F, Fp);
    
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
    if (g_dbg_F) g_dbg_F[n1] += F;
    if (g_dbg_ADP) g_dbg_ADP[n1] += adp_energy;
    g_Fp[n1] = Fp;
    
    // Debug: Print detailed values for first few atoms
    if (n1 < 3) {
      printf("Atom %d: type=%d, rho=%.6f, F=%.6f, adp_energy=%.6f, total_E=%.6f\n",
             n1, type1, rho, F, adp_energy, F + adp_energy);
      printf("  mu: %.6f %.6f %.6f, mu_squared=%.6f\n", mu_x, mu_y, mu_z, mu_squared);
      printf("  lambda_diag: %.6f %.6f %.6f, lambda_offdiag: %.6f %.6f %.6f\n",
             lambda_xx, lambda_yy, lambda_zz, lambda_xy, lambda_xz, lambda_yz);
    }
    
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
  const double rc,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_shift,
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
  double* g_pe,
  double* g_dbg_PAIR)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  
  if (n1 < N2) {
    int NN = g_NN[n1];
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
  if (g_shift != nullptr) {
    int sx, sy, sz;
    const int code = g_shift[n1 + N * i1];
    decode_neighbor_shift(code, sx, sy, sz);
    if (code != 0) {
      const double shift_x = sx * box.cpu_h[0] + sy * box.cpu_h[1] + sz * box.cpu_h[2];
      const double shift_y = sx * box.cpu_h[3] + sy * box.cpu_h[4] + sz * box.cpu_h[5];
      const double shift_z = sx * box.cpu_h[6] + sy * box.cpu_h[7] + sz * box.cpu_h[8];
      delx -= shift_x;
      dely -= shift_y;
      delz -= shift_z;
    }
  } else {
    apply_mic(box, delx, dely, delz);
  }
  double d12 = sqrt(delx * delx + dely * dely + delz * delz);
      
      if (d12 < rc && d12 > 1e-12) {
        double d12_inv = 1.0 / d12;
        
        double pp = d12 / dr + 1.0;
        int m = (int)pp; if (m < 1) m = 1; if (m > nr - 1) m = nr - 1;
        double frac = pp - m; if (frac > 1.0) frac = 1.0;
        int ir = m - 1;
        
        int rho1_base = type1 * nr;
        int rho2_base = type2 * nr;
        int pair_base = get_pair_index(type1, type2, Nelements) * nr;
        
        // Get interpolated values and derivatives
        double rho1_val, rho1_deriv, rho2_val, rho2_deriv;
  double phi_rphi_val, phi_rphi_deriv, u2r_val, u2r_deriv, w2r_val, w2r_deriv;
        
        // rho1_*: density at atom i due to j (uses rho of type j)
        interpolate_adp(rho_r_a, rho_r_b, rho_r_c, rho_r_d,
                       rho2_base + ir, frac, dr, rho1_val, rho1_deriv);
        // rho2_*: density at atom j due to i (uses rho of type i)
        interpolate_adp(rho_r_a, rho_r_b, rho_r_c, rho_r_d,
                       rho1_base + ir, frac, dr, rho2_val, rho2_deriv);
        interpolate_adp(phi_r_a, phi_r_b, phi_r_c, phi_r_d,
                       pair_base + ir, frac, dr, phi_rphi_val, phi_rphi_deriv);
        interpolate_adp(u_r_a, u_r_b, u_r_c, u_r_d,
                       pair_base + ir, frac, dr, u2r_val, u2r_deriv);
        interpolate_adp(w_r_a, w_r_b, w_r_c, w_r_d,
                       pair_base + ir, frac, dr, w2r_val, w2r_deriv);
          // Convert r*phi(r) back to phi(r) and calculate z2r' - z2r/r which equals r*phi'(r)
          // phi_rphi_val is z2r = r*phi(r) from the table
          // phi_rphi_deriv is dz2r/dr from interpolation
          double rinv = 1.0 / d12;
          double phi_val = phi_rphi_val * rinv;  // phi(r) = z2r / r
          
          // ADP setfl stores u(r) and w(r) directly
          double u_val   = u2r_val;
          double u_deriv = u2r_deriv;
          double w_val   = w2r_val;
          double w_deriv = w2r_deriv;
          
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
          // 1+2. Central force contribution (pair + embedding density) in EAM/ADP form
          // phi'(r) = d(r*phi)/dr / r - (r*phi)/r^2
          double phi_deriv = phi_rphi_deriv * rinv - phi_rphi_val * (rinv * rinv);
          // Embedding density force: psip = Fp[i]*rhojp + Fp[j]*rhoip
          double psip = g_Fp[n1] * rho1_deriv + g_Fp[n2] * rho2_deriv;
          double scalar = -(phi_deriv + psip); // multiply by (del/r) below
          
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
          
          // Total force components: central (scalar * del/r) + ADP
          double force_total_x = scalar * delx * d12_inv + adpx;
          double force_total_y = scalar * dely * d12_inv + adpy;
          double force_total_z = scalar * delz * d12_inv + adpz;
          
          fx += force_total_x;
          fy += force_total_y;
          fz += force_total_z;
          
          // Add pair potential energy with 0.5 factor to avoid double counting (as in EAM)
          pe += 0.5 * phi_val;
          
          // Debug: Print pair energy for first atom
          if (n1 == 0) {
            printf("  Pair %d-%d: r=%.4f, phi=%.6f, pair_E=%.6f\n",
                   n1, n2, d12, phi_val, 0.5 * phi_val);
          }

          // Per-atom virial following LAMMPS ev_tally_xyz convention:
          // v = del \otimes f, where del = x_i - x_j (consistent with force calculation)
          const double v0 = delx * force_total_x;  // xx component
          const double v1 = dely * force_total_y;  // yy component  
          const double v2 = delz * force_total_z;  // zz component
          const double v3 = delx * force_total_y;  // xy component
          const double v4 = delx * force_total_z;  // xz component
          const double v5 = dely * force_total_z;  // yz component
          
          // GPUMD uses bidirectional neighbor lists (each pair appears in both atoms' lists)
          // Therefore, we need 0.5 factor to avoid double counting in stress calculation
          const double half = 0.5;
          s_sxx += half * v0;
          s_syy += half * v1;
          s_szz += half * v2;
          s_sxy += half * v3;
          s_sxz += half * v4;
          s_syz += half * v5; 
          s_syx += half * v3; // yx mirrors xy
          s_szx += half * v4; // zx mirrors xz
          s_szy += half * v5; // zy mirrors yz
        }
      }
    
    g_fx[n1] += fx;
    g_fy[n1] += fy;
    g_fz[n1] += fz;
  // Save virial tensor components to global array (consistent with EAM/NEP convention)
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
    if (g_dbg_PAIR) g_dbg_PAIR[n1] += pe;
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
  const int* neighbor_shift_ptr = nullptr;
  
  // Map user element types to ADP file element indices
  const int* type_ptr;
  if (element_mapping.empty()) {
    // Use original type array directly
    type_ptr = type.data();
  } else {
    // Map types using element mapping
    const int block_size = 256;
    const int grid_map = (number_of_atoms - 1) / block_size + 1;
    map_element_types<<<grid_map, block_size>>>(
      number_of_atoms,
      type.data(),
      adp_data.mapped_type.data(),
      adp_data.element_mapping_gpu.data(),
      static_cast<int>(element_mapping.size()));
    GPU_CHECK_KERNEL
    type_ptr = adp_data.mapped_type.data();
  }
  
  // Build neighbor list with configurable algorithm selection
  {
    int nbins[3];
    bool small_box = box.get_num_bins(0.5 * adp_data.rc, nbins);
    const bool extended_x = box.pbc_x && box.thickness_x > 0.0 && adp_data.rc > 0.5 * box.thickness_x;
    const bool extended_y = box.pbc_y && box.thickness_y > 0.0 && adp_data.rc > 0.5 * box.thickness_y;
    const bool extended_z = box.pbc_z && box.thickness_z > 0.0 && adp_data.rc > 0.5 * box.thickness_z;
    const bool requires_extended_images = extended_x || extended_y || extended_z;

    // Choose neighbor algorithm based on user preference and box size
    bool use_brute_force = !use_linear_neighbor_ || small_box || requires_extended_images;
    const int max_neighbors = adp_data.NL.size() / number_of_atoms;
    neighbor_shift_ptr = nullptr;
    
    if (!use_brute_force) {
      // Use O(N) linear complexity cell list algorithm
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
    } else {
      // Use O(N^2) brute force algorithm for small boxes or when explicitly requested
      const double* gx = position_per_atom.data();
      const double* gy = position_per_atom.data() + number_of_atoms;
      const double* gz = position_per_atom.data() + number_of_atoms * 2;
      const int block = 256;
      const int grid = (N2 - N1 - 1) / block + 1;
      build_neighbor_ON2<<<grid, block>>>(
        box,
        number_of_atoms,
        N1,
        N2,
        adp_data.rc * adp_data.rc,
        gx, gy, gz,
        adp_data.NN.data(),
        adp_data.NL.data(),
        adp_data.NL_shift.data(),
        max_neighbors);
      GPU_CHECK_KERNEL
      neighbor_shift_ptr = adp_data.NL_shift.data();
    }
  }


  // Zero only internal accumulators needed for this step; top-level framework already zeroed force/potential/virial.
  adp_data.mu.fill(0.0);      // size 3N
  adp_data.lambda.fill(0.0);  // size 6N
  adp_data.Fp.fill(0.0);      // size N
  // Zero debug accumulators per compute call to avoid double counting across pre-run and run
#if ADP_DEBUG_PRINT
  adp_data.dbg_F.fill(0.0);
  adp_data.dbg_ADP.fill(0.0);
  adp_data.dbg_PAIR.fill(0.0);
#endif
  
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
    adp_data.rc,
    box,
    adp_data.NN.data(),
  adp_data.NL.data(),
  neighbor_shift_ptr,
    type_ptr,
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
    potential_per_atom.data(),
    adp_data.dbg_F.data(),
    adp_data.dbg_ADP.data());
  GPU_CHECK_KERNEL
  
  // Step 2: Calculate forces
  find_force_adp_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    adp_data.Nelements,
    adp_data.nr,
    adp_data.dr,
    adp_data.rc,
    box,
    adp_data.NN.data(),
  adp_data.NL.data(),
  neighbor_shift_ptr,
    type_ptr,
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
    potential_per_atom.data(),
    adp_data.dbg_PAIR.data());
  GPU_CHECK_KERNEL

#if ADP_DEBUG_PRINT
  // Reduce and print energy breakdown for quick diagnostics
  std::vector<double> hF(number_of_atoms), hA(number_of_atoms), hP(number_of_atoms);
  adp_data.dbg_F.copy_to_host(hF.data());
  adp_data.dbg_ADP.copy_to_host(hA.data());
  adp_data.dbg_PAIR.copy_to_host(hP.data());
  double sF=0, sA=0, sP=0, sT=0;
  for (int i=0;i<number_of_atoms;++i){ sF+=hF[i]; sA+=hA[i]; sP+=hP[i]; }
  std::vector<double> hE(number_of_atoms);
  potential_per_atom.copy_to_host(hE.data());
  for (int i=0;i<number_of_atoms;++i) sT+=hE[i];

  std::printf("=== GPUMD ADP energy breakdown ===\n");
  std::printf("Total energy:      %.12f eV\n", sT);
  std::printf("  Pair energy:     %.12f eV\n", sP);
  std::printf("  Embedding energy:%.12f eV\n", sF);
  std::printf("  ADP energy:      %.12f eV\n", sA);

  std::vector<int> hNN(number_of_atoms);
  adp_data.NN.copy_to_host(hNN.data());
  long long total_neighbors = 0;
  for (int i = 0; i < number_of_atoms; ++i) total_neighbors += hNN[i];
  double avg_neighbors = number_of_atoms > 0 ? static_cast<double>(total_neighbors) / number_of_atoms : 0.0;
  std::printf("Average neighbor count: %.6f\n", avg_neighbors);
  for (int i = 0; i < 3 && i < number_of_atoms; ++i) {
    std::printf("  NN[%d] = %d\n", i, hNN[i]);
  }

  std::vector<int> hNL;
  hNL.resize(adp_data.NL.size());
  adp_data.NL.copy_to_host(hNL.data());
  for (int atom = 0; atom < 3 && atom < number_of_atoms; ++atom) {
    std::printf("  Neighbors of atom %d:", atom);
    for (int k = 0; k < hNN[atom]; ++k) {
      int neighbor = hNL[k * number_of_atoms + atom];
      std::printf(" %d", neighbor);
    }
    std::printf("\n");
  }

  FILE* fpb=fopen("adp_energy_breakdown.txt","w");
  if(fpb){ fprintf(fpb,"EF=%.12g EADP=%.12g EPAIR=%.12g ETOT=%.12g\n", sF, sA, sP, sT); fclose(fpb);} 
  // Also dump per-atom contributions for detailed comparison (F, ADP, PAIR, TOTAL)
  FILE* fpp = fopen("adp_energy_per_atom.txt","w");
  if (fpp) {
    for (int i = 0; i < number_of_atoms; ++i) {
      fprintf(fpp, "% .15e % .15e % .15e % .15e\n", hF[i], hA[i], hP[i], hE[i]);
    }
    fclose(fpp);
  }
#endif
}
