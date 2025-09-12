// CUDA neighbor list (matrix method) inspired by torch_neighborlist.py
// Implements a fixed-width neighbor matrix NL (column-major, [MN, N]) and NN per atom.
// Uses GPUMD Box + cell-list routines for performance and MIC correctness.

#pragma once

#include "model/box.cuh"
#include "force/neighbor.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <cuda_runtime.h>

// Kernel: build neighbor list limited to max_neighbors (MN) using existing cell list data
static __global__ void ker_build_neighbor_matrix(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int * __restrict__ cell_counts,
  const int * __restrict__ cell_count_sum,
  const int * __restrict__ cell_contents,
  int * __restrict__ NN,
  int * __restrict__ NL,
  const double * __restrict__ x,
  const double * __restrict__ y,
  const double * __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square,
  const int MN)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];

  int count = 0;

  int cell_id, cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 1 : 0;
  const int y_lim = box.pbc_y ? 1 : 0;
  const int x_lim = box.pbc_x ? 1 : 0;

  for (int k = -z_lim; k <= z_lim; ++k) {
    for (int j = -y_lim; j <= y_lim; ++j) {
      for (int i = -x_lim; i <= x_lim; ++i) {
        int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
        if (cell_id_x + i < 0) neighbor_cell += nx;
        if (cell_id_x + i >= nx) neighbor_cell -= nx;
        if (cell_id_y + j < 0) neighbor_cell += ny * nx;
        if (cell_id_y + j >= ny) neighbor_cell -= ny * nx;
        if (cell_id_z + k < 0) neighbor_cell += nz * ny * nx;
        if (cell_id_z + k >= nz) neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
        const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = cell_contents[num_atoms_previous_cells + m];
          if (n2 >= N1 && n2 < N2 && n1 != n2) {
            double dx = x[n2] - x1;
            double dy = y[n2] - y1;
            double dz = z[n2] - z1;
            apply_mic(box, dx, dy, dz);
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < cutoff_square) {
              if (count < MN) NL[count * N + n1] = n2; // column-major
              ++count;
            }
          }
        }
      }
    }
  }

  NN[n1] = min(count, MN);
}

// Host wrapper: build matrix neighbor list using cell list
// - Operates entirely on device arrays passed by caller (from GPUMD data structures)
// - Keeps NL in column-major order [MN, N] as used in GPUMD
inline void build_neighbor_list_torch_style(
  int N,
  const double *d_x,
  const double *d_y,
  const double *d_z,
  Box &box,
  double cutoff,
  int max_neighbors,
  int *d_NN,
  int *d_NL)
{
  // Choose a smaller cell size to ensure neighbors lie within adjacent cells (like GPUMD does)
  const double rc_cell = 0.5 * cutoff;
  int num_bins[3];
  box.get_num_bins(rc_cell, num_bins);

  // Temporary cell list storage
  GPU_Vector<int> cell_count, cell_count_sum, cell_contents;
  cell_count.resize(num_bins[0] * num_bins[1] * num_bins[2]);
  cell_count_sum.resize(num_bins[0] * num_bins[1] * num_bins[2]);
  cell_contents.resize(N);

  // Build a packed position array [x, y, z] with length 3N (device-to-device copy)
  GPU_Vector<double> pos;
  pos.resize(3 * N);
  // Device-to-device copies into packed layout
  CHECK(gpuMemcpy(pos.data() + 0 * N, d_x, sizeof(double) * N, gpuMemcpyDeviceToDevice));
  CHECK(gpuMemcpy(pos.data() + 1 * N, d_y, sizeof(double) * N, gpuMemcpyDeviceToDevice));
  CHECK(gpuMemcpy(pos.data() + 2 * N, d_z, sizeof(double) * N, gpuMemcpyDeviceToDevice));

  // Now rebuild cell list using the non-stream overload that takes packed positions
  find_cell_list(
    rc_cell,
    num_bins,
    box,
    pos,
    cell_count,
    cell_count_sum,
    cell_contents);

  // Clear outputs before filling
  CHECK(gpuMemset(d_NN, 0, sizeof(int) * N));
  // NL is written sparsely by columns; leaving garbage beyond NN is fine for our use.

  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 2.0 / cutoff; // consistent with rc_cell = 0.5*cutoff
  const double cutoff_sq = cutoff * cutoff;

  ker_build_neighbor_matrix<<<grid_size, block_size>>>(
    box,
    N,
    0,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    d_NN,
    d_NL,
    d_x,
    d_y,
    d_z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv,
    cutoff_sq,
    max_neighbors);
  GPU_CHECK_KERNEL

  // Sort each column's neighbor indices for determinism
  const int MN = max_neighbors;
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int)>>>(N, d_NN, d_NL);
  GPU_CHECK_KERNEL
}
