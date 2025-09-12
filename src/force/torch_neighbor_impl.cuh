#pragma once
#include <cuda_runtime.h>
#include <cmath>

// Simplified torch-style neighbor list for ADP integration
// Based on the PyTorch algorithm but optimized for GPUMD integration

struct TorchNeighborConfig {
    double cutoff;
    int max_neighbors;
    bool use_torch_style;
};

// CUDA kernels for torch-style neighbor list
__constant__ double d_torch_cutoff;
__constant__ double d_torch_cutoff_sq;
__constant__ int d_torch_max_neighbors;

// Wrap positions (simplified for cubic boxes)
__device__ void torch_wrap_position(double& x, double& y, double& z, 
                                   double lx, double ly, double lz) {
    const double eps = 1e-7;
    
    // Wrap using fractional coordinates
    double fx = x / lx + eps;
    fx = fx - floor(fx);
    x = (fx - eps) * lx;
    
    double fy = y / ly + eps;
    fy = fy - floor(fy);
    y = (fy - eps) * ly;
    
    double fz = z / lz + eps;
    fz = fz - floor(fz);
    z = (fz - eps) * lz;
}

// Torch-style neighbor finding kernel
__global__ void find_neighbors_torch_style(
    int N,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    double lx, double ly, double lz,
    int* NN, int* NL)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 >= N) return;
    
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    
    // Wrap position
    torch_wrap_position(x1, y1, z1, lx, ly, lz);
    
    int neighbor_count = 0;
    
    // Calculate padding based on cutoff (like PyTorch)
    double volume = lx * ly * lz;
    int padding_a = max(1, (int)ceil(d_torch_cutoff * ly * lz / volume));
    int padding_b = max(1, (int)ceil(d_torch_cutoff * lz * lx / volume));
    int padding_c = max(1, (int)ceil(d_torch_cutoff * lx * ly / volume));
    
    // Search through periodic images (similar to PyTorch padding)
    for (int pa = -padding_a; pa <= padding_a && neighbor_count < d_torch_max_neighbors; pa++) {
        for (int pb = -padding_b; pb <= padding_b && neighbor_count < d_torch_max_neighbors; pb++) {
            for (int pc = -padding_c; pc <= padding_c && neighbor_count < d_torch_max_neighbors; pc++) {
                
                // Check all other atoms in this periodic image
                for (int n2 = 0; n2 < N && neighbor_count < d_torch_max_neighbors; n2++) {
                    if (n1 == n2 && pa == 0 && pb == 0 && pc == 0) continue;  // Skip self
                    
                    double x2 = x[n2] + pa * lx;
                    double y2 = y[n2] + pb * ly;
                    double z2 = z[n2] + pc * lz;
                    
                    double dx = x2 - x1;
                    double dy = y2 - y1;
                    double dz = z2 - z1;
                    
                    double dist_sq = dx * dx + dy * dy + dz * dz;
                    
                    if (dist_sq < d_torch_cutoff_sq && dist_sq > 1e-8) {
                        // Store in column-major format like GPUMD
                        NL[neighbor_count * N + n1] = n2;
                        neighbor_count++;
                    }
                }
            }
        }
    }
    
    NN[n1] = neighbor_count;
}

// Host wrapper for torch-style neighbor list
void build_torch_neighbor_list(
    int N,
    const double* h_x, const double* h_y, const double* h_z,
    double lx, double ly, double lz,
    double cutoff, int max_neighbors,
    int* h_NN, int* h_NL)
{
    // Copy constants to GPU
    cudaMemcpyToSymbol(d_torch_cutoff, &cutoff, sizeof(double));
    double cutoff_sq = cutoff * cutoff;
    cudaMemcpyToSymbol(d_torch_cutoff_sq, &cutoff_sq, sizeof(double));
    cudaMemcpyToSymbol(d_torch_max_neighbors, &max_neighbors, sizeof(int));
    
    // Allocate device memory
    double *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_z, N * sizeof(double));
    
    cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(double), cudaMemcpyHostToDevice);
    
    int *d_NN, *d_NL;
    cudaMalloc(&d_NN, N * sizeof(int));
    cudaMalloc(&d_NL, max_neighbors * N * sizeof(int));
    
    cudaMemset(d_NN, 0, N * sizeof(int));
    cudaMemset(d_NL, 0, max_neighbors * N * sizeof(int));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    find_neighbors_torch_style<<<grid_size, block_size>>>(
        N, d_x, d_y, d_z, lx, ly, lz, d_NN, d_NL);
    
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_NN, d_NN, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_NL, d_NL, max_neighbors * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_NN); cudaFree(d_NL);
    
    printf("Torch-style neighbor list: ");
    int total_neighbors = 0;
    for (int i = 0; i < N; i++) {
        total_neighbors += h_NN[i];
    }
    printf("avg_neighbors=%.1f\n", (double)total_neighbors / N);
}
