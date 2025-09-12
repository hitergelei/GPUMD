#pragma once

// Torch-style neighbor list implementation for ADP
// Based on PyTorch neighborlist algorithm for more accurate neighbor counting

void build_neighbor_list_torch_style(
    int N,
    const double* h_x, const double* h_y, const double* h_z,
    double lx, double ly, double lz,
    double cutoff,
    int max_neighbors,
    int* h_NN, int* h_NL);

// Utility structure for torch-style neighbor list
struct TorchNeighborList {
    int* NN;            // neighbor count per atom
    int* NL;            // neighbor list (column-major: [MN, N])
    int N;              // number of atoms
    int MN;             // max neighbors per atom
    double cutoff;      // cutoff distance
    
    void allocate(int num_atoms, int max_neighbors) {
        N = num_atoms;
        MN = max_neighbors;
        NN = new int[N];
        NL = new int[MN * N];
    }
    
    void deallocate() {
        delete[] NN;
        delete[] NL;
    }
};
