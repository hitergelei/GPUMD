# GCMC (Grand Canonical Monte Carlo) Implementation for GPUMD

## Overview
This implementation provides a comprehensive Grand Canonical Monte Carlo (GCMC) ensemble for GPUMD, supporting insertion, deletion, and displacement moves with full GPU acceleration and advanced sampling techniques.

## Features

### Core GCMC Functionality
- **Particle Insertion**: Random insertion of atoms with overlap checking and energy-based acceptance
- **Particle Deletion**: Random deletion with proper GCMC acceptance criteria
- **Particle Displacement**: Metropolis-based atomic displacement moves
- **GPU Acceleration**: All critical operations optimized for GPU execution

### Advanced Sampling Methods
- **Cluster Moves**: Collective displacement of atom clusters for enhanced sampling
- **Volume Changes**: NPT-like volume fluctuations
- **Adaptive Scaling**: Automatic adjustment of move parameters based on acceptance rates
- **Parallel Optimization**: Batch overlap checking and energy calculations

### System Validation
- **Overlap Detection**: Prevents unphysical atom configurations
- **Energy Conservation**: Validates energy calculation consistency
- **State Validation**: Comprehensive system state checking
- **Error Handling**: Robust error detection and recovery

## Usage

### Basic Setup
```cpp
// Initialize species and parameters
std::vector<std::string> species = {"Cu", "Ag"};
std::vector<int> types = {0, 1};
std::vector<double> chemical_potentials = {-3.5, -2.8}; // eV
double max_displacement = 0.1; // Angstroms
int num_mc_steps = 1000;

// Create GCMC ensemble
MC_Ensemble_GCMC gcmc(
    param, num_param,
    num_mc_steps,
    species,
    types,
    chemical_potentials,
    max_displacement
);

// Run GCMC
gcmc.compute(md_step, temperature, atom, box, groups, grouping_method, group_id);
```

### Key Parameters

#### Chemical Potentials
- Controls the insertion/deletion balance for each species
- Higher chemical potential favors insertion
- Units: eV (consistent with GPUMD energy units)

#### Temperature
- Controls thermal fluctuations
- Affects acceptance probabilities for all move types
- Units: Kelvin

#### Maximum Displacement
- Controls the step size for atomic displacements
- Automatically adjusted to maintain ~50% acceptance rate
- Units: Angstroms

### Output Files

#### gcmc.out
Contains detailed statistics for each MD step:
- Overall acceptance rate
- Number of active atoms
- Species-specific insertion/deletion rates
- Displacement acceptance rates
- Current maximum displacement
- Temperature
- Per-species atom counts

Example output:
```
# GCMC Statistics Output
# Step Overall_Acceptance N_active Insertion_Rate Deletion_Rate Displacement_Rate Max_Displacement Temperature N_Cu N_Ag
0  0.450  1250  0.25  0.30  0.55  0.095  300.0  800  450
1  0.465  1248  0.28  0.32  0.52  0.098  300.0  798  450
```

## Implementation Details

### Energy Calculations
- Uses GPUMD's NEP (Neural Evolution Potential) for accurate energy evaluation
- Supports all potential types available in GPUMD
- Efficient local energy calculations for MC moves

### Boundary Conditions
- Full support for periodic boundary conditions
- Proper minimum image convention for all distance calculations
- Automatic handling of triclinic simulation cells

### Memory Management
- Dynamic atom array resizing
- GPU memory optimization
- Efficient data transfer between CPU and GPU

### Parallel Optimization
- Batch processing of insertion candidates
- Shared memory optimization for overlap checking
- Reduction algorithms for energy summation

## Performance Considerations

### GPU Utilization
- Most computationally intensive operations run on GPU
- Optimized kernel launches with appropriate block sizes
- Minimized CPU-GPU data transfers

### Scalability
- Linear scaling with system size for most operations
- Efficient neighbor list management
- Batch processing for improved throughput

### Memory Usage
- Configurable maximum atom limit
- Dynamic memory allocation
- GPU memory pooling for frequent allocations

## Advanced Features

### Adaptive Algorithms
- Automatic displacement scaling based on acceptance rates
- Dynamic load balancing for multi-component systems
- Intelligent move type selection

### Enhanced Sampling
- Cluster move algorithms for correlated motion
- Volume fluctuations for pressure effects
- Framework for bias potential integration

### Analysis Tools
- Real-time crystallization detection
- Energy conservation monitoring
- System state validation

## Limitations and Future Work

### Current Limitations
- Local energy calculations not fully optimized
- Limited bias potential implementations
- No parallel tempering support yet

### Planned Enhancements
- Enhanced local energy algorithms
- Advanced bias potential methods (metadynamics, umbrella sampling)
- Multi-replica parallel tempering
- Machine learning-guided move proposals

## Troubleshooting

### Common Issues
1. **Low acceptance rates**: Adjust chemical potentials or temperature
2. **System instability**: Check for atom overlaps, reduce displacement magnitude
3. **Memory errors**: Increase max_atoms parameter or reduce system size
4. **Energy inconsistencies**: Verify potential parameters and neighbor lists

### Error Messages
- "Cannot use small box for GCMC": Increase simulation cell size
- "System validation failed": Check for atom overlaps or infinite coordinates
- "Atom overlap detected": Reduce displacement magnitude or insertion rate

## References
1. Frenkel, D. & Smit, B. Understanding Molecular Simulation (Academic Press, 2002)
2. Adams, D. J. Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid. Mol. Phys. 29, 307-311 (1975)
3. Sadigh, B. et al. Scalable parallel Monte Carlo algorithm for atomistic simulations. Phys. Rev. B 85, 184203 (2012)
