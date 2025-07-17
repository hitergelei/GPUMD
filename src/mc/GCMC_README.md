# GCMC (Grand Canonical Monte Carlo) Implementation for GPUMD

## Overview
This implementation provides a comprehensive Grand Canonical Monte Carlo (GCMC) ensemble for GPUMD, supporting insertion, deletion, and displacement moves with full GPU acceleration and advanced sampling techniques. The implementation includes state-of-the-art enhanced sampling methods, real-time analysis tools, and comprehensive performance monitoring capabilities for materials science applications.

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
- **Wang-Landau Sampling**: Enhanced exploration of density space with automatic convergence
- **Umbrella Sampling**: Biased sampling along coordination number order parameter
- **Parallel Tempering**: Temperature exchange for improved sampling efficiency

### System Validation
- **Overlap Detection**: Prevents unphysical atom configurations
- **Energy Conservation**: Validates energy calculation consistency
- **State Validation**: Comprehensive system state checking
- **Error Handling**: Robust error detection and recovery
- **Performance Monitoring**: Real-time statistics and diagnostics
- **Checkpoint System**: State saving and recovery capabilities

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

### Advanced Usage Examples

#### Enhanced Sampling Setup
```cpp
// Enable advanced sampling methods
MC_Ensemble_GCMC gcmc(param, num_param, num_mc_steps, species, types, mu, max_disp);

// The enhanced sampling methods are automatically applied during computation:
// - Wang-Landau sampling every 5000 steps for density exploration
// - Umbrella sampling with coordination number windows
// - Parallel tempering with temperature ladder
// - Adaptive chemical potential adjustment
// - Real-time crystallization detection

// Load previous checkpoint if available
gcmc.load_checkpoint("gcmc_checkpoint.dat");

// Run simulation with automatic statistics
gcmc.compute(md_step, temperature, atom, box, groups, grouping_method, group_id);

// Print performance statistics
gcmc.print_performance_statistics();
```

#### Bias Potential Applications
```cpp
// The implementation includes several bias potentials:
// 1. Density bias: Maintains target density during simulation
// 2. Surface bias: Applies energy penalty near surfaces
// 3. Coordination bias: Used in umbrella sampling

// These are automatically applied and statistics are saved to bias_statistics.out
```

#### Crystallization Monitoring
```cpp
// Real-time crystallization detection is performed every 5000 steps
// Results are saved to crystallization.out with:
// - Q4 and Q6 bond order parameters
// - Structure type identification (FCC, BCC, HCP, liquid)
// - Crystal fraction evolution
// - Automatic warnings for significant crystallization
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

#### bias_statistics.out
Records bias potential information:
- System density and volume
- Surface atom counts
- Energy bias contributions
- Real-time bias statistics

#### wang_landau_dos.out
Wang-Landau density of states:
- Density bins and corresponding DOS values
- Convergence information
- Final density distribution

#### umbrella_sampling.out
Umbrella sampling data:
- Coordination number windows
- Target vs actual coordination
- Bias energy contributions

#### crystallization.out
Crystallization analysis:
- Bond order parameters (Q4, Q6)
- Structure type identification
- Crystal fraction evolution
- FCC/BCC/HCP counts

#### gcmc_checkpoint.dat
System checkpoint for restart:
- All statistical counters
- Current chemical potentials
- Adaptive parameters

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
- Integrated bias potential framework for enhanced sampling
- Energy conservation monitoring and validation

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
- Wang-Landau sampling for density space exploration
- Umbrella sampling along coordination number
- Adaptive chemical potential adjustment
- Multi-temperature parallel tempering

### Analysis Tools
- Real-time crystallization detection using bond order parameters
- Energy conservation monitoring
- System state validation
- Performance statistics and diagnostics
- Automatic checkpoint generation
- Structure type identification (FCC/BCC/HCP)
- Coordination number analysis

## Limitations and Future Work

### Current Limitations
- Local energy calculations use simplified approach for efficiency
- Bias potential methods are demonstration implementations
- Parallel tempering limited to single replica with temperature ladder
- Wang-Landau and umbrella sampling use basic order parameters

### Planned Enhancements
- Full local energy calculations with partial NEP evaluation
- Advanced metadynamics implementation
- Multi-replica parallel tempering with replica exchange
- Machine learning-guided move proposals
- Advanced order parameters for complex phase detection
- GPU-accelerated bias potential calculations

## Troubleshooting

### Common Issues
1. **Low acceptance rates**: Adjust chemical potentials or temperature
2. **System instability**: Check for atom overlaps, reduce displacement magnitude
3. **Memory errors**: Increase max_atoms parameter or reduce system size
4. **Energy inconsistencies**: Verify potential parameters and neighbor lists
5. **Poor sampling**: Enable enhanced sampling methods (Wang-Landau, umbrella sampling)
6. **Crystallization warnings**: Monitor bond order parameters and adjust conditions

### Error Messages
- "Cannot use small box for GCMC": Increase simulation cell size
- "System validation failed": Check for atom overlaps or infinite coordinates
- "Atom overlap detected": Reduce displacement magnitude or insertion rate
- "Energy calculation inconsistency": Check potential parameters and system state
- "Wang-Landau sampling converged": Normal completion of enhanced sampling
- "Significant crystallization detected": System may be approaching phase transition

### Performance Tips
1. **Use checkpoints**: Save and restore system state for long runs
2. **Monitor statistics**: Check acceptance rates and adjust parameters accordingly
3. **Enable bias potentials**: Use density or surface bias for specific applications
4. **Tune enhanced sampling**: Adjust Wang-Landau and umbrella sampling parameters
5. **GPU optimization**: Ensure adequate GPU memory for large systems

## References
1. Frenkel, D. & Smit, B. Understanding Molecular Simulation (Academic Press, 2002)
2. Adams, D. J. Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid. Mol. Phys. 29, 307-311 (1975)
3. Sadigh, B. et al. Scalable parallel Monte Carlo algorithm for atomistic simulations. Phys. Rev. B 85, 184203 (2012)
4. Wang, F. & Landau, D. P. Efficient, multiple-range random walk algorithm to calculate the density of states. Phys. Rev. Lett. 86, 2050-2053 (2001)
5. Torrie, G. M. & Valleau, J. P. Nonphysical sampling distributions in Monte Carlo free-energy estimation: Umbrella sampling. J. Comp. Phys. 23, 187-199 (1977)
6. Steinhardt, P. J., Nelson, D. R. & Ronchetti, M. Bond-orientational order in liquids and glasses. Phys. Rev. B 28, 784-805 (1983)
