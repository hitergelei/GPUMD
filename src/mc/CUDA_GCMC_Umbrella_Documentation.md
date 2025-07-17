# CUDA GCMC with Umbrella Sampling Enhancement

## Overview

This implementation enhances the CUDA-accelerated Grand Canonical Monte Carlo (GCMC) framework with umbrella sampling functionality based on the LAMMPS `fix_gcmc_umbrella` implementation. Umbrella sampling biases the simulation toward specific atom counts, enabling enhanced sampling of rare configurations and better convergence for systems with strong adsorption/desorption barriers.

## Theoretical Background

### Umbrella Sampling Theory

Umbrella sampling introduces a bias potential to enhance sampling around a target configuration. For GCMC simulations, this typically means biasing toward a specific number of atoms of a particular species.

The umbrella bias potential is:
```
U_bias(N) = 0.5 * k * (N - N₀)²
```

Where:
- `N` = current number of atoms of target species
- `N₀` = target number of atoms
- `k` = umbrella force constant (spring constant)

### Modified Acceptance Criteria

The umbrella bias modifies the standard GCMC acceptance criteria:

**For insertion moves:**
```
ln(P_acc) = μβ - ΔE·β + ln(V/(N+1)) + umbrella_bias
```

**For deletion moves:**
```
ln(P_acc) = -μβ - ΔE·β + ln(N/V) + umbrella_bias
```

Where the umbrella bias contribution is:
```
umbrella_bias = -0.5 * k * [(N_after - N₀)² - (N_before - N₀)²]
```

This follows the LAMMPS implementation exactly.

## Implementation Details

### Key Classes and Methods

#### MC_Ensemble_CUDA_GCMC Class Extensions

**New Member Variables:**
```cpp
// Umbrella sampling parameters
bool enable_umbrella_sampling = false;
bool enable_adaptive_umbrella = false;
int umbrella_target_atoms = 0;
double umbrella_force_constant = 1.0;
double umbrella_bias_energy = 0.0;
int target_type = 0; // Atom type for umbrella sampling
```

**Core Umbrella Methods:**
1. `umbrella_sampling_cuda()` - Main umbrella sampling routine
2. `calculate_umbrella_bias_energy()` - Compute bias potential
3. `apply_umbrella_bias_to_insertion()` - Modify insertion acceptance
4. `apply_umbrella_bias_to_deletion()` - Modify deletion acceptance
5. `update_umbrella_parameters()` - Update bias energy and parameters
6. `adaptive_umbrella_tuning()` - Automatic force constant optimization
7. `write_umbrella_statistics()` - Output umbrella sampling statistics

### Integration with GCMC Operations

The umbrella sampling is integrated into the core GCMC operations:

1. **During Insertion Attempts:**
   - Count current atoms of target type
   - Calculate umbrella bias for N → N+1 transition
   - Add bias to acceptance probability

2. **During Deletion Attempts:**
   - Count current atoms of target type
   - Calculate umbrella bias for N → N-1 transition
   - Add bias to acceptance probability

3. **After Each Monte Carlo Step:**
   - Update umbrella bias energy
   - Apply adaptive tuning if enabled
   - Write statistics periodically

### Adaptive Force Constant Tuning

The implementation includes adaptive tuning of the umbrella force constant:

```cpp
void MC_Ensemble_CUDA_GCMC::adaptive_umbrella_tuning()
{
  double acceptance_rate = (num_accepted_insertions + num_accepted_deletions) / 
                          (attempted_insertions + attempted_deletions);
  
  // Reduce force constant if acceptance too low
  if (acceptance_rate < 0.1 && umbrella_force_constant > 0.1) {
    umbrella_force_constant *= 0.95;
  }
  
  // Increase force constant if acceptance too high
  if (acceptance_rate > 0.7 && umbrella_force_constant < 10.0) {
    umbrella_force_constant *= 1.05;
  }
}
```

## Usage Examples

### Basic Umbrella Sampling Setup

```cpp
MC_Ensemble_CUDA_GCMC gcmc;

// Enable umbrella sampling
gcmc.enable_umbrella_sampling = true;
gcmc.target_type = 0;                    // First atom type
gcmc.umbrella_target_atoms = 50;         // Target 50 atoms
gcmc.umbrella_force_constant = 2.0;      // Spring constant
gcmc.enable_adaptive_umbrella = true;    // Adaptive tuning

// Configure species and chemical potentials
std::vector<std::string> species = {"Ar"};
std::vector<double> mu = {-2.5};
gcmc.set_chemical_potentials(species, mu);

// Run simulation
gcmc.sample(atom, box, groups, grouping_method, group_id, temperature);
```

### Multiple Umbrella Windows

For constructing free energy profiles, run multiple simulations with different target counts:

```cpp
std::vector<int> targets = {10, 20, 30, 40, 50};
for (int target : targets) {
  MC_Ensemble_CUDA_GCMC gcmc;
  gcmc.enable_umbrella_sampling = true;
  gcmc.umbrella_target_atoms = target;
  gcmc.umbrella_force_constant = 2.0;
  
  // Run simulation for this window
  gcmc.sample(atom, box, groups, grouping_method, group_id, temperature);
}
```

## Parameter Selection Guidelines

### Force Constant (k)

- **Small k (0.1-1.0):** Weak bias, broad sampling around target
- **Medium k (1.0-5.0):** Moderate bias, good for most applications
- **Large k (5.0-20.0):** Strong bias, tight sampling around target

Start with k ≈ 1-2 and use adaptive tuning.

### Target Atom Count (N₀)

Choose based on the physical phenomenon of interest:
- Adsorption isotherms: Multiple windows covering expected range
- Phase transitions: Target counts near transition regions
- Rare events: Intermediate states along reaction coordinate

### Adaptive Tuning

Enable adaptive tuning for:
- Unknown optimal force constants
- Long simulations where conditions change
- Automatic optimization of sampling efficiency

## Output and Analysis

The implementation provides several types of output:

### Statistics Output
```
UMBRELLA STEP 10000: Target=50 Current=48 BiasEnergy=2.000 ForceConstant=2.000
```

### Key Quantities to Monitor

1. **Current atom count vs. target:** Should fluctuate around target
2. **Bias energy:** Minimum at target count
3. **Acceptance rates:** Should be reasonable (10-70%)
4. **Force constant evolution:** With adaptive tuning

## Comparison with LAMMPS Implementation

This implementation follows the LAMMPS `fix_gcmc_umbrella` closely:

### Similarities
- Identical bias potential formula
- Same acceptance criterion modification
- Compatible parameter definitions
- Similar adaptive tuning concepts

### Enhancements
- Full GPU acceleration with CUDA
- Batch processing for better GPU utilization
- Integration with NEP potential calculator
- Advanced memory management for large systems

## Performance Considerations

### GPU Optimization
- Umbrella calculations use minimal GPU memory
- Bias computations are CPU-based (lightweight)
- No significant impact on CUDA kernel performance

### Computational Overhead
- Umbrella bias calculation: O(1) per move
- Atom counting: O(N) per MC step
- Overall overhead: < 5% of total simulation time

## Best Practices

1. **Start Simple:** Begin with single windows before multiple windows
2. **Use Adaptive Tuning:** Let the algorithm optimize force constants
3. **Monitor Statistics:** Check bias energy and acceptance rates
4. **Validate Results:** Compare with unbiased simulations when possible
5. **Choose Appropriate Targets:** Based on physical insight

## Future Enhancements

Potential improvements for future versions:

1. **GPU-accelerated atom counting** for large systems
2. **Parallel umbrella windows** within single simulation
3. **Replica exchange** between umbrella windows
4. **Advanced bias potentials** (non-harmonic forms)
5. **Integration with other enhanced sampling methods**

## References

1. LAMMPS fix_gcmc_umbrella documentation
2. Torrie, G. M. & Valleau, J. P. Nonphysical sampling distributions in Monte Carlo free-energy estimation: Umbrella sampling. J. Comput. Phys. 23, 187-199 (1977).
3. Kumar, S. et al. THE weighted histogram analysis method for free-energy calculations on biomolecules. I. The method. J. Comput. Chem. 13, 1011-1021 (1992).

---

**Note:** This implementation is designed for research and educational purposes. For production simulations, validate results against known benchmarks and consider the specific requirements of your system.
