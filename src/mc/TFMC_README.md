# Time-stamped Force-bias Monte Carlo (tfMC) Implementation for GPUMD

## Overview

This CUDA implementation of tfMC is based on the LAMMPS fix_tfmc and adapted for GPUMD's MC framework.

## Files

- `mc_ensemble_tfmc.cuh` - Header file defining the MC_Ensemble_TFMC class
- `mc_ensemble_tfmc.cu` - CUDA implementation of tfMC algorithm

## Usage in run.in

### Basic Syntax

```
mc tfmc N_md N_mc T_initial T_final d_max seed [com x y z] [rot] [group ...]
```

### Parameters

- `N_md`: Number of MD steps between MC trials
- `N_mc`: Number of MC trials per MC step
- `T_initial`: Initial temperature (K)
- `T_final`: Final temperature (K)
- `d_max`: Maximum displacement length (Angstroms)
- `seed`: Random number seed (positive integer)

### Optional Keywords

- `com x y z`: Fix center of mass motion in x, y, z directions (0 or 1 for each)
- `rot`: Fix rotational motion
- `group method_id group_id`: Apply MC only to specified group

### Examples

#### Example 1: Basic tfMC

```
mc tfmc 100 50 300.0 300.0 0.20 12345
```

Run tfMC with:
- 100 MD steps between MC trials
- 50 MC trials each time
- Temperature 300 K
- Maximum displacement 0.20 Å
- Random seed 12345

#### Example 2: tfMC with COM fixing

```
mc tfmc 100 50 300.0 300.0 0.20 12345 com 1 1 1
```

Same as above, but fix center of mass motion in all directions.

#### Example 3: tfMC with rotation fixing

```
mc tfmc 100 50 300.0 300.0 0.20 12345 rot
```

Same as Example 1, but also fix rotational motion.

#### Example 4: tfMC with group selection

**model.xyz** (定义group属性):
```xyz
1000
Lattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3:group:I:1
Cu 15.0 15.0 15.0 0
Cu 16.0 15.0 15.0 0
Cu 5.0 5.0 5.0 1
Cu 6.0 5.0 5.0 1
...
```

**run.in**:
```
mc tfmc 100 50 300.0 300.0 0.20 12345 group 0 1
```

Apply tfMC only to atoms in group 1 (grouping method 0).

**说明**: 
- `group:I:1` 在 model.xyz 的 Properties 中声明有group属性
- 每行最后一个整数为该原子的group ID (0, 1, 2, ...)
- 不需要在 run.in 中定义group，GPUMD自动从model.xyz读取

## Algorithm Details

### Displacement Generation

For each atom i and dimension j, the displacement is generated according to:

1. Compute γ = F_j · d_i / (2kT)
2. Sample ξ uniformly from [-1, 1]
3. Accept ξ with probability P_acc(ξ, γ)
4. displacement = ξ · d_i

where d_i = d_max · (m_min/m_i)^0.25

### Acceptance Probability

```
if ξ < 0:
    P_acc = [exp(2ξγ)·exp(γ) - exp(-γ)] / [exp(γ) - exp(-γ)]
elif ξ > 0:
    P_acc = [exp(-γ) - exp(2ξγ)·exp(-γ)] / [exp(γ) - exp(-γ)]
else:
    P_acc = 1
```

### COM and Rotation Removal

- COM motion is removed by computing center of mass displacement and subtracting it from all atoms
- Rotation removal computes angular momentum and removes rotational component (simplified version)

## Implementation Notes

1. **Force Calculation**: Forces must be computed before calling tfMC. The forces are already available in atom.fx, atom.fy, atom.fz from the MD integrator.

2. **Mass Scaling**: Displacement length is scaled by (m_min/m_i)^0.25 to account for different atomic masses.

3. **GPU Parallelization**: All displacement generation and motion removal operations are parallelized on GPU using CUDA kernels.

4. **Random Number Generation**: Uses cuRAND for on-device random number generation.

## References

1. K. M. Bal and E. C. Neyts, "Merging Metadynamics into Hyperdynamics: Accelerated Molecular Simulations Reaching Time Scales from Microseconds to Seconds", J. Chem. Theory Comput. 11, 4545 (2015).

2. LAMMPS fix tfmc documentation and source code

## Compilation

Add the following to the makefile:

```makefile
mc/mc_ensemble_tfmc.o: mc/mc_ensemble_tfmc.cu mc/mc_ensemble_tfmc.cuh
	$(NVCC) $(CFLAGS) -c mc/mc_ensemble_tfmc.cu -o mc/mc_ensemble_tfmc.o
```

And include `mc/mc_ensemble_tfmc.o` in the linking step.

## Limitations and Future Work

1. Rotation removal is not fully implemented (requires inertia tensor calculation)
2. Group support needs testing
3. Performance optimization for large systems
4. Acceptance rate statistics output
5. Enhanced error checking and validation

## Testing

To test the tfMC implementation:

1. Prepare a NEP potential
2. Create a run.in file with tfMC command
3. Run GPUMD
4. Monitor for proper displacement generation and COM fixing

Example test system: simple bulk material at 300 K with small displacement steps.
