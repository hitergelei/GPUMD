# GPUMD tfMC Examples

Time-stamped Force-bias Monte Carlo (tfMC) simulation examples for GPUMD.

## What is tfMC?

tfMC (time-stamped force-bias Monte Carlo) is a hybrid Monte Carlo-Molecular Dynamics method that:
- Uses force-biased sampling (no energy calculation needed)
- Accelerates rare event sampling by 10²-10⁴× vs. standard MD
- Accepts all MC moves uniformly (no Metropolis rejection)
- Ideal for barrier crossing, diffusion, and configuration space exploration

## Files Overview

### Basic Examples

1. **run_tfmc_basic.in** - Simplest tfMC simulation
   - Standard setup without constraints
   - Good starting point

2. **run_tfmc_com_fixed.in** - Fix center of mass motion
   - Prevents system drift
   - `com 1 1 1`: Fix all three directions

3. **run_tfmc_rotation_fixed.in** - Remove rotational motion
   - Uses unwrapped coordinates (accurate for all system sizes)
   - Prevents rigid-body rotation

4. **run_tfmc_full_constraint.in** - Both COM and rotation fixed
   - **Recommended for most simulations**
   - Most physically realistic for isolated systems

### Advanced Examples

5. **run_tfmc_annealing.in** - Temperature annealing
   - Cool from 500K → 100K
   - Useful for structure optimization
   - Finding low-energy configurations

6. **run_tfmc_group.in** - Group-specific MC
   - Apply MC only to selected atoms
   - Requires `group:I:1` in model.xyz Properties
   - More efficient than full system MC

7. **run_tfmc_partial_com.in** - Partial COM constraint
   - Fix only selected directions (e.g., `com 1 1 0`)
   - Useful for surfaces or layered systems

8. **run_tfmc_rare_event.in** - Large displacement sampling
   - Larger d_max (0.5 Å) for barrier crossing
   - Fewer trials, larger jumps
   - Ideal for rare events

9. **run_tfmc_high_frequency.in** - Rapid equilibration
   - Very frequent MC steps (N_md=10)
   - Many trials per step (N_mc=100)
   - Fast thermalization

```txt
9个run.in示例：

run_tfmc_basic.in - 基础用法
run_tfmc_com_fixed.in - 固定质心
run_tfmc_rotation_fixed.in - 移除转动
run_tfmc_full_constraint.in - 完整约束（推荐）
run_tfmc_annealing.in - 温度退火
run_tfmc_group.in - 群组选择
run_tfmc_partial_com.in - 部分质心约束
run_tfmc_rare_event.in - 稀有事件采样
run_tfmc_high_frequency.in - 高频快速平衡
```

## Command Syntax

```bash
mc tfmc N_md N_mc T_initial T_final d_max seed [com x y z] [rot] [group method_id group_id]
ensemble nve  # MUST use NVE ensemble for tfMC
```

### Required Parameters

- **N_md**: MD steps between MC trials (e.g., 100)
- **N_mc**: Number of MC trials per MC step (e.g., 50)
- **T_initial**: Initial temperature in K (e.g., 300.0)
- **T_final**: Final temperature in K (e.g., 300.0)
- **d_max**: Maximum displacement in Angstrom (e.g., 0.20)
- **seed**: Random number seed (positive integer)

**IMPORTANT**: tfMC controls temperature through MC sampling. You MUST use `ensemble nve` (not NVT/NPT). Using NVT will cause double temperature control and incorrect sampling.

### Optional Keywords

- **com x y z**: Fix COM motion (0 or 1 for each direction)
  - `com 1 1 1`: Fix all directions
  - `com 1 1 0`: Fix x,y; free z
  
- **rot**: Remove rotational motion
  - Uses unwrapped coordinates (matches LAMMPS)
  - Accurate for all system sizes
  
- **group method_id group_id**: Apply MC to specific group
  - Requires group definition in model.xyz
  - More efficient for selective sampling

## Parameter Guidelines

### N_md (MD steps between MC)
- **Small (10-50)**: Rapid equilibration, high overhead
- **Medium (100-200)**: Balanced (recommended)
- **Large (500-1000)**: Less overhead, slower equilibration

### N_mc (MC trials per step)
- **Small (10-20)**: Fast, less thorough
- **Medium (50-100)**: Balanced (recommended)
- **Large (200-500)**: Thorough, slower

### d_max (Maximum displacement)
- **Small (0.1-0.2 Å)**: Local exploration, refinement
- **Medium (0.2-0.3 Å)**: General sampling (recommended)
- **Large (0.5-1.0 Å)**: Rare events, barrier crossing

### Temperature
- **Constant**: `T_initial = T_final` (e.g., 300.0 300.0)
- **Annealing**: `T_initial > T_final` (e.g., 500.0 100.0)
- **Heating**: `T_initial < T_final` (e.g., 100.0 500.0)

## Model Requirements

### Standard model.xyz
```
250
Lattice="..." Properties=species:S:1:pos:R:3
Te 3.379 3.216 3.349
Pb 6.510 6.440 3.349
...
```

### With group support
```
250
Lattice="..." Properties=species:S:1:pos:R:3:group:I:1
Te 3.379 3.216 3.349 0
Pb 6.510 6.440 3.349 1
...
```
Last column = group ID (0, 1, 2, ...)

## Typical Use Cases

### 1. Equilibration at Fixed Temperature
```bash
mc tfmc 100 50 300.0 300.0 0.20 12345 com 1 1 1 rot
ensemble nve
```

### 2. Simulated Annealing
```bash
mc tfmc 100 50 500.0 100.0 0.20 12345 com 1 1 1 rot
ensemble nve
```

### 3. Vacancy Diffusion
```bash
mc tfmc 50 20 800.0 800.0 0.50 12345 com 1 1 1 rot
ensemble nve
```

### 4. Surface Relaxation
```bash
mc tfmc 100 50 300.0 300.0 0.15 12345 com 1 1 0
ensemble nve
```

### 5. Selective Sampling (e.g., only mobile species)
```bash
mc tfmc 100 50 300.0 300.0 0.25 12345 group 0 1 com 1 1 1
ensemble nve
```

**Note**: All tfMC simulations require `ensemble nve`. tfMC controls temperature internally through MC sampling, so using NVT/NPT thermostats would interfere with the algorithm.

## Comparison with LAMMPS

GPUMD tfMC is **fully equivalent** to LAMMPS fix_tfmc:

| Feature | LAMMPS | GPUMD |
|---------|--------|-------|
| Core algorithm | ✅ | ✅ Same |
| Mass scaling | ✅ | ✅ Same |
| COM removal | ✅ | ✅ Same |
| Rotation removal | ✅ | ✅ Same (unwrapped coords) |
| PBC handling | ✅ | ✅ Same |
| Temperature control | Fixed | ✅ **Variable** (better!) |
| Performance | CPU | ✅ **GPU** (10-100x faster!) |
| Group support | Mask | ✅ **More efficient** |

## Performance Tips

1. **Adjust N_md based on system size**:
   - Small systems (< 1000 atoms): N_md = 10-50
   - Large systems (> 10000 atoms): N_md = 100-500

2. **Adjust d_max based on problem**:
   - Local refinement: 0.1-0.2 Å
   - Barrier crossing: 0.5-1.0 Å

3. **Use group support** when possible:
   - Only sample relevant atoms
   - Reduces computation

4. **Monitor output**:
   - Check energy drift
   - Check temperature stability
   - Verify COM/rotation removal

## Frequently Asked Questions (FAQ)

### Q: Can I use NVT or NPT ensemble with tfMC?
**A: NO! You MUST use NVE ensemble only.**

Reasons:
- tfMC already controls temperature through MC sampling
- Using NVT creates **double temperature control** → incorrect sampling
- MD steps need to conserve energy (microcanonical) for tfMC to work correctly
- NVT/NPT thermostats interfere with the force-biased MC algorithm

Correct usage:
```bash
mc tfmc 100 50 300.0 300.0 0.20 12345
ensemble nve  # Always use NVE!
```

### Q: How does tfMC control temperature without a thermostat?
**A**: tfMC uses MC steps to sample velocities from Maxwell-Boltzmann distribution at target temperature. The `T_initial` and `T_final` parameters define the temperature schedule.

### Q: Can I do variable temperature (annealing)?
**A**: Yes! Set different `T_initial` and `T_final`:
```bash
mc tfmc 100 50 500.0 100.0 0.20 12345  # Cool from 500K to 100K
```

## Troubleshooting

### Issue: System drifts
**Solution**: Use `com 1 1 1`

### Issue: System rotates
**Solution**: Use `rot`

### Issue: Temperature unstable
**Solution**: Reduce d_max or increase N_md

### Issue: Too slow
**Solution**: Increase N_md, reduce N_mc

### Issue: Not sampling enough
**Solution**: Increase N_mc or d_max

### Issue: Used NVT ensemble by mistake
**Solution**: Change `ensemble nvt_xxx` to `ensemble nve`. tfMC controls temperature internally.

## References

1. K. M. Bal and E. C. Neyts, J. Chem. Theory Comput. 11, 4545 (2015)
2. LAMMPS fix_tfmc documentation
3. GPUMD documentation

## Contact

For questions or issues:
- GPUMD GitHub: https://github.com/brucefan1983/GPUMD
- tfMC branch: https://github.com/hitergelei/GPUMD

---
**Note**: All examples assume you have a valid `nep.txt` potential file and `model.xyz` structure file in the same directory.
