#!/usr/bin/env python3
"""
Compare energy components between LAMMPS and GPUMD for ADP potential
"""

# From GPUMD output:
print("=== GPUMD Results ===")
gpumd_total = -433.27131307
gpumd_per_atom_F = -4.757949  # embedding energy per atom
gpumd_per_atom_adp = 0.0      # ADP energy per atom
n_atoms = 54

gpumd_total_F = n_atoms * gpumd_per_atom_F
gpumd_total_adp = n_atoms * gpumd_per_atom_adp
gpumd_pair = gpumd_total - gpumd_total_F - gpumd_total_adp

print(f"Total energy: {gpumd_total:.6f}")
print(f"Embedding energy (F): {gpumd_total_F:.6f}")
print(f"ADP energy: {gpumd_total_adp:.6f}")
print(f"Pair energy: {gpumd_pair:.6f}")
print(f"Per atom - F: {gpumd_per_atom_F:.6f}, ADP: {gpumd_per_atom_adp:.6f}, Pair: {gpumd_pair/n_atoms:.6f}")

print("\n=== LAMMPS Results ===")
lammps_total = -437.38253050195868354

print(f"Total energy: {lammps_total:.6f}")
print(f"Energy difference (LAMMPS - GPUMD): {lammps_total - gpumd_total:.6f}")
print(f"Per atom difference: {(lammps_total - gpumd_total)/n_atoms:.6f}")

# Let's analyze what might be wrong
print("\n=== Analysis ===")
print("Possible issues:")
print("1. Different spline interpolation methods")
print("2. Different handling of r*phi(r) format")  
print("3. Different ADP energy calculation")
print("4. Different cutoff handling")

# From LAMMPS debug output, we see ADP components are non-zero
# but GPUMD shows ADP energy = 0
print("\nFrom LAMMPS output, we see non-zero lambda values:")
print("This suggests ADP terms should contribute to energy")
print("But GPUMD shows ADP energy = 0, indicating a calculation issue")