#!/usr/bin/env bash
set -euo pipefail

# Clean previous outputs to avoid appending
rm -f thermo.out force.out observer*.xyz observer*.out dump.xyz dump.*.xyz gpumd_stdout.txt || true

# Run GPUMD in this directory
../../src/gpumd > gpumd_stdout.txt 2>&1 || {
  echo "Error: failed to run gpumd. See gpumd_stdout.txt for details." >&2
  exit 1
}

# Compare with LAMMPS reference
if command -v python3 >/dev/null 2>&1; then
  python3 compare_gpumd_vs_lammps.py || {
    echo "Warning: comparison script failed." >&2
    exit 1
  }
else
  echo "python3 not found; skipping comparison." >&2
fi

echo "Done. Thermo in thermo.out, forces in force.out."

