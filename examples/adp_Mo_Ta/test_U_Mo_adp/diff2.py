#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# Read GPUMD per-atom forces and energies from dump.xyz
with open('dump.xyz') as f:
    lines = f.read().splitlines()
N = int(lines[0].strip())
header = lines[1]
if 'forces:R:3' not in header or 'energy_atom:R:1' not in header:
    raise RuntimeError('dump.xyz missing forces or energy_atom columns. Use: dump_exyz 1 0 1 1')

gp_list = []
for s in lines[2:2+N]:
    toks = s.split()
    # species x y z fx fy fz energy_atom
    fx, fy, fz = map(float, toks[4:7])
    e = float(toks[7])
    gp_list.append((fx, fy, fz, e))
gp = np.array(gp_list, dtype=float)

# Read LAMMPS per-atom forces and energies from lammps_adp_reference.dump
dump = Path('lammps_adp_reference.dump').read_text().splitlines()
try:
    start = next(i for i, line in enumerate(dump) if line.startswith('ITEM: ATOMS'))
except StopIteration:
    raise RuntimeError('Could not find "ITEM: ATOMS" header in lammps_adp_reference.dump')

lm_list = []
for s in dump[start+1:start+1+N]:
    toks = s.split()
    # Assume last 4 tokens are fx, fy, fz, c_pe
    fx, fy, fz, e = map(float, toks[-4:])
    lm_list.append((fx, fy, fz, e))
lm = np.array(lm_list, dtype=float)

# Compute differences
df = gp[:, :3] - lm[:, :3]
de = gp[:, 3]   - lm[:, 3]
l2 = np.linalg.norm(df, axis=1)

print('N =', N)
print('Force L2 mean: {:.6e}  max: {:.6e}  at atom {}'.format(l2.mean(), l2.max(), int(np.argmax(l2))+1))
print('Energy sums  GPUMD: {:.8f}  LAMMPS: {:.8f}  dE: {:.8f}'.format(gp[:,3].sum(), lm[:,3].sum(), de.sum()))
print('Energy mean diff: {:.6e}  std: {:.6e}'.format(de.mean(), de.std()))

# Show top-5 largest force and energy offenders
idx_f = np.argsort(-l2)[:5]
idx_e = np.argsort(-np.abs(de))[:5]
print('\nTop-5 force L2 offenders (atom, L2, dfx, dfy, dfz):')
for i in idx_f:
    print('{:3d}  {: .6e}  {: .6e} {: .6e} {: .6e}'.format(int(i)+1, l2[i], df[i,0], df[i,1], df[i,2]))
print('\nTop-5 energy offenders (atom, de):')
for i in idx_e:
    print('{:3d}  {: .6e}'.format(int(i)+1, de[i]))
