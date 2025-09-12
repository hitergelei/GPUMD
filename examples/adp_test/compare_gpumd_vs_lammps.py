#!/usr/bin/env python3
import re
import numpy as np
from pathlib import Path

def read_gpumd_force(path='force.out'):
    f = np.loadtxt(path)
    if f.ndim == 1:
        f = f.reshape(1, -1)
    return f  # shape: (N, 3)

def read_gpumd_thermo(path='thermo.out'):
    # Columns: T, Ek, Ep, sxx, syy, szz, syz, sxz, sxy, h00..h22 (9 entries)
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr  # shape: (steps, 18)

def parse_lammps_dump_forces(path='lammps_adp_reference.dump', timestep=0):
    ids = []
    forces = []
    reading = False
    current_step = None
    with open(path, 'r') as fh:
        for line in fh:
            if line.startswith('ITEM: TIMESTEP'):
                current_step = int(next(fh).strip())
                reading = False
                continue
            if current_step == timestep and line.startswith('ITEM: ATOMS'):
                # Expect columns: id type x y z fx fy fz c_pe
                reading = True
                continue
            if reading:
                if line.startswith('ITEM:'):
                    # next block
                    break
                toks = line.split()
                if len(toks) < 9:
                    continue
                ids.append(int(toks[0]))
                fx, fy, fz = map(float, toks[5:8])
                forces.append([fx, fy, fz])
    if not ids:
        raise RuntimeError('No forces parsed from LAMMPS dump at timestep {}'.format(timestep))
    order = np.argsort(np.array(ids))
    return np.array(forces)[order]

def parse_lammps_dump_pe_sum(path='lammps_adp_reference.dump', timestep=0):
    pe = []
    reading = False
    current_step = None
    with open(path, 'r') as fh:
        for line in fh:
            if line.startswith('ITEM: TIMESTEP'):
                current_step = int(next(fh).strip())
                reading = False
                continue
            if current_step == timestep and line.startswith('ITEM: ATOMS'):
                reading = True
                continue
            if reading:
                if line.startswith('ITEM:'):
                    break
                toks = line.split()
                if len(toks) < 9:
                    continue
                pe.append(float(toks[8]))
    if not pe:
        raise RuntimeError('No per-atom energy parsed at timestep {}'.format(timestep))
    return float(np.sum(pe))

def parse_lammps_log_pe(path='log.lammps'):
    txt = Path(path).read_text(errors='ignore')
    m = re.search(r'Potential energy:\s*([\-+eE0-9\.]+)', txt)
    return float(m.group(1)) if m else None

def main():
    # GPUMD
    g_force = read_gpumd_force('force.out')
    g_thermo = read_gpumd_thermo('thermo.out')
    g_pe = float(g_thermo[-1, 2])  # third column
    g_stress = g_thermo[-1, 3:9]   # sxx, syy, szz, syz, sxz, sxy (converted units)

    # LAMMPS
    l_force = parse_lammps_dump_forces('lammps_adp_reference.dump', timestep=0)
    l_pe_sum = parse_lammps_dump_pe_sum('lammps_adp_reference.dump', timestep=0)
    l_pe_log = parse_lammps_log_pe('log.lammps')

    # Force comparison
    if g_force.shape != l_force.shape:
        raise RuntimeError(f'Force shape mismatch: GPUMD {g_force.shape} vs LAMMPS {l_force.shape}')
    diff = g_force - l_force
    max_abs = np.max(np.abs(diff))
    l2 = np.linalg.norm(diff, axis=1)
    print('Forces: max|Δ| = {:.3e}, mean|Δ| = {:.3e}'.format(max_abs, float(np.mean(l2))))

    # Energy comparison
    print('Potential energy GPUMD (thermo): {:.8f}'.format(g_pe))
    print('Potential energy LAMMPS (sum c_pe): {:.8f}'.format(l_pe_sum))
    if l_pe_log is not None:
        print('Potential energy LAMMPS (log): {:.8f}'.format(l_pe_log))
    print('ΔE (GPUMD - LAMMPS sum c_pe): {:.3e}'.format(g_pe - l_pe_sum))

    # Stress comparison: requires LAMMPS to output pxx,pyy,pzz,pxy,pxz,pyz for exact compare
    print('GPUMD stress [sxx, syy, szz, syz, sxz, sxy]:')
    print('  ', ' '.join(f'{v:.6f}' for v in g_stress))
    print('Note: To compare stress, set in LAMMPS input:\n'
          '  thermo_style custom step temp pe ke pxx pyy pzz pxy pxz pyz\n'
          'and re-run to obtain matching components.')

if __name__ == '__main__':
    main()

