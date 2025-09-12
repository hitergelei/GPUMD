#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np

def read_gpumd_thermo(path='thermo.out'):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # columns: T, Ek, Ep, sxx, syy, szz, syz, sxz, sxy, h00..h22
    return {
        'T': float(arr[-1, 0]),
        'Ek': float(arr[-1, 1]),
        'Ep': float(arr[-1, 2]),
        'sxx': float(arr[-1, 3]),
        'syy': float(arr[-1, 4]),
        'szz': float(arr[-1, 5]),
        'syz': float(arr[-1, 6]),
        'sxz': float(arr[-1, 7]),
        'sxy': float(arr[-1, 8]),
    }

def parse_lammps_log(path='log.lammps'):
    # Look for thermo header and the first data row after it (run 0)
    lines = Path(path).read_text(errors='ignore').splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Step') and ('PotEng' in line) and ('Pxx' in line):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError('Could not find thermo header in log.lammps')
    # Next non-empty, non-comment line should be the data row
    j = header_idx + 1
    while j < len(lines) and (not lines[j].strip() or lines[j].startswith('#')):
        j += 1
    toks = lines[j].split()
    # Expect: Step PotEng Lx Ly Lz Press Pxx Pyy Pzz c_eatoms
    step = int(toks[0])
    pe = float(toks[1])
    pxx_bar = float(toks[6])
    pyy_bar = float(toks[7])
    pzz_bar = float(toks[8])
    # Convert bar -> GPa
    conv = 1e-4
    return {
        'step': step,
        'Ep': pe,
        'pxx_GPa': pxx_bar * conv,
        'pyy_GPa': pyy_bar * conv,
        'pzz_GPa': pzz_bar * conv,
    }

def main():
    g = read_gpumd_thermo('thermo.out')
    l = parse_lammps_log('log.lammps')

    print('LAMMPS vs GPUMD (Mo ADP, single-step)')
    print('Energy (eV):   LAMMPS {:.8f}   GPUMD {:.8f}   dE = {:+.3e}'.format(l['Ep'], g['Ep'], g['Ep']-l['Ep']))
    print('Stress GPa:    Pxx  LAMMPS {:.6f}   GPUMD {:.6f}   d = {:+.3e}'.format(l['pxx_GPa'], g['sxx'], g['sxx']-l['pxx_GPa']))
    print('               Pyy  LAMMPS {:.6f}   GPUMD {:.6f}   d = {:+.3e}'.format(l['pyy_GPa'], g['syy'], g['syy']-l['pyy_GPa']))
    print('               Pzz  LAMMPS {:.6f}   GPUMD {:.6f}   d = {:+.3e}'.format(l['pzz_GPa'], g['szz'], g['szz']-l['pzz_GPa']))
    print('Note: Off-diagonal LAMMPS stresses were not dumped; only diagonals are compared.')

if __name__ == '__main__':
    main()

