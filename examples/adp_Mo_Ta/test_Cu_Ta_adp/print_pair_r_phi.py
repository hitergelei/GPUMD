from pathlib import Path
import sys
import numpy as np
root = Path('/home/hjchen/projects/GPUMD/examples/adp_Mo_Ta/test_Cu_Ta_adp')
sys.path.append(str(root))
from Analytic_ADP_torch_version import load_adp_setfl, read_ovito_lammps
from torch_neigh import TorchNeighborList

adp = load_adp_setfl(root/'Ta.adp.txt')
atoms = read_ovito_lammps(root/'model_Ta_dump_idx10.dat')
cutoff = adp.rc + 1e-6
neighbor_list = TorchNeighborList(cutoff=cutoff)
pairs, pair_diff, pair_dist = neighbor_list(atoms)
mask = pairs[:,0] < pairs[:,1]
pair_i = pairs[mask,0].astype(int)
pair_j = pairs[mask,1].astype(int)
pair_vec = pair_diff[mask]
pair_dist = pair_dist[mask]

rho_vals = adp.rho.evaluate(pair_dist)
phi_vals = adp.phi.evaluate(pair_dist)

print('total unique pairs', len(pair_i))
print('per atom count average', len(pair_i)*2/len(atoms))

idx0 = np.where(pair_i==0)[0]
print('pairs for atom 0:', len(idx0))
for k in idx0[:15]:
    j = int(pair_j[k])
    r = pair_dist[k]
    phi = phi_vals[k]
    print(f'pair 0-{j}: r={r:.4f}, phi={phi:.6f}')


#---------
from pathlib import Path
import sys
import numpy as np
root = Path('/home/hjchen/projects/GPUMD/examples/adp_Mo_Ta/test_Cu_Ta_adp')
sys.path.append(str(root))
from Analytic_ADP_torch_version import load_adp_setfl, read_ovito_lammps
from torch_neigh import TorchNeighborList

adp = load_adp_setfl(root/'Ta.adp.txt')
atoms = read_ovito_lammps(root/'model_Ta_dump_idx10.dat')
cutoff = adp.rc + 1e-6
neighbor_list = TorchNeighborList(cutoff=cutoff)
_, _, pair_dist_all = neighbor_list(atoms)
# keep unique pairs (i<j)
pairs, _, d = neighbor_list(atoms)
mask = pairs[:,0] < pairs[:,1]
d = pair_dist_all[mask]
print('max distance', d.max())
print('cutoff', adp.rc)
print('min margin', adp.rc - d.max())
print('min distance > 6.0 count', np.sum(d>6.0))
print('min distance > 6.1 count', np.sum(d>6.1))
print('min distance > 6.14 count', np.sum(d>6.14))
print('Total pairs', len(d))


#--------------
from pathlib import Path
import sys
import numpy as np
root = Path('/home/hjchen/projects/GPUMD/examples/adp_Mo_Ta/test_Cu_Ta_adp')
sys.path.append(str(root))
from Analytic_ADP_torch_version import load_adp_setfl, read_ovito_lammps
from torch_neigh import TorchNeighborList

adp = load_adp_setfl(root/'Ta.adp.txt')
atoms = read_ovito_lammps(root/'model_Ta_dump_idx10.dat')
for cutoff in [5.0,5.2,5.4,5.6,5.8,6.0,6.15]:
    neighbor_list = TorchNeighborList(cutoff=cutoff)
    pairs, _, _ = neighbor_list(atoms)
    mask = pairs[:,0] < pairs[:,1]
    total = mask.sum()
    avg = total*2/len(atoms)
    print(cutoff, avg)

#-------------
from pathlib import Path, sys
import numpy as np
root = Path('/home/hjchen/projects/GPUMD/examples/adp_Mo_Ta/test_Cu_Ta_adp')
sys.path.append(str(root))
from Analytic_ADP_torch_version import load_adp_setfl, read_ovito_lammps
from torch_neigh import TorchNeighborList

atoms = read_ovito_lammps(root/'model_Ta_dump_idx10.dat')
for cutoff in np.linspace(5.6, 5.8, 9):
    neighbor_list = TorchNeighborList(cutoff=float(cutoff))
    pairs, _, _ = neighbor_list(atoms)
    mask = pairs[:,0] < pairs[:,1]
    total = mask.sum()
    avg = total*2/len(atoms)
    print(f"{cutoff:.3f} -> {avg}")