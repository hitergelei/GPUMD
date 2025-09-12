from tokenize import Double
import numpy as np
import ase.units
from ase.calculators.calculator import Calculator
import os

# 源代码版本：https://github.com/qzhu2017/PyXtal_FF/blob/master/pyxtal_ff/calculator/lammpslib.py
# 其他版本：https://github.com/libAtoms/pymatnest/blob/master/lammpslib.py

def is_upper_triangular(arr, atol=1e-8):
    """基于numpy的上三角形矩阵测试"""
    # must be (n*n) matrix
    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]
    return np.allclose(np.tril(arr, k=-1), 0., atol=atol) and \
           np.all(np.diag(arr) >= 0.0)


def convert_cell(ase_cell):
    """
    Convert a parallelepiped (forming right hand basis)
    to lower triangular matrix LAMMPS can accept. This
    function transposes cell matrix so the bases are column vectors
    """
    cell = ase_cell.T
    if not is_upper_triangular(cell):
        tri_mat = np.zeros((3, 3))
        A = cell[:, 0]
        B = cell[:, 1]
        C = cell[:, 2]
        tri_mat[0, 0] = np.linalg.norm(A)
        Ahat = A / np.linalg.norm(A)
        AxBhat = np.cross(A, B) / np.linalg.norm(np.cross(A, B))
        tri_mat[0, 1] = np.dot(B, Ahat)
        tri_mat[1, 1] = np.linalg.norm(np.cross(Ahat, B))
        tri_mat[0, 2] = np.dot(C, Ahat)
        tri_mat[1, 2] = np.dot(C, np.cross(AxBhat, Ahat))
        tri_mat[2, 2] = np.linalg.norm(np.dot(C, AxBhat))
        # create and save the transformation for coordinates
        volume = np.linalg.det(ase_cell)
        trans = np.array([np.cross(B, C), np.cross(C, A), np.cross(A, B)])
        trans /= volume
        coord_transform = np.dot(tri_mat, trans)
        return tri_mat, coord_transform
    else:
        return cell, None


class LAMMPSlib(Calculator):

    def __init__(self, lmp, lmpcmds=None, path='tmp',
                 lmp_file=None, ntyp=None, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)
        self.lmp = lmp
        # print("------", lmp)
        self.lmp_file = lmp_file
        self.folder = path
        self.ntyp = ntyp
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.lammps_data = self.folder + '/data.lammps'
        self.lammps_in = self.folder + '/in.lammps'
        self.lmpcmds = lmpcmds
        self.paras = []
        for para in lmpcmds:
            self.paras.append(para)

    def __str__(self):
        s = "\nLAMMPS calculator with pyxtal_ff force field\n"
        return s

    def __repr__(self):
        return str(self)

    def calculate(self, atoms):
        """
        prepare lammps .in file and data file
        write_lammps_data(filename, self.atoms, )
        """
        boundary = ''
        for i in range(3):
            if atoms.pbc[i]:
                boundary += 'p '
            else:
                boundary += 'f '
        if boundary in ['f f p ', 'p p f ']:  # needs some work later
            boundary = 'p p p '
        self.boundary = boundary
        self.write_lammps_data(atoms)
        self.write_lammps_in()
        self.lmp.file(self.lammps_in)

        # Extract the forces and energy
        self.lmp.command('variable pxx equal pxx')
        self.lmp.command('variable pyy equal pyy')
        self.lmp.command('variable pzz equal pzz')
        self.lmp.command('variable pxy equal pxy')
        self.lmp.command('variable pxz equal pxz')
        self.lmp.command('variable pyz equal pyz')
        self.lmp.command('variable fx atom fx')
        self.lmp.command('variable fy atom fy')
        self.lmp.command('variable fz atom fz')
        self.lmp.command('variable pe equal pe')

        pos = np.array(
            [x for x in self.lmp.gather_atoms("x", 1, 3)]).reshape(-1, 3)

        self.energy = self.lmp.extract_variable('pe', None, 0)
        # extract_global用法 :https://matsci.org/t/accessing-boundary-dimensions-in-python/32543/2
        xlo = self.lmp.extract_global("boxxlo", 1)   # 等价写法print(self.lmp.extract_box()[0][0])
        # print("xlo = ", xlo)  
        xhi = self.lmp.extract_global("boxxhi", 1)  # 等价写法print(self.lmp.extract_box()[1][0])
        # print("xhi = ", xhi)
        ylo = self.lmp.extract_global("boxylo", 1)
        yhi = self.lmp.extract_global("boxyhi", 1)
        zlo = self.lmp.extract_global("boxzlo", 1)
        zhi = self.lmp.extract_global("boxzhi", 1)
        xy = self.lmp.extract_global("xy", 1)
        yz = self.lmp.extract_global("yz", 1)
        xz = self.lmp.extract_global("xz", 1)
        unitcell = np.array([[xhi - xlo, xy, xz],
                             [0, yhi - ylo, yz],
                             [0, 0, zhi - zlo]]).T
        #-----hjchen---
        # print('xhi = ', xhi, 'xlo = ', xlo)
        # print("unitcell = ", unitcell)

        stress = np.empty(6)
        stress_vars = ['pxx', 'pyy', 'pzz', 'pyz', 'pxz', 'pxy']

        for i, var in enumerate(stress_vars):
            stress[i] = self.lmp.extract_variable(var, None, 0)

        stress_mat = np.zeros((3, 3))
        stress_mat[0, 0] = stress[0]
        stress_mat[1, 1] = stress[1]
        stress_mat[2, 2] = stress[2]
        stress_mat[1, 2] = stress[3]
        stress_mat[2, 1] = stress[3]
        stress_mat[0, 2] = stress[4]
        stress_mat[2, 0] = stress[4]
        stress_mat[0, 1] = stress[5]
        stress_mat[1, 0] = stress[5]
        stress[0] = stress_mat[0, 0]
        stress[1] = stress_mat[1, 1]
        stress[2] = stress_mat[2, 2]
        stress[3] = stress_mat[1, 2]
        stress[4] = stress_mat[0, 2]
        stress[5] = stress_mat[0, 1]

        self.stress = -stress * 1e5 * ase.units.Pascal
        f = (np.array(self.lmp.gather_atoms("f", 1, 3)).reshape(-1, 3) *
             (ase.units.eV / ase.units.Angstrom))
        self.forces = f.copy()
        atoms.positions = pos.copy()
        atoms.cell = unitcell.copy()
        self.atoms = atoms.copy()

    def write_lammps_in(self):
        if self.lmp_file is not None:
            os.system('cp ' + self.lmp_file + ' ' + self.folder + '/in.lammps')
        else:
            with open(self.lammps_in, 'w') as fh:
                fh.write('clear\n')
                fh.write('units metal\n')
                fh.write('boundary {:s}\n'.format(self.boundary))
                fh.write("atom_style atomic\n")
                fh.write('read_data {:s}\n'.format(self.lammps_data))
                fh.write('\n### interactions\n')
                for para in self.paras:
                    fh.write("{:s}\n".format(para))
                fh.write('neighbor 1.0 bin\n')
                fh.write('neigh_modify  every 1  delay 1  check yes\n')
                fh.write('thermo_style custom pe pxx pyy pzz pyz pxz pxy\n')
                fh.write('thermo_modify flush yes\n')
                fh.write('thermo 1\n')
                fh.write('run 1\n')

    def write_lammps_data(self, atoms):
        # QZ: assign the tags by sorting the element
        # eg [Si, Si, Si, O, O] will be [2, 2, 2, 1, 1]
        # Si, O will be sorted alphabetically
        atom_types = np.array(atoms.get_tags())  # [1]*len(atoms)
        if sum(atom_types) == 0:
            symbols = atoms.get_chemical_symbols()
            ordered_eles = sorted(list(set(symbols)))
            ntypes = len(ordered_eles)
            tags = []
            for symbol in symbols:
                for t, e in enumerate(ordered_eles):
                    if e == symbol:
                        tags.append(t + 1)
                        break
        else:
            ntypes = len(np.unique(atom_types))
            tags = atom_types

            # force n types
        if self.ntyp is not None:
            ntypes = self.ntyp

        with open(self.lammps_data, 'w') as fh:
            comment = 'lammpslib autogenerated data file'
            fh.write(comment.strip() + '\n\n')
            fh.write('{:d} atoms\n'.format(len(atoms)))
            fh.write('{:d} atom types\n'.format(ntypes))
            cell, coord_transform = convert_cell(atoms.get_cell())
            fh.write('\n')
            fh.write('{0:16.8e} {1:16.8e} xlo xhi\n'.format(0.0, cell[0, 0]))
            fh.write('{0:16.8e} {1:16.8e} ylo yhi\n'.format(0.0, cell[1, 1]))
            fh.write('{0:16.8e} {1:16.8e} zlo zhi\n'.format(0.0, cell[2, 2]))
            fh.write('{0:16.8e} {1:16.8e} {2:16.8e} xy xz yz\n'
                     ''.format(cell[0, 1], cell[0, 2], cell[1, 2]))

            fh.write('\n\nAtoms \n\n')
            for i, (typ, pos) in enumerate(
                    zip(tags, atoms.get_positions())):
                if coord_transform is not None:
                    pos = np.dot(coord_transform, pos.transpose())
                fh.write('{:4d} {:4d} {:16.8e} {:16.8e} {:16.8e}\n'
                         .format(i + 1, typ, pos[0], pos[1], pos[2]))

    def update(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms)

    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        return self.energy

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()

    def get_stress(self, atoms):
        self.update(atoms)
        return self.stress.copy()


if __name__ == '__main__':
    from ase.io import read
    from ase.build import bulk
    from lammps import lammps

    lammps_name=''
    comm=None
    log_file='lammps.log'
    cmd_args = ['-echo', 'log', '-log', log_file,
                '-screen', 'none', '-nocite']
    lmp = lammps(lammps_name, cmd_args, comm)

    # struc = bulk('Al', 'fcc', cubic=True)

    data = read('/home/hjchen/projects/al_empirical_potentials/tests/Mo_adp_Baysian_opt/datasets/GAP/db_Mo.extxyz',':')
    dft_energy = [ei.get_potential_energy() for ei  in data ]
    # print("dft: ", dft_energy[-10:])


    adp_energy = []
    i = 0
    # for i in range(len(data)):
    for i in range(1):
        
        struc = data[i]
        i = i + 1



        struc.set_tags([1]*len(struc))
        parameters = ["mass              1 42",
                    "pair_style  adp",
                    "pair_coeff * * ./hjchen-Mo.adp Mo"
                    ]
        # parameters = [
        #     "mass            1 42",
        #     'pair_style      meam/spline',
        #      'pair_coeff      * * Park_MEAM_Mo_2012.spline Mo'
        #      ]

        lammps = LAMMPSlib(lmp=lmp, lmpcmds=parameters)
        struc.set_calculator(lammps)
        # print('positions: ')
        # print(struc.get_positions())
        print('energy: ')
        print(struc.get_potential_energy())
        adp_energy.append(struc.get_potential_energy())

        # print('force: ')
        # print(struc.get_forces())
        # print('stress: ')
        # print(struc.get_stress())

        if i == 10:
            exit()

    # print("---------------result compare report:")
    # for dft_e, adp_e in zip(dft_energy, adp_energy):
    #     print("dft_E = {};  adp_E = {}".format(dft_e, adp_e))
        
    print("adp_energy:", adp_energy)