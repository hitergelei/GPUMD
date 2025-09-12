import torch
from ase import Atoms
from ase.io import read, write
from ase.build import bulk, make_supercell
from ase.neighborlist import NeighborList
from ase.calculators.eam import EAM
import matplotlib.pyplot as plt
import numpy as np
import asap3
from torch_neigh import TorchNeighborList
from adp_func import emb, emb_vec, rho, phi_AA, u11, w11

# vesin也提供了近邻列表的计算
# https://github.com/Luthaf/vesin



def asap3_get_neighborlist(cutoff, atoms):      
    nl = asap3.FullNeighborList(cutoff, atoms)
    pair_i_idx = []
    pair_j_idx = []
    n_diff = []
    for i in range(len(atoms)):
        indices, diff, _ = nl.get_neighbors(i)
        pair_i_idx += [i] * len(indices)               # local index of pair i
        pair_j_idx.append(indices)   # local index of pair j
        n_diff.append(diff)

    pair_j_idx = np.concatenate(pair_j_idx)
    pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
    n_diff = np.concatenate(n_diff)
    
    return pairs, n_diff


# TODO: 方法1： （ADP势的力的解析式的计算方法） 有待于实现和ase中类似的计算方法
def angular_forces(self, mu_i, mu, lam_i, lam, r, rvec, form1, form2):
    # calculate the extra components for the adp forces
    # rvec are the relative positions to atom i
    psi = np.zeros(mu.shape)
    for gamma in range(3):
        term1 = (mu_i[gamma] - mu[:, gamma]) * self.d[form1][form2](r)

        term2 = np.sum((mu_i - mu) *
                        self.d_d[form1][form2](r)[:, np.newaxis] *
                        (rvec * rvec[:, gamma][:, np.newaxis] /
                        r[:, np.newaxis]), axis=1)

        term3 = 2 * np.sum((lam_i[:, gamma] + lam[:, :, gamma]) *
                            rvec * self.q[form1][form2](r)[:, np.newaxis],
                            axis=1)
        term4 = 0.0
        for alpha in range(3):
            for beta in range(3):
                rs = rvec[:, alpha] * rvec[:, beta] * rvec[:, gamma]
                term4 += ((lam_i[alpha, beta] + lam[:, alpha, beta]) *
                            self.d_q[form1][form2](r) * rs) / r

        term5 = ((lam_i.trace() + lam.trace(axis1=1, axis2=2)) *
                    (self.d_q[form1][form2](r) * r +
                    2 * self.q[form1][form2](r)) * rvec[:, gamma]) / 3.

        # the minus for term5 is a correction on the adp
        # formulation given in the 2005 Mishin Paper and is posted
        # on the NIST website with the AlH potential
        psi[:, gamma] = term1 + term2 + term3 + term4 - term5

    return np.sum(psi, axis=0)



def TorchNeigh(atoms_obj: Atoms, cutoff: float):
    print("\n----------------TorchNeighborList方法")
    # 实例化邻居列表类
    cutoff = cutoff  # 设置截断距离
    neighbor_list = TorchNeighborList(cutoff=cutoff)

    # 计算近邻列表
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)
    
    print("len(pairs) = ", len(pairs))
    # pairs包含原子对的索引，pair_diff是原子对之间的向量差，pair_dist是原子对之间的距离
    print("Pairs:\n", pairs)
    print("Pair differences形状：", pair_diff.shape)
    # print("Pair differences:\n", pair_diff)
    # print("Pair distances:\n", pair_dist)
    print("Pair distances形状: ", pair_dist.shape)


def Mo_adp_test():
    Mo_adp = EAM(potential="./test_Mo_Ta_adp/Mo.adp", cutoff=6.5000000000000000e+00, skin=0, elements=['Mo'], form='adp')
    # # Mo_adp.plot('Mo_adp_test')  # 触发 Mo_rij.txt和Mo_Φij.txt数据的本地保存
    
    atoms_obj = read("Mo_structs_0.dat", format="lammps-data")
    atoms_obj.set_chemical_symbols(['Mo' for _ in range(len(atoms_obj))])
    Mo_adp.calculate(properties='energy', atoms=atoms_obj)   # 测试例子：调试ase的源代码，查看我的计算结果是否正确

 
    # cutoff = 3  # 截断半径
    cutoff = 6.5000000000000000e+00  # Mo的adp势文件中实际的截断半径
    TorchNeigh(atoms_obj, cutoff=cutoff)

    neighbor_list = TorchNeighborList(cutoff=cutoff)   # 已经设置好了，计算截断半径内的粒子对(i, j)
    # 计算近邻列表
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)
    # 测试asap3的计算结果，也是一样的
    # pairs_asap3, pair_diff_asap3 = asap3_get_neighborlist(cutoff, atoms_obj)  

    print("\n<--------u11(rij)*dx, u11(rij)*dy, u11(rij)*dz-------->\n")

    # TODO: 目前mu_i_alpha和lam_i_alpha_beta的设计，只支持纯元素
    # 将 u11 扩展为形状 (3456, 1)
    print(pair_diff*u11(pair_dist)[:, np.newaxis])    # mu_i^α  # 这里没有累加，只是单纯的相乘。 或者这样写：u11(pair_dist)[:, np.newaxis] * pair_diff
    print((pair_diff*u11(pair_dist)[:, np.newaxis]).shape)  # u11(rij)*rij   # 3个方向: u11(rij)*dx, u11(rij)*dy, u11(rij)*dz
    mu_i_alpha = np.concatenate((pairs, pair_diff*u11(pair_dist)[:, np.newaxis]), axis=1) # mu_i_alpha[:, :2].astype(int)查看第1,2列为粒子对(i,j)的索引
    
    
    print("\n<--------w11(rij)*rij^αβ (α=0,1,2; β=0,1,2；代表x,y,z这3个方向)-------->\n")
    # 计算rij^αβ所有组合(对于每个粒子对(i, j)，可以得到9个分量。 resut这里看成是rij^α与rij^β的两两互乘，组成的数组
    result = np.einsum('ij,ik->ijk', pair_diff, pair_diff)  # 每个粒子对，都有一个(3x3)的矩阵（该矩阵类似于协方差矩阵一样）
    print('result.shape = ', result.shape)  # (N, 3, 3)  这里为(i, j)粒子对的个数
    

    #------------for plotting----------
    rij_list = []
    phi_AA_list = []  # Φ(rij)
    rho_i_list = []   # Electron Density rho(r) 
    emb_ener_list = [] # F(rho_i)
    _phi_AA_values = np.array([phi_AA(rij) for rij in pair_dist])
    _rho_i_values = np.array([rho(rij) for rij in pair_dist])
    _emb_ener_values = np.array([emb(rho_i) for rho_i in _rho_i_values])
    #------------for plotting----------
    
    cfg_energy = 0.0
    embedding_energy = 0.0  # 公式的第1项：∑F(rho_i)
    embed_ener_list = []
    pair_energy = 0.0       # 公式的第2项：∑Φ(rij) / 2
    mu_energy = 0.0         # 公式的第3项：∑mu_i^α^2 / 2
    lam_energy = 0.0        # 公式的第4项：∑lam_i^αβ^2 / 2
    trace_energy = 0.0      # 公式的第5项：-∑νi^2 / 6
    total_density = np.zeros(len(atoms_obj))
    # 参考：ase.calculators.eam.EAM中的函数和lammps的pair_adp.cpp代码
    mu = np.zeros([len(atoms_obj), 3])    # dx, dy, dz三个方向： ∑u11*dx, ∑u11*dy, ∑u11*dz # 注意：只统计[i,j]之间的,不需要再统计[j,i]之间的
    _lam = np.zeros([len(atoms_obj), 1, 6])  #  在写法1中：对应lammps代码中∑lam_i^αβ^2 / 2中dx*dx, dy*dy, dz*dz, dy*dz, dx*dz, dx*dy的写法
    lam_maxtri_9 = np.zeros([len(atoms_obj), 3, 3])  #  在写法2中：对应lammps代码中∑lam_i^αβ^2 / 2
    
    cfg_forces_analytic = np.zeros((len(atoms_obj), 3))   # 方法1：根据ADP势的力解析形式:得到该构型中每个原子的力
    cfg_forces_autodiff = np.zeros((len(atoms_obj), 3))   # 方法2：根据能量对rij的自动微分求导计算:得到该构型中每个原子的力


    if True:
        for atom_i in range(len(atoms_obj)):
            # 例如：选择atom_idx索引为0的粒子对
            idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
            
            phi_AA_values = phi_AA(pair_dist[idx_mask])
            pair_energy += np.sum(phi_AA_values) / 2.    # 公式的第2项：∑Φ(rij) / 2
            rho_i_values = rho(pair_dist[idx_mask])  #  ρ(rij)  对应ase中的self.electron_density[j_index](r[nearest][use])
            density = np.sum(rho_i_values, axis=0)   # ρ(i)  选择对应原子的rho_i
            total_density[atom_i] = np.sum(rho(pair_dist[idx_mask]), axis=0)   # 测试用： 每个中心原子i对应的ρ(i)的结果。
            emb_ener_values = emb_vec(density)  # emb_vec表示向量化的emb函数: F(rho_i)
            # TODO:  这里np.sum(emb_ener_values)应该没必要用np.sum()函数, 因为emb_ener_values本身就是一个标量
            # embedding_energy += np.sum(emb_ener_values)   # 公式的第1项：∑F(rho_i) 
            embedding_energy += emb_ener_values   # 公式的第1项：∑F(rho_i)
            embed_ener_list.append(emb_ener_values.tolist())   # 用于测试ase的结果: 每个中心原子i对应的F(rho_i))
            #------------

            
            mu_arr = mu_i_alpha[idx_mask][:, 2:5]   # 对应x, y, z三个方向的mu_i^α (α=0,1,2分别表示x, y, z方向)
            mu[atom_i] = np.sum(mu_arr, axis=0)  # keepdims可不加？结果也一样 # np.sum(mu_arr, axis=0, keepdims=True)
            
            # ---------------写法(1): 用于6分量的写法--------未实现成功!------------
            # TODO: 可以参考：https://gitlab.com/atomicrex/atomicrex/-/blob/master/src/arexlib/potentials/AngularDependentPotential.cpp?ref_type=heads#L35
            # # lam_i_alpha_beta为（N, 8）数组，包含了对角线元素. 前2列对应粒子对(i, j)，后6列对应lammps代码中dx*dx, dy*dy, dz*dz, dy*dz, dx*dz, dx*dy的写法

            # lam_arr = lam_i_alpha_beta[idx_mask]   # lam_arr[:, :2].astype(int) # 第1,2列为粒子对(i,j)的索引
            # lam_arr_alpha_alpha = lam_arr[:, 2:5]  # 对应dx*dx, dy*dy, dz*dz  公式(6)中的niu_i （未累加时）
            # lam_arr_alpha_beta = lam_arr[:, 5:8]   # 对应dy*dz, dx*dz, dx*dy. 公式(5)中的lam_i^αβ （未累加时）
            # _lam[atom_i][:, 0:3] = 0.5*np.sum(lam_arr_alpha_alpha*lam_arr_alpha_alpha, axis=0)  # (N, 3) # np.sum(np.square(lam_arr_alpha_alpha), axis=0)
            # _lam[atom_i][:, 3:6] = 1.0*np.sum(lam_arr_alpha_beta*lam_arr_alpha_beta, axis=0)    # (N, 3)
            # print("00000000-------------00000000-------------")
            
            # ------------------------------写法(2): 用于9分量的写法--------------------------------------
            # # lam_arr为（N, 3, 3）数组，包含了对角线元素. 对应x, y, z三个方向的lam_i^αβ 
            rvec= pair_diff[idx_mask]
            # 初始化 lam 矩阵
            lam = np.zeros([rvec.shape[0], 3, 3])
            r = pair_dist[idx_mask]
            qr = w11(r)
            # for alpha in range(3):
            #     for beta in range(3):
            #         lam[:, alpha, beta] += qr * rvec[:, alpha] * rvec[:, beta]
            
            # 使用 np.einsum 替换alpha和beta这2个for循环
            lam = np.einsum('i,ij,ik->ijk', qr, rvec, rvec)
            lam_maxtri_9[atom_i] = np.sum(lam, axis=0)  # 9个分量的矩阵. 对应ase中的self.lam[i]
            print(lam_maxtri_9[0])
            print(lam_maxtri_9.shape)
            # #-----------------------------------------------------------------------------------------

        print("\n------> embed_ener_list(即，每个中心原子i对应的F(rho_i)) ------ \n", embed_ener_list)  
        print("\n------> total_density(即，每个中心原子i对应的ρ(i))值----- \n", total_density)   
        print("mu = \n", mu)
        print("mu.shape = ", mu.shape)
        # 如果要自动微分求导，请确保mu是一个torch.Tensor类型的对象而不是普通的numpy数组  #torch.sum(mu ** 2) / 2.
        mu_energy += np.sum(mu ** 2) / 2.   # 公式的第3项：∑mu_i^α^2 / 2


        lam_energy += np.sum(lam_maxtri_9 ** 2) / 2.     # 公式的第4项：∑lam_i^αβ^2 / 2
        for i in range(len(atoms_obj)):  # this is the atom to be embedded
            trace_energy -= np.sum(lam_maxtri_9[i].trace() ** 2) / 6.     # 公式中的第5项： -∑νi^2 / 6

        
        cfg_energy =  pair_energy + embedding_energy + mu_energy + lam_energy + trace_energy
        print("cfg_energy = ", cfg_energy)












            

if __name__ == "__main__":
    Mo_adp_test()
    # Ta_adp_test()













"""
In [17]: # 假设以下变量已经定义
    ...: atoms_obj = [0, 1, 2, 3, 4]  # 示例原子对象列表
    ...: mu_i_alpha = np.array([
    ...:     [0, 0, 1.0, 2.0, 3.0],
    ...:     [0, 0, 4.0, 5.0, 6.0],
    ...:     [0, 0, 7.0, 8.0, 9.0],
    ...:     [1, 0, 1.5, 2.5, 3.5],
    ...:     [1, 0, 4.5, 5.5, 6.5],
    ...:     [1, 0, 7.5, 8.5, 9.5],
    ...:     [2, 0, 1.0, 2.0, 3.0],
    ...:     [2, 0, 4.0, 5.0, 6.0],
    ...:     [2, 0, 7.0, 8.0, 9.0]
    ...: ])
    ...:
    ...: # 初始化 mu 数组
    ...: mu = np.zeros((len(atoms_obj), 3))
    ...:
    ...: for atom_i in range(len(atoms_obj)):
    ...:     # 选择对应原子的索引
    ...:     idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
    ...:     mu_arr = mu_i_alpha[idx_mask][:, 2:5]  # 对应 x, y, z 三个方向的 mu_i^α (α=0,1,2 分别表示 x, y, z 方向)
    ...:     mu[atom_i] = np.sum(mu_arr, axis=0, keepdims=True)  # keepdims 可以不加，但保留维度更方便后续操作
    ...:
    ...: print("mu = \n", mu)
    ...: print("mu.shape = ", mu.shape)
mu =
 [[12.  15.  18. ]
 [13.5 16.5 19.5]
 [12.  15.  18. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]]
mu.shape =  (5, 3)
"""


