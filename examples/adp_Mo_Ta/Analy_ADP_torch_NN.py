import torch
import torch.nn as nn
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

# 定义神经网络模型
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 确保输入数据至少为一维
        if not x.dim() >= 1:
            x = x.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# TorchNeighborList 方法
def TorchNeigh(atoms_obj: Atoms, cutoff: float):
    print("\n----------------TorchNeighborList方法")
    # 实例化邻居列表类
    neighbor_list = TorchNeighborList(cutoff=cutoff)

    # 计算近邻列表
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)
    
    print("len(pairs) = ", len(pairs))
    # pairs 包含原子对的索引，pair_diff 是原子对之间的向量差，pair_dist 是原子对之间的距离
    print("Pairs:\n", pairs)
    print("Pair differences形状：", pair_diff.shape)
    print("Pair distances形状: ", pair_dist.shape)

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def Mo_adp_test():
    Mo_adp = EAM(potential="./test_Mo_Ta_adp/Mo.adp", cutoff=6.5000000000000000e+00, skin=0, elements=['Mo'], form='adp')
    # Mo_adp.plot('Mo_adp_test')  # 触发 Mo_rij.txt 和 Mo_Φij.txt 数据的本地保存
    
    # atoms_obj = read("Mo_structs_0.dat", format="lammps-data")   # 原来的
    atoms_obj = read("./extxyz_Mo/all_186_Mo_structs-with-54atoms.extxyz", format="extxyz", index=127)   # 读取的是extxyz格式的文件
    atoms_obj.set_chemical_symbols(['Mo' for _ in range(len(atoms_obj))])
    Mo_adp.calculate(properties='energy', atoms=atoms_obj)   # 测试例子：调试ase的源代码，查看我的计算结果是否正确

    cutoff = 6.5000000000000000e+00  # Mo 的 adp 势文件中实际的截断半径
    TorchNeigh(atoms_obj, cutoff=cutoff)

    neighbor_list = TorchNeighborList(cutoff=cutoff)   # 已经设置好了，计算截断半径内的粒子对(i, j)

    # 设置随机数种子
    setup_seed(20)
    
    # 计算近邻列表
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)

    print("\n<--------u11(rij)*dx, u11(rij)*dy, u11(rij)*dz-------->")
    print(pair_diff * u11(pair_dist)[:, np.newaxis])
    print((pair_diff * u11(pair_dist)[:, np.newaxis]).shape)

    mu_i_alpha = np.concatenate((pairs, pair_diff * u11(pair_dist)[:, np.newaxis]), axis=1)

    print("\n<--------w11(rij)*rij^αβ (α=0,1,2; β=0,1,2；代表x,y,z这3个方向)-------->")
    result = np.einsum('ij,ik->ijk', pair_diff, pair_diff)
    print('result.shape = ', result.shape)

    rij_list = []
    phi_AA_list = []  # Φ(rij)
    rho_i_list = []   # Electron Density rho(r) 
    emb_ener_list = [] # F(rho_i)

    cfg_energy = 0.0
    embedding_energy = 0.0  # 公式的第1项：∑F(rho_i)
    embed_ener_list = []
    pair_energy = 0.0       # 公式的第2项：∑Φ(rij) / 2
    mu_energy = 0.0         # 公式的第3项：∑mu_i^α^2 / 2
    lam_energy = 0.0        # 公式的第4项：∑lam_i^αβ^2 / 2
    trace_energy = 0.0      # 公式的第5项：-∑νi^2 / 6
    total_density = np.zeros(len(atoms_obj))
    # 参考：ase.calculators.eam.EAM 中的函数和 lammps 的 pair_adp.cpp 代码
    mu = np.zeros([len(atoms_obj), 3])    # dx, dy, dz 三个方向： ∑u11*dx, ∑u11*dy, ∑u11*dz # 注意：只统计[i,j]之间的,不需要再统计[j,i]之间的
    lam_maxtri_9 = np.zeros([len(atoms_obj), 3, 3])  #  在写法2中：对应lammps代码中∑lam_i^αβ^2 / 2

    cfg_forces_analytic = np.zeros((len(atoms_obj), 3))   # 方法1：根据ADP势的力解析形式:得到该构型中每个原子的力
    cfg_forces_autodiff = np.zeros((len(atoms_obj), 3))   # 方法2：根据能量对rij的自动微分求导计算:得到该构型中每个原子的力

    # 创建神经网络模型实例
    embedding_network = EmbeddingNetwork(input_size=1, hidden_size=10, output_size=1)

    # 加载预训练的权重（如果有的话）
    # embedding_network.load_state_dict(torch.load('embedding_network.pth'))

    # 替换解析式中的emb_vec函数
    def emb_vec_nn(density):
        # 确保密度数据至少为一维
        density = np.atleast_1d(density)
        density_tensor = torch.tensor(density, dtype=torch.float32)
        emb_ener_values = embedding_network(density_tensor).detach().numpy()  # 使用 detach()
        return emb_ener_values
    

    _phi_AA_values = np.array([phi_AA(rij) for rij in pair_dist])
    _rho_i_values = np.array([rho(rij) for rij in pair_dist])
    _emb_ener_values = np.array([emb(rho_i) for rho_i in _rho_i_values])

    for atom_i in range(len(atoms_obj)):
        # 例如：选择 atom_idx 索引为 0 的粒子对
        idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
        
        phi_AA_values = phi_AA(pair_dist[idx_mask])
        pair_energy += np.sum(phi_AA_values) / 2.    # 公式的第2项：∑Φ(rij) / 2
        rho_i_values = rho(pair_dist[idx_mask])  # ρ(rij)  对应ase中的self.electron_density[j_index](r[nearest][use])
        density = np.sum(rho_i_values, axis=0)   # ρ(i)  选择对应原子的rho_i
        total_density[atom_i] = np.sum(rho(pair_dist[idx_mask]), axis=0)   # 测试用： 每个中心原子i对应的ρ(i)的结果。
        # emb_ener_values = emb_vec_nn(density)  # emb_vec表示向量化的emb函数: F(rho_i)
        emb_ener_values = emb_vec(density)
        embedding_energy += np.sum(emb_ener_values)   # 公式的第1项：∑F(rho_i)
        embed_ener_list.append(emb_ener_values.tolist())   # 用于测试ase的结果: 每个中心原子i对应的F(rho_i))
        #------------

        mu_arr = mu_i_alpha[idx_mask][:, 2:5]   # 对应x, y, z三个方向的mu_i^α (α=0,1,2分别表示x, y, z方向)
        mu[atom_i] = np.sum(mu_arr, axis=0)  # keepdims可不加？结果也一样 # np.sum(mu_arr, axis=0, keepdims=True)

        rvec = pair_diff[idx_mask]
        r = pair_dist[idx_mask]
        qr = w11(r)
        lam = np.einsum('i,ij,ik->ijk', qr, rvec, rvec)
        lam_maxtri_9[atom_i] = np.sum(lam, axis=0)

    mu_energy += np.sum(mu ** 2) / 2   # 公式的第3项：∑mu_i^α^2 / 2
    lam_energy += np.sum(lam_maxtri_9 ** 2) / 2     # 公式的第4项：∑lam_i^αβ^2 / 2
    for i in range(len(atoms_obj)):  # this is the atom to be embedded
        # 公式的第5项：-∑νi^2 / 6
        trace_energy -= np.sum(lam_maxtri_9[i].trace()**2) / 6

    # 计算总能量
    cfg_energy = embedding_energy + pair_energy + mu_energy + lam_energy + trace_energy

    # 输出各个能量项
    print(f"Embedding Energy: {embedding_energy}")
    print(f"Pair Energy: {pair_energy}")
    print(f"Mu Energy: {mu_energy}")
    print(f"Lam Energy: {lam_energy}")
    print(f"Trace Energy: {trace_energy}")
    print(f"Total Configuration Energy: {cfg_energy}")






    # TODO: 计算配置的力  ---------代码是AI自动生成的，需要调整以适应实际情况---------
    # # 方法1：根据ADP势的力解析形式
    # for atom_i in range(len(atoms_obj)):
    #     idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i
    #     rvec = pair_diff[idx_mask]
    #     r = pair_dist[idx_mask]
    #     phi_AA_values = phi_AA(r)
    #     rho_i_values = rho(r)
    #     emb_ener_values = emb_vec(np.sum(rho_i_values))

    #     # 计算F'(rho_i)
    #     emb_prime = embedding_network(torch.tensor(emb_ener_values, dtype=torch.float32)).backward(torch.ones_like(emb_ener_values))
    #     emb_prime = emb_prime.numpy()

    #     # 计算力
    #     d_emb_d_rho = emb_prime * rho_i_values
    #     d_phi_AA_dr = -phi_AA_values / r
    #     d_rho_dr = -rho_i_values / r

    #     # 更新力
    #     cfg_forces_analytic[atom_i] += np.sum(d_emb_d_rho * d_rho_dr + d_phi_AA_dr * rvec, axis=0)

    # # 方法2：根据能量对rij的自动微分求导计算
    # # 这里可以使用PyTorch的自动微分特性来计算力
    # # 假设我们已经定义了所有相关的张量并将其设置为requires_grad=True
    # # 以下代码仅作示例，具体实现可能需要调整以适应实际情况
    # energy_tensor = torch.tensor(cfg_energy, requires_grad=True)
    # forces_tensor = torch.autograd.grad(energy_tensor, [pair_dist], grad_outputs=torch.ones_like(energy_tensor), create_graph=True)[0]

    # # 将梯度转换为numpy数组
    # cfg_forces_autodiff = forces_tensor.detach().numpy()

    # # 输出计算的力
    # print(f"Analytic Forces: {cfg_forces_analytic}")
    # print(f"Automatic Differentiation Forces: {cfg_forces_autodiff}")

if __name__ == "__main__":
    Mo_adp_test()
