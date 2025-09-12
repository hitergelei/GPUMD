import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from ase.io import read
from ase.build import bulk, make_supercell
from ase.neighborlist import NeighborList
from ase.calculators.eam import EAM
import numpy as np
from torch_neigh import TorchNeighborList
from adp_func import emb, emb_vec, rho, phi_AA, u11, w11

# 定义神经网络模型
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if not x.dim() >= 1:
            x = x.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# TorchNeighborList 方法
def TorchNeigh(atoms_obj: Atoms, cutoff: float):
    neighbor_list = TorchNeighborList(cutoff=cutoff)
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)
    print("len(pairs) = ", len(pairs))
    print("Pairs:\n", pairs)
    print("Pair differences形状：", pair_diff.shape)
    print("Pair distances形状: ", pair_dist.shape)

def calculate_energy(atoms_obj, embedding_network):
    cutoff = 6.5000000000000000e+00
    TorchNeigh(atoms_obj, cutoff=cutoff)

    neighbor_list = TorchNeighborList(cutoff=cutoff)
    pairs, pair_diff, pair_dist = neighbor_list(atoms_obj)

    mu_i_alpha = np.concatenate((pairs, pair_diff * u11(pair_dist)[:, np.newaxis]), axis=1)

    rij_list = []
    phi_AA_list = []
    rho_i_list = []
    emb_ener_list = []

    cfg_energy = 0.0
    embedding_energy = 0.0
    pair_energy = 0.0
    mu_energy = 0.0
    lam_energy = 0.0
    trace_energy = 0.0
    total_density = np.zeros(len(atoms_obj))
    mu = np.zeros([len(atoms_obj), 3])
    lam_maxtri_9 = np.zeros([len(atoms_obj), 3, 3])

    for atom_i in range(len(atoms_obj)):
        idx_mask = mu_i_alpha[:, 0].astype(int) == atom_i

        phi_AA_values = phi_AA(pair_dist[idx_mask])
        pair_energy += np.sum(phi_AA_values) / 2.

        rho_i_values = rho(pair_dist[idx_mask])
        density = np.sum(rho_i_values, axis=0)
        total_density[atom_i] = np.sum(rho(pair_dist[idx_mask]), axis=0)

        emb_ener_values = emb_vec(density)
        embedding_energy += np.sum(emb_ener_values)

        mu_arr = mu_i_alpha[idx_mask][:, 2:5]
        mu[atom_i] = np.sum(mu_arr, axis=0)

        rvec = pair_diff[idx_mask]
        r = pair_dist[idx_mask]
        qr = w11(r)
        lam = np.einsum('i,ij,ik->ijk', qr, rvec, rvec)
        lam_maxtri_9[atom_i] = np.sum(lam, axis=0)

    mu_energy += np.sum(mu ** 2) / 2.
    lam_energy += np.sum(lam_maxtri_9 ** 2) / 2.

    for i in range(len(atoms_obj)):
        trace_energy -= np.sum(total_density[i] ** 2) / 6.

    cfg_energy = embedding_energy + pair_energy + mu_energy + lam_energy + trace_energy
    return cfg_energy

def train_model(data_loader, embedding_network, optimizer, criterion):
    embedding_network.train()
    total_loss = 0.0

    for atoms_obj, true_energy in data_loader:
        optimizer.zero_grad()

        predicted_energy = calculate_energy(atoms_obj, embedding_network)
        loss = criterion(torch.tensor(predicted_energy), torch.tensor(true_energy))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)




# TODO: 还未实现批量读取构型文件，这里的atoms_obj为单个构型中的原子数
def main():
    # 读取多个构型文件
    structures = read("/home/centos/hjchen/GNN_painn/neighborlist_code/src/extxyz_Mo/all_186_Mo_structs-with-54atoms.extxyz", index=":", format="extxyz")
    print("energy = ", structures[0].info['REF_energy'])  # ase 3.23.1之后的版本，只识别数据集里带REF_的物理量

    # 计算每个构型的真实能量
    Mo_adp = EAM(potential="./test_Mo_Ta_adp/Mo.adp", cutoff=6.5000000000000000e+00, skin=0, elements=['Mo'], form='adp')
    true_energies = [Mo_adp.get_potential_energy(atoms=atoms_obj) for atoms_obj in structures]  # 不使用Mo.adp势文件计算的能量

    # 创建数据集
    dataset = list(zip(structures, true_energies))

    # 定义数据加载器
    data_loader = [(atoms_obj, true_energy) for atoms_obj, true_energy in dataset]

    # 创建神经网络模型实例
    embedding_network = EmbeddingNetwork(input_size=1, hidden_size=10, output_size=1)

    # 定义优化器和损失函数
    optimizer = optim.Adam(embedding_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        avg_loss = train_model(data_loader, embedding_network, optimizer, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()